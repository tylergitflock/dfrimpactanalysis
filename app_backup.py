import io
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="DFR Impact Analysis", layout="wide")

# -----------------------------
# UI – Uploads & Assumptions
# -----------------------------
st.title("DFR Impact Analysis")

with st.sidebar:
    st.header("1) Upload CSVs")
    raw_file   = st.file_uploader("Raw Call Data CSV", type=["csv"], key="raw")
    agency_file= st.file_uploader("Agency Call Types CSV", type=["csv"], key="agency")
    launch_file= st.file_uploader("Launch Locations CSV", type=["csv"], key="launch")
    alpr_file  = st.file_uploader("ALPR Data CSV (optional)", type=["csv"], key="alpr")
    audio_file = st.file_uploader("Audio Hits CSV (optional)", type=["csv"], key="audio")

    st.header("2) Assumptions")
    fte_hours   = st.number_input("Full Time Work Year (hours)", value=2080, step=1)
    officer_cost= st.number_input("Full Time Equivalent Cost of an Officer ($)", value=127940, step=1000, format="%d")
    cancel_rate = st.number_input("Drone Cancellation Rate (e.g., 0.11 = 11%)", value=0.11, min_value=0.0, max_value=1.0, step=0.01, format="%.2f")
    drone_speed = st.number_input("Drone Speed (mph)", value=51.0, step=1.0, format="%.0f")
    drone_range = st.number_input("Drone Range (miles)", value=3.5, step=0.1, format="%.1f")

st.markdown("---")

# -----------------------------
# Helpers (robust & vectorized)
# -----------------------------
EPOCH_1899 = np.datetime64("1899-12-30")

def parse_time_series(s: pd.Series) -> pd.Series:
    """
    Accepts Excel serial day numbers (e.g., 45474.05472) OR timestamp text (e.g., 2024-01-01 01:20:48).
    Returns pandas datetime64[ns].
    """
    s = s.astype("object")
    # try numeric (Excel serial days)
    num = pd.to_numeric(s, errors="coerce")
    dt_from_num = pd.to_datetime(EPOCH_1899) + pd.to_timedelta(num * 86400, unit="s")
    # try text
    as_text = s.where(num.isna(), None)  # None where numeric parsed
    dt_from_text = pd.to_datetime(as_text, errors="coerce", utc=False)

    # prefer whichever parsed
    dt = dt_from_num.where(~num.isna(), dt_from_text)
    return dt

def haversine_min_distance_miles(lat, lon, launches):
    """
    Vectorized: compute min Haversine distance from each (lat,lon) to any launch.
    launches: ndarray shape (M,2) [lat, lon]
    Returns: ndarray shape (N,) min distance in miles (np.nan if no launches or no lat/lon)
    """
    if launches.size == 0:
        return np.full(lat.shape, np.nan)

    # Build result as +inf then reduce launch by launch to keep memory low
    res = np.full(lat.shape, np.inf, dtype="float64")

    # radians
    lat1 = np.radians(lat)
    lon1 = np.radians(lon)

    for la, lo in launches:
        la_r = np.radians(la)
        lo_r = np.radians(lo)
        dlat = la_r - lat1
        dlon = lo_r - lon1
        a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(la_r)*(np.sin(dlon/2.0)**2)
        d = 2.0 * 3958.8 * np.arcsin(np.sqrt(a))  # miles
        res = np.minimum(res, d)

    # Where lat/lon invalid, set NaN
    res = np.where(np.isnan(lat) | np.isnan(lon), np.nan, res)
    return res

def fmt_mmss(seconds: float|int|None) -> str:
    if seconds is None or not np.isfinite(seconds):
        return ""
    seconds = int(round(seconds))
    m, s = divmod(seconds, 60)
    return f"{m}:{s:02d}"

def fmt_hhmmss(seconds: float|int|None) -> str:
    if seconds is None or not np.isfinite(seconds):
        return ""
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"

def try_read_csv(file, **kwargs):
    return pd.read_csv(file, **kwargs) if file is not None else None

def detect_alpr_totals_and_clearable_cols(df_alpr: pd.DataFrame):
    """
    ALPR sheet: A=address, B=Lat, C=Long, remaining columns vary.
    Last column is TOTAL hits per site.
    'Clearable' alert columns: headers containing 'custom' or 'stolen' (case-insensitive).
    Returns: (total_col_name, clearable_cols_list)
    """
    cols = list(df_alpr.columns)
    if len(cols) < 4:
        return None, []
    total_col = cols[-1]
    clearable = [c for c in cols[3:-1] if ("custom" in c.lower() or "stolen" in c.lower())]
    return total_col, clearable

# -----------------------------
# Load data
# -----------------------------
raw_df     = try_read_csv(raw_file)
agency_df  = try_read_csv(agency_file)
launch_df  = try_read_csv(launch_file)
alpr_df    = try_read_csv(alpr_file)
audio_df   = try_read_csv(audio_file)

# Validate required uploads
missing = []
if raw_df is None:     missing.append("Raw Call Data")
if agency_df is None:  missing.append("Agency Call Types")
if launch_df is None:  missing.append("Launch Locations")
if missing:
    st.info(f"Upload required CSVs in the sidebar: {', '.join(missing)}.")
    st.stop()

# -----------------------------
# Normalize column names & types
# -----------------------------
# Raw Call Data expected cols (support both your older and “Time …” names)
# We will map to canonical names: create, dispatch, arrive, close, type, priority, lat, lon
raw_cols = {c.lower(): c for c in raw_df.columns}

def pick_col(*aliases):
    for a in aliases:
        if a.lower() in raw_cols:
            return raw_cols[a.lower()]
    return None

col_create  = pick_col("Call Create", "Time Call Entered Queue")
col_arrive  = pick_col("First Arrive", "Time First Unit Arrived")
col_close   = pick_col("Call Close", "Fixed Time Call Closed")
col_type    = pick_col("Call Type")
col_pri     = pick_col("Call Priority")
col_lat     = pick_col("Lat")
col_lon     = pick_col("Lon")

required = [col_create, col_arrive, col_close, col_type, col_pri, col_lat, col_lon]
if any(c is None for c in required):
    st.error("Raw Call Data is missing one or more required columns: "
             "'Call Create'/'Time Call Entered Queue', 'First Arrive'/'Time First Unit Arrived', "
             "'Call Close'/'Fixed Time Call Closed', 'Call Type', 'Call Priority', 'Lat', 'Lon'.")
    st.stop()

# Parse times
create_dt = parse_time_series(raw_df[col_create])
arrive_dt = parse_time_series(raw_df[col_arrive])
close_dt  = parse_time_series(raw_df[col_close])

# Compute patrol response (create→arrive) and on-scene (close→arrive) in seconds
patrol_sec  = (arrive_dt - create_dt).dt.total_seconds()
onscene_sec = (close_dt - arrive_dt).dt.total_seconds()

# Lat/Lon numeric
lat = pd.to_numeric(raw_df[col_lat], errors="coerce")
lon = pd.to_numeric(raw_df[col_lon], errors="coerce")

# Launch coordinates ndarray
launch_coords = np.asarray(
    launch_df[[launch_df.columns[2], launch_df.columns[3]]].astype(float).values
) if launch_df.shape[1] >= 4 else np.empty((0,2))

# Distance to nearest launch & drone ETA
dist_mi = haversine_min_distance_miles(lat.values.astype(float), lon.values.astype(float), launch_coords)
drone_eta_sec = (dist_mi / max(drone_speed, 1e-9)) * 3600.0

# Agency flags
agency_df = agency_df.rename(columns=lambda c: c.strip())
if not set(["Call Type","DFR Response (Y/N)","Clearable (Y/N)"]).issubset(agency_df.columns):
    st.error("Agency Call Types must have columns: 'Call Type', 'DFR Response (Y/N)', 'Clearable (Y/N)'.")
    st.stop()

agency_df["_key"] = agency_df["Call Type"].astype(str).str.upper().str.strip()
dfr_map = set(agency_df.loc[agency_df["DFR Response (Y/N)"].astype(str).str.upper().str.strip()=="Y","_key"])
clr_map = set(agency_df.loc[agency_df["Clearable (Y/N)"].astype(str).str.upper().str.strip()=="Y","_key"])

call_type_up = raw_df[col_type].astype(str).str.upper().str.strip()

# -----------------------------
# Build filtered datasets
# -----------------------------
df_all = pd.DataFrame({
    "create_dt": create_dt,
    "arrive_dt": arrive_dt,
    "close_dt":  close_dt,
    "patrol_sec": patrol_sec,
    "onscene_sec": onscene_sec,
    "lat": lat,
    "lon": lon,
    "dist_mi": dist_mi,
    "drone_eta_sec": drone_eta_sec,
    "call_type_up": call_type_up,
    "priority": raw_df[col_pri].astype(str).str.strip()
})

# 1) Compute patrol_sec = arrive - create in seconds
df_all["patrol_sec"] = (df_all["arrive_dt"] - df_all["create_dt"]).dt.total_seconds()

# 2) DFR Only: must be flagged Y, have valid times, AND patrol_sec > 0
mask_dfr = (
    df_all["call_type_up"].isin(dfr_map)
    & df_all["patrol_sec"].notna()
    & (df_all["patrol_sec"] > 0)
)
dfr_only = df_all.loc[mask_dfr].copy()

# 3) In Range: DFR Only + valid distance + within drone_range
mask_in_range = (
    dfr_only["dist_mi"].notna()
    & (dfr_only["dist_mi"] <= drone_range)
)
in_range = dfr_only.loc[mask_in_range].copy()

# Clearable: In Range + clearable Y
mask_clr = in_range["call_type_up"].isin(clr_map)
clearable = in_range.loc[mask_clr].copy()

# -----------------------------
# ALPR & Audio (optional)
# -----------------------------
alpr_sites_in_range = 0
alpr_hits_in_range  = 0
alpr_weighted_eta   = np.nan
alpr_clearable_hits_in_range = 0

if alpr_df is not None and not alpr_df.empty:
    # Expect B=Lat, C=Long, last column is total hits, clearable columns contain 'custom' or 'stolen'
    alpr = alpr_df.copy()
    # Lat/Lon columns (B,C) => index 1,2
    if alpr.shape[1] < 3:
        st.warning("ALPR Data: expected at least 3 columns (Address, Lat, Long, ...). Skipping ALPR.")
    else:
        alpr_lat = pd.to_numeric(alpr.iloc[:,1], errors="coerce")
        alpr_lon = pd.to_numeric(alpr.iloc[:,2], errors="coerce")
        total_col, clearable_cols = detect_alpr_totals_and_clearable_cols(alpr)
        if total_col not in alpr.columns:
            st.warning("ALPR Data: could not detect total hits column (last col). Skipping ALPR.")
        else:
            total_hits = pd.to_numeric(alpr[total_col], errors="coerce").fillna(0.0).values
            # Compute nearest launch distance and ETA
            alpr_dist = haversine_min_distance_miles(alpr_lat.values.astype(float), alpr_lon.values.astype(float), launch_coords)
            inr = (alpr_dist <= drone_range) & np.isfinite(alpr_dist)
            alpr_sites_in_range = int(inr.sum())
            alpr_hits_in_range  = int(total_hits[inr].sum())
            # Weighted ETA by hits
            etas = (alpr_dist / max(drone_speed,1e-9)) * 3600.0
            num = (etas[inr] * total_hits[inr]).sum()
            den = total_hits[inr].sum()
            alpr_weighted_eta = (num/den) if den > 0 else np.nan
            # Clearable ALPR hits (sum of all 'custom' or 'stolen' alert columns)
            if clearable_cols:
                clr_hits = alpr[clearable_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1).values
                alpr_clearable_hits_in_range = int(clr_hits[inr].sum())

audio_sites_in_range = 0
audio_hits_in_range  = 0
audio_weighted_eta   = np.nan

if audio_df is not None and not audio_df.empty:
    # Expect: C=lat, D=long, E=hits per site
    if audio_df.shape[1] < 5:
        st.warning("Audio Hits: expected at least 5 columns (… , Lat, Long, Hits). Skipping Audio.")
    else:
        aud_lat  = pd.to_numeric(audio_df.iloc[:,2], errors="coerce")
        aud_lon  = pd.to_numeric(audio_df.iloc[:,3], errors="coerce")
        aud_hits = pd.to_numeric(audio_df.iloc[:,4], errors="coerce").fillna(0.0).values
        aud_dist = haversine_min_distance_miles(aud_lat.values.astype(float), aud_lon.values.astype(float), launch_coords)
        inr = (aud_dist <= drone_range) & np.isfinite(aud_dist)
        audio_sites_in_range = int(inr.sum())
        audio_hits_in_range  = int(aud_hits[inr].sum())
        etas = (aud_dist / max(drone_speed,1e-9)) * 3600.0
        num = (etas[inr] * aud_hits[inr]).sum()
        den = aud_hits[inr].sum()
        audio_weighted_eta = (num/den) if den > 0 else np.nan

# Combined hits for “DFR Responses to ALPR and Audio within Range”
dfr_alpr_audio_in_range = alpr_hits_in_range + audio_hits_in_range

# -----------------------------
# Metrics (match your order)
# -----------------------------
def avg_safe(series):
    s = pd.to_numeric(series, errors="coerce")
    s = s[np.isfinite(s)]
    return float(s.mean()) if len(s) else np.nan

# Averages
avg_drone_eta        = avg_safe(in_range["drone_eta_sec"])
avg_patrol_dfr       = avg_safe(dfr_only["patrol_sec"])
avg_scene_all_dfr    = avg_safe(dfr_only["onscene_sec"])
avg_patrol_in        = avg_safe(in_range["patrol_sec"])
avg_patrol_p1        = avg_safe(in_range.loc[in_range["priority"]=="1","patrol_sec"])
avg_drone_p1         = avg_safe(in_range.loc[in_range["priority"]=="1","drone_eta_sec"])
avg_scene_clearable  = avg_safe(clearable["onscene_sec"])

# Counts
total_cfs            = len(df_all)
total_potential_dfr  = len(dfr_only)
in_range_count       = len(in_range)
p1_in_range_count    = int((in_range["priority"]=="1").sum())
clearable_count      = len(clearable)

# First on scene %
if len(in_range):
    first_on_scene_pct = float(np.mean(in_range["drone_eta_sec"].values < in_range["patrol_sec"].values) * 100.0)
else:
    first_on_scene_pct = np.nan

# % Decrease vs patrol
pct_decrease = ((avg_patrol_in - avg_drone_eta) / avg_patrol_in * 100.0) if (avg_patrol_in and np.isfinite(avg_patrol_in) and avg_patrol_in>0) else np.nan

# Expected cleared (your rule = apply rate to in-range flights + alerts)
expected_cfs_cleared = int(round((in_range_count + dfr_alpr_audio_in_range) * cancel_rate))
expected_clearable_cfs_cleared = expected_cfs_cleared  # same rule per your latest decision

# Officers & ROI based on avg on-scene for clearable
time_saved_sec = (avg_scene_clearable or 0.0) * expected_clearable_cfs_cleared
officers_fte   = (time_saved_sec/3600.0)/fte_hours if fte_hours>0 else np.nan
roi_usd        = officers_fte * officer_cost if np.isfinite(officers_fte) else np.nan

# -----------------------------
# Report Values table (37 rows)
# -----------------------------
rows = [
    ("DFR Responses within Range",                                in_range_count,                          "int"),
    ("DFR Responses to ALPR and Audio within Range",              dfr_alpr_audio_in_range,                 "int"),
    ("Expected DFR Drone Response Times by Location",             avg_drone_eta,                           "mmss"),
    ("Expected First on Scene %",                                 first_on_scene_pct,                      "pct"),
    ("Expected CFS Cleared",                                      expected_cfs_cleared,                    "int"),
    ("Number of Officers - Force Multiplication",                 officers_fte,                            "2dec"),
    ("ROI from Potential Calls Cleared",                          roi_usd,                                 "usd"),
    ("Total CFS",                                                 total_cfs,                               "int"),
    ("Total Potential DFR Calls",                                 total_potential_dfr,                     "int"),
    ("DFR Responses within Range",                                in_range_count,                          "int"),
    ("DFR Responses to ALPR and Audio within Range",              dfr_alpr_audio_in_range,                 "int"),
    ("Total Potential DFR Calls",                                 total_potential_dfr,                     "int"),
    ("Avg Disp + Patrol Response Time to DFR Calls",              avg_patrol_dfr,                          "mmss"),
    ("Avg Time on Scene ALL DFR Calls",                           avg_scene_all_dfr,                       "mmss"),
    ("Expected DFR Drone Response Times by Location",             avg_drone_eta,                           "mmss"),
    ("Expected First on Scene %",                                 first_on_scene_pct,                      "pct"),
    ("Expected Decrease in Response Times",                       pct_decrease,                            "pct"),
    ("Avg Disp + Patrol Response Time to In-Range Calls",         avg_patrol_in,                           "mmss"),
    ("Expected DFR Drone Response Times by Location",             avg_drone_eta,                           "mmss"),
    ("Total DFR Calls In Range that are priority 1",              p1_in_range_count,                       "int"),
    ("Avg Disp + Patrol Response Time to In-Range P1 Calls",      avg_patrol_p1,                           "mmss"),
    ("Expected DFR Drone Response Times to P1 Calls",             avg_drone_p1,                            "mmss"),
    ("Hotspot location number of DFR calls within range",         np.nan,                                  "int"),   # placeholder
    ("Avg Disp + Pat to hotspot within range",                    np.nan,                                  "mmss"),  # placeholder
    ("Avg Expected drone response time to hotspot",               np.nan,                                  "mmss"),  # placeholder
    ("Number of ALPR Locations within range",                     alpr_sites_in_range,                     "int"),
    ("Number of Hits within range",                               alpr_hits_in_range,                      "int"),
    ("Expected response time to ALPR data",                       alpr_weighted_eta,                       "mmss"),
    ("Number of Audio Locations",                                 audio_sites_in_range,                    "int"),
    ("Number of hits within range",                               audio_hits_in_range,                     "int"),
    ("Avg expected resp time within range Audio Hits",            audio_weighted_eta,                      "mmss"),
    ("Expected Clearable CFS Cleared",                            expected_clearable_cfs_cleared,          "int"),
    ("Number of Officers - Force Multiplication",                 officers_fte,                            "2dec"),
    ("ROI from Potential Calls Cleared",                          roi_usd,                                 "usd"),
    ("Total clearable CFS within range",                          clearable_count,                         "int"),
    ("Total time spent on Clearable CFS",                         clearable["onscene_sec"].sum() if len(clearable) else np.nan, "hhmmss"),
    ("Avg Time on Scene – Clearable Calls",                       avg_scene_clearable,                     "mmss"),
]

def pretty_value(val, kind):
    if kind=="int":
        return f"{int(val):,}" if pd.notna(val) else ""
    if kind=="usd":
        return f"${int(round(val)):,}" if pd.notna(val) else ""
    if kind=="2dec":
        return f"{val:.2f}" if pd.notna(val) else ""
    if kind=="pct":
        return f"{val:.0f}%" if pd.notna(val) else ""
    if kind=="mmss":
        return fmt_mmss(val) if pd.notna(val) else ""
    if kind=="hhmmss":
        return fmt_hhmmss(val) if pd.notna(val) else ""
    return "" if pd.isna(val) else str(val)

report_df = pd.DataFrame(
    {
        "Metric": [r[0] for r in rows],
        "Function": [  # keep names for traceability; can be blanks if you prefer
            "inRangeCount","dfrAlprAudioInRange","avgDroneResp","expectedFirstOnPct",
            "expectedCfsCleared","officerFtes","roiPotentialCalls","totalCfs","totalPotentialDfr",
            "inRangeCount","dfrAlprAudioInRange","totalPotentialDfr","avgDispPatrol","avgSceneTime",
            "avgDroneResp","expectedFirstOnPct","pctDecrease","avgDispInRange","avgDroneResp",
            "p1Count","avgDispP1","avgDroneP1","hotspotCount","avgDispHotspot","avgDroneHotspot",
            "alprLocCount","alprHits","expRespAlpr","audioLocCount","audioHits","expRespAudioHits",
            "expectedClearableCfsCleared","officerFtes","roiPotentialCalls","clearableCount",
            "timeSpentClearable","avgSceneClr"
        ],
        "Description": [
            "Calls within drone range",
            "Sum of ALPR+Audio hits in range",
            "Avg drone ETA (sec) for in-range calls",
            "% in-range where drone < patrol",
            "Flights+alerts × cancellation rate",
            "FTEs saved by clearable calls",
            "$ savings from FTE reduction",
            "All calls for service",
            "DFR-eligible call count",
            "(dup) Calls within drone range",
            "(dup) Sum of hits",
            "(dup) DFR-eligible call count",
            "Avg patrol secs (create→arrive) for DFR calls",
            "Avg scene secs (close−arrive) for DFR calls",
            "(dup) Avg drone ETA",
            "(dup) % in-range where drone < patrol",
            "% reduction vs patrol response",
            "Avg patrol secs in-range",
            "(3rd) Avg drone ETA",
            "Priority-1 in-range call count",
            "Avg patrol secs P1",
            "Avg drone ETA P1",
            "Hotspot calls count",
            "Avg patrol secs to hotspot",
            "Avg drone ETA to hotspot",
            "ALPR sites in range",
            "ALPR hit count",
            "Hits-weighted avg drone ETA to ALPR",
            "Audio sites in range",
            "Audio hit count",
            "Hits-weighted avg drone ETA to audio",
            "(dup) Flights+alerts × cancellation rate",
            "(dup) FTEs saved",
            "(dup) $ savings",
            "Clearable calls in range",
            "Sum on-scene secs for clearable calls",
            "Avg scene secs for clearable calls",
        ],
        "Result": [pretty_value(r[1], r[2]) for r in rows],
    }
)

# -----------------------------
# Display & Downloads
# -----------------------------
st.subheader("Report Values")
st.dataframe(report_df, use_container_width=True)

# ESRI CSVs (with lat/lon)
def df_to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("DFR Only")
    if len(dfr_only):
        st.caption(f"Rows: {len(dfr_only):,}")
        st.download_button("Download DFR Only (CSV)", df_to_csv_bytes(dfr_only[["lat","lon","patrol_sec","drone_eta_sec","onscene_sec","priority","call_type_up"]]), file_name="dfr_only.csv", mime="text/csv")
    else:
        st.caption("No rows.")
with col2:
    st.subheader("DFR In Range")
    if len(in_range):
        st.caption(f"Rows: {len(in_range):,}")
        st.download_button("Download In Range (CSV)", df_to_csv_bytes(in_range[["lat","lon","patrol_sec","drone_eta_sec","onscene_sec","priority","call_type_up"]]), file_name="dfr_in_range.csv", mime="text/csv")
    else:
        st.caption("No rows.")
with col3:
    st.subheader("DFR Clearable")
    if len(clearable):
        st.caption(f"Rows: {len(clearable):,}")
        st.download_button("Download Clearable (CSV)", df_to_csv_bytes(clearable[["lat","lon","patrol_sec","drone_eta_sec","onscene_sec","priority","call_type_up"]]), file_name="dfr_clearable.csv", mime="text/csv")
    else:
        st.caption("No rows.")

# -----------------------------
# Diagnostics (expand if needed)
# -----------------------------
with st.expander("Diagnostics (click to expand)"):
    st.write("Counts:", {
        "total_cfs": total_cfs,
        "total_potential_dfr": total_potential_dfr,
        "in_range_count": in_range_count,
        "clearable_count": clearable_count,
        "p1_in_range_count": p1_in_range_count,
    })
    st.write("ALPR:", {
        "sites_in_range": alpr_sites_in_range,
        "hits_in_range": alpr_hits_in_range,
        "weighted_eta_sec": float(alpr_weighted_eta) if pd.notna(alpr_weighted_eta) else None,
        "clearable_hits_in_range": alpr_clearable_hits_in_range,
    })
    st.write("Audio:", {
        "sites_in_range": audio_sites_in_range,
        "hits_in_range": audio_hits_in_range,
        "weighted_eta_sec": float(audio_weighted_eta) if pd.notna(audio_weighted_eta) else None,
    })
    st.write("Averages (sec):", {
        "avg_drone_eta": avg_drone_eta,
        "avg_patrol_dfr": avg_patrol_dfr,
        "avg_scene_all_dfr": avg_scene_all_dfr,
        "avg_patrol_in": avg_patrol_in,
        "avg_patrol_p1": avg_patrol_p1,
        "avg_drone_p1": avg_drone_p1,
        "avg_scene_clearable": avg_scene_clearable,
    })

import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

# ─── Sidebar: heatmap controls ────────────────────────────────────────────────
st.sidebar.header("3) Heatmap Settings")
heat_radius = st.sidebar.slider(
    "Heatmap point radius", min_value=1, max_value=50, value=15, help="Radius of each point"
)
heat_blur = st.sidebar.slider(
    "Heatmap blur", min_value=1, max_value=50, value=25, help="Amount of blur"
)

# ─── Maps ────────────────────────────────────────────────────────────────────
def render_map(df_points, show_circles=False, heatmap=False, title="Map"):
    st.subheader(title)
    # center on first launch or mean of points
    if show_circles and launch_coords.size:
        center = [float(launch_coords[0][0]), float(launch_coords[0][1])]
    elif not df_points.empty:
        center = [float(df_points["lat"].mean()), float(df_points["lon"].mean())]
    else:
        center = [39.0, -98.5]  # fallback

    m = folium.Map(location=center, zoom_start=10)

    # draw drone‐range circles
    if show_circles:
        meters_per_mile = 1609.34
        for la, lo in launch_coords:
            folium.Circle(
                location=(float(la), float(lo)),
                radius=drone_range * meters_per_mile,
                color="blue",
                weight=2,
                fill=False,
            ).add_to(m)

    # add either markers or a heatmap
    if heatmap and not df_points.empty:
        heat_data = df_points[["lat","lon"]].dropna().values.tolist()
        HeatMap(heat_data, radius=heat_radius, blur=heat_blur).add_to(m)
    else:
        # scatter points
        for _, row in df_points.iterrows():
            folium.CircleMarker(
                location=(row["lat"], row["lon"]),
                radius=3,
                color="red",
                fill=True,
                fill_opacity=0.6,
            ).add_to(m)

    st_folium(m, width=800, height=500)

# 1) Total DFR calls city-wide (scatter of ALL DFR-only points)
render_map(
    df_points=dfr_only,
    show_circles=False,
    heatmap=False,
    title="All DFR Calls (City-wide)"
)

# 2) Just the drone-range circles (no points, no heat)
render_map(
    df_points=pd.DataFrame(),  # empty
    show_circles=True,
    heatmap=False,
    title="3.5-mile Drone Range around Launch Locations"
)

# 3) Heatmap of DFR calls within range
render_map(
    df_points=in_range,
    show_circles=False,
    heatmap=True,
    title="Heatmap: All In-Range DFR Calls"
)

# 4) Heatmap of priority-1 in-range calls
render_map(
    df_points=in_range.loc[in_range["priority"]=="1"],
    show_circles=False,
    heatmap=True,
    title="Heatmap: Priority-1 In-Range Calls"
)

# 5) Heatmap of ALPR locations
# we already parsed alpr_dist, so reconstruct a small DF
if alpr_df is not None:
    alpr_plot = pd.DataFrame({
        "lat": pd.to_numeric(alpr_df.iloc[:,1], errors="coerce"),
        "lon": pd.to_numeric(alpr_df.iloc[:,2], errors="coerce")
    })
    render_map(
        df_points=alpr_plot,
        show_circles=False,
        heatmap=True,
        title="Heatmap: ALPR Locations"
    )

# 6) Heatmap of clearable calls within range
render_map(
    df_points=clearable,
    show_circles=False,
    heatmap=True,
    title="Heatmap: Clearable In-Range Calls"
)

# 7) Heatmap of Audio Hit Locations
if audio_df is not None:
    # build a small DataFrame of lat/lon from your Audio Hits CSV
    audio_plot = pd.DataFrame({
        "lat": pd.to_numeric(audio_df.iloc[:,2], errors="coerce"),
        "lon": pd.to_numeric(audio_df.iloc[:,3], errors="coerce")
    }).dropna()

    render_map(
        df_points=audio_plot,
        show_circles=False,
        heatmap=True,
        title="Heatmap: Audio Locations"
    )
