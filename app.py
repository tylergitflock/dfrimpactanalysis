import io
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
import folium
from folium.plugins import HeatMap
import math

st.set_page_config(page_title="DFR Impact Analysis", layout="wide")

# ─── Detect available data‐editor API ────────────────────────────────────────
if hasattr(st, "data_editor"):
    _EDITOR = st.data_editor
elif hasattr(st, "experimental_data_editor"):
    _EDITOR = st.experimental_data_editor
else:
    _EDITOR = None

# ─── Helpers ─────────────────────────────────────────────────────────────────
EPOCH_1899 = np.datetime64("1899-12-30")

def parse_time_series(s: pd.Series) -> pd.Series:
    # 1) Turn everything into strings, strip out commas (thousands separators)
    s_str = s.astype(str).str.replace(",", "")
    # 2) Try numeric → Excel serial date
    num   = pd.to_numeric(s_str, errors="coerce")
    dt    = pd.to_datetime(EPOCH_1899) + pd.to_timedelta(num * 86400, unit="s")
    # 3) If numeric failed, fall back to parsing any actual datetime text
    txt   = pd.to_datetime(s_str.where(num.isna(), None), errors="coerce")
    # 4) Use dt when we got a number, otherwise use txt
    return dt.where(~num.isna(), txt)


def haversine_min(lat_arr, lon_arr, launches):
    res = np.full(lat_arr.shape, np.inf)
    lat1, lon1 = np.radians(lat_arr), np.radians(lon_arr)
    for la, lo in launches:
        la_r, lo_r = np.radians(la), np.radians(lo)
        dlat, dlon = la_r - lat1, lo_r - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(la_r)*(np.sin(dlon/2)**2)
        d = 2 * 3958.8 * np.arcsin(np.sqrt(a))
        res = np.minimum(res, d)
    res[np.isnan(lat_arr) | np.isnan(lon_arr)] = np.nan
    return res

def average(arr):
    vals = [v for v in arr if isinstance(v, (int, float)) and np.isfinite(v)]
    return float(np.mean(vals)) if vals else np.nan

def fmt_mmss(sec):
    if not np.isfinite(sec): return ""
    s = int(round(sec))
    m, s = divmod(s, 60)
    return f"{m}:{s:02d}"

def fmt_hhmmss(sec):
    if not np.isfinite(sec): return ""
    s = int(round(sec))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"

def pretty_value(val, kind):
    if pd.isna(val): return ""
    if kind=="int": return f"{int(val):,}"
    if kind=="usd": return f"${int(round(val)):,}"
    if kind=="2dec": return f"{val:.2f}"
    if kind=="pct": return f"{val:.0f}%"
    if kind=="mmss": return fmt_mmss(val)
    if kind=="hhmmss": return fmt_hhmmss(val)
    return str(val)

def to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")

def auto_heat_params(df, max_radius=50, max_blur=50):
    """
    Given a DataFrame of points, return a (radius, blur) pair 
    that scales inversely with point count so sparse maps still show color.
    """
    n = len(df)
    if n <= 1:
        return max_radius, max_blur
    base = max_radius * math.sqrt(1000 / n)
    radius = int(min(max_radius, max(5, base)))
    blur   = int(min(max_blur, max(5, base)))
    return radius, blur

# ─── 0) PROGRESS BAR ──────────────────────────────────────────────────────────
progress = st.sidebar.progress(0)

# ─── 1) SIDEBAR: UPLOADS & EDITORS ───────────────────────────────────────────
st.title("DFR Impact Analysis")

st.sidebar.header("1) Raw Call Data")
raw_file = st.sidebar.file_uploader("Upload Raw Call Data CSV", type=["csv"])
if not raw_file:
    st.sidebar.warning("Please upload Raw Call Data to proceed.")
    st.stop()
raw_df = pd.read_csv(raw_file)
raw_df_orig = raw_df.copy()
# ── normalize raw Call Type for lookup ──────────────────────────────────────
raw_df["Call Type"] = (
    raw_df["Call Type"]
      .astype(str)
      .str.strip()
      .str.upper()
)
progress.progress(10)

# 📌 keep the total number of CAD events before any filtering
raw_count = len(raw_df)

# ─── EXTRA STEP: BUILD & OFFER A “Call Types” TEMPLATE ────────────────────────
raw_types = (
    raw_df["Call Type"]
      .dropna()
      .unique()
)
raw_types.sort()

template = pd.DataFrame({
    "Call Type":           raw_types,
    "DFR Response (Y/N)":  ["" for _ in raw_types],
    "Clearable (Y/N)":     ["" for _ in raw_types]
})

csv_bytes = template.to_csv(index=False).encode("utf-8")
st.sidebar.download_button(
    "📥 Download Call-Types Template",
    data=csv_bytes,
    file_name="call_types_template.csv",
    mime="text/csv",
    help="Fill in Y/N for each row, then re-upload under “Agency Call Types.”"
)

# ─── 2) Launch Locations ────────────────────────────────────────────────────
st.sidebar.header("2) Launch Locations")

# 2a) Upload a CSV or edit in place:
launch_file = st.sidebar.file_uploader(
    "Upload Launch Locations CSV (with any of: Name, Address, Lat, Lon)",
    type=["csv"],
    key="launch_csv"
)
if launch_file:
    launch_df = pd.read_csv(launch_file)
else:
    if _EDITOR is None:
        st.sidebar.error("Upgrade Streamlit or upload a CSV with launch locations.")
        st.stop()
    launch_df = _EDITOR(
        pd.DataFrame(columns=["Location Name","Address","Lat","Lon"]),
        num_rows="dynamic",
        use_container_width=True,
        key="launch_editor"
    )

# 2b) Normalize column names and ensure all four exist
launch_df.columns = [c.strip() for c in launch_df.columns]
for col in ["Location Name","Address","Lat","Lon"]:
    if col not in launch_df:
        launch_df[col] = ""

# ─── 2c) Geocode any rows with Address but missing Lat/Lon ──────────────────
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

geolocator = Nominatim(user_agent="dfrimpact")
geocode    = RateLimiter(geolocator.geocode, min_delay_seconds=1)

@st.cache_data(show_spinner=False)
def lookup(addr):
    try:
        loc = geocode(addr)
        return (loc.latitude, loc.longitude) if loc else (None, None)
    except:
        return (None, None)

to_geocode = launch_df["Address"].notna() & (
    pd.to_numeric(launch_df["Lat"], errors="coerce").isna() |
    pd.to_numeric(launch_df["Lon"], errors="coerce").isna()
)
for idx in launch_df.loc[to_geocode].index:
    lat, lon = lookup(launch_df.at[idx, "Address"])
    launch_df.at[idx, "Lat"]  = lat
    launch_df.at[idx, "Lon"]  = lon

# show the updated table so you can verify any blanks
st.sidebar.subheader("Geocoded Launch Locations")
st.sidebar.dataframe(launch_df)

# 2d) Final validation: every row must now have numeric Lat & Lon
valid_coords = (
    pd.to_numeric(launch_df["Lat"], errors="coerce").notna() &
    pd.to_numeric(launch_df["Lon"], errors="coerce").notna()
)
if not valid_coords.all():
    bad = launch_df.loc[~valid_coords, ["Location Name","Address","Lat","Lon"]]
    st.sidebar.error(
        "Some rows still lack valid Lat/Lon:\n" +
        bad.to_csv(index=False)
    )
    st.stop()

# 2e) Convert and store launch_coords for downstream use
launch_df["Lat"] = pd.to_numeric(launch_df["Lat"], errors="coerce")
launch_df["Lon"] = pd.to_numeric(launch_df["Lon"], errors="coerce")
launch_coords   = list(launch_df[["Lat","Lon"]].itertuples(index=False, name=None))

progress.progress(30)

# ─── 3) Agency Call Types ──────────────────────────────────────────────────
st.sidebar.header("3) Agency Call Types")
ag_file = st.sidebar.file_uploader(
    "Upload Agency Call Types CSV", 
    type=["csv"], 
    key="agency_csv"
)
if not ag_file:
    st.sidebar.error("Please upload your Agency Call Types CSV.")
    st.stop()

agency_df = pd.read_csv(ag_file)
# normalize & preview
agency_df["Call Type"]          = agency_df["Call Type"].astype(str).str.strip().str.upper()
agency_df["DFR Response (Y/N)"] = agency_df["DFR Response (Y/N)"].astype(str).str.strip().str.upper()
agency_df["Clearable (Y/N)"]    = agency_df["Clearable (Y/N)"].astype(str).str.strip().str.upper()

st.sidebar.subheader("Agency Call Types Preview")
st.sidebar.dataframe(agency_df.head())

progress.progress(50)

agency_df["Call Type"]          = agency_df["Call Type"].astype(str).str.strip().str.upper()
agency_df["DFR Response (Y/N)"] = agency_df["DFR Response (Y/N)"].astype(str).str.strip().str.upper()
agency_df["Clearable (Y/N)"]    = agency_df["Clearable (Y/N)"].astype(str).str.strip().str.upper()

st.sidebar.header("4) Assumptions")
fte_hours    = st.sidebar.number_input("Full Time Work Year (hrs)", value=2080, step=1)
officer_cost = st.sidebar.number_input("Officer Cost per FTE ($)", value=127940, step=1000, format="%d")
cancel_rate  = st.sidebar.number_input("Drone Cancellation Rate (0–1)", value=0.11, step=0.01, format="%.2f")
drone_speed  = st.sidebar.number_input("Drone Speed (mph)", value=51.0, step=1.0)
drone_range  = st.sidebar.number_input("Drone Range (miles)", value=3.5, step=0.1)
progress.progress(70)

st.sidebar.header("5) ALPR & Audio (optional)")
alpr_file  = st.sidebar.file_uploader("Upload ALPR Data CSV", type=["csv"])
audio_file = st.sidebar.file_uploader("Upload Audio Hits CSV", type=["csv"])
progress.progress(80)

# ─── 6) Hotspot Area ──────────────────────────────────────────
st.sidebar.header("6) Hotspot Area")

# 1) Address input
hotspot_address = st.sidebar.text_input(
    "Enter Hotspot Address (0.5 mi radius)",
    help="e.g. “123 Main St, Anytown, USA”"
)

# 2) Manual fallback fields (only used if geocoding fails)
hotspot_lat_manual = st.sidebar.number_input("Manual Latitude (if geocoding fails)",  format="%.6f")
hotspot_lon_manual = st.sidebar.number_input("Manual Longitude (if geocoding fails)", format="%.6f")

hotspot_coords: list[tuple[float,float]] = []
if hotspot_address:
    coords = lookup(hotspot_address)  # (lat, lon) or (None, None)
    valid_geo = (
        coords is not None
        and isinstance(coords[0], (int,float)) and np.isfinite(coords[0])
        and isinstance(coords[1], (int,float)) and np.isfinite(coords[1])
    )
    if valid_geo:
        hotspot_coords = [coords]
    elif np.isfinite(hotspot_lat_manual) and np.isfinite(hotspot_lon_manual):
        hotspot_coords = [(float(hotspot_lat_manual), float(hotspot_lon_manual))]
    else:
        st.sidebar.warning("Could not geocode that address. Enter lat/lon manually if needed.")
        
# ─── 2) PARSE & COMPUTE ───────────────────────────────────────────────────────
col_map = {c.lower():c for c in raw_df.columns}
def pick(*alts):
    for a in alts:
        if a.lower() in col_map:
            return col_map[a.lower()]
    return None

col_create = pick("Call Create","Time Call Entered Queue")
col_arrive = pick("First Arrive","Time First Unit Arrived")
col_close  = pick("Call Close","Fixed Time Call Closed")
col_type   = pick("Call Type")
col_pri    = pick("Call Priority","Priority")
col_lat    = pick("Lat")
col_lon    = pick("Lon","Long")
col_dispatch = pick(
  "First Dispatch",
  "Time First Unit Assigned",
  "First Unit Assigned",
  "Time First Dispatch"
)

for c in [col_create, col_dispatch, col_arrive, col_close, col_type, col_pri, col_lat, col_lon]:
    if c is None:
        st.error("Missing required Raw Call Data columns.")
        st.stop()

create_dt   = parse_time_series(raw_df[col_create])
dispatch_dt = parse_time_series(raw_df[col_dispatch])
arrive_dt   = parse_time_series(raw_df[col_arrive])
close_dt    = parse_time_series(raw_df[col_close])

# — compute raw response seconds
response_sec = (arrive_dt - create_dt).dt.total_seconds()

# — validity mask: drop self-initiated, negatives, tiny times, NaTs
valid = (
     (dispatch_dt  > create_dt)         # real dispatch
  &  (arrive_dt    > dispatch_dt)       # real arrival
  &  (arrive_dt    > create_dt)         # no weird negatives
  &  (response_sec > 5)                 # ignore <= 5s “glitches”
  &  dispatch_dt.notna()
  &  arrive_dt.notna()
  &  create_dt.notna()
)

# — filter down your DataFrame & all series
raw_df      = raw_df.loc[valid].copy()
create_dt   = create_dt[valid]
dispatch_dt = dispatch_dt[valid]
arrive_dt   = arrive_dt[valid]
close_dt    = close_dt[valid]

patrol_sec  = (arrive_dt - create_dt).dt.total_seconds()
onscene_sec = (close_dt  - arrive_dt).dt.total_seconds()
lat = pd.to_numeric(raw_df[col_lat], errors="coerce")
lon = pd.to_numeric(raw_df[col_lon], errors="coerce")

progress.progress(85)

# build a proper Lat/Lon array by column name, after any geocoding has run
try:
    raw_coords = launch_df[["Lat","Lon"]].astype(float).values

    # ─── DROP any bad coordinates so Folium never sees a NaN
    launch_coords = [
        (lat, lon)
        for lat, lon in raw_coords
        if np.isfinite(lat) and np.isfinite(lon)
    ]

    if not launch_coords:
        st.sidebar.error("No valid launch locations available to draw the drone-range circle.")
        st.stop()

except Exception:
    st.sidebar.error("Couldn't parse ‘Lat’ and ‘Lon’ from Launch Locations — please ensure those columns exist and contain numeric values.")
    st.stop()

dist_mi       = haversine_min(lat.values, lon.values, launch_coords)
drone_eta_sec = dist_mi / max(drone_speed,1e-9) * 3600
progress.progress(90)

df_all = pd.DataFrame({
    "patrol_sec":    patrol_sec,
    "onscene_sec":   onscene_sec,
    "lat":           lat,
    "lon":           lon,
    "dist_mi":       dist_mi,
    "drone_eta_sec": drone_eta_sec,
    "call_type_up":  raw_df[col_type].astype(str).str.upper().str.strip(),
    "priority":      raw_df[col_pri].astype(str).str.strip(),
})

# normalize every column name to lowercase (and strip any stray spaces)
df_all.columns = df_all.columns.str.lower().str.strip()

# ─── 2f) Drop any rows with missing or non-positive patrol response
df_all = df_all[df_all["patrol_sec"] > 0].copy()

# ─── 3a) DEFINE DFR & CLEARABLE SETS & SUBSETS ───────────────────────────────
dfr_map = set(
    agency_df.loc[
        agency_df["DFR Response (Y/N)"].str.upper() == "Y",
        "Call Type"
    ]
    .str.upper().str.strip()
)
clr_map = set(
    agency_df.loc[
        agency_df["Clearable (Y/N)"].str.upper() == "Y",
        "Call Type"
    ]
    .str.upper().str.strip()
)

dfr_only  = df_all[
    df_all["call_type_up"].isin(dfr_map)
].copy()

in_range  = dfr_only[
    dfr_only["dist_mi"] <= drone_range
].copy()

clearable = in_range[
    in_range["call_type_up"].isin(clr_map)
].copy()
# ─── end subsets ─────────────────────────────────────────────────────────────

# now your existing hotspot code follows:
hotspot_count = 0
hotspot_avg_patrol = float("nan")
hotspot_avg_drone = float("nan")

if hotspot_coords:
    # compute every row’s distance to that single hotspot point
    all_hot_dists = haversine_min(
        df_all["lat"].values,
        df_all["lon"].values,
        hotspot_coords
    )
    # restrict to the DFR-only calls (or whichever set you prefer)
    mask = (
        (all_hot_dists <= 0.5)
        & df_all["patrol_sec"].gt(0)
        & df_all["call_type_up"].isin(dfr_map)
    )
    hotspot_df = df_all.loc[mask].copy()

    hotspot_count      = len(hotspot_df)
    hotspot_avg_patrol = average(hotspot_df["patrol_sec"])
    hotspot_avg_drone  = average(hotspot_df["drone_eta_sec"])
progress.progress(95)
# ─── 3) OPTIONAL ALPR & AUDIO METRICS (new ALPR format) ────────────────────
alpr_df = pd.read_csv(alpr_file) if alpr_file else None

if alpr_df is not None:
    try:
        st.sidebar.write("ALPR rows loaded:", alpr_df.shape[0])
        st.sidebar.write("ALPR columns:", alpr_df.columns.tolist())
        st.sidebar.write("ALPR sample rows:", alpr_df.head(3))
    except Exception as e:
        st.sidebar.error(f"Debug block failed: {e}")

audio_df = pd.read_csv(audio_file) if audio_file else None


# initialize metrics
alpr_sites = alpr_hits = alpr_eta = 0
audio_sites = audio_hits = audio_eta = 0
audio_pts   = None

# --- ALPR metrics (sites = all in-range; hits/ETA = whitelist in-range) ---
alpr_sites = alpr_hits = 0
alpr_eta   = np.nan
alpr_pts   = pd.DataFrame()

if alpr_df is not None and not alpr_df.empty:
    # Columns per new export:
    # A: Camera Name (idx 0)
    # D: Reason      (idx 3)
    # E: Latitude    (idx 4)
    # F: Longitude   (idx 5)
    # G: Sum of hits (idx 6)

    cam_names = alpr_df.iloc[:, 0].astype(str).str.strip()
    reasons   = alpr_df.iloc[:, 3].astype(str).str.upper().str.strip()
    lat_a     = pd.to_numeric(alpr_df.iloc[:, 4], errors="coerce").values
    lon_a     = pd.to_numeric(alpr_df.iloc[:, 5], errors="coerce").values
    hits      = pd.to_numeric(alpr_df.iloc[:, 6], errors="coerce").fillna(0).values

    # distance to nearest launch & in-range mask
    dist = haversine_min(lat_a, lon_a, launch_coords)
    in_range_mask = (dist <= drone_range) & np.isfinite(dist)

    # 1) SITES: count ALL unique camera names that are in range (no reason filter)
    alpr_sites = int(pd.Series(cam_names[in_range_mask]).nunique())

    # whitelist for hits / ETA only
    alpr_reason_set = {
        "GANG OR SUSPECTED TERRORIST",
        "MISSING PERSON",
        "HOTLIST HITS",
        "SEX OFFENDER",
        "STOLEN PLATE",
        "STOLEN VEHICLE",
        "VIOLENT PERSON"
    }
    ok_reason = pd.Series(reasons).isin(alpr_reason_set).values
    ok_mask   = in_range_mask & ok_reason

    # 2) HITS: only whitelisted reasons in range
    alpr_hits = int(hits[ok_mask].sum())

    # 3) ETA (hits-weighted) for whitelisted, in-range rows
    etas_sec = dist / max(drone_speed, 1e-9) * 3600
    denom = hits[ok_mask].sum()
    alpr_eta = float((etas_sec[ok_mask] * hits[ok_mask]).sum() / denom) if denom > 0 else np.nan

    # heatmap points (all valid coords for map, no filters)
    alpr_pts = pd.DataFrame({
        "lat": pd.to_numeric(alpr_df.iloc[:, 4], errors="coerce"),
        "lon": pd.to_numeric(alpr_df.iloc[:, 5], errors="coerce")
    }).dropna()

# --- Audio metrics (stats only for in-range; heatmap uses ALL valid points) ---
audio_sites = audio_hits = audio_eta = 0
audio_pts   = None

if audio_df is not None and not audio_df.empty:
    # Flexible column picking (case/space-insensitive)
    def pick_col(df, names):
        colmap = {c.lower().strip(): c for c in df.columns}
        for n in names:
            if n.lower().strip() in colmap:
                return colmap[n.lower().strip()]
        return None

    # Expected headers (with tolerant fallbacks)
    addr_col = pick_col(audio_df, ["Address", "Site", "Location"])
    lat_col  = pick_col(audio_df, ["Hit Latitude", "Latitude", "Lat"])
    lon_col  = pick_col(audio_df, ["Hit Longitude", "Longitude", "Lon", "Long", "Lng"])
    cnt_col  = pick_col(audio_df, ["Number of hits", "Count", "Count of Audio Hit Id", "Hits", "Hit_Count"])

    # Hard-fail if we’re missing required columns
    missing = []
    if addr_col is None: missing.append("Address")
    if lat_col  is None: missing.append("Hit Latitude")
    if lon_col  is None: missing.append("Hit Longitude")
    if cnt_col  is None: missing.append("Number of hits")
    if missing:
        st.sidebar.error("Audio CSV is missing required column(s): " + ", ".join(missing))
    else:
        # Parse columns
        addr  = audio_df[addr_col].astype(str).str.strip()
        lat_b = pd.to_numeric(audio_df[lat_col], errors="coerce")
        lon_b = pd.to_numeric(audio_df[lon_col], errors="coerce")
        hits  = pd.to_numeric(audio_df[cnt_col], errors="coerce").fillna(0)

        # Valid rows must have coords
        valid = lat_b.notna() & lon_b.notna()
        addr_v = addr[valid]
        lat_v  = lat_b[valid]
        lon_v  = lon_b[valid]
        hits_v = hits[valid]

        # Distances to nearest launch + in-range mask (stats use in-range only)
        dist2   = haversine_min(lat_v.values, lon_v.values, launch_coords)
        in_rng2 = (dist2 <= drone_range) & np.isfinite(dist2)

        # ── METRICS (IN-RANGE) ───────────────────────────────────────────────
        # 1) Unique audio locations (addresses) in range
        audio_sites = int(addr_v[in_rng2].nunique())

        # 2) Total hits in range
        audio_hits = int(hits_v[in_rng2].sum())

        # 3) Hits-weighted average ETA in range
        etas_sec = dist2 / max(drone_speed, 1e-9) * 3600  # seconds
        w_sum    = hits_v[in_rng2].sum()
        audio_eta = float((etas_sec[in_rng2] * hits_v[in_rng2]).sum() / w_sum) if w_sum > 0 else np.nan

        # ── DEBUG: total unfiltered hits (all valid rows, regardless of range) ─
        total_audio_hits = int(hits_v.sum())
        st.sidebar.write(f"Total unfiltered Audio hits: {total_audio_hits:,}")

        # ── HEATMAP DATA = ALL VALID POINTS (not only in-range) ───────────────
        audio_pts = pd.DataFrame({
            "lat":   lat_v.values,
            "lon":   lon_v.values,
            "count": hits_v.values,   # optional intensity for HeatMap
        })
else:
    # keep previous total display consistent even when no file
    st.sidebar.write("Total unfiltered Audio hits: 0")

# combine for your overall “DFR + ALPR + Audio” metric
dfr_alpr_audio = alpr_hits + audio_hits

# --- Unfiltered totals for ALPR + Audio (for the report/debug) ---
total_alpr_hits = 0
if alpr_df is not None and not alpr_df.empty:
    # treat col 1/2 as lat/lon; only count rows that look like real points
    alpr_lat_ok = pd.to_numeric(alpr_df.iloc[:, 1], errors="coerce").notna()
    total_alpr_hits = int(
        pd.to_numeric(alpr_df.iloc[:, 3], errors="coerce").fillna(0)[alpr_lat_ok].sum()
    )

total_audio_hits = 0
if audio_df is not None and not audio_df.empty:
    # find a hits column robustly (your new export uses "Number of hits")
    lc = {c.strip().lower(): c for c in audio_df.columns}
    hits_col = None
    for k in ["number of hits", "count of audio hit id", "hits", "count"]:
        if k in lc:
            hits_col = lc[k]; break
    total_audio_hits = int(pd.to_numeric(audio_df[hits_col], errors="coerce").fillna(0).sum()) if hits_col else len(audio_df)

total_alpr_audio = total_alpr_hits + total_audio_hits

# optional sidebar debug
st.sidebar.write(f"Total unfiltered ALPR hits: {total_alpr_hits:,}")
st.sidebar.write(f"Total unfiltered Audio hits: {total_audio_hits:,}")

# ─── 4) METRICS & REPORT ─────────────────────────────────────────────────────
total_cfs   = raw_count
total_dfr   = len(dfr_only)
in_count    = len(in_range)
clr_count   = len(clearable)
p1_count    = int((in_range["priority"]=="1").sum())

avg_drone    = average(in_range["drone_eta_sec"])
avg_patrol   = average(dfr_only["patrol_sec"])
avg_scene    = average(dfr_only["onscene_sec"])
avg_in       = average(in_range["patrol_sec"])
avg_p1_pat   = average(in_range.loc[in_range["priority"]=="1","patrol_sec"])
avg_p1_drone = average(in_range.loc[in_range["priority"]=="1","drone_eta_sec"])
avg_clr      = average(clearable["onscene_sec"])

first_on_pct = float(np.mean(in_range["drone_eta_sec"] < in_range["patrol_sec"]) * 100) if in_count else np.nan
pct_dec      = ((avg_in-avg_drone)/avg_in*100) if avg_in>0 else np.nan
exp_cleared  = int(round((in_count + dfr_alpr_audio) * cancel_rate))
time_saved   = avg_clr * exp_cleared
officers     = (time_saved/3600)/fte_hours if fte_hours>0 else np.nan
roi          = officers * officer_cost if np.isfinite(officers) else np.nan

rows = [
    ("Total CFS",                                           total_cfs,       "int"),
    ("Total ALPR + Audio Hits ",                            total_alpr_audio,"int"),
    ("DFR Responses within Range",                          in_count,        "int"),
    ("DFR Responses to ALPR and Audio within Range",        dfr_alpr_audio,  "int"),
    ("Expected DFR Drone Response Times by Location",       avg_drone,       "mmss"),
    ("Expected First on Scene %",                           first_on_pct,    "pct"),
    ("Expected CFS Cleared",                                exp_cleared,     "int"),
    ("Number of Officers - Force Multiplication",           officers,        "2dec"),
    ("ROI from Potential Calls Cleared",                    roi,             "usd"),
    ("Total CFS",                                           total_cfs,       "int"),
    ("Total Potential DFR Calls",                           total_dfr,       "int"),
    ("DFR Responses within Range",                          in_count,        "int"),  # repeated on purpose per original
    ("DFR Responses to ALPR and Audio within Range",        dfr_alpr_audio,  "int"),
    ("Total Potential DFR Calls",                           total_dfr,       "int"),  # repeated on purpose per original
    ("Avg Disp + Patrol Response Time to DFR Calls",        avg_patrol,      "mmss"),
    ("Avg Time on Scene ALL DFR Calls",                     avg_scene,       "mmss"),
    ("Expected DFR Drone Response Times by Location",       avg_drone,       "mmss"),
    ("Expected First on Scene %",                           first_on_pct,    "pct"),
    ("Expected Decrease in Response Times",                 pct_dec,         "pct"),
    ("Avg Disp + Patrol Response Time to In-Range Calls",   avg_in,          "mmss"),
    ("Expected DFR Drone Response Times by Location",       avg_drone,       "mmss"),  # repeated again by original order
    ("Total DFR Calls In Range that are priority 1",        p1_count,        "int"),
    ("Avg Disp + Patrol Response Time to In-Range P1 Calls",avg_p1_pat,      "mmss"),
    ("Expected DFR Drone Response Times to P1 Calls",       avg_p1_drone,    "mmss"),
    ("Hotspot location number of DFR calls within range",   hotspot_count,      "int"),
    ("Avg Disp + Pat to hotspot within range",              hotspot_avg_patrol, "mmss"),
    ("Avg Expected drone response time to hotspot",         hotspot_avg_drone,  "mmss"),
    ("Number of ALPR Locations within range",               alpr_sites,      "int"),
    ("Number of Hits within range",                         alpr_hits,       "int"),
    ("Expected response time to ALPR data",                 alpr_eta,        "mmss"),
    ("Number of Audio Locations",                           audio_sites,     "int"),
    ("Number of hits within range",                         audio_hits,      "int"),
    ("Avg expected resp time within range Audio Hits",      audio_eta,       "mmss"),
    ("Expected Clearable CFS Cleared",                      exp_cleared,     "int"),   # same as earlier, used again
    ("Number of Officers - Force Multiplication",           officers,        "2dec"),  # same again per original
    ("ROI from Potential Calls Cleared",                    roi,             "usd"),   # same again per original
    ("Total clearable CFS within range",                    clr_count,       "int"),
    ("Total time spent on Clearable CFS",                   clr_count * avg_clr, "hhmmss"),
    ("Avg Time on Scene – Clearable Calls",                 avg_clr,         "mmss"),
]

report_df = pd.DataFrame({
    "Metric": [r[0] for r in rows],
    "Result": [pretty_value(r[1],r[2]) for r in rows],
})
st.subheader("Report Values")
st.dataframe(report_df, use_container_width=True)

# ─── AUDIT MODE ──────────────────────────────────────────────────────────────
with st.sidebar.expander("🔎 Audit Mode", expanded=False):
    audit_on = st.checkbox("Enable audit diagnostics", value=False)

if audit_on:
    st.markdown("### Audit — Core Subsets")
    def n(x): return 0 if x is None else (len(x) if hasattr(x, "__len__") else int(x))

    c1, c2, c3 = st.columns(3)
    c1.metric("Raw rows (uploaded)", f"{raw_count:,}")
    c2.metric("df_all (validity filtered)", f"{len(df_all):,}")
    c3.metric("DFR-eligible (dfr_only)", f"{len(dfr_only):,}")

    # --- CFS filtering audit ---
    with st.expander("CFS filtering audit (pre-filter data)", expanded=False):
        col_map0 = {c.lower(): c for c in raw_df_orig.columns}
        def pick0(*alts):
            for a in alts:
                if a.lower() in col_map0:
                    return col_map0[a.lower()]
            return None

        c_create0 = pick0("Call Create", "Time Call Entered Queue")
        c_dispatch0 = pick0("First Dispatch", "Time First Unit Assigned", "First Unit Assigned", "Time First Dispatch")
        c_arrive0 = pick0("First Arrive", "Time First Unit Arrived")

        if c_create0 and c_dispatch0 and c_arrive0:
            create_all   = parse_time_series(raw_df_orig[c_create0])
            dispatch_all = parse_time_series(raw_df_orig[c_dispatch0])
            arrive_all   = parse_time_series(raw_df_orig[c_arrive0])
            response_all = (arrive_all - create_all).dt.total_seconds()

            missing_ts        = create_all.isna() | dispatch_all.isna() | arrive_all.isna()
            bad_dispatch      = dispatch_all <= create_all
            bad_arr_vs_dis    = arrive_all <= dispatch_all
            bad_arr_vs_create = arrive_all <= create_all
            too_fast          = response_all <= 5

            fail_any = missing_ts | bad_dispatch | bad_arr_vs_dis | bad_arr_vs_create | too_fast
            pass_all = ~fail_any

            only_missing    = missing_ts & ~(bad_dispatch | bad_arr_vs_dis | bad_arr_vs_create | too_fast)
            only_dispatch   = bad_dispatch & ~(missing_ts | bad_arr_vs_dis | bad_arr_vs_create | too_fast)
            only_arr_vs_dis = bad_arr_vs_dis & ~(missing_ts | bad_dispatch | bad_arr_vs_create | too_fast)
            only_arr_vs_cr  = bad_arr_vs_create & ~(missing_ts | bad_dispatch | bad_arr_vs_dis | too_fast)
            only_fast       = too_fast & ~(missing_ts | bad_dispatch | bad_arr_vs_dis | bad_arr_vs_create)
            multi_fail      = fail_any & ~(only_missing | only_dispatch | only_arr_vs_dis | only_arr_vs_cr | only_fast)

            st.write(f"Raw rows: {len(raw_df_orig):,}")
            st.write(f"Rows passing validity: {int(pass_all.sum()):,}  (df_all rows = {len(df_all):,})")
            st.write(f"Rows failing ANY rule: {int(fail_any.sum()):,}")

            st.markdown("**Failed exactly one rule:**")
            st.write(f"• Missing timestamp(s): {int(only_missing.sum()):,}")
            st.write(f"• Dispatch not after create: {int(only_dispatch.sum()):,}")
            st.write(f"• Arrive not after dispatch: {int(only_arr_vs_dis.sum()):,}")
            st.write(f"• Arrive not after create: {int(only_arr_vs_cr.sum()):,}")
            st.write(f"• Response ≤ 5s: {int(only_fast.sum()):,}")
            st.write(f"• Failed multiple rules (overlap): {int(multi_fail.sum()):,}")
        else:
            st.warning("CFS audit skipped: missing Create, Dispatch, or Arrive column.")

    # --- Main subset metrics ---
    c1, c2, c3 = st.columns(3)
    c1.metric("In-range (<= range mi)", f"{len(in_range):,}")
    c2.metric("Clearable in-range", f"{len(clearable):,}")
    c3.metric("Hotspot DFR (≤ 0.5 mi)", f"{hotspot_count:,}")

    st.markdown("### Audit — Audio & ALPR")
    c1, c2, c3 = st.columns(3)
    c1.metric("Audio sites (in-range)", f"{audio_sites:,}")
    c2.metric("Audio hits (in-range)", f"{audio_hits:,}")
    c3.metric("Audio ETA (mm:ss)", pretty_value(audio_eta, "mmss"))

    c1, c2, c3 = st.columns(3)
    c1.metric("ALPR sites (in-range)", f"{alpr_sites:,}")
    c2.metric("ALPR hits (in-range, reason-filtered)", f"{alpr_hits:,}")
    c3.metric("ALPR ETA (mm:ss)", pretty_value(alpr_eta, "mmss"))

    st.markdown("### Audit — Spot-check filters")
    st.write("- **DFR map values** (first 15):", list(sorted(dfr_map))[:15])
    st.write("- **Clearable map values** (first 15):", list(sorted(clr_map))[:15])

    st.markdown("**Examples feeding each metric (first 5 rows)**")
    st.write("• `in_range` (DFR Responses within Range / first-on-scene / avg drone time):")
    st.dataframe(in_range[["lat", "lon", "dist_mi", "drone_eta_sec", "patrol_sec", "call_type_up", "priority"]].head())

    st.write("• `clearable` (clearable metrics):")
    st.dataframe(clearable[["onscene_sec", "call_type_up", "priority", "dist_mi"]].head())

    if audio_pts is not None:
        st.write("• Audio (raw valid rows used for stats; intensity=Number of Hits):")
        st.dataframe(audio_pts.head())

    try:
        _alat = pd.to_numeric(alpr_df["Latitude"], errors="coerce")
        _alon = pd.to_numeric(alpr_df["Longitude"], errors="coerce")
        _hits = pd.to_numeric(alpr_df["Sum"], errors="coerce").fillna(0)
        _rsn  = alpr_df["Reason"].astype(str).str.upper().str.strip()
        _valid = _alat.notna() & _alon.notna()
        _dist  = haversine_min(_alat[_valid].values, _alon[_valid].values, launch_coords)
        _inrng = (_dist <= drone_range) & np.isfinite(_dist)
        _dfprev = pd.DataFrame({
            "lat": _alat[_valid].values,
            "lon": _alon[_valid].values,
            "dist_mi": _dist,
            "hits": _hits[_valid].values,
            "reason_up": _rsn[_valid].values,
            "in_range": _inrng,
            "whitelist_reason": _rsn[_valid].isin({
                "GANG OR SUSPECTED TERRORIST", "MISSING PERSON", "HOTLIST HITS",
                "SEX OFFENDER", "STOLEN PLATE", "STOLEN VEHICLE", "VIOLENT PERSON"
            }).values
        }).head(10)
        st.write("• ALPR preview (distance, in-range, whitelist flags):")
        st.dataframe(_dfprev)
    except Exception:
        pass

    st.markdown("### Audit — Key derived values")
    st.write({
        "dfr_alpr_audio (in-range hits)": int(dfr_alpr_audio),
        "Expected CFS Cleared": int(exp_cleared),
        "Officers (FTE)": officers,
        "ROI": roi,
    })

# ─── 5) CSV DOWNLOADS ────────────────────────────────────────────────────────
st.subheader("ESRI CSV Exports")
cols = ["lat","lon","patrol_sec","drone_eta_sec","onscene_sec","priority","call_type_up"]
c1,c2,c3 = st.columns(3)
with c1:
    st.download_button("Download DFR Only", to_csv_bytes(dfr_only[cols]), "dfr_only.csv")
with c2:
    st.download_button("Download In Range", to_csv_bytes(in_range[cols]), "in_range.csv")
with c3:
    st.download_button("Download Clearable", to_csv_bytes(clearable[cols]), "clearable.csv")
progress.progress(100)

# ─── 6) MAPS & HEATMAPS ──────────────────────────────────────────────────────
st.header("Maps & Heatmaps")

def render_map(
    df_pts,
    heat=False,
    title="",
    key=None,
    heat_radius=15,
    heat_blur=25,
    show_circle=False,
    launch_coords=None,
    hotspot_center=None,
    hotspot_radius=None,
):
    st.subheader(title)

    # drop any NaNs so folium never errors
    if {"lat","lon"}.issubset(df_pts.columns):
        df_pts = df_pts.dropna(subset=["lat","lon"])

    # determine map center
    if show_circle and launch_coords:
        center = [float(launch_coords[0][0]), float(launch_coords[0][1])]
    elif not df_pts.empty:
        center = [float(df_pts["lat"].mean()), float(df_pts["lon"].mean())]
    else:
        center = [0.0, 0.0]

    m = folium.Map(location=center, zoom_start=10)

    # blue 3.5 mi drone-range circle
    if show_circle and launch_coords:
        for la, lo in launch_coords:
            folium.Circle(
                location=(la, lo),
                radius=drone_range * 1609.34,
                color="blue",
                fill=False
            ).add_to(m)

    # red 0.5 mi hotspot circle
    if hotspot_center and hotspot_radius:
        folium.Circle(
            location=hotspot_center,
            radius=hotspot_radius * 1609.34,
            color="red",
            weight=3,
            fill=False
        ).add_to(m)

        # heat or points
    if heat and not df_pts.empty:
        # if you've passed in a 'count' column, use it as intensity
        if "count" in df_pts.columns:
            data = df_pts[["lat", "lon", "count"]].values.tolist()
        else:
            data = df_pts[["lat", "lon"]].values.tolist()

        HeatMap(
            data,
            radius=heat_radius,
            blur=heat_blur
        ).add_to(m)
    else:
        for _, r in df_pts.iterrows():
            folium.CircleMarker(
                location=(r["lat"], r["lon"]),
                radius=3,
                color="red",
                fill=True,
                fill_opacity=0.6
            ).add_to(m)

    st_folium(m, width=800, height=500, key=key)


# prepare full-city subsets
all_dfr       = dfr_only.copy()
all_p1        = all_dfr[all_dfr["priority"]=="1"]
all_clearable = all_dfr[all_dfr["call_type_up"].isin(clr_map)]

# compute hotspot subset (0.5 mi) if provided
if hotspot_coords:
    hs_lat = all_dfr["lat"].values
    hs_lon = all_dfr["lon"].values
    hs_dist = haversine_min(hs_lat, hs_lon, hotspot_coords)
    hotspot_calls = all_dfr[hs_dist <= 0.5].copy()
else:
    hotspot_calls = pd.DataFrame()


# 6a) Heatmap: All DFR Calls
r0, b0 = auto_heat_params(all_dfr)
r_all = st.sidebar.slider("All DFR Calls Heat Radius", 1, 50, value=r0, key="all_r")
b_all = st.sidebar.slider("All DFR Calls Heat Blur",   1, 50, value=b0, key="all_b")
render_map(
    all_dfr,
    heat=True,
    heat_radius=r_all,
    heat_blur=b_all,
    title="Heatmap: All DFR Calls",
    key="map_all_heat",
    show_circle=True,
    launch_coords=launch_coords
)

# 6b) 3.5-mile Drone-Range Circle Only
render_map(
    pd.DataFrame(),
    heat=False,
    title="3.5-mile Drone Range",
    key="map_range_circle",
    show_circle=True,
    launch_coords=launch_coords
)

# 6c) Heatmap: All P1 DFR Calls
r1, b1 = auto_heat_params(all_p1)
r_p1 = st.sidebar.slider("P1 DFR Heat Radius", 1, 50, value=r1, key="p1_r")
b_p1 = st.sidebar.slider("P1 DFR Heat Blur",   1, 50, value=b1, key="p1_b")
render_map(
    all_p1,
    heat=True,
    heat_radius=r_p1,
    heat_blur=b_p1,
    title="Heatmap: All P1 DFR Calls",
    key="map_p1_heat",
    show_circle=True,
    launch_coords=launch_coords
)

# 6d) Heatmap: All DFR Calls + Hotspot Overlay
if hotspot_coords:
    # auto-compute a sensible default radius/blur
    r_hs, b_hs = auto_heat_params(all_dfr)

    # let the user tweak them
    r_hs = st.sidebar.slider(
        "Hotspot Heat Radius", 1, 50, value=r_hs, key="hs_r"
    )
    b_hs = st.sidebar.slider(
        "Hotspot Heat Blur",   1, 50, value=b_hs, key="hs_b"
    )

    # now render exactly the same heatmap as “All DFR Calls,” but with a red .5 mi hotspot circle
    render_map(
        all_dfr,
        heat=True,
        heat_radius=r_hs,
        heat_blur=b_hs,
        title="Heatmap: All DFR Calls + Hotspot",
        key="map_hotspot_heat",
        show_circle=True,                     # your 3.5 mi blue circle
        launch_coords=launch_coords,
        hotspot_center=hotspot_coords[0],     # safe because we’re inside `if hotspot_coords:`
        hotspot_radius=0.5                     # miles
    )

# 6e) Heatmap: All Clearable DFR Calls
r2, b2 = auto_heat_params(all_clearable)
r_cl = st.sidebar.slider("Clearable Heat Radius", 1, 50, value=r2, key="clr_r")
b_cl = st.sidebar.slider("Clearable Heat Blur",   1, 50, value=b2, key="clr_b")
render_map(
    all_clearable,
    heat=True,
    heat_radius=r_cl,
    heat_blur=b_cl,
    title="Heatmap: All Clearable DFR Calls",
    key="map_clearable_heat",
    show_circle=True,
    launch_coords=launch_coords
)

# ─── 6f) Heatmap: ALPR Locations ─────────────────────────────────────────────
if alpr_df is not None:
    # build a DataFrame of all valid ALPR points (E=4, F=5 in new export)
    alpr_pts = pd.DataFrame({
        "lat": pd.to_numeric(alpr_df.iloc[:, 4], errors="coerce"),
        "lon": pd.to_numeric(alpr_df.iloc[:, 5], errors="coerce")
    }).dropna()

    # allow radius/blur tweaks
    r_al = st.sidebar.slider("ALPR Heat Radius", 1, 50, value=6, key="alpr_r")
    b_al = st.sidebar.slider("ALPR Heat Blur",   1, 50, value=4, key="alpr_b")

    # always render the circle + whatever points you have
    render_map(
        alpr_pts,
        heat=True,
        heat_radius=r_al,
        heat_blur=b_al,
        title="Heatmap: ALPR Locations",
        key="map_alpr_heat",
        show_circle=True,
        launch_coords=launch_coords
    )

# 6g) Heatmap: Audio Locations (using the cleaned audio_pts from above)
if audio_pts is not None and not audio_pts.empty:
    # allow the user to tweak radius/blur
    r_au = st.sidebar.slider("Audio Heat Radius", 1, 50, value=4, key="audio_r")
    b_au = st.sidebar.slider("Audio Heat Blur",   1, 50, value=4, key="audio_b")

    render_map(
        audio_pts,                     # your DataFrame with 'lat' & 'lon'
        heat=True,
        heat_radius=r_au,
        heat_blur=b_au,
        title="Heatmap: Audio Locations",
        key="map_audio_heat",
        show_circle=True,
        launch_coords=launch_coords
    )
else:
    st.sidebar.info("No audio points to display on the heatmap.")
