import io
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
import folium
from folium.plugins import HeatMap

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
    s_obj = s.astype("object")
    num   = pd.to_numeric(s_obj, errors="coerce")
    dt    = pd.to_datetime(EPOCH_1899) + pd.to_timedelta(num * 86400, unit="s")
    txt   = pd.to_datetime(s_obj.where(num.isna(), None), errors="coerce")
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
progress.progress(10)

# ─── 2) Launch Locations ────────────────────────────────────────────────────
st.sidebar.header("2) Launch Locations")

# 2a) Choose manual vs CSV
mode = st.sidebar.radio(
    "How would you like to enter launch locations?",
    ("By Coordinates", "By Address")
)

# 2b) Or upload a CSV instead
launch_file = st.sidebar.file_uploader(
    "Or upload Launch Locations CSV", 
    type=["csv"], 
    key="launch_csv"
)
if launch_file:
    launch_df = pd.read_csv(launch_file)
else:
    # 2c) Manual-entry table
    if _EDITOR is None:
        st.sidebar.error("Upgrade Streamlit or upload a CSV.")
        st.stop()

    if mode == "By Coordinates":
        launch_df = _EDITOR(
            pd.DataFrame(columns=["Location Name","Lat","Lon"]),
            num_rows="dynamic", 
            use_container_width=True
        )
    else:  # By Address
        launch_df = _EDITOR(
            pd.DataFrame(columns=["Location Name","Address"]),
            num_rows="dynamic", 
            use_container_width=True
        )

# 2d) Only geocode when in “By Address” mode
if mode == "By Address":
    from geopy.geocoders import Nominatim
    from geopy.extra.rate_limiter import RateLimiter

    geolocator = Nominatim(user_agent="dfrimpact")
    geocode    = RateLimiter(geolocator.geocode, min_delay_seconds=1)

    @st.cache_data(show_spinner=False)
    def lookup(addr):
        loc = geocode(addr)
        return (loc.latitude, loc.longitude) if loc else (None, None)

    # Build a list of (lat, lon) tuples—skip empty
    coords = launch_df["Address"].fillna("").apply(
        lambda a: lookup(a) if str(a).strip() else (None, None)
    )
    coords_df = pd.DataFrame(
        coords.tolist(),
        columns=["Lat","Lon"],
        index=launch_df.index
    )
    launch_df[["Lat","Lon"]] = coords_df
    # ─── Debug: show what we geocoded ─────────────────────────────────────────────
st.sidebar.subheader("Geocoded Launch Locations")


# 2e) Final validation
if not {"Lat","Lon"}.issubset(launch_df.columns):
    st.sidebar.error("You must supply Lat & Lon (directly or via Address).")
    st.stop()

progress.progress(30)

st.sidebar.header("3) Agency Call Types")
agency_file = st.sidebar.file_uploader("Upload Agency Call Types CSV", type=["csv"])
if agency_file:
    agency_df = pd.read_csv(agency_file)
    req = {"Call Type","DFR Response (Y/N)","Clearable (Y/N)"}
    if not req.issubset(agency_df.columns):
        st.sidebar.error(f"Agency CSV must include {req}")
        st.stop()
else:
    if _EDITOR is None:
        st.sidebar.error("Upgrade Streamlit or upload Agency CSV.")
        st.stop()
    types = sorted(raw_df["Call Type"].astype(str).unique())
    base = pd.DataFrame({
        "Call Type": types,
        "DFR Response (Y/N)": [False]*len(types),
        "Clearable (Y/N)"    : [False]*len(types),
    })
    agency_df = _EDITOR(base, num_rows="dynamic", use_container_width=True)
progress.progress(50)

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
col_pri    = pick("Call Priority")
col_lat    = pick("Lat")
col_lon    = pick("Lon")

for c in [col_create,col_arrive,col_close,col_type,col_pri,col_lat,col_lon]:
    if c is None:
        st.error("Missing required Raw Call Data columns.")
        st.stop()

create_dt = parse_time_series(raw_df[col_create])
arrive_dt = parse_time_series(raw_df[col_arrive])
close_dt  = parse_time_series(raw_df[col_close])

patrol_sec  = (arrive_dt - create_dt).dt.total_seconds()
onscene_sec = (close_dt  - arrive_dt).dt.total_seconds()
lat = pd.to_numeric(raw_df[col_lat], errors="coerce")
lon = pd.to_numeric(raw_df[col_lon], errors="coerce")
progress.progress(85)

# build a proper Lat/Lon array by column name, after any geocoding has run
try:
    launch_coords = launch_df[["Lat","Lon"]].astype(float).values
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

dfr_map = set(agency_df.loc[agency_df["DFR Response (Y/N)"].astype(str).str.upper()=="Y","Call Type"]
              .str.upper().str.strip())
clr_map = set(agency_df.loc[agency_df["Clearable (Y/N)"].astype(str).str.upper()=="Y","Call Type"]
              .str.upper().str.strip())

dfr_only  = df_all[df_all["call_type_up"].isin(dfr_map) & df_all["patrol_sec"].gt(0)].copy()
in_range  = dfr_only[dfr_only["dist_mi"].le(drone_range)].copy()
clearable = in_range[in_range["call_type_up"].isin(clr_map)].copy()
progress.progress(95)

# ─── 3) OPTIONAL ALPR & AUDIO METRICS ────────────────────────────────────────
alpr_df  = pd.read_csv(alpr_file)  if alpr_file  else None
audio_df = pd.read_csv(audio_file) if audio_file else None

alpr_sites=alpr_hits=alpr_eta=0
if alpr_df is not None and alpr_df.shape[1]>=4:
    cols = list(alpr_df.columns)
    total_col = cols[-1]
    lat_a = pd.to_numeric(alpr_df.iloc[:,1], errors="coerce")
    lon_a = pd.to_numeric(alpr_df.iloc[:,2], errors="coerce")
    hits  = pd.to_numeric(alpr_df[total_col], errors="coerce").fillna(0).values
    dist  = haversine_min(lat_a.values, lon_a.values, launch_coords)
    ok    = (dist<=drone_range)&np.isfinite(dist)
    alpr_sites = int(ok.sum())
    alpr_hits  = int(hits[ok].sum())
    etas = dist/ max(drone_speed,1e-9)*3600
    alpr_eta   = float((etas[ok]*hits[ok]).sum()/hits[ok].sum()) if hits[ok].sum()>0 else np.nan

audio_sites=audio_hits=audio_eta=0
if audio_df is not None and audio_df.shape[1]>=5:
    lat_b = pd.to_numeric(audio_df.iloc[:,2], errors="coerce")
    lon_b = pd.to_numeric(audio_df.iloc[:,3], errors="coerce")
    hits2 = pd.to_numeric(audio_df.iloc[:,4], errors="coerce").fillna(0).values
    dist2 = haversine_min(lat_b.values, lon_b.values, launch_coords)
    ok2   = (dist2<=drone_range)&np.isfinite(dist2)
    audio_sites = int(ok2.sum())
    audio_hits  = int(hits2[ok2].sum())
    etas2 = dist2/ max(drone_speed,1e-9)*3600
    audio_eta   = float((etas2[ok2]*hits2[ok2]).sum()/hits2[ok2].sum()) if hits2[ok2].sum()>0 else np.nan

dfr_alpr_audio = alpr_hits + audio_hits

# ─── 4) METRICS & REPORT ─────────────────────────────────────────────────────
total_cfs   = len(df_all)
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
    ("DFR Responses within Range",         in_count,       "int"),
    ("DFR Responses to ALPR and Audio",    dfr_alpr_audio, "int"),
    ("Expected Drone Response Time",       avg_drone,      "mmss"),
    ("Expected First on Scene %",          first_on_pct,   "pct"),
    ("Expected CFS Cleared",               exp_cleared,    "int"),
    ("Officers Saved (FTE)",               officers,       "2dec"),
    ("ROI from Calls Cleared",             roi,            "usd"),
    ("Total CFS",                          total_cfs,      "int"),
    ("Total Potential DFR Calls",          total_dfr,      "int"),
    ("Avg Patrol Response Time (All)",     avg_patrol,     "mmss"),
    ("Avg Scene Time (All)",               avg_scene,      "mmss"),
    ("Expected Decrease vs Patrol %",      pct_dec,        "pct"),
    ("Avg Patrol (In-Range)",              avg_in,         "mmss"),
    ("P1 In-Range Count",                  p1_count,       "int"),
    ("Avg Patrol (P1 In-Range)",           avg_p1_pat,     "mmss"),
    ("Expected Drone P1 ETA",              avg_p1_drone,   "mmss"),
    ("ALPR Sites In-Range",                alpr_sites,     "int"),
    ("ALPR Hits In-Range",                 alpr_hits,      "int"),
    ("ALPR ETA Weighted",                  alpr_eta,       "mmss"),
    ("Audio Sites In-Range",               audio_sites,    "int"),
    ("Audio Hits In-Range",                audio_hits,     "int"),
    ("Audio ETA Weighted",                 audio_eta,      "mmss"),
    ("Total Clearable In-Range",           clr_count,      "int"),
    ("Avg Scene Time (Clearable)",         avg_clr,        "mmss"),
]

report_df = pd.DataFrame({
    "Metric": [r[0] for r in rows],
    "Result": [pretty_value(r[1],r[2]) for r in rows],
})
st.subheader("Report Values")
st.dataframe(report_df, use_container_width=True)

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
    heat_blur=25
):
    
    st.subheader(title)
    if show_circle and launch_coords and len(launch_coords):
        center = [float(launch_coords[0][0]), float(launch_coords[0][1])]
    elif not df_pts.empty:
        center = [float(df_pts["lat"].mean()), float(df_pts["lon"].mean())]
    else:
        center = [39.0,-98.5]
    m = folium.Map(location=center, zoom_start=10)
    # ── draw the 3.5 mi drone range circle + epicenter icon ────────────────
    for la, lo in launch_coords:
        folium.Circle(
            location=(la, lo),
            radius=drone_range * 1609.34,
            color="blue",
            fill=False
        ).add_to(m)
        folium.Marker(
            location=(la, lo),
            icon=folium.Icon(icon="info-sign")
        ).add_to(m)
    if heat and not df_pts.empty:
        HeatMap(
            df_pts[["lat","lon"]].values.tolist(),
            radius=heat_radius,
            blur=heat_blur
        ).add_to(m)

    else:
        for _,r in df_pts.iterrows():
            folium.CircleMarker(location=(r["lat"],r["lon"]),
                                radius=3, color="red", fill=True, fill_opacity=0.6).add_to(m)
    st_folium(m, width=800, height=500, key=key)

# Scatter of all DFR calls
render_map(
    dfr_only,
    heat=False,
    title="All DFR Calls (Scatter)",
    key="map_all_scatter"
)

# 3.5‑mile circle only
render_map(
    pd.DataFrame(),
    heat=False,
    title="3.5‑mile Drone Range",
    key="map_range_circle"
)

# In‑Range Heatmap + its own sliders
r1 = st.sidebar.slider("In‑Range Heat Radius", 1, 50, 15, key="r1")
b1 = st.sidebar.slider("In‑Range Heat Blur",   1, 50, 25, key="b1")
render_map(
    in_range,
    heat=True,
    title="Heatmap: In‑Range Calls",
    key="map_in_heat",
    heat_radius=r1,
    heat_blur=b1
)

# P1 In‑Range Heatmap + its own sliders
r2 = st.sidebar.slider("P1 In‑Range Heat Radius", 1, 50, 15, key="r2")
b2 = st.sidebar.slider("P1 In‑Range Heat Blur",   1, 50, 25, key="b2")
render_map(
    in_range[in_range["priority"]=="1"],
    heat=True,
    title="Heatmap: P1 In‑Range",
    key="map_p1_heat",
    heat_radius=r2,
    heat_blur=b2
)

# ALPR Heatmap + its own sliders (if present)
if alpr_df is not None:
    alpr_plot = pd.DataFrame({
        "lat": pd.to_numeric(alpr_df.iloc[:,1],errors="coerce"),
        "lon": pd.to_numeric(alpr_df.iloc[:,2],errors="coerce")
    }).dropna()
    r3 = st.sidebar.slider("ALPR Heat Radius", 1, 50, 15, key="r3")
    b3 = st.sidebar.slider("ALPR Heat Blur",   1, 50, 25, key="b3")
    render_map(
        alpr_plot,
        heat=True,
        title="Heatmap: ALPR Locations",
        key="map_alpr_heat",
        heat_radius=r3,
        heat_blur=b3
    )

# Clearable Heatmap + its own sliders
r4 = st.sidebar.slider("Clearable Heat Radius", 1, 50, 15, key="r4")
b4 = st.sidebar.slider("Clearable Heat Blur",   1, 50, 25, key="b4")
render_map(
    clearable,
    heat=True,
    title="Heatmap: Clearable Calls",
    key="map_clearable_heat",
    heat_radius=r4,
    heat_blur=b4
)

# Audio Heatmap + its own sliders (if present)
if audio_df is not None:
    audio_plot = pd.DataFrame({
        "lat": pd.to_numeric(audio_df.iloc[:,2],errors="coerce"),
        "lon": pd.to_numeric(audio_df.iloc[:,3],errors="coerce")
    }).dropna()
    r5 = st.sidebar.slider("Audio Heat Radius", 1, 50, 15, key="r5")
    b5 = st.sidebar.slider("Audio Heat Blur",   1, 50, 25, key="b5")
    render_map(
        audio_plot,
        heat=True,
        title="Heatmap: Audio Locations",
        key="map_audio_heat",
        heat_radius=r5,
        heat_blur=b5
    )
