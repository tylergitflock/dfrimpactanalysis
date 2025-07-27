# app.py

import io
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
import folium
from folium.plugins import HeatMap

st.set_page_config(page_title="DFR Impact Analysis", layout="wide")

# ─── Helpers ─────────────────────────────────────────────────────────────────
EPOCH_1899 = np.datetime64("1899-12-30")

def parse_time_series(s: pd.Series) -> pd.Series:
    s = s.astype("object")
    num = pd.to_numeric(s, errors="coerce")
    dt_num = pd.to_datetime(EPOCH_1899) + pd.to_timedelta(num * 86400, unit="s")
    txt   = pd.to_datetime(s.where(num.isna(), None), errors="coerce")
    return dt_num.where(~num.isna(), txt)

def haversine_min(lat_arr, lon_arr, launches):
    res = np.full(lat_arr.shape, np.inf)
    lat1, lon1 = np.radians(lat_arr), np.radians(lon_arr)
    for la, lo in launches:
        la_r, lo_r = np.radians(la), np.radians(lo)
        dlat, dlon = la_r - lat1, lo_r - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(la_r)*(np.sin(dlon/2)**2)
        d = 2 * 3958.8 * np.arcsin(np.sqrt(a))
        res = np.minimum(res, d)
    res[np.isnan(lat_arr)|np.isnan(lon_arr)] = np.nan
    return res

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
    if kind=="int":
        return f"{int(val):,}" if pd.notna(val) else ""
    if kind=="usd":
        return f"${int(round(val)):,}" if pd.notna(val) else ""
    if kind=="2dec":
        return f"{val:.2f}" if pd.notna(val) else ""
    if kind=="pct":
        return f"{val:.0f}%" if pd.notna(val) else ""
    if kind=="mmss":
        return fmt_mmss(val)
    if kind=="hhmmss":
        return fmt_hhmmss(val)
    return "" if pd.isna(val) else str(val)

def average(arr):
    nums = [v for v in arr if isinstance(v,(int,float)) and np.isfinite(v)]
    return float(np.mean(nums)) if nums else np.nan

# ─── 0) PROGRESS BAR ──────────────────────────────────────────────────────────
progress = st.sidebar.progress(0)

# ─── 1) SIDEBAR: UPLOAD RAW DATA ──────────────────────────────────────────────
st.title("DFR Impact Analysis")
with st.sidebar.expander("1) Raw Call Data & Launch Locations", expanded=True):
    raw_file    = st.file_uploader("Raw Call Data CSV", type=["csv"])
    launch_file = st.file_uploader("Launch Locations CSV (optional)", type=["csv"])
progress.progress(10)

# ─── 2) SIDEBAR: ASSUMPTIONS ─────────────────────────────────────────────────
with st.sidebar.expander("2) Assumptions", expanded=False):
    fte_hours    = st.number_input("Full Time Work Year (hours)", value=2080, step=1)
    officer_cost = st.number_input("Officer Cost per FTE ($)", value=127940, step=1000, format="%d")
    cancel_rate  = st.number_input("Drone Cancellation Rate (0–1)", value=0.11, step=0.01, format="%.2f")
    drone_speed  = st.number_input("Drone Speed (mph)", value=51.0, step=1.0)
    drone_range  = st.number_input("Drone Range (miles)", value=3.5, step=0.1)
progress.progress(20)

# ─── 3) SIDEBAR: LAUNCH LOCATIONS ─────────────────────────────────────────────
if launch_file:
    launch_df = pd.read_csv(launch_file)
else:
    st.sidebar.write("Or enter Launch Locations manually:")
    launch_df = st.sidebar.experimental_data_editor(
        pd.DataFrame(columns=["Location Name","Lat","Lon"]),
        num_rows="dynamic", use_container_width=True
    )
if launch_df.shape[1] < 3:
    st.error("Launch Locations must have columns: Location Name, Lat, Lon.")
    st.stop()
progress.progress(30)

# ─── 4) SIDEBAR: AGENCY CALL TYPES ────────────────────────────────────────────
agency_file = st.sidebar.file_uploader("Agency Call Types CSV (optional)", type=["csv"])
if agency_file:
    agency_df = pd.read_csv(agency_file)
    req = {"Call Type","DFR Response (Y/N)","Clearable (Y/N)"}
    if not req.issubset(agency_df.columns):
        st.error(f"Agency CSV must include: {req}")
        st.stop()
else:
    # read raw types once we have raw_df
    agency_df = None
progress.progress(35)

# ─── 5) SIDEBAR: ALPR & AUDIO ─────────────────────────────────────────────────
alpr_file  = st.sidebar.file_uploader("ALPR Data CSV (optional)", type=["csv"])
audio_file = st.sidebar.file_uploader("Audio Hits CSV (optional)", type=["csv"])
progress.progress(40)

# ─── 6) LOAD RAW CALL DATA ───────────────────────────────────────────────────
if not raw_file:
    st.info("Please upload Raw Call Data to proceed.")
    st.stop()
raw_df = pd.read_csv(raw_file)
progress.progress(45)

# ─── 7) BUILD AGENCY DF IF NEEDED ────────────────────────────────────────────
if agency_df is None:
    types = sorted(raw_df["Call Type"].astype(str).unique())
    agency_df = pd.DataFrame({
        "Call Type": types,
        "DFR Response (Y/N)": False,
        "Clearable (Y/N)": False
    })
agency_df = st.sidebar.experimental_data_editor(
    agency_df, num_rows="dynamic", use_container_width=True
)
dfr_map = set(agency_df.loc[agency_df["DFR Response (Y/N)"].astype(str).str.upper()=="Y","Call Type"].str.upper().str.strip())
clr_map = set(agency_df.loc[agency_df["Clearable (Y/N)"].astype(str).str.upper()=="Y","Call Type"].str.upper().str.strip())
progress.progress(50)

# ─── 8) PARSE TIMES & COORDS ─────────────────────────────────────────────────
# map raw columns
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
        st.error("Missing one of required columns in Raw Call Data.")
        st.stop()

create_dt = parse_time_series(raw_df[col_create])
arrive_dt = parse_time_series(raw_df[col_arrive])
close_dt  = parse_time_series(raw_df[col_close])

patrol_sec  = (arrive_dt - create_dt).dt.total_seconds()
onscene_sec = (close_dt  - arrive_dt).dt.total_seconds()
lat = pd.to_numeric(raw_df[col_lat], errors="coerce")
lon = pd.to_numeric(raw_df[col_lon], errors="coerce")
progress.progress(60)

# ─── 9) DISTANCES & DRONE ETA ────────────────────────────────────────────────
launch_coords = launch_df.iloc[:,1:3].astype(float).values
dist_mi       = haversine_min(lat.values, lon.values, launch_coords)
drone_eta_sec = dist_mi / max(drone_speed,1e-9) * 3600
progress.progress(70)

# ───10) BUILD master DF & FILTERS ────────────────────────────────────────────
df_all = pd.DataFrame({
    "patrol_sec":    patrol_sec,
    "onscene_sec":   onscene_sec,
    "lat":           lat,
    "lon":           lon,
    "dist_mi":       dist_mi,
    "drone_eta_sec": drone_eta_sec,
    "call_type_up":  raw_df[col_type].astype(str).str.upper().str.strip(),
    "priority":      raw_df[col_pri].astype(str).str.strip()
})

mask_dfr    = df_all["call_type_up"].isin(dfr_map) & df_all["patrol_sec"].notna() & (df_all["patrol_sec"]>0)
dfr_only    = df_all.loc[mask_dfr].copy()

mask_in     = dfr_only["dist_mi"].notna() & (dfr_only["dist_mi"] <= drone_range)
in_range    = dfr_only.loc[mask_in].copy()

mask_clr    = in_range["call_type_up"].isin(clr_map)
clearable   = in_range.loc[mask_clr].copy()
progress.progress(80)

# ───11) OPTIONAL ALPR & AUDIO METRICS ────────────────────────────────────────
alpr_df  = pd.read_csv(alpr_file)  if alpr_file  else None
audio_df = pd.read_csv(audio_file) if audio_file else None

# ALPR
alpr_sites=alpr_hits=alpr_eta=0
if alpr_df is not None and alpr_df.shape[1]>=4:
    cols = list(alpr_df.columns)
    total_col, clear_cols = cols[-1], [c for c in cols[3:-1] if "custom" in c.lower() or "stolen" in c.lower()]
    lat_a = pd.to_numeric(alpr_df.iloc[:,1], errors="coerce")
    lon_a = pd.to_numeric(alpr_df.iloc[:,2], errors="coerce")
    hits  = pd.to_numeric(alpr_df[total_col], errors="coerce").fillna(0).values
    dist  = haversine_min(lat_a.values, lon_a.values, launch_coords)
    ok    = (dist<=drone_range)&np.isfinite(dist)
    alpr_sites = int(ok.sum())
    alpr_hits  = int(hits[ok].sum())
    etas = dist/ max(drone_speed,1e-9)*3600
    alpr_eta   = float((etas[ok]*hits[ok]).sum()/hits[ok].sum()) if hits[ok].sum()>0 else np.nan

# Audio
audio_sites=audio_hits=audio_eta=0
if audio_df is not None and audio_df.shape[1]>=5:
    lat_a = pd.to_numeric(audio_df.iloc[:,2], errors="coerce")
    lon_a = pd.to_numeric(audio_df.iloc[:,3], errors="coerce")
    hits  = pd.to_numeric(audio_df.iloc[:,4], errors="coerce").fillna(0).values
    dist  = haversine_min(lat_a.values, lon_a.values, launch_coords)
    ok    = (dist<=drone_range)&np.isfinite(dist)
    audio_sites = int(ok.sum())
    audio_hits  = int(hits[ok].sum())
    etas = dist/ max(drone_speed,1e-9)*3600
    audio_eta   = float((etas[ok]*hits[ok]).sum()/hits[ok].sum()) if hits[ok].sum()>0 else np.nan

dfr_alpr_audio = alpr_hits + audio_hits
progress.progress(90)

# ───12) CALCULATE METRICS ────────────────────────────────────────────────────
total_cfs   = len(df_all)
total_dfr   = len(dfr_only)
in_count    = len(in_range)
clr_count   = len(clearable)
p1_count    = int((in_range["priority"]=="1").sum())

avg_drone   = average(in_range["drone_eta_sec"])
avg_patrol  = average(dfr_only["patrol_sec"])
avg_scene   = average(dfr_only["onscene_sec"])
avg_in      = average(in_range["patrol_sec"])
avg_p1_pat  = average(in_range.loc[in_range["priority"]=="1","patrol_sec"])
avg_p1_drone= average(in_range.loc[in_range["priority"]=="1","drone_eta_sec"])
avg_clr     = average(clearable["onscene_sec"])

first_on_pct= float(np.mean(in_range["drone_eta_sec"] < in_range["patrol_sec"]) * 100) if in_count else np.nan
pct_dec     = ((avg_in-avg_drone)/avg_in*100) if avg_in>0 else np.nan
exp_cleared = int(round((in_count + dfr_alpr_audio) * cancel_rate))
exp_clr     = exp_cleared
time_saved  = (avg_clr or 0)*exp_clr
officers    = (time_saved/3600)/fte_hours if fte_hours>0 else np.nan
roi         = officers * officer_cost if np.isfinite(officers) else np.nan

# ───13) BUILD REPORT TABLE ──────────────────────────────────────────────────
rows = [
    ("DFR Responses within Range",           in_count,       "int"),
    ("DFR Responses to ALPR and Audio within Range", dfr_alpr_audio, "int"),
    ("Expected DFR Drone Response Times by Location", avg_drone, "mmss"),
    ("Expected First on Scene %",            first_on_pct,   "pct"),
    ("Expected CFS Cleared",                 exp_cleared,    "int"),
    ("Number of Officers - Force Multiplication",officers, "2dec"),
    ("ROI from Potential Calls Cleared",     roi,            "usd"),
    ("Total CFS",                            total_cfs,      "int"),
    ("Total Potential DFR Calls",            total_dfr,      "int"),
    ("DFR Responses within Range",           in_count,       "int"),
    ("DFR Responses to ALPR and Audio within Range", dfr_alpr_audio, "int"),
    ("Total Potential DFR Calls",            total_dfr,      "int"),
    ("Avg Disp + Patrol Response Time to DFR Calls", avg_patrol, "mmss"),
    ("Avg Time on Scene ALL DFR Calls",      avg_scene,      "mmss"),
    ("Expected DFR Drone Response Times by Location",avg_drone, "mmss"),
    ("Expected First on Scene %",            first_on_pct,   "pct"),
    ("Expected Decrease in Response Times",  pct_dec,        "pct"),
    ("Avg Disp + Patrol Response Time to In-Range Calls", avg_in, "mmss"),
    ("Expected DFR Drone Response Times by Location",avg_drone, "mmss"),
    ("Total DFR Calls In Range that are priority 1", p1_count, "int"),
    ("Avg Disp + Patrol Response Time to In-Range P1 Calls", avg_p1_pat, "mmss"),
    ("Expected DFR Drone Response Times to P1 Calls", avg_p1_drone, "mmss"),
    ("3.5-mile Drone Range",               np.nan,         ""),
    ("Heatmap: All DFR Calls",              np.nan,         ""),
    ("Heatmap: Priority-1 In-Range Calls",  np.nan,         ""),
    ("Number of ALPR Locations within range", alpr_sites,  "int"),
    ("Number of Hits within range",          alpr_hits,    "int"),
    ("Expected response time to ALPR data",  alpr_eta,     "mmss"),
    ("Number of Audio Locations",            audio_sites,  "int"),
    ("Number of hits within range",          audio_hits,   "int"),
    ("Avg expected resp time within range Audio Hits", audio_eta, "mmss"),
    ("Expected Clearable CFS Cleared",       exp_clr,       "int"),
    ("Number of Officers - Force Multiplication",officers, "2dec"),
    ("ROI from Potential Calls Cleared",     roi,            "usd"),
    ("Total clearable CFS within range",     clr_count,     "int"),
    ("Total time spent on Clearable CFS",    clearable["onscene_sec"].sum() if clr_count else np.nan, "hhmmss"),
    ("Avg Time on Scene – Clearable Calls",  avg_clr,       "mmss"),
]

report_df = pd.DataFrame({
    "Metric":    [r[0] for r in rows],
    "Result":    [pretty_value(r[1],r[2]) for r in rows],
})
st.subheader("Report Values")
st.dataframe(report_df, use_container_width=True)
progress.progress(95)

# ───14) ESRI CSV DOWNLOADS ──────────────────────────────────────────────────
def to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")

st.subheader("ESRI CSV Exports")
cols = ["lat","lon","patrol_sec","drone_eta_sec","onscene_sec","priority","call_type_up"]
c1,c2,c3 = st.columns(3)
with c1:
    st.download_button("Download DFR Only", to_csv_bytes(dfr_only[cols]), "dfr_only.csv")
with c2:
    st.download_button("Download In Range", to_csv_bytes(in_range[cols]), "dfr_in_range.csv")
with c3:
    st.download_button("Download Clearable", to_csv_bytes(clearable[cols]), "dfr_clearable.csv")
progress.progress(100)

# ───15) MAPS ─────────────────────────────────────────────────────────────────
st.header("Maps & Heatmaps")
heat_radius = st.sidebar.slider("Heat radius",1,50,15)
heat_blur   = st.sidebar.slider("Heat blur",1,50,25)

def render_map(df_pts, show_circle=False, heat=False, title=""):
    st.subheader(title)
    if show_circle and len(launch_coords):
        center = [float(launch_coords[0][0]), float(launch_coords[0][1])]
    elif not df_pts.empty:
        center = [float(df_pts["lat"].mean()), float(df_pts["lon"].mean())]
    else:
        center = [39.0,-98.5]
    m = folium.Map(location=center, zoom_start=10)
    if show_circle:
        for la,lo in launch_coords:
            folium.Circle(location=(la,lo),
                          radius=drone_range*1609.34,
                          color="blue",fill=False).add_to(m)
    if heat and not df_pts.empty:
        heat_data = df_pts[["lat","lon"]].dropna().values.tolist()
        HeatMap(heat_data, radius=heat_radius, blur=heat_blur).add_to(m)
    else:
        for _,r in df_pts.iterrows():
            folium.CircleMarker(location=(r["lat"],r["lon"]),
                                radius=3,color="red",fill=True,fill_opacity=0.6).add_to(m)
    st_folium(m, width=800, height=500)

render_map(dfr_only, heat=False, title="All DFR Calls (Scatter)")
render_map(pd.DataFrame(), show_circle=True, heat=False, title="3.5-mi Drone Range")
render_map(in_range, heat=True, title="Heatmap: In-Range Calls")
render_map(in_range[in_range["priority"]=="1"], heat=True, title="Heatmap: P1 In-Range")
if alpr_df is not None:
    alpr_plot = pd.DataFrame({
        "lat": pd.to_numeric(alpr_df.iloc[:,1],errors="coerce"),
        "lon": pd.to_numeric(alpr_df.iloc[:,2],errors="coerce")
    }).dropna()
    render_map(alpr_plot, heat=True, title="Heatmap: ALPR Locations")
render_map(clearable, heat=True, title="Heatmap: Clearable Calls")
if audio_df is not None:
    audio_plot = pd.DataFrame({
        "lat": pd.to_numeric(audio_df.iloc[:,2],errors="coerce"),
        "lon": pd.to_numeric(audio_df.iloc[:,3],errors="coerce")
    }).dropna()
    render_map(audio_plot, heat=True, title="Heatmap: Audio Locations")
