import io
import os
import json
import time
import uuid
import tempfile
from io import BytesIO
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
import folium
from folium.plugins import HeatMap
import math
import zipfile, re

# ─── Page Setup (must be first Streamlit call) ───────────────────────────────
st.set_page_config(page_title="DFR Impact Analysis", layout="wide")

# Optional: tz for local timestamps (comment out if unavailable)
try:
    from tzlocal import get_localzone
except Exception:
    get_localzone = None

# Writable default; can be overridden with env var RUNS_DIR
BASE_DIR = os.environ.get("RUNS_DIR", os.path.join(tempfile.gettempdir(), "dfr_runs"))
os.makedirs(BASE_DIR, exist_ok=True)

def slugify(name: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in (name or "")).strip("_") or "unknown_agency"

def list_runs():
    rows = []
    if not os.path.isdir(BASE_DIR): 
        return rows
    for agency in sorted(os.listdir(BASE_DIR)):
        apath = os.path.join(BASE_DIR, agency)
        if not os.path.isdir(apath): 
            continue
        for stamp in sorted(os.listdir(apath), reverse=True):
            rpath = os.path.join(apath, stamp)
            if os.path.isdir(rpath):
                rows.append({"agency": agency, "stamp": stamp, "path": rpath})
    return rows

def save_run(agency_name, config_dict, metrics_dict, input_files_dict,
             map_images=None, pdf_bytes=None):
    # Attach local timestamp + timezone if tzlocal is available
    try:
        if get_localzone is not None:
            local_tz = get_localzone()
            now_local = datetime.now(local_tz)
            config_dict["run_time_iso_local"] = now_local.isoformat()
            config_dict["run_timezone"] = str(local_tz)
    except Exception:
        # non-fatal
        pass

    agency_slug = slugify(agency_name)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    rdir = os.path.join(BASE_DIR, agency_slug, stamp)

    os.makedirs(os.path.join(rdir, "inputs"), exist_ok=True)
    os.makedirs(os.path.join(rdir, "maps"), exist_ok=True)

    with open(os.path.join(rdir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)
    with open(os.path.join(rdir, "metrics.json"), "w") as f:
        json.dump(metrics_dict, f, indent=2)

    for name, fobj in (input_files_dict or {}).items():
        if fobj is not None:
            try:
                fobj.seek(0)
            except Exception:
                pass
            with open(os.path.join(rdir, "inputs", name), "wb") as out:
                out.write(fobj.read())

    if map_images:
        for name, png_bytes in map_images.items():
            with open(os.path.join(rdir, "maps", name), "wb") as out:
                out.write(png_bytes)

    if pdf_bytes:
        with open(os.path.join(rdir, "report.pdf"), "wb") as out:
            out.write(pdf_bytes)

    return rdir

def _read_bytes(path):
    with open(path, "rb") as f:
        return BytesIO(f.read())

def extract_agency_name(zip_filename: str) -> str:
    """
    From 'Fort Worth PD - ZIP.zip' -> 'Fort Worth PD'.
    Falls back to the stem if ' - ' isn't present.
    """
    if not zip_filename:
        return ""
    base = os.path.basename(zip_filename)
    stem = os.path.splitext(base)[0]
    return stem.split(" - ")[0].strip() if " - " in stem else stem.strip()

# === REPLAY: single canonical loader (PUT THIS ONCE, HERE) ===================
REPLAY = st.session_state.get("replay_dir")
replay_cfg = st.session_state.get("replay_config", {})
replay_inputs = {}

if REPLAY:
    inp_dir = os.path.join(REPLAY, "inputs")

    def _maybe(fname):
        p = os.path.join(inp_dir, fname)
        return _read_bytes(p) if os.path.exists(p) else None

    # These names MUST match what save_run() writes into /inputs/
    replay_inputs = {
        "raw":    _maybe("raw_calls.csv"),
        "agency": _maybe("agency_call_types.csv"),
        "launch": _maybe("launch_locations.csv"),
        "alpr":   _maybe("alpr.csv"),
        "audio":  _maybe("audio.csv"),
    }

    # Optional: tiny debug so you can see what's loaded
    st.sidebar.caption("Replaying saved inputs…")
    st.sidebar.code({k: bool(v) for k, v in replay_inputs.items()})

    # One back button for replay mode
    if st.sidebar.button("⬅️ Back to Start"):
        for k in ("replay_dir", "replay_config", "viewing_saved"):
            st.session_state.pop(k, None)
        st.rerun()

# === QUICK VIEW OF A SAVED RUN (no re-run) ==============================
if st.session_state.get("viewing_saved") and st.session_state.get("loaded_run_dir"):
    import os, json
    run_dir = st.session_state["loaded_run_dir"]
    cfg_p = os.path.join(run_dir, "config.json")
    met_p = os.path.join(run_dir, "metrics.json")

    cfg = {}
    metrics = {}
    try:
        if os.path.exists(cfg_p):
            with open(cfg_p, "r") as f: cfg = json.load(f)
        if os.path.exists(met_p):
            with open(met_p, "r") as f: metrics = json.load(f)
    except Exception as e:
        st.error(f"Failed to load saved files: {e}")

    # Header
    st.title("Saved Report (no re-run)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Agency",   cfg.get("agency_name", "—"))
    c2.metric("Run by",   cfg.get("analyst_name", "—"))
    c3.metric("When",     cfg.get("run_time_iso_local", cfg.get("run_time_iso", "—")))

    # Simple table of saved metrics
    if metrics:
        import pandas as pd
        df = pd.DataFrame(
            [{"Metric": k, "Value": v} for k, v in metrics.items()]
        )
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No metrics.json found in this run.")

    # Back button
    if st.button("⬅️ Back to start"):
        for k in ("viewing_saved", "loaded_run_dir", "loaded_config"):
            st.session_state.pop(k, None)
        try:
            st.rerun()
        except Exception:
            st.experimental_rerun()

    st.stop()  # IMPORTANT: don’t run the rest of the app

# ─── Page Setup ───────────────────────────────────────────────────────────────
# Show a banner when replaying a saved run
if st.session_state.get("replay_dir"):
    st.info(f"Replaying saved run from: {st.session_state['replay_dir']}")

# ─── Detect available data‐editor API ─────────────────────────────────────────
if hasattr(st, "data_editor"):
    _EDITOR = st.data_editor
elif hasattr(st, "experimental_data_editor"):
    _EDITOR = st.experimental_data_editor
else:
    _EDITOR = None

# ─── Helper Functions ─────────────────────────────────────────────────────────
EPOCH_1899 = np.datetime64("1899-12-30")

def parse_time_series(s: pd.Series) -> pd.Series:
    s_str = s.astype(str).str.replace(",", "")
    num = pd.to_numeric(s_str, errors="coerce")
    dt = pd.to_datetime(EPOCH_1899) + pd.to_timedelta(num * 86400, unit="s")
    txt = pd.to_datetime(s_str.where(num.isna(), None), errors="coerce")
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
    n = len(df)
    if n <= 1:
        return max_radius, max_blur
    base = max_radius * math.sqrt(1000 / n)
    radius = int(min(max_radius, max(5, base)))
    blur = int(min(max_blur, max(5, base)))
    return radius, blur

# ─── Landing Page UI ─────────────────────────────────────────────────────────
with st.sidebar.expander("🧭 Start", expanded=True):
    mode = st.radio("Choose:", ["Start new report", "Open past report"])

from datetime import datetime
import os, json

# --- Past runs selector & actions (REPLACE your current block with this) ---
if mode == "Open past report":
    runs = list_runs()
    if not runs:
        st.info("No past runs found yet.")
        st.stop()

    # Nice labels for the dropdown
    def _label(row):
        try:
            dt = datetime.strptime(row["stamp"], "%Y%m%d-%H%M%S")
            when = dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            when = row["stamp"]
        return f"{row['agency']} / {when}"

    idx = st.selectbox(
        "Select a past run",
        options=range(len(runs)),
        format_func=lambda i: _label(runs[i]),
    )
    r = runs[idx]

    # Load config/metrics (config may be missing; that's fine)
    cfg_path = os.path.join(r["path"], "config.json")
    met_path = os.path.join(r["path"], "metrics.json")

    cfg = {}
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r") as f:
                cfg = json.load(f)
        except Exception:
            cfg = {}

    # --- Safe fallbacks for the header metrics
    agency_name = cfg.get("agency_name", r["agency"])
    run_by      = cfg.get("analyst_name", "Unknown")

    # Prefer saved local time (if present), else use the folder stamp
    if cfg.get("run_time_iso_local"):
        try:
            dt_local = datetime.fromisoformat(cfg["run_time_iso_local"])
            when_str = dt_local.strftime("%b %d, %Y — %I:%M %p")
            tz_short = cfg.get("run_timezone", "")
            if tz_short:
                when_str += f" {tz_short}"
        except Exception:
            try:
                run_dt = datetime.strptime(r["stamp"], "%Y%m%d-%H%M%S")
                when_str = run_dt.strftime("%b %d, %Y — %I:%M %p")
            except Exception:
                when_str = r["stamp"]
    else:
        try:
            run_dt = datetime.strptime(r["stamp"], "%Y%m%d-%H%M%S")
            when_str = run_dt.strftime("%b %d, %Y — %I:%M %p")
        except Exception:
            when_str = r["stamp"]

    # Display summary of the saved run
    c1, c2, c3 = st.columns(3)
    c1.metric("Agency", agency_name)
    c2.metric("Run by", run_by)
    c3.metric("When", when_str)

    # Actions: view saved (no compute) OR re-run with stored inputs
    c1b, c2b = st.columns(2)
    view_btn  = c1b.button("View saved metrics (no re-run)")
    rerun_btn = c2b.button("Re-run this report with stored inputs", type="primary")

    if view_btn:
        st.session_state["viewing_saved"] = True
        st.session_state["loaded_run_dir"] = r["path"]
        st.session_state["loaded_config"]  = cfg
        st.rerun()

    if rerun_btn:
        st.session_state["replay_dir"]    = r["path"]
        st.session_state["replay_config"] = cfg   # pass assumptions/names/etc.
        st.session_state.pop("viewing_saved", None)
        st.success("Replaying this run with the saved CSVs and current code…")
        st.rerun()

# st.sidebar.write("DEBUG state:", dict(st.session_state))


# ─── 0) PROGRESS BAR ──────────────────────────────────────────────────────────
progress = st.sidebar.progress(0)

# ─── 0) Optional ZIP Upload (Pre-populates other sections) ───────────────────
import zipfile
from io import BytesIO

# 0a) Helper exists ALWAYS (even if no ZIP uploaded)
def _find_csv_by_partial(partial: str):
    """Return BytesIO of a CSV from an uploaded ZIP by partial filename match."""
    files = st.session_state.get("zip_files")
    if not files:
        return None
    for name, data in files.items():
        if partial.lower() in name.lower():
            return BytesIO(data)
    return None

st.sidebar.header("0) Bulk ZIP Upload (optional)")
bundle_zip_file = st.sidebar.file_uploader(
    "Upload ZIP containing all CSV files (optional)",
    type=["zip"],
    key="bundle_zip"   # make sure this key is used ONLY here in the app
)

# 0b) If a ZIP is uploaded, cache its CSV bytes in session_state
if bundle_zip_file is not None:
    with zipfile.ZipFile(bundle_zip_file, "r") as zf:
        st.session_state["zip_files"] = {
            name: zf.read(name)
            for name in zf.namelist()
            if name.lower().endswith(".csv")
        }
    st.session_state["agency_name_guess"] = extract_agency_name(bundle_zip_file.name)
    st.sidebar.success("ZIP file processed — files loaded into their sections below.")

# 0c) Pre-populate replay_inputs from ZIP (no-ops if no ZIP)
replay_inputs["raw"]    = _find_csv_by_partial("Raw Call Data")
replay_inputs["agency"] = _find_csv_by_partial("Agency Call Types")
replay_inputs["launch"] = _find_csv_by_partial("Launch Locations")
replay_inputs["alpr"]   = _find_csv_by_partial("LPR Hits by Camera")
replay_inputs["audio"]  = _find_csv_by_partial("Audio Hits Aggregated")

# ─── 1) SIDEBAR: UPLOADS & EDITORS ───────────────────────────────────────────


# RAW
st.sidebar.header("1) Raw Call Data")
raw_file = (
    replay_inputs.get("raw") or
    st.sidebar.file_uploader("Upload Raw Call Data CSV", type=["csv"], key="raw_csv")
)

if not raw_file:
    zip_loaded  = bool(st.session_state.get("zip_files"))
    zip_matched = any(replay_inputs.get(k) is not None for k in ["raw","agency","launch","alpr","audio"])
    if zip_loaded and not zip_matched:
        st.sidebar.error("ZIP uploaded, but no expected CSVs were recognized. Check filenames. See ZIP contents above.")
    else:
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
    file_name="Agency Call Types.csv",
    mime="text/csv",
    help="Fill in Y/N for each row, then re-upload under “Agency Call Types.”"
)

# ─── 2) Launch Locations ────────────────────────────────────────────────────
st.sidebar.header("2) Launch Locations")

# Source priority: REPLAY → manual upload → inline editor
launch_src = None
launch_file = None

if replay_inputs.get("launch"):
    launch_file = replay_inputs["launch"]
    launch_src = "replay"
else:
    launch_file = st.sidebar.file_uploader(
        "Upload Launch Locations CSV (with any of: Name, Address, Lat, Lon, Type)",
        type=["csv"],
        key="launch_csv"
    )
    launch_src = "upload" if launch_file else None

# ─── Launch Locations Loader ──────────────────────────────

def _load_launch_locations_csv(file_obj):
    """
    Supports two formats:
      NEW:  Row1: A1='Agency Name', B1='<name>'
             Row2: A2='Sq. Mi.',     B2='<area>'
             Row3: headers for the table
             Row4+: data
      OLD:  Plain CSV with headers in first row.
    Returns: (launch_df, meta_agency_name_or_None, meta_city_area_sqmi_or_None)
    """
    import pandas as pd
    from io import BytesIO

    raw = file_obj.read()
    try:
        # Try to sniff "new" format
        df0 = pd.read_csv(BytesIO(raw), header=None, dtype=str, keep_default_na=False)
        looks_new = False
        if df0.shape[0] >= 3 and df0.shape[1] >= 2:
            a1 = str(df0.iat[0, 0]).strip().lower()
            a2 = str(df0.iat[1, 0]).strip().lower()
            looks_new = ("agency" in a1 and "name" in a1) and ("sq" in a2 and "mi" in a2)

        if looks_new:
            meta_agency = (df0.iat[0, 1] or "").strip() or None
            try:
                meta_sqmi = float(str(df0.iat[1, 1]).replace(",", "").strip())
            except Exception:
                meta_sqmi = None

            headers = df0.iloc[2].tolist()
            data = df0.iloc[3:].copy()
            data.columns = headers
            return data.reset_index(drop=True), meta_agency, meta_sqmi

        # fallback: old style
        df_old = pd.read_csv(BytesIO(raw))
        return df_old, None, None

    except Exception:
        df_old = pd.read_csv(BytesIO(raw))
        return df_old, None, None

if launch_file is not None:
    st.sidebar.success(
        "Loaded launch locations from saved run." if launch_src == "replay"
        else "Using uploaded launch locations file."
    )
    # Load the CSV and pull metadata in one place
    def _load_launch_locations_csv(file):
    # Read metadata only
    meta_df = pd.read_csv(file, nrows=2, header=None)
    agency   = str(meta_df.iat[0,1]).strip() if pd.notna(meta_df.iat[0,1]) else None
    sqmi     = float(meta_df.iat[1,1]) if pd.notna(meta_df.iat[1,1]) else None
    
    # Reset pointer and read the actual launch data (skip 2 rows)
    file.seek(0)
    df = pd.read_csv(file, skiprows=2)
    return df, agency, sqmi
    try:
        launch_file.seek(0)  # reset pointer just in case
    except Exception:
        pass
else:
    if _EDITOR is None:
        st.sidebar.error("Upgrade Streamlit or upload a CSV with launch locations.")
        st.stop()
    launch_df = _EDITOR(
        pd.DataFrame(columns=["Location Name","Address","Lat","Lon","Type"]),
        num_rows="dynamic",
        use_container_width=True,
        key="launch_editor"
    )
    _meta_agency, _meta_sqmi = None, None

# ─── Safely push metadata into session state ─────────────────────────────

if _meta_agency:
    if "agency_name" not in st.session_state:
        st.session_state["agency_name"] = _meta_agency
    elif not (st.session_state.get("agency_name") or "").strip():
        st.session_state["agency_name"] = _meta_agency

if _meta_sqmi and _meta_sqmi > 0:
    if "city_area_sqmi" not in st.session_state:
        st.session_state["city_area_sqmi"] = float(_meta_sqmi)
    elif not st.session_state.get("city_area_sqmi"):
        st.session_state["city_area_sqmi"] = float(_meta_sqmi)

# 2b) Normalize headers and make sure required columns exist
launch_df.columns = [c.strip() for c in launch_df.columns]

# Accept common synonyms
if "Location Name" not in launch_df.columns and "Locations" in launch_df.columns:
    launch_df.rename(columns={"Locations": "Location Name"}, inplace=True)
if "Lon" not in launch_df.columns and "Long" in launch_df.columns:
    launch_df.rename(columns={"Long": "Lon"}, inplace=True)

for col in ["Location Name","Address","Lat","Lon","Type"]:
    if col not in launch_df.columns:
        launch_df[col] = ""

# Save column names for later pricing logic
st.session_state["launch_columns"] = list(launch_df.columns)

# Split rows by Type (case/space tolerant)
_type = launch_df["Type"].astype(str).str.strip().str.lower()
is_launch  = _type.eq("launch location")
is_hotspot = _type.eq("hotspot address")

launch_rows  = launch_df.loc[is_launch].copy()
hotspot_rows = launch_df.loc[is_hotspot].copy()

# Stash hotspot addresses so Step 6 can prefill the text input
st.session_state["hotspot_addresses"] = (
    hotspot_rows["Address"].dropna().astype(str).str.strip().tolist()
)

# ─── 2c) Geocode: only for LAUNCH rows that have Address but missing Lat/Lon ─
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

import re
def normalize_address(addr: str) -> str:
    if not isinstance(addr, str):
        return addr
    replacements = {
        r"\bSt\b": "Street", r"\bRd\b": "Road", r"\bAve\b": "Avenue",
        r"\bBlvd\b": "Boulevard", r"\bDr\b": "Drive", r"\bLn\b": "Lane",
        r"\bHwy\b": "Highway", r"\bPkwy\b": "Parkway", r"\bCt\b": "Court",
        r"\bPl\b": "Place", r"\bSq\b": "Square",
    }
    out = addr
    for pat, rep in replacements.items():
        out = re.sub(pat, rep, out, flags=re.IGNORECASE)
    out = re.sub(r"\b\d{5}(?:-\d{4})?\b", "", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out

to_geocode = launch_rows["Address"].notna() & (
    pd.to_numeric(launch_rows["Lat"], errors="coerce").isna() |
    pd.to_numeric(launch_rows["Lon"], errors="coerce").isna()
)

for idx in launch_rows.loc[to_geocode].index:
    clean_addr = normalize_address(launch_rows.at[idx, "Address"])
    lat, lon = lookup(clean_addr)
    launch_rows.at[idx, "Lat"] = lat
    launch_rows.at[idx, "Lon"] = lon

# Preview (separate views help sanity‑check)
st.sidebar.subheader("Launch Locations (geocoded)")
st.sidebar.dataframe(launch_rows[["Location Name","Address","Lat","Lon"]], use_container_width=True)
if not hotspot_rows.empty:
    st.sidebar.subheader("Hotspot Addresses (from CSV)")
    st.sidebar.dataframe(hotspot_rows[["Location Name","Address"]], use_container_width=True)

# 2d) Validation: require coords ONLY for launch rows
valid_coords = (
    pd.to_numeric(launch_rows["Lat"], errors="coerce").notna() &
    pd.to_numeric(launch_rows["Lon"], errors="coerce").notna()
)
if not valid_coords.all():
    bad = launch_rows.loc[~valid_coords, ["Location Name","Address","Lat","Lon"]]
    st.sidebar.error(
        "Some LAUNCH rows lack valid Lat/Lon:\n" + bad.to_csv(index=False)
    )
    st.stop()

# 2e) Build launch_coords for downstream use (maps/ETA)
launch_rows["Lat"] = pd.to_numeric(launch_rows["Lat"], errors="coerce")
launch_rows["Lon"] = pd.to_numeric(launch_rows["Lon"], errors="coerce")
launch_coords = list(launch_rows[["Lat","Lon"]].itertuples(index=False, name=None))

# 0d) Agency Name (from ZIP name or manual)
st.sidebar.header("2) Agency Name")

# Initialize once from Launch-CSV metadata or ZIP guess
if "agency_name" not in st.session_state or not (st.session_state["agency_name"] or "").strip():
    st.session_state["agency_name"] = (
        st.session_state.get("agency_name") 
        or (_meta_agency or "") 
        or st.session_state.get("agency_name_guess", "")
    )

# The widget OWNS this key. Do not write to this key anywhere else.
st.sidebar.text_input("Enter Agency Name", key="agency_name")

AGENCY_NAME = (st.session_state.get("agency_name") or "").strip()

if AGENCY_NAME:
    st.markdown(f"# {AGENCY_NAME}")
    st.markdown("## DFR Impact Analysis")
else:
    st.title("DFR Impact Analysis")

progress.progress(30)

# ─── 3) Agency Call Types ──────────────────────────────────────────────────
st.sidebar.header("3) Agency Call Types")

# Source priority: REPLAY → ZIP bundle → manual upload
ag_src = None
ag_file = None

if replay_inputs.get("agency"):
    ag_file = replay_inputs["agency"]
    ag_src = "replay"
else:
    ag_file = st.sidebar.file_uploader(
        "Upload Agency Call Types CSV",
        type=["csv"],
        key="agency_csv"
    )
    ag_src = "upload" if ag_file else None

if not ag_file:
    st.sidebar.error("Please provide Agency Call Types (replay, bundle, or upload).")
    st.stop()

if ag_src == "replay":
    st.sidebar.success("Loaded agency call types from saved run.")
else:
    st.sidebar.info("Using uploaded agency call types file.")

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
# Prefer values remembered from replay_config → assumptions
a = replay_cfg.get("assumptions", {}) or {}
fte_hours    = st.sidebar.number_input("Full Time Work Year (hrs)", value=int(a.get("fte_hours", 2080)), step=1)
officer_cost = st.sidebar.number_input("Officer Cost per FTE ($)", value=int(a.get("officer_cost_usd", 127940)), step=1000, format="%d")
cancel_rate  = st.sidebar.number_input("Drone Cancellation Rate (0–1)", value=float(a.get("cancel_rate", 0.11)), step=0.01, format="%.2f")
drone_speed  = st.sidebar.number_input("Drone Speed (mph)", value=float(a.get("drone_speed_mph", 51.0)), step=1.0)
drone_range  = st.sidebar.number_input("Drone Range (miles)", value=float(a.get("drone_range_miles", 3.5)), step=0.1)
progress.progress(70)

# ─── Agency details (saved with each run) ────────────────────────────────────
with st.sidebar.expander("Agency details", expanded=True):
    analyst_name = st.text_input("Analyst (optional)", value="", key="analyst_name")
    run_notes = st.text_area("Run notes (optional)", height=80, key="run_notes")

st.sidebar.header("5) ALPR & Audio (optional)")

# Optional: configure via env var; falls back to placeholder
ALPR_AUDIO_DB_URL = os.getenv(
    "ALPR_AUDIO_DB_URL",
    "https://app.sigmacomputing.com/flock-safety/workbook/pdq-request-lpr-gsd-hits-21CqYotoy7AkzXcViUnzmp?:nodeId=UDm7Bv0y6f&:customView=cb177da1-cb59-4e1d-826d-f166721bd4db"
)

st.sidebar.markdown(
    f'**Data source:** <a href="{ALPR_AUDIO_DB_URL}" target="_blank">Open ALPR/Audio database ↗</a>',
    unsafe_allow_html=True
)

# === ALPR ===
alpr_src = None
alpr_file = None

if replay_inputs.get("alpr"):
    alpr_file = replay_inputs["alpr"]
    alpr_src = "replay"
else:
    alpr_file = st.sidebar.file_uploader("Upload ALPR Data CSV", type=["csv"])
    alpr_src = "upload" if alpr_file else None

if alpr_src == "replay":
    st.sidebar.success("Loaded ALPR CSV from saved run.")

# === Audio ===
audio_src = None
audio_file = None

if replay_inputs.get("audio"):
    audio_file = replay_inputs["audio"]
    audio_src = "replay"
else:
    audio_file = st.sidebar.file_uploader("Upload Audio Hits CSV", type=["csv"])
    audio_src = "upload" if audio_file else None

if audio_src == "replay":
    st.sidebar.success("Loaded Audio CSV from saved run.")

# ─── 6) Hotspot Area ──────────────────────────────────────────
st.sidebar.header("6) Hotspot Area")

# 1) Address input, prefilled from CSV if available
default_hotspot = (st.session_state.get("hotspot_addresses") or [None])[0]
hotspot_address = st.sidebar.text_input(
    "Enter Hotspot Address (0.5 mi radius)",
    value=default_hotspot if default_hotspot else "",
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
audio_df = pd.read_csv(audio_file) if audio_file else None

# --- Unfiltered totals for ALPR + Audio (for the report/debug) ---
total_alpr_hits  = 0
total_audio_hits = 0

if alpr_df is not None and not alpr_df.empty:
    # Sum all hits in "Sum of hits" col (col 6 index if not renamed)
    hits_col = "Sum of hits" if "Sum of hits" in alpr_df.columns else alpr_df.columns[6]
    total_alpr_hits = pd.to_numeric(alpr_df[hits_col], errors="coerce").fillna(0).sum()

if audio_df is not None and not audio_df.empty:
    # Sum all hits in "Number of hits" col (col F index if not renamed)
    hits_col = "Number of hits" if "Number of hits" in audio_df.columns else audio_df.columns[5]
    total_audio_hits = pd.to_numeric(audio_df[hits_col], errors="coerce").fillna(0).sum()

total_alpr_audio = int(total_alpr_hits + total_audio_hits)

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


        # ── HEATMAP DATA = ALL VALID POINTS (not only in-range) ───────────────
        audio_pts = pd.DataFrame({
            "lat":   lat_v.values,
            "lon":   lon_v.values,
            "count": hits_v.values,   # optional intensity for HeatMap
        })

# combine for your overall “DFR + ALPR + Audio” metric
dfr_alpr_audio = alpr_hits + audio_hits

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



# ─── Auto-save this run ──────────────────────────────────────────────────────
try:
    # Build a clean metrics dict with raw numbers (not pretty strings)
    metrics_dict = {
        "total_cfs": int(total_cfs),
        "total_alpr_audio_hits_citywide": int(total_alpr_audio) if 'total_alpr_audio' in locals() else None,
        "dfr_responses_within_range": int(in_count),
        "dfr_responses_to_alpr_audio_within_range": int(dfr_alpr_audio),
        "expected_drone_eta_sec": float(avg_drone) if np.isfinite(avg_drone) else None,
        "expected_first_on_scene_pct": float(first_on_pct) if np.isfinite(first_on_pct) else None,
        "expected_cfs_cleared": int(exp_cleared),
        "officers_fte": float(officers) if np.isfinite(officers) else None,
        "roi_usd": float(roi) if np.isfinite(roi) else None,
        "total_potential_dfr_calls": int(total_dfr),
        "avg_patrol_to_dfr_sec": float(avg_patrol) if np.isfinite(avg_patrol) else None,
        "avg_scene_time_dfr_sec": float(avg_scene) if np.isfinite(avg_scene) else None,
        "avg_inrange_patrol_sec": float(avg_in) if np.isfinite(avg_in) else None,
        "expected_decrease_response_times_pct": float(pct_dec) if np.isfinite(pct_dec) else None,
        "p1_inrange_count": int(p1_count),
        "avg_p1_patrol_sec": float(avg_p1_pat) if np.isfinite(avg_p1_pat) else None,
        "avg_p1_drone_eta_sec": float(avg_p1_drone) if np.isfinite(avg_p1_drone) else None,
        "hotspot_count": int(hotspot_count),
        "hotspot_avg_patrol_sec": float(hotspot_avg_patrol) if np.isfinite(hotspot_avg_patrol) else None,
        "hotspot_avg_drone_eta_sec": float(hotspot_avg_drone) if np.isfinite(hotspot_avg_drone) else None,
        "alpr_sites_in_range": int(alpr_sites) if 'alpr_sites' in locals() else 0,
        "alpr_hits_in_range_reason_filtered": int(alpr_hits) if 'alpr_hits' in locals() else 0,
        "alpr_eta_sec": float(alpr_eta) if ('alpr_eta' in locals() and np.isfinite(alpr_eta)) else None,
        "audio_sites_in_range": int(audio_sites) if 'audio_sites' in locals() else 0,
        "audio_hits_in_range": int(audio_hits) if 'audio_hits' in locals() else 0,
        "audio_eta_sec": float(audio_eta) if ('audio_eta' in locals() and np.isfinite(audio_eta)) else None,
        "clearable_in_range": int(clr_count),
        "avg_clearable_scene_sec": float(avg_clr) if np.isfinite(avg_clr) else None,
        "total_time_on_clearable_sec": float(clr_count * avg_clr) if np.isfinite(avg_clr) else None,
    }

    # Config/context so we can reproduce the run later
    config_dict = {
        "agency_name": AGENCY_NAME or "unknown_agency",
        "analyst_name": (analyst_name if 'analyst_name' in locals() else st.session_state.get("analyst_name", "")),
        "notes": (run_notes if 'run_notes' in locals() else st.session_state.get("run_notes", "")),
        "run_time_iso": datetime.now().isoformat(),
        "assumptions": {
            "fte_hours": int(fte_hours),
            "officer_cost_usd": int(officer_cost),
            "cancel_rate": float(cancel_rate),
            "drone_speed_mph": float(drone_speed),
            "drone_range_miles": float(drone_range),
        },
        "launch_sites_count": len(launch_coords) if 'launch_coords' in locals() else 0,
        "hotspot": {
            "address": hotspot_address if 'hotspot_address' in locals() else None,
            "coords": list(hotspot_coords[0]) if (hotspot_coords) else None,
            "radius_miles": 0.5 if hotspot_coords else None,
        },
        "dfr_map_values_sample": list(sorted(dfr_map))[:25],
        "clearable_map_values_sample": list(sorted(clr_map))[:25],
        "raw_counts": {
            "raw_uploaded_rows": int(raw_count),
            "df_all_rows": int(len(df_all)),
            "dfr_only_rows": int(len(dfr_only)),
            "in_range_rows": int(len(in_range)),
            "clearable_rows": int(len(clearable)),
        },
        "app_version": "auto-logger-v1",
    }

    # Inputs we’ll save a copy of (so a past run can be replayed)
    input_files_dict = {
        "raw_calls.csv": raw_file,
        "agency_call_types.csv": ag_file,
        # If no CSV upload for launch sites, save the in-memory DataFrame
        "launch_locations.csv": (
            launch_file if 'launch_file' in locals() and launch_file is not None
            else BytesIO(launch_df.to_csv(index=False).encode("utf-8"))
        ),
        "alpr.csv": alpr_file,
        "audio.csv": audio_file,
    }

    # If a ZIP was uploaded, save it as well for replay
    if bundle_zip_file is not None:
        input_files_dict["bundle.zip"] = bundle_zip_file

    # Rewind streams so save_run can read them from the start
    for _f in input_files_dict.values():
        if _f is not None and hasattr(_f, "seek"):
            try:
                _f.seek(0)
            except Exception:
                pass

    run_dir = save_run(
        AGENCY_NAME or "unknown_agency",
        config_dict=config_dict,
        metrics_dict=metrics_dict,
        input_files_dict=input_files_dict,
        map_images=None,
        pdf_bytes=None
    )

    st.sidebar.success(f"📦 Run saved: {run_dir}")
    st.session_state["last_run_dir"] = run_dir

except Exception as e:
    st.sidebar.warning(f"Couldn’t auto-save this run: {e}")

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

# ─── 5.5) TOP SUMMARY (matches PDF headline metrics) ─────────────────────────
st.markdown("---")
st.markdown("### Summary")

def metric_row(*pairs):
    cols = st.columns(len(pairs))
    for c, (label, value, kind) in zip(cols, pairs):
        c.metric(label, pretty_value(value, kind))

# Row 1 — headline ops metrics
metric_row(
    ("Total CFS", total_cfs, "int"),
    ("Total Potential DFR Calls", total_dfr, "int"),
    ("DFR Responses within Range", in_count, "int"),
    ("DFR + ALPR/Audio (in-range)", dfr_alpr_audio, "int"),
)

# Row 2 — time/impact highlights
metric_row(
    ("Expected DFR Drone Response (avg)", avg_drone, "mmss"),
    ("Avg Patrol Response to In-Range Calls", avg_in, "mmss"),
    ("Expected First on Scene %", first_on_pct, "pct"),
    ("Expected Decrease in Response Times", pct_dec, "pct"),
)

# Row 3 — outcomes
metric_row(
    ("Expected CFS Cleared", exp_cleared, "int"),
    ("Officers (FTE) Saved", officers, "2dec"),
    ("ROI (USD)", roi, "usd"),
    ("Clearable CFS In Range", clr_count, "int"),
)

# ─── 6) MAPS & HEATMAPS ──────────────────────────────────────────────────────
st.markdown("---")
st.header("Maps & Heatmaps")

def metrics_under(title, *pairs):
    st.caption(title)
    cols = st.columns(len(pairs))
    for c, (label, value, kind) in zip(cols, pairs):
        c.metric(label, pretty_value(value, kind))

# Prep subsets you already built
all_dfr       = dfr_only.copy()
all_p1        = all_dfr[all_dfr["priority"] == "1"]
all_clearable = all_dfr[all_dfr["call_type_up"].isin(clr_map)]

# Hotspot subset (0.5 mi)
if hotspot_coords:
    hs_dist = haversine_min(all_dfr["lat"].values, all_dfr["lon"].values, hotspot_coords)
    hotspot_calls = all_dfr[hs_dist <= 0.5].copy()
else:
    hotspot_calls = pd.DataFrame()

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

# 6a) Heatmap: All DFR Calls
r0, b0 = auto_heat_params(all_dfr)
r_all = st.sidebar.slider("All DFR Calls Heat Radius", 1, 50, value=r0, key="all_r")
b_all = st.sidebar.slider("All DFR Calls Heat Blur",   1, 50, value=b0, key="all_b")
render_map(
    all_dfr,
    heat=True,
    heat_radius=r_all, heat_blur=b_all,
    title="",
    key="map_all_heat",
    show_circle=True,
    launch_coords=launch_coords
)
metrics_under(
    "All DFR Calls — key stats",
    ("Total Potential DFR Calls", total_dfr, "int"),
    ("DFR within Range", in_count, "int"),
    ("Avg Patrol (DFR)", avg_patrol, "mmss"),
    ("Avg DFR Drone Response", avg_drone, "mmss"),
    ("First on Scene %", first_on_pct, "pct"),
    ("Decrease in Response Times", pct_dec, "pct"),
)

st.markdown("---")

# 6b) 3.5-mile Drone Range (circle only)
render_map(
    pd.DataFrame(),
    heat=False,
    title="",
    key="map_range_circle",
    show_circle=True,
    launch_coords=launch_coords
)
metrics_under(
    "Range — key stats",
    ("DFR within Range", in_count, "int"),
    ("Avg DFR Drone Response (in-range)", avg_drone, "mmss"),
    ("P1 Calls in Range", p1_count, "int"),
    ("Avg P1 Patrol (in-range)", avg_p1_pat, "mmss"),
    ("Avg P1 Drone Response (in-range)", avg_p1_drone, "mmss"),
)

st.markdown("---")

# 6c) Heatmap: P1 DFR Calls
r_p1 = st.sidebar.slider("P1 DFR Heat Radius", 1, 50, value=9, key="p1_r")
b_p1 = st.sidebar.slider("P1 DFR Heat Blur",   1, 50, value=9, key="p1_b")
render_map(
    all_p1,
    heat=True,
    heat_radius=r_p1, heat_blur=b_p1,
    title="",
    key="map_p1_heat",
    show_circle=True,
    launch_coords=launch_coords
)
metrics_under(
    "P1 — key stats",
    ("P1 Calls in Range", p1_count, "int"),
    ("Avg P1 Patrol (in-range)", avg_p1_pat, "mmss"),
    ("Avg P1 Drone Response (in-range)", avg_p1_drone, "mmss"),
)

st.markdown("---")

# 6d) Heatmap: All DFR Calls + Hotspot Overlay
if hotspot_coords:
    r_hs, b_hs = auto_heat_params(all_dfr)
    r_hs = st.sidebar.slider("Hotspot Heat Radius", 1, 50, value=r_hs, key="hs_r")
    b_hs = st.sidebar.slider("Hotspot Heat Blur",   1, 50, value=b_hs, key="hs_b")

    render_map(
        all_dfr,
        heat=True,
        heat_radius=r_hs, heat_blur=b_hs,
        title="",
        key="map_hotspot_heat",
        show_circle=True,
        launch_coords=launch_coords,
        hotspot_center=hotspot_coords[0],
        hotspot_radius=0.5
    )
    metrics_under(
        "Hotspot — key stats (≤ 0.5 mi)",
        ("DFR Calls in Hotspot", hotspot_count, "int"),
        ("Avg Patrol (hotspot)", hotspot_avg_patrol, "mmss"),
        ("Avg Drone Response (hotspot)", hotspot_avg_drone, "mmss"),
    )
    st.markdown("---")

# 6e) Heatmap: Clearable DFR Calls
r2, b2 = auto_heat_params(all_clearable)
r_cl = st.sidebar.slider("Clearable Heat Radius", 1, 50, value=r2, key="clr_r")
b_cl = st.sidebar.slider("Clearable Heat Blur",   1, 50, value=b2, key="clr_b")
render_map(
    all_clearable,
    heat=True,
    heat_radius=r_cl, heat_blur=b_cl,
    title="",
    key="map_clearable_heat",
    show_circle=True,
    launch_coords=launch_coords
)
metrics_under(
    "Clearable — key stats",
    ("Clearable CFS in Range", clr_count, "int"),
    ("Avg Time on Scene (clearable)", avg_clr, "mmss"),
    ("Expected CFS Cleared", exp_cleared, "int"),
    ("Officers (FTE)", officers, "2dec"),
    ("ROI (USD)", roi, "usd"),
)

st.markdown("---")

# 6f) Heatmap: ALPR Locations
if alpr_df is not None:
    alpr_pts = pd.DataFrame({
        "lat": pd.to_numeric(alpr_df.iloc[:, 4], errors="coerce"),
        "lon": pd.to_numeric(alpr_df.iloc[:, 5], errors="coerce")
    }).dropna()

    r_al = st.sidebar.slider("ALPR Heat Radius", 1, 50, value=6, key="alpr_r")
    b_al = st.sidebar.slider("ALPR Heat Blur",   1, 50, value=4, key="alpr_b")

    render_map(
        alpr_pts,
        heat=True,
        heat_radius=r_al, heat_blur=b_al,
        title="",
        key="map_alpr_heat",
        show_circle=True,
        launch_coords=launch_coords
    )
    metrics_under(
        "ALPR — key stats (in-range rules applied for metrics)",
        ("ALPR Sites (in-range)", alpr_sites, "int"),
        ("Hits (whitelist, in-range)", alpr_hits, "int"),
        ("Avg Drone Response (hits-weighted)", alpr_eta, "mmss"),
    )
    st.markdown("---")

# 6g) Heatmap: Audio Locations
if audio_pts is not None and not audio_pts.empty:
    r_au = st.sidebar.slider("Audio Heat Radius", 1, 50, value=4, key="audio_r")
    b_au = st.sidebar.slider("Audio Heat Blur",   1, 50, value=4, key="audio_b")

    render_map(
        audio_pts,
        heat=True,
        heat_radius=r_au, heat_blur=b_au,
        title="",
        key="map_audio_heat",
        show_circle=True,
        launch_coords=launch_coords
    )
    metrics_under(
        "Audio — key stats (in-range rules applied for metrics)",
        ("Audio Locations (in-range)", audio_sites, "int"),
        ("Audio Hits (in-range)", audio_hits, "int"),
        ("Avg Drone Response (hits-weighted)", audio_eta, "mmss"),
    )
else:
    st.sidebar.info("No audio points to display on the heatmap.")


# ─── 7) PRICING ───────────────────────────────────────────────────────────────
st.markdown("---")
st.header("Pricing")

# --- Yearly unit prices (no CapEx) ---
PRICE_PER_DOCK_BY_TYPE = {
    "DOCK 3": 50000,
    "ALPHA": 125000,
    "DELTA": 300000,
}
PRICE_PER_RADAR = 150000  # yearly

# Ensure optional pricing columns exist in launch_rows
# Expected columns: ["Location Name","Address","Lat","Lon","Dock Type","Number of Docks","Number of Radar"]
for col, default in [("Dock Type", ""), ("Number of Docks", 0), ("Number of Radar", 0)]:
    if col not in launch_rows.columns:
        launch_rows[col] = default

# Normalize and coerce
_lr = launch_rows.copy()
_lr["Dock Type"] = _lr["Dock Type"].astype(str).str.strip().str.upper()
_lr["Number of Docks"] = pd.to_numeric(_lr["Number of Docks"], errors="coerce").fillna(0).astype(int)
_lr["Number of Radar"] = pd.to_numeric(_lr["Number of Radar"], errors="coerce").fillna(0).astype(int)

# Compute per-site costs
def dock_unit_price(dock_type):
    return PRICE_PER_DOCK_BY_TYPE.get(str(dock_type).upper().strip(), 0)

_lr["Dock Unit Price"]   = _lr["Dock Type"].map(dock_unit_price)
_lr["Dock Yearly Cost"]  = (_lr["Number of Docks"] * _lr["Dock Unit Price"]).astype(int)
_lr["Radar Yearly Cost"] = (_lr["Number of Radar"] * PRICE_PER_RADAR).astype(int)
_lr["Site Yearly Total"] = (_lr["Dock Yearly Cost"] + _lr["Radar Yearly Cost"]).astype(int)

# Totals
total_launch_sites = len(_lr)
total_docks  = int(_lr["Number of Docks"].sum())
total_radars = int(_lr["Number of Radar"].sum())
list_total   = int(_lr["Site Yearly Total"].sum())

# Discount input (sidebar)
st.sidebar.header("Pricing Options")
discount_pct = st.sidebar.number_input("Discount (%)", min_value=0.0, max_value=100.0, value=0.0, step=1.0, key="pricing_discount_pct")
discount_rate = float(discount_pct) / 100.0
discount_amount = int(round(list_total * discount_rate))
discounted_total = int(list_total - discount_amount)

# Recommended dock type(s): based on CSV values present
present_types = [t for t in _lr["Dock Type"].unique().tolist() if t]
if len(present_types) > 1:
    top_type = (_lr.groupby("Dock Type")["Number of Docks"].sum().sort_values(ascending=False).index.tolist() or [""])[0]
    recommended_label = f"{', '.join(present_types)}  (top: {top_type})"
else:
    recommended_label = present_types[0] if present_types else "—"

# --- Pricing map (its own rendering) ---
st.subheader("Pricing Map (Launch Sites + Range)")
pricing_pts = pd.DataFrame({
    "lat": pd.to_numeric(_lr["Lat"], errors="coerce"),
    "lon": pd.to_numeric(_lr["Lon"], errors="coerce")
}).dropna(subset=["lat","lon"])
render_map(
    pricing_pts,
    heat=False,
    title="",
    key="map_pricing",
    show_circle=True,
    launch_coords=launch_coords
)

# --- Summary row ---
def _fmt_usd(x): return f"${x:,.0f}"

c1, c2, c3, c4 = st.columns(4)
c1.metric("Launch Locations", f"{total_launch_sites:,}")
c2.metric("Total Docks", f"{total_docks:,}")
c3.metric("Total Radars", f"{total_radars:,}")
c4.metric("Recommended Dock Type(s)", recommended_label)

# Always show list cost
c5, c6 = st.columns(2)
c5.metric("Yearly Cost (List)", _fmt_usd(list_total))

# Show discounted cost only if >0
if discount_pct > 0:
    c6.metric(f"Yearly Cost (Discounted {int(discount_pct)}%)", _fmt_usd(discounted_total))

# --- Expandable per-site breakdown ---
with st.expander("Per-site pricing details"):
    detail_df = _lr[[
        "Location Name", "Address", "Dock Type", "Number of Docks", "Dock Unit Price",
        "Dock Yearly Cost", "Number of Radar", "Radar Yearly Cost", "Site Yearly Total"
    ]].copy()
    # Nice formatting for display
    _money_cols = ["Dock Unit Price", "Dock Yearly Cost", "Radar Yearly Cost", "Site Yearly Total"]
    for mc in _money_cols:
        detail_df[mc] = detail_df[mc].map(lambda v: _fmt_usd(int(v)))

    st.dataframe(detail_df, use_container_width=True)

    # Totals row (rendered separately under the table)
    st.caption(
        f"**Totals:** Docks={total_docks:,} • Radars={total_radars:,} • "
        f"List={_fmt_usd(list_total)} • Discount={_fmt_usd(discount_amount)} • "
        f"Discounted Total={_fmt_usd(discounted_total)}"
    )

# ─── 7) COMPARISON (no GeoPandas; city limits from call data) ────────────────
st.markdown("---")
st.header("Comparison")

import math
import numpy as np
import pandas as pd
import folium
from streamlit_folium import st_folium
from shapely.geometry import Point, MultiPoint
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from pyproj import Transformer
import alphashape

SQM_PER_SQMI = 2_589_988.110336

# ---------- UTM helpers ----------
def _utm_epsg_for(lat_deg: float, lon_deg: float) -> int:
    zone = int((lon_deg + 180) // 6) + 1
    north = lat_deg >= 0
    return (32600 if north else 32700) + zone

def _make_transformers(lat_arr, lon_arr):
    lat0 = float(np.nanmean(lat_arr)) if len(lat_arr) else 0.0
    lon0 = float(np.nanmean(lon_arr)) if len(lon_arr) else 0.0
    epsg = _utm_epsg_for(lat0, lon0)
    fwd = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    inv = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    return fwd, inv

def _project_points(fwd: Transformer, lats, lons):
    x, y = fwd.transform(np.asarray(lons, dtype=float), np.asarray(lats, dtype=float))
    return np.asarray(x), np.asarray(y)

def _unproject_points(inv: Transformer, xs, ys):
    lon, lat = inv.transform(np.asarray(xs, dtype=float), np.asarray(ys, dtype=float))
    return np.asarray(lat), np.asarray(lon)

# ---------- City limits from call data (concave hull in meters) ----------
def calls_concave_hull_utm(
    lat, lon, eps_m=None, min_samples=20,
    alpha=None, buffer_smooth_m=150, simplify_m=60
):
    lat = pd.to_numeric(pd.Series(lat), errors="coerce")
    lon = pd.to_numeric(pd.Series(lon), errors="coerce")
    mask = lat.notna() & lon.notna()
    if mask.sum() < 10:
        return None, 0.0, (None, None)

    fwd, inv = _make_transformers(lat[mask].values, lon[mask].values)
    X, Y = _project_points(fwd, lat[mask].values, lon[mask].values)

    finite = np.isfinite(X) & np.isfinite(Y)
    X, Y = X[finite], Y[finite]
    if len(X) < max(3, min_samples):
        return None, 0.0, (fwd, inv)

    XY = np.c_[X, Y]

    # de-dupe near-identical points to stabilize Delaunay
    XY = np.unique(np.round(XY, 1), axis=0)
    if len(XY) < 3:
        return None, 0.0, (fwd, inv)

    # DBSCAN outlier removal (auto eps with clamp)
    if eps_m is None:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=2).fit(XY)
        dists, _ = nn.kneighbors(XY)
        med = float(np.median(dists[:, 1]))
        eps = float(np.clip(med if np.isfinite(med) and med > 0 else 50.0, 50.0, 400.0))
    else:
        eps = float(eps_m)

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(XY)
    XYk = XY[labels == np.bincount(labels[labels >= 0]).argmax()] if (labels >= 0).any() else XY
    if len(XYk) < 3:
        return None, 0.0, (fwd, inv)

    # --- radial 90th-percentile trim to drop long tentacles between towns ---
    cx, cy = XYk.mean(axis=0)
    r = np.hypot(XYk[:,0]-cx, XYk[:,1]-cy)
    cut = np.quantile(r, 0.90)
    XYt = XYk[r <= cut]
    if len(XYt) >= 10:
        XYk = XYt  # only trim if enough points remain

    # alpha from local spacing
    if alpha is None:
        from sklearn.neighbors import NearestNeighbors
        nn2 = NearestNeighbors(n_neighbors=2).fit(XYk)
        d2, _ = nn2.kneighbors(XYk)
        nn_med = float(np.median(d2[:, 1]))
        nn_med = nn_med if np.isfinite(nn_med) and nn_med > 0 else 50.0
        alpha = 1.0 / nn_med

    # build polygon; safe fallback to convex hull
    try:
        pts = [Point(float(x), float(y)) for x, y in XYk]
        poly = alphashape.alphashape(pts, alpha)
    except Exception:
        poly = None
    if not poly or poly.is_empty:
        hull = MultiPoint([Point(float(x), float(y)) for x, y in XYk]).convex_hull
        if getattr(hull, "geom_type", "") == "LineString":
            hull = hull.buffer(25.0)
        poly = hull

    if getattr(poly, "geom_type", "") == "MultiPolygon":
        poly = max(poly.geoms, key=lambda p: p.area)

    # tighter smoothing → less area inflation
    poly = poly.buffer(buffer_smooth_m).simplify(simplify_m).buffer(-buffer_smooth_m).buffer(0)

    # optional micro-erosion to pull in the outline a block or two
    poly = poly.buffer(-60).buffer(60)

    if poly.is_empty:
        return None, 0.0, (fwd, inv)

    area_sqmi = float(poly.area / SQM_PER_SQMI)
    return poly, area_sqmi, (fwd, inv)

# ---------- Our coverage (union of launch circles in meters) ----------
def union_launch_circles_utm(launch_latlon, radius_mi, fwd: Transformer | None):
    if not launch_latlon:
        return None, 0.0
    if fwd is None:
        lats = np.array([la for la, _ in launch_latlon], dtype=float)
        lons = np.array([lo for _, lo in launch_latlon], dtype=float)
        fwd, _ = _make_transformers(lats, lons)

    r_m = float(radius_mi) * 1609.34
    xs, ys = _project_points(fwd,
                             np.array([la for la, _ in launch_latlon], dtype=float),
                             np.array([lo for _, lo in launch_latlon], dtype=float))
    circles = [Point(float(x), float(y)).buffer(r_m) for x, y in zip(xs, ys)]
    poly = circles[0]
    for c in circles[1:]:
        poly = poly.union(c)
    if not poly or poly.is_empty:
        return None, 0.0
    return poly, float(poly.area / SQM_PER_SQMI)

# ---------- Outline conversion for folium ----------

def polygon_outline_latlon(poly_utm, inv: Transformer):
    if poly_utm is None or poly_utm.is_empty or inv is None:
        return []
    if poly_utm.geom_type == "MultiPolygon":
        poly_utm = max(poly_utm.geoms, key=lambda p: p.area)
    xs, ys = poly_utm.exterior.coords.xy
    lat, lon = _unproject_points(inv, np.asarray(xs), np.asarray(ys))
    return list(zip(lat.tolist(), lon.tolist()))

# ---------------- Hardcoded specs/prices/ranges (from your CSV) --------------
PLATFORMS = {
    "Flock Aerodome M350": {
        "price_per_dock": 150000,
        "docks_per_location": 1,
        "range_mi": 3.5,
        "specs": {
            "Pricing / Dock / Year (2-Year Contract)": "$150,000",
            "Number of Docks / Location": "1",
            "Real-world Speed (MPH)": "51 mph",
            "Response Time (1 Mile) (sec)": "41 sec",
            "Real-world On-scene Time (min)": "35 min",
            "Hit License Plate at 400ft Alt": "1000 ft",
            "Effectively Fly at 400ft Alt": "Yes",
            "Night Vision": "Yes",
            "Integrations": "Flock911 AD, Flock LPR, Flock NOVA, Flock Audio, CAD, Inflight LPR, Flock OS / Fusus, Evidence.com",
        },
    },
    "Flock Aerodome Dock 3": {
        "price_per_dock": 50000,
        "docks_per_location": 2,
        "range_mi": 3.5,
        "specs": {
            "Pricing / Dock / Year (2-Year Contract)": "$50,000",
            "Number of Docks / Location": "2",
            "Real-world Speed (MPH)": "47 mph",
            "Response Time (1 Mile) (sec)": "47 sec",
            "Real-world On-scene Time (min)": "40 min",
            "Hit License Plate at 400ft Alt": "700 ft",
            "Effectively Fly at 400ft Alt": "Yes",
            "Night Vision": "Yes",
            "Integrations": "Flock911 AD, Flock LPR, Flock NOVA, Flock Audio, CAD, Inflight LPR, Flock OS / Fusus, Evidence.com",
        },
    },
    "Flock Aerodome Alpha": {
        "price_per_dock": 125000,
        "docks_per_location": 1,
        "range_mi": 3.5,
        "specs": {
            "Pricing / Dock / Year (2-Year Contract)": "$125,000",
            "Number of Docks / Location": "1",
            "Real-world Speed (MPH)": "60 mph",
            "Response Time (1 Mile) (sec)": "36 sec",
            "Real-world On-scene Time (min)": "50 min",
            "Hit License Plate at 400ft Alt": "1000 ft",
            "Effectively Fly at 400ft Alt": "Yes",
            "Night Vision": "Yes",
            "Integrations": "Flock911 AD, Flock LPR, Flock NOVA, Flock Audio, CAD, Inflight LPR, Flock OS / Fusus, Evidence.com",
        },
    },
    "Flock Aerodome Delta": {
        "price_per_dock": 300000,
        "docks_per_location": 1,
        "range_mi": 15.0,
        "specs": {
            "Pricing / Dock / Year (2-Year Contract)": "$300,000",
            "Number of Docks / Location": "1",
            "Real-world Speed (MPH)": "100 mph",
            "Response Time (1 Mile) (sec)": "24 sec",
            "Real-world On-scene Time (min)": "120 min",
            "Hit License Plate at 400ft Alt": "1000 ft",
            "Effectively Fly at 400ft Alt": "Yes",
            "Night Vision": "Yes",
            "Integrations": "Flock911 AD, Flock LPR, Flock NOVA, Flock Audio, CAD, Inflight LPR, Flock OS / Fusus, Evidence.com",
        },
    },
    # Competitors
    "Skydio X10": {
        "price_per_dock": 50000,
        "docks_per_location": 3,
        "range_mi": 2.0,
        "specs": {
            "Pricing / Dock / Year (2-Year Contract)": "$50,000",
            "Number of Docks / Location": "3",
            "Real-world Speed (MPH)": "30 mph",
            "Response Time (1 Mile) (sec)": "157 sec",
            "Real-world On-scene Time (min)": "15 min",
            "Hit License Plate at 400ft Alt": "300 ft",
            "Effectively Fly at 400ft Alt": "No",
            "Night Vision": "No",
            "Integrations": "Evidence.com, Flock OS / Fusus",
        },
    },
    "Brinc Responder": {
        "price_per_dock": 75000,
        "docks_per_location": 3,
        "range_mi": 2.0,
        "specs": {
            "Pricing / Dock / Year (2-Year Contract)": "$75,000",
            "Number of Docks / Location": "3",
            "Real-world Speed (MPH)": "30 mph",
            "Response Time (1 Mile) (sec)": "157 sec",
            "Real-world On-scene Time (min)": "15 min",
            "Hit License Plate at 400ft Alt": "200 ft",
            "Effectively Fly at 400ft Alt": "No",
            "Night Vision": "No",
            "Integrations": "None",
        },
    },
    "Paladin Dock 3": {
        "price_per_dock": 50000,
        "docks_per_location": 2,
        "range_mi": 2.0,
        "specs": {
            "Pricing / Dock / Year (2-Year Contract)": "$50,000",
            "Number of Docks / Location": "2",
            "Real-world Speed (MPH)": "33 mph",
            "Response Time (1 Mile) (sec)": "61 sec",
            "Real-world On-scene Time (min)": "35 min",
            "Hit License Plate at 400ft Alt": "700 ft",
            "Effectively Fly at 400ft Alt": "No",
            "Night Vision": "Yes",
            "Integrations": "None",
        },
    },
    "Dronesense Dock 3": {
        "price_per_dock": 40000,
        "docks_per_location": 2,
        "range_mi": 2.0,
        "specs": {
            "Pricing / Dock / Year (2-Year Contract)": "$40,000",
            "Number of Docks / Location": "2",
            "Real-world Speed (MPH)": "33 mph",
            "Response Time (1 Mile) (sec)": "61 sec",
            "Real-world On-scene Time (min)": "40 min",
            "Hit License Plate at 400ft Alt": "700 ft",
            "Effectively Fly at 400ft Alt": "Yes",
            "Night Vision": "Yes",
            "Integrations": "Axon Air, Evidence.com, Flock OS / Fusus",
        },
    },
}

COMPETITOR_OPTIONS = [
    "Skydio X10",
    "Brinc Responder",
    "Paladin Dock 3",
    "Dronesense Dock 3",
]

# ---------------- Pull counts & detected dock types from Launch CSV ----------
def _get_col(df, *names):
    colmap = {c.lower().strip(): c for c in df.columns}
    for n in names:
        c = colmap.get(n.lower().strip())
        if c:
            return df[c]
    return None

dock_type_col = _get_col(launch_rows, "Dock Type", "Drone Type", "Dock Type or Drone Type")
docks_col     = _get_col(launch_rows, "Number of Docks")
radars_col    = _get_col(launch_rows, "Number of Radar")

launch_count = len(launch_rows)
total_docks  = int(pd.to_numeric(docks_col, errors="coerce").fillna(0).sum()) if docks_col is not None else 0
total_radars = int(pd.to_numeric(radars_col, errors="coerce").fillna(0).sum()) if radars_col is not None else 0

def _normalized_dock_types():
    if dock_type_col is None or dock_type_col.empty:
        return ["Flock Aerodome Dock 3"]
    vals = dock_type_col.astype(str).fillna("").str.strip()
    vals = vals.replace("", "Flock Aerodome Dock 3")
    out = []
    for v in vals:
        v2 = v.replace("Flock Aerodome ", "").strip()
        if v2.upper() in ["M350", "DOCK 3", "ALPHA", "DELTA"]:
            label = f"Flock Aerodome {('Dock 3' if v2.upper() == 'DOCK 3' else v2.upper().title())}"
        else:
            label = "Flock Aerodome Dock 3"
        out.append(label)
    return out

detected_types_list = sorted(set(_normalized_dock_types()))
is_multi = len(detected_types_list) > 1
aerodome_title = f"Flock Aerodome — {'Multi-platform' if is_multi else detected_types_list[0].split('Flock Aerodome ',1)[-1]}"

# Effective range & area estimates
our_eff_range = max(PLATFORMS[t]["range_mi"] for t in detected_types_list) if detected_types_list else 3.5
OUR_AREA_SQMI_EST = len(launch_coords) * math.pi * (our_eff_range ** 2)

# Build polygons
calls_poly_utm, CALLS_AREA_SQMI, (fwd_calls, inv_calls) = calls_concave_hull_utm(
    df_all["lat"].values, df_all["lon"].values,
    eps_m=None,  # auto
    min_samples=20
)
our_poly_utm, OUR_CIRCLES_AREA_SQMI = union_launch_circles_utm(
    launch_coords, our_eff_range, fwd_calls
)

# Choose placement mask:
# default = city coverage from calls; if our coverage is smaller, constrain to ours
if calls_poly_utm is None:
    PLACEMENT_POLY_UTM = our_poly_utm
    TARGET_AREA_SQMI = OUR_CIRCLES_AREA_SQMI
    target_label = f"Target area (our coverage fallback): {TARGET_AREA_SQMI:.2f} sq mi"
else:
    if OUR_CIRCLES_AREA_SQMI < CALLS_AREA_SQMI:
        PLACEMENT_POLY_UTM = our_poly_utm
        TARGET_AREA_SQMI   = OUR_CIRCLES_AREA_SQMI
        target_label = f"Target area (our coverage smaller): {TARGET_AREA_SQMI:.2f} sq mi"
    else:
        PLACEMENT_POLY_UTM = calls_poly_utm
        TARGET_AREA_SQMI   = CALLS_AREA_SQMI
        target_label = f"Target area (call-derived city): {TARGET_AREA_SQMI:.2f} sq mi"
st.caption(target_label)

# ---------------- City limits default (sidebar overrides) --------------------
CITY_AREA_SQMI_DEFAULT = float(CALLS_AREA_SQMI) if (CALLS_AREA_SQMI and CALLS_AREA_SQMI > 0) else None

with st.sidebar.expander("City limits (default = from call data)", expanded=False):
    if CITY_AREA_SQMI_DEFAULT:
        st.caption(f"Detected from call data hull: ~{CITY_AREA_SQMI_DEFAULT:.2f} sq mi")
    else:
        st.caption("Call-data hull unavailable (not enough points).")

    city_query = st.text_input("(Optional) Search Wikidata city name (e.g., \"Boise, ID\")", key="cmp_city_query")
    # (leave your Wikidata button block as-is)

    current_default = st.session_state.get("city_area_sqmi", CITY_AREA_SQMI_DEFAULT or 0.0)
    manual_city_area = st.number_input(
        "City area (sq mi — leave as default to use call-data hull)",
        value=float(current_default), min_value=0.0, step=1.0, format="%.2f"
    )
    st.session_state["city_area_sqmi"] = manual_city_area

# ---- EFFECTIVE city area used for competitor math (NO override later) ----
CITY_AREA_SQMI = float(st.session_state.get("city_area_sqmi", 0.0)) or None
EFFECTIVE_CITY_AREA = CITY_AREA_SQMI or CALLS_AREA_SQMI                   # <- this is the scalar we will use

# Placement polygon we draw = call hull if present, else our union-of-circles
PLACEMENT_POLY_UTM = calls_poly_utm or our_poly_utm

# Scalar target area for competitor count = EFFECTIVE city area (no re-assignment later)
TARGET_AREA_SQMI = EFFECTIVE_CITY_AREA or OUR_AREA_SQMI_EST

target_label = f"Target area (city limits used for math): {TARGET_AREA_SQMI:.2f} sq mi"
st.caption(target_label)

# ---------------- Aerodome yearly pricing (no discount UI in Comparison) -----
RADAR_PRICE = 150000

DOCK_PRICES = {
    "Dock 3": 50000,
    "Alpha": 125000,
    "Delta": 300000,
    "M350": 150000,
}

def _dock_price_for_row(row):
    v = str(row.get(dock_type_col.name, "") if dock_type_col is not None else "").strip()
    v = v.replace("Flock Aerodome ", "")
    v = (v or "Dock 3").strip()
    return DOCK_PRICES.get(v, DOCK_PRICES["Dock 3"])

def compute_our_yearly_price_no_discount():
    if launch_rows.empty:
        return 0
    _rows = launch_rows.copy()
    _rows["_docks"] = pd.to_numeric(docks_col, errors="coerce").fillna(0) if docks_col is not None else 0
    _rows["_radars"] = pd.to_numeric(radars_col, errors="coerce").fillna(0) if radars_col is not None else 0
    _rows["_dock_price"] = _rows.apply(_dock_price_for_row, axis=1)
    base = int((_rows["_docks"] * _rows["_dock_price"]).sum() + _rows["_radars"].sum() * RADAR_PRICE)
    return int(base)

our_base = compute_our_yearly_price_no_discount()

# ---------------- Competitor math ----------------
def circle_area_sqmi(radius_mi: float) -> float:
    return math.pi * (radius_mi ** 2)

def round_locations(x: float) -> int:
    if x <= 1.0:
        return 1
    frac = x - math.floor(x)
    return max(1, math.floor(x) + (1 if frac >= 0.35 else 0))

def competitor_plan(comp_name: str, target_area_sqmi: float):
    cfg = PLATFORMS[comp_name]
    per_loc_area = circle_area_sqmi(cfg["range_mi"])
    raw_needed = (target_area_sqmi / per_loc_area) if per_loc_area > 0 else 1.0
    locations = round_locations(raw_needed)
    yearly_cost = locations * cfg["docks_per_location"] * cfg["price_per_dock"]
    return {
        "locations": locations,
        "yearly_cost": yearly_cost,
        "per_location_area_sqmi": per_loc_area,
        "docks_per_location": cfg["docks_per_location"],
        "price_per_dock": cfg["price_per_dock"],
        "range_mi": cfg["range_mi"],
    }

# ---------- Place competitor centers INSIDE polygon (KMeans on calls) ----------
def place_sites_kmeans_in_polygon(lat, lon, polygon_utm, n_sites, fwd: Transformer, inv: Transformer, rng=42):
    if polygon_utm is None or n_sites < 1 or fwd is None or inv is None:
        return []
    lat = pd.to_numeric(pd.Series(lat), errors="coerce")
    lon = pd.to_numeric(pd.Series(lon), errors="coerce")
    mask = lat.notna() & lon.notna()
    if mask.sum() == 0:
        return []
    X, Y = _project_points(fwd, lat[mask].values, lon[mask].values)
    pts_in = [(x, y) for x, y in zip(X, Y) if polygon_utm.contains(Point(float(x), float(y)))]
    if not pts_in:
        return []
    XY = np.asarray(pts_in, dtype=float)
    n = max(1, min(n_sites, len(XY)))
    km = KMeans(n_clusters=n, n_init=10, random_state=rng).fit(XY)
    cx, cy = km.cluster_centers_[:, 0], km.cluster_centers_[:, 1]
    lat_c, lon_c = _unproject_points(inv, cx, cy)
    return list(zip(lat_c.tolist(), lon_c.tolist()))

# ---------------- Panel renderer ----------------
def panel(title, product_names_list, is_left=True, competitor=None):
    with st.container(border=True):
        st.subheader(title)

        if is_left:
            # LEFT: our heatmap + blue range circles
            render_map(
                df_all,                         # ← if you prefer only DFR calls, change to dfr_only
                heat=True,
                heat_radius=8, heat_blur=12,
                title="",
                key=f"cmp_map_L_{title}",
                show_circle=True,
                launch_coords=launch_coords
            )
        else:
            # RIGHT: competitor centers chosen inside polygon via K-Means on calls
            plan = competitor_plan(competitor, TARGET_AREA_SQMI)
            required_locs = plan["locations"]
            comp_range_mi = plan["range_mi"]

            fwd = fwd_calls if fwd_calls else _make_transformers(df_all["lat"].values, df_all["lon"].values)[0]
            inv = inv_calls if inv_calls else _make_transformers(df_all["lat"].values, df_all["lon"].values)[1]
            centers = place_sites_kmeans_in_polygon(
                df_all["lat"].values, df_all["lon"].values,
                PLACEMENT_POLY_UTM, n_sites=required_locs,
                fwd=fwd, inv=inv
            )

            if launch_coords:
                lat0, lon0 = float(launch_coords[0][0]), float(launch_coords[0][1])
            elif len(df_all) > 0:
                lat0, lon0 = float(df_all["lat"].mean()), float(df_all["lon"].mean())
            else:
                lat0, lon0 = 0.0, 0.0

            m = folium.Map(location=[lat0, lon0], zoom_start=11)

            # Red competitor circles
            comp_r_m = comp_range_mi * 1609.34
            for (la, lo) in centers:
                folium.Circle(location=(la, lo), radius=comp_r_m, color="red", weight=3, fill=False).add_to(m)
                folium.CircleMarker(location=(la, lo), radius=3, color="red", fill=True).add_to(m)

            # Our blue coverage for context
            for la, lo in launch_coords:
                folium.Circle(location=(la, lo), radius=our_eff_range * 1609.34, color="blue", weight=2, fill=False, opacity=0.35).add_to(m)

            # City-limits outline (purple)
            outline_latlon = polygon_outline_latlon(PLACEMENT_POLY_UTM, inv_calls)
            if outline_latlon:
                folium.PolyLine(locations=[(lt, ln) for lt, ln in outline_latlon],
                                color="purple", weight=4, opacity=0.8).add_to(m)

            st_folium(m, width=800, height=500, key=f"cmp_map_R_{competitor}")

            # Headline metrics
            comp_docks_per_loc = PLATFORMS[competitor]["docks_per_location"]
            total_comp_docks = required_locs * comp_docks_per_loc
            c1, c2, c3 = st.columns(3)
            c1.metric("Required Locations", f"{required_locs:,}")
            c2.metric("Total Docks", f"{total_comp_docks:,}")
            c3.metric("Yearly Cost", f"${plan['yearly_cost']:,}")
            st.caption(
                f"Docks/location: {comp_docks_per_loc} • "
                f"Per-location area: {plan['per_location_area_sqmi']:.2f} sq mi • "
                f"Radius: {comp_range_mi:.2f} mi"
            )

        # Specs list
        def render_specs(pname: str):
            specs = PLATFORMS[pname]["specs"]
            rows = [
                "Pricing / Dock / Year (2-Year Contract)",
                "Number of Docks / Location",
                "Real-world Speed (MPH)",
                "Response Time (1 Mile) (sec)",
                "Real-world On-scene Time (min)",
                "Hit License Plate at 400ft Alt",
                "Effectively Fly at 400ft Alt",
                "Night Vision",
                "Integrations",
            ]
            for r in rows:
                if r in specs:
                    st.write(f"**{r}**: {specs[r]}")

        if is_multi:
            if is_left:
                st.markdown("**Detected Aerodome Platforms:** " + ", ".join(detected_types_list))
                for p in detected_types_list:
                    st.markdown(f"**{p}**")
                    render_specs(p)
                    st.markdown("---")
        else:
            render_specs(product_names_list[0] if product_names_list else competitor)

# ---- Controls row + two panels ---------------------------------------------
topL, topR = st.columns([3, 2])
with topL:
    # show the single target label once (not twice)
    st.caption(target_label)
with topR:
    comp_choice = st.selectbox("Compare against", COMPETITOR_OPTIONS, index=0, key="cmp_choice")

L, R = st.columns(2)
with L:
    panel(aerodome_title, detected_types_list if is_multi else detected_types_list[:1], is_left=True)
with R:
    panel(comp_choice, [], is_left=False, competitor=comp_choice)
