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
from folium.features import DivIcon

# --- FAA ESRI overlay (hardcoded) -------------------------------------------
import requests  # add if not already imported

FAA_LAYER_URL = "https://services6.arcgis.com/ssFJjBXIUyZDrSYZ/arcgis/rest/services/FAA_UAS_FacilityMap_Data/FeatureServer/0"

def _data_bbox(lat_series, lon_series, pad=0.05):
    """Return (xmin, ymin, xmax, ymax) in lon/lat with padding."""
    lat = pd.to_numeric(pd.Series(lat_series), errors="coerce").dropna()
    lon = pd.to_numeric(pd.Series(lon_series), errors="coerce").dropna()
    if lat.empty or lon.empty:
        return None
    return (
        float(lon.min() - pad),
        float(lat.min() - pad),
        float(lon.max() + pad),
        float(lat.max() + pad),
    )

def load_esri_geojson(layer_url: str, bbox=None, out_sr=4326):
    """Query an ArcGIS FeatureServer layer and return GeoJSON (as a dict)."""
    query_url = layer_url.rstrip("/") + "/query"
    params = {
        "f": "geojson",
        "where": "1=1",
        "outFields": "*",
        "outSR": out_sr,
    }
    if bbox:
        xmin, ymin, xmax, ymax = bbox
        params.update({
            "geometry": json.dumps({
                "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax,
                "spatialReference": {"wkid": 4326}
            }),
            "geometryType": "esriGeometryEnvelope",
            "spatialRel": "esriSpatialRelIntersects",
            "inSR": 4326,
        })
    r = requests.get(query_url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def add_faa_overlay_if_enabled(m: folium.Map, lat_list, lon_list, pad=0.05):
    """
    Adds FAA UAS grid to the map with transparent squares + altitude labels.
    - Clears any older 'FAA grid' layers first so styles don’t get masked by a previous draw.
    - Colors by CEILING using the FAA renderer palette.
    """
    try:
        if not st.session_state.get("show_faa", True):
            return

        bbox = _data_bbox(lat_list, lon_list, pad=pad)
        if not bbox:
            return

        # Remove any existing FAA overlays so we don’t stack old blue tiles on top
        try:
            for key in list(m._children.keys()):
                child = m._children[key]
                if getattr(child, "layer_name", "") in {"FAA grid", "FAA ceilings"}:
                    del m._children[key]
        except Exception:
            pass

        gj = load_esri_geojson(FAA_LAYER_URL, bbox=bbox)

        # Debug: how many FAA features returned in current bbox
        if st.session_state.get("faa_debug"):
            try:
                feat_ct = len(gj.get("features", []))
                st.sidebar.info(f"FAA: loaded {feat_ct} features in bbox")
                print("FAA loaded features:", feat_ct)
            except Exception as e:
                st.sidebar.warning(f"FAA debug count failed: {e}")

        # Heuristic: find the attribute that holds the ceiling (0..400)
        def _get_alt_ft(props: dict):
            if props is None:
                return None
            for key in ("CEILING", "ceiling"):
                if key in props and props[key] is not None:
                    try:
                        return float(props[key])
                    except Exception:
                        pass
            for k in ("UASFM_CEILING","GRID_VALUE","grid_value","MAX_ALT","MAX_AGL","GridCellCeiling","gridCeiling","altitude","ALT"):
                if k in props and props[k] is not None:
                    try:
                        return float(props[k])
                    except Exception:
                        pass
            return None

        # FAA color scheme by the service’s renderer (exact matches)
        _FAA_COLORS = {
            0:   "#B53535",  # red
            50:  "#BCFCB6",  # pale green
            100: "#E69800",  # orange
            150: "#FCCFB8",  # peach
            200: "#FFFFBE",  # yellow
            250: "#D4D9A1",  # khaki
            300: "#A7C7B3",  # gray-green
            350: "#BDD8FC",  # light blue (not always present)
            400: "#65A843",  # green
        }

        def _faa_color_for(alt):
            """Use the exact bucket if present; else fall back by stepping down to the nearest lower bucket."""
            try:
                a = int(round(float(alt)))
            except Exception:
                return "#D9E3F0"
            if a in _FAA_COLORS:
                return _FAA_COLORS[a]
            for level in sorted(_FAA_COLORS.keys(), reverse=True):
                if a >= level:
                    return _FAA_COLORS[level]
            return "#D9E3F0"

        def _style(_feat):
            props = _feat.get("properties", {}) or {}
            alt = _get_alt_ft(props)
            if st.session_state.get("faa_debug"):
                print(f"FAA style → CEILING raw={props.get('CEILING')} parsed={alt}")
            return {
                "color": "#222222",
                "weight": 0.7,
                "opacity": 0.8,
                "fill": True,
                "fillColor": _faa_color_for(alt),
                "fillOpacity": 0.30,
            }

        # Add the grid polygons first
        poly_layer = folium.GeoJson(
            gj,
            name="FAA grid",
            style_function=_style,
        )
        poly_layer.add_to(m)

        # Label each cell with its ceiling at the centroid (always-on labels).
        features = gj.get("features", [])
        n = len(features)
        step = 1 if n <= 4000 else (2 if n <= 8000 else 3)

        label_group = folium.FeatureGroup(name="FAA ceilings", show=True)
        for i in range(0, n, step):
            ft = features[i]
            geom = ft.get("geometry") or {}
            if geom.get("type") != "Polygon":
                continue
            coords = geom.get("coordinates")
            if not coords:
                continue
            ring = coords[0]
            try:
                xs = [pt[0] for pt in ring]
                ys = [pt[1] for pt in ring]
                cx = sum(xs) / len(xs)
                cy = sum(ys) / len(ys)
            except Exception:
                continue
            alt = _get_alt_ft(ft.get("properties", {}))
            if alt is None:
                continue
            folium.Marker(
                location=[cy, cx],
                icon=DivIcon(
                    html=(
                        '<div style="font-size:11px; font-weight:700; color:#cc0000; '
                        'text-shadow:-1px -1px 0 #ffffff,1px -1px 0 #ffffff,'
                        '-1px 1px 0 #ffffff,1px 1px 0 #ffffff">'
                        f"{int(round(alt))}</div>"
                    ),
                    icon_size=(0, 0),
                    icon_anchor=(0, 0),
                    class_name="",
                ),
            ).add_to(label_group)
        label_group.add_to(m)

        # Keep it collapsible
        folium.LayerControl(collapsed=True).add_to(m)

    except Exception as e:
        if st.session_state.get("faa_debug"):
            st.sidebar.error(f"FAA overlay error: {e}")
            try:
                import traceback
                print("FAA overlay error:", e)
                print(traceback.format_exc())
            except Exception:
                pass
        return

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


def parse_flexible_time(s: pd.Series) -> pd.Series:
    """
    Accepts Excel serials (handled by parse_time_series) OR clock-like strings:
    'HH:MM:SS(.ms)', 'H:MM', 'MM:SS(.ms)'. Returns datetimes anchored to 1970-01-01.
    """
    import re
    s_str = s.astype(str).str.strip()

    # 1) Try the existing parser first (keeps current behavior for standard files)
    try:
        dt_try = parse_time_series(s)
        # consider it a success if it yields at least a few valid rows
        if dt_try.notna().sum() >= max(3, int(0.2 * len(s))):
            return dt_try
    except Exception:
        pass

    # 2) Fallback: interpret clock-like text as seconds from an origin
    origin = pd.Timestamp("1970-01-01")

    def to_seconds(text: str):
        t = text.strip()
        if not t or t.upper() in ("NAN", "NA", "NONE"):
            return None
        m = re.match(r'^(\d{1,2}):([0-5]?\d):([0-5]?\d(?:\.\d+)?)$', t)  # H:MM:SS(.s)
        if m:
            h = int(m.group(1)); mm = int(m.group(2)); ss = float(m.group(3))
            return h*3600 + mm*60 + ss
        m = re.match(r'^([0-5]?\d):([0-5]?\d(?:\.\d+)?)$', t)            # MM:SS(.s)
        if m:
            mm = int(m.group(1)); ss = float(m.group(2))
            return mm*60 + ss
        m = re.match(r'^(\d{1,2}):([0-5]?\d)$', t)                       # H:MM
        if m:
            h = int(m.group(1)); mm = int(m.group(2))
            return h*3600 + mm*60
        try:
            val = float(t)  # plain seconds
            return val if np.isfinite(val) else None
        except Exception:
            return None

    secs = s_str.map(to_seconds)
    td   = pd.to_timedelta(pd.to_numeric(secs, errors="coerce"), unit="s")
    out  = origin + td
    out[td.isna()] = pd.NaT
    return out

# --- Flexible raw call parser: supports single-timestamp schema -------------
def parse_calls_flexible(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts the minimal schema (by header OR by position):
      Headers or order: [Call Create, Call Type, Call Priority, Lat, Long]
    Returns a normalized dataframe with the columns your pipeline expects:
      lat, lon, call_type_up, call_type, priority, patrol_sec, drone_eta_sec,
      onscene_sec, dispatch_sec, arrive_sec, close_sec, call_create (if parseable), count.
    If the input does not match this minimal schema, returns the original df unchanged.
    """
    if not isinstance(df_in, pd.DataFrame) or df_in.empty:
        return df_in

    df = df_in.copy()
    # Normalize header names (strip only; do NOT lowercase in-place to avoid side-effects)
    df.columns = [str(c).replace("\u00A0", " ").strip() for c in df.columns]
    lower_map = {c.lower(): c for c in df.columns}

    # Helper: find column by preferred names; if not found, return None
    def col_any(names):
        for n in names:
            k = str(n).lower()
            if k in lower_map:
                return lower_map[k]
        return None

    # 1) Try header-based resolution for minimal schema
    c_create = col_any(["call create", "created", "create", "date/time created", "incident create", "created time"])  
    c_type   = col_any(["call type", "type", "nature", "problem", "event type"])  
    c_pri    = col_any(["call priority", "priority", "pri", "pr"])  
    c_lat    = col_any(["lat", "latitude", "y"])  
    c_lon    = col_any(["long", "lon", "longitude", "x"])  

    # 2) If headers not found, fall back to strict 5-column positional schema
    if any(x is None for x in [c_type, c_pri, c_lat, c_lon]):
        if len(df.columns) >= 5:
            # assume order: Call Create, Call Type, Call Priority, Lat, Long
            c_create = c_create or df.columns[0]
            c_type   = c_type   or df.columns[1]
            c_pri    = c_pri    or df.columns[2]
            c_lat    = c_lat    or df.columns[3]
            c_lon    = c_lon    or df.columns[4]

    # 3) Validate we truly have the minimal schema; else bail out unchanged
    if any(x is None for x in [c_type, c_pri, c_lat, c_lon]):
        return df_in

    # Build output
    out = pd.DataFrame()

    # Timestamp (optional): keep if parseable, but don't discard rows if it's NaT
    if c_create is not None:
        out["call_create"] = pd.to_datetime(df[c_create], errors="coerce", infer_datetime_format=True)
    else:
        out["call_create"] = pd.NaT

    # Basic fields
    out["call_type_raw"] = df[c_type].astype(str).str.strip()
    out["lat"]           = pd.to_numeric(df[c_lat], errors="coerce")
    out["lon"]           = pd.to_numeric(df[c_lon], errors="coerce")

    # Priority: keep raw; accept only digits 1..5 anywhere in the cell (tolerant to prefixes like 'P1')
    out["priority_raw"] = df[c_pri]
    _digits = pd.Series(df[c_pri]).astype(str).str.replace("\u00A0", " ", regex=False).str.strip().str.extract(r"(?<!\d)([1-5])(?!\d)")[0]
    _num = pd.to_numeric(_digits, errors="coerce")
    out["priority"] = _num.where(_num.isin([1, 2, 3, 4, 5]))
    try:
        out["priority"] = out["priority"].astype("Int64")  # nullable integer
    except Exception:
        pass

    # Standardize call type for mapping
    out["call_type_up"] = out["call_type_raw"].fillna("").str.strip().str.upper()
    out["call_type"]    = out["call_type_up"]

    # Response metrics cannot be derived → ensure present and numeric zeros
    for col in [
        "patrol_sec", "drone_eta_sec", "onscene_sec",
        "dispatch_sec", "arrive_sec", "close_sec",
    ]:
        out[col] = 0

    # For heatmaps/aggregations
    out["count"] = 1

    # Keep all usable geocoded rows; DO NOT require call_create (can be NaT)
    out = out.dropna(subset=["lat", "lon"]).reset_index(drop=True)

    # Convenience for time-of-day / weekday filters (non-breaking)
    if "call_create" in out.columns:
        try:
            out["create_hour"]    = out["call_create"].dt.hour
            out["create_weekday"] = out["call_create"].dt.weekday  # Mon=0
        except Exception:
            out["create_hour"] = pd.NA
            out["create_weekday"] = pd.NA

    # Flag that we are operating on the minimal 5-column schema
    try:
        st.session_state["raw_is_minimal"] = True
    except Exception:
        pass

    # Mark minimal-mode and set eligibility on returned rows
    try:
        st.session_state["eligibility_mode"] = "minimal"
        st.session_state["force_minimal_validity"] = True
        st.session_state["suppress_full_validity"] = True
    except Exception:
        pass
    out["is_dfr_eligible"] = True

    return out

# --- Minimal schema validity override helpers ------------------------------
def _minimal_validity_mask(df: pd.DataFrame) -> pd.Series:
    """
    For single-timestamp / 5-column data, we can't use time-delta checks.
    Keep rows that have usable geo + type + parsed priority.
    """
    has_geo  = df["lat"].notna() & df["lon"].notna()
    # Prefer normalized 'call_type_up' if present; else fallback to raw
    _ctype = df.get("call_type_up", df.get("call_type", df.get("call_type_raw", "")))
    has_type = _ctype.astype(str).str.strip().ne("")
    pri = pd.to_numeric(df.get("priority"), errors="coerce")
    has_pri = pri.notna()
    return (has_geo & has_type & has_pri)

def _apply_minimal_validity(df_all: pd.DataFrame):
    """
    If the minimal schema flag is set, bypass time-based validity checks and
    set session_state['df_all_valid'] and ['df_dfr_only'] so downstream code
    can continue unchanged.
    """
    if not st.session_state.get("raw_is_minimal", False):
        return  # only act for minimal schema runs

    if not isinstance(df_all, pd.DataFrame) or df_all.empty:
        return

    try:
        mask_valid = _minimal_validity_mask(df_all)
        df_all_valid = df_all.loc[mask_valid].copy()
        df_dfr_only  = df_all_valid.copy()

        # Backfill missing time columns with call_create **plus offsets** so deltas are > 0
        # Offsets: dispatch +1s, arrive +2s, close +3s
        call_create_series = df_dfr_only.get("call_create")
        for tcol, offset_sec in ("first_dispatch", 1), ("first_arrive", 2), ("call_close", 3):
            if tcol not in df_dfr_only.columns:
                df_dfr_only[tcol] = pd.NaT
            if call_create_series is not None:
                filled = df_dfr_only[tcol]
                # If missing or not strictly after call_create, set to call_create + offset
                needs = filled.isna()
                try:
                    needs |= (filled <= call_create_series)
                except Exception:
                    needs = filled.isna()
                df_dfr_only.loc[needs, tcol] = (
                    pd.to_datetime(call_create_series, errors="coerce") + pd.to_timedelta(offset_sec, unit="s")
                )

        # Ensure numeric metric columns exist and are minimally positive where required
        metric_defaults = {
            "patrol_sec": 0,
            "drone_eta_sec": 0,
            "dispatch_sec": 1,
            "arrive_sec": 1,
            "onscene_sec": 1,
            "close_sec": 1,
            "count": 1,
        }
        for col, default_val in metric_defaults.items():
            if col not in df_dfr_only.columns:
                df_dfr_only[col] = default_val
            else:
                df_dfr_only[col] = pd.to_numeric(df_dfr_only[col], errors="coerce").fillna(default_val)
            if col in {"dispatch_sec", "arrive_sec", "onscene_sec", "close_sec"}:
                df_dfr_only.loc[df_dfr_only[col] <= 0, col] = 1

        # Normalize types & priority if needed
        if "call_type_up" not in df_dfr_only.columns:
            _raw = df_dfr_only.get("call_type", df_dfr_only.get("call_type_raw", ""))
            df_dfr_only["call_type_up"] = pd.Series(_raw).astype(str).str.upper().str.strip()
        if "priority" in df_dfr_only.columns:
            _pr = pd.to_numeric(df_dfr_only["priority"], errors="coerce")
            try:
                df_dfr_only["priority"] = _pr.astype("Int64")  # keep <NA> for non-1..5; no remap/coercion
            except Exception:
                df_dfr_only["priority"] = _pr

        # Explicitly mark DFR eligibility for minimal schema
        df_all_valid["is_dfr_eligible"] = True
        df_dfr_only["is_dfr_eligible"] = True

        # Lock: downstream code should NOT overwrite this decision
        try:
            st.session_state["eligibility_mode"] = "minimal"
            st.session_state["force_minimal_validity"] = True
            st.session_state["suppress_full_validity"] = True
            st.session_state["final_validity_locked"] = True
        except Exception:
            pass

        # Expose subsets for downstream use (after backfills)
        st.session_state["df_all_valid"] = df_all_valid
        st.session_state["df_dfr_only"]  = df_dfr_only
    except Exception as e:
        # Non-fatal; leave downstream to handle with existing logic
        st.warning(f"Minimal-schema validity override encountered an issue: {e}")


# --- Final circuit breaker for minimal mode ----------------------------------
def enforce_minimal_guard(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hard bypass for any late-stage validity recomputations when running minimal schema.
    If minimal mode is active, force all common validity/eligibility flags and masks
    to True so no downstream logic can silently drop rows.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df
    if not st.session_state.get("raw_is_minimal", False):
        return df

    # Normalize a handful of common flag/mask column names used across the app
    force_true_cols = [
        "is_dfr_eligible",
        "is_valid",
        "valid_flag",
        "dfr_valid",
        "eligible",
        "_valid",
    ]
    for col in force_true_cols:
        if col in df.columns:
            try:
                df[col] = True
            except Exception:
                pass

    # Also stash a global, all-True mask that downstream code can reuse
    try:
        st.session_state["eligibility_mode"] = "minimal"
        st.session_state["suppress_full_validity"] = True
        st.session_state["final_validity_locked"] = True
        st.session_state["minimal_true_mask"] = pd.Series(True, index=df.index)
    except Exception:
        pass

    return df

# === Canonicalize calls dataframe to downstream schema ==========================
def canonicalize_calls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Canonicalize ANY raw-calls dataframe to the schema expected downstream.
    This is a non-destructive normalizer: it renames common variants and
    fills missing required columns (with safe defaults), so the rest of the
    pipeline can operate unchanged.

    Target schema (column names):
      call_create, first_dispatch, first_arrive, call_close,
      call_type, call_type_up, priority, lat, lon,
      patrol_sec, drone_eta_sec, onscene_sec, dispatch_sec, arrive_sec, close_sec,
      count
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df

    out = df.copy()

    # --- 1) Clean header text (strip spaces & NBSP) -------------------------
    cleaned = []
    for c in out.columns:
        c2 = c
        if not isinstance(c2, str):
            c2 = str(c2)
        c2 = c2.replace("\u00A0", " ").strip()
        cleaned.append(c2)
    out.columns = cleaned

    # Build a case-insensitive lookup of existing columns
    lower_map = {c.lower(): c for c in out.columns}

    def lookup(*cands):
        """Return the present column (original case) matching any candidate names."""
        for cand in cands:
            key = str(cand).lower()
            if key in lower_map:
                return lower_map[key]
        return None

    # --- 2) Compute a rename map to canonical names -------------------------
    rename_map = {}
    # Times
    c_create  = lookup("call create", "created", "create", "date/time created", "incident create", "created time")
    c_disp    = lookup("first dispatch", "dispatch", "first_dispatched", "time dispatched", "first dispatched")
    c_arr     = lookup("first arrive", "first arrived", "arrive", "arrival", "time arrived", "first on scene")
    c_close   = lookup("call close", "closed", "time closed", "closed time")

    # Type / Priority
    c_type    = lookup("call type", "type", "nature", "problem", "event type")
    c_pri     = lookup("call priority", "priority", "pri", "pr")

    # Coordinates
    c_lat     = lookup("lat", "latitude", "y")
    c_lon     = lookup("lon", "long", "longitude", "x")

    if c_create: rename_map[c_create] = "call_create"
    if c_disp:   rename_map[c_disp]   = "first_dispatch"
    if c_arr:    rename_map[c_arr]    = "first_arrive"
    if c_close:  rename_map[c_close]  = "call_close"
    if c_type:   rename_map[c_type]   = "call_type"
    if c_pri:    rename_map[c_pri]    = "priority"
    if c_lat:    rename_map[c_lat]    = "lat"
    if c_lon:    rename_map[c_lon]    = "lon"

    if rename_map:
        out = out.rename(columns=rename_map)
        # refresh lookup after rename
        lower_map = {c.lower(): c for c in out.columns}

    # --- 3) Ensure ALL required columns exist (fill safe defaults) ----------
    required_defaults = {
        "call_create":   pd.NaT,
        "first_dispatch": pd.NaT,
        "first_arrive":   pd.NaT,
        "call_close":     pd.NaT,
        "call_type":      "",
        "priority":       "",
        "lat":            np.nan,
        "lon":            np.nan,
        # response metrics used downstream (seconds)
        "patrol_sec":     0,
        "drone_eta_sec":  0,
        "onscene_sec":    0,
        "dispatch_sec":   0,
        "arrive_sec":     0,
        "close_sec":      0,
        # heatmap/weights
        "count":          1,
    }
    for col, default in required_defaults.items():
        if col not in out.columns:
            out[col] = default

    # --- 4) Parse datetimes if present --------------------------------------
    for tcol in ["call_create", "first_dispatch", "first_arrive", "call_close"]:
        if tcol in out.columns:
            # Only coerce strings/numbers; leave existing datetimes as-is
            try:
                out[tcol] = pd.to_datetime(out[tcol], errors="coerce", infer_datetime_format=True)
            except Exception:
                pass

    # --- Midnight rollover fix ---------------------------------------------------
    # If a later timestamp is earlier than the previous one, assume it crossed midnight.
    for prev_col, next_col in [
        ("call_create",   "first_dispatch"),
        ("first_dispatch","first_arrive"),
        ("first_arrive",  "call_close"),
    ]:
        if prev_col in out.columns and next_col in out.columns:
            try:
                prev_dt = pd.to_datetime(out[prev_col], errors="coerce")
                next_dt = pd.to_datetime(out[next_col], errors="coerce")
                mask = prev_dt.notna() & next_dt.notna() & (next_dt < prev_dt)
                if mask.any():
                    out.loc[mask, next_col] = next_dt.loc[mask] + pd.Timedelta(days=1)
            except Exception:
                pass

    # --- 5) Normalize priority to numeric only if already 1..5; else leave NaN ---
    pr_src = None
    if "priority" in out.columns:
        pr_src = out["priority"].astype(str)
    elif "priority_raw" in out.columns:
        pr_src = out["priority_raw"].astype(str)

    if pr_src is not None:
        digits = pr_src.str.replace("\u00A0", " ", regex=False).str.strip().str.extract(r"(?<!\d)([1-5])(?!\d)")[0]
        pr_num = pd.to_numeric(digits, errors="coerce")
        pr_num = pr_num.where(pr_num.isin([1, 2, 3, 4, 5]))
        try:
            out["priority"] = pr_num.astype("Int64")
        except Exception:
            out["priority"] = pr_num

    # Helper label for widgets; only populated when priority is valid
    try:
        out["priority_band"] = out["priority"].map({1:"P1",2:"P2",3:"P3",4:"P4",5:"P5"})
    except Exception:
        pass

    # --- 6) Lat/Lon numeric & drop invalid rows -----------------------------
    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
    out = out.dropna(subset=["lat", "lon"]).reset_index(drop=True)

    # --- 7) call_type_up uppercase mirror -----------------------------------
    out["call_type_up"] = (
        out["call_type"].astype(str).str.strip().str.upper()
        if "call_type" in out.columns
        else out.get("call_type_up", pd.Series([], dtype=str)).astype(str).str.strip().str.upper()
    )

    # --- 8) Ensure numeric metric columns exist (if present, coerce numeric) -
    for c in ["patrol_sec", "drone_eta_sec", "onscene_sec", "dispatch_sec", "arrive_sec", "close_sec", "count"]:
        if c in out.columns:
            series_num = pd.to_numeric(out[c], errors="coerce").fillna(0)
            # clamp any negative durations to 0 to avoid negative averages/formatting
            series_num = series_num.where(series_num >= 0, 0)
            # keep ints for downstream display where appropriate
            try:
                out[c] = series_num.astype(int)
            except Exception:
                out[c] = series_num

    # --- 9) Convenience: hour/weekday (non-breaking) ------------------------
    if "call_create" in out.columns:
        try:
            out["create_hour"]    = out["call_create"].dt.hour
            out["create_weekday"] = out["call_create"].dt.weekday
        except Exception:
            out["create_hour"] = pd.NA
            out["create_weekday"] = pd.NA

        # If we're in minimal mode, pre-populate validity subsets so the rest of the
    # pipeline (which expects them) continues unchanged.
    try:
        _apply_minimal_validity(out)
    except Exception:
        pass

    # Guard: if minimal mode, re-assert backfills and eligibility so later filters can't undo it
    if st.session_state.get("raw_is_minimal", False):
        # Ensure time columns exist and are strictly increasing from call_create
        call_create_series = out.get("call_create")
        for _t, _off in ("first_dispatch", 1), ("first_arrive", 2), ("call_close", 3):
            if _t not in out.columns:
                out[_t] = pd.NaT
            if call_create_series is not None:
                filled = out[_t]
                needs = filled.isna()
                try:
                    needs |= (filled <= call_create_series)
                except Exception:
                    needs = filled.isna()
                out.loc[needs, _t] = (
                    pd.to_datetime(call_create_series, errors="coerce") + pd.to_timedelta(_off, unit="s")
                )
        for _m in ("dispatch_sec", "arrive_sec", "onscene_sec", "close_sec"):
            if _m not in out.columns:
                out[_m] = 1
            else:
                out[_m] = pd.to_numeric(out[_m], errors="coerce").fillna(1)
                out.loc[out[_m] <= 0, _m] = 1
        if "count" not in out.columns:
            out["count"] = 1
        out["is_dfr_eligible"] = True
        try:
            st.session_state["eligibility_mode"] = "minimal"
            st.session_state["suppress_full_validity"] = True
            st.session_state["final_validity_locked"] = True
        except Exception:
            pass

    # --- Optional debug: show why rows are/aren't valid ---------------------
    if st.session_state.get("debug_validity", False):
        try:
            pr_nonnull = pd.to_numeric(out.get("priority"), errors="coerce").notna().sum()
        except Exception:
            pr_nonnull = 0
        lat_ok  = out.get("lat").notna().sum() if "lat" in out.columns else 0
        lon_ok  = out.get("lon").notna().sum() if "lon" in out.columns else 0
        type_ok = out.get("call_type_up", pd.Series([], dtype=str)).astype(str).str.strip().ne("").sum()

        st.subheader("🔎 Raw/Validity debug")
        st.write({
            "raw_rows": int(len(out)),
            "lat_non_null": int(lat_ok),
            "lon_non_null": int(lon_ok),
            "type_non_blank": int(type_ok),
            "priority_non_null": int(pr_nonnull),
            "minimal_mode": bool(st.session_state.get("raw_is_minimal", False)),
        })

        if st.session_state.get("raw_is_minimal", False):
            try:
                _mask = _minimal_validity_mask(out)
                st.write({"valid_rows_minimal_mask": int(_mask.sum())})
            except Exception as _e:
                st.write({"minimal_mask_error": str(_e)})

        # show a small sample so you can visually confirm columns
        st.dataframe(out.head(15), use_container_width=True)

    # Safety net: never downgrade eligibility in minimal mode if a lock is set
    if st.session_state.get("final_validity_locked", False) and st.session_state.get("eligibility_mode") == "minimal":
        out["is_dfr_eligible"] = True
    # Final circuit breaker so no late-stage filters can undo minimal eligibility
    out = enforce_minimal_guard(out)
    return out

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
    # Robust average: accept pandas/NumPy scalars and coerce safely
    try:
        s = pd.Series(arr)
    except Exception:
        try:
            s = pd.to_numeric(arr, errors="coerce")
        except Exception:
            return np.nan
    vals = pd.to_numeric(s, errors="coerce")
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if len(vals) else np.nan

def fmt_mmss(sec):
    if not np.isfinite(sec):
        return ""
    # guard against negative display — treat negatives as 0 for UI
    if sec < 0:
        sec = 0
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


# --- ALPR clearance rate accessor (single source of truth for downstream) ---
def get_alpr_clear_rate(default: float = 0.05) -> float:
    """Central accessor for ALPR clearance rate assumption."""
    try:
        val = float(st.session_state.get("alpr_clear_rate", default))
        # Clamp to [0,1] to avoid accidental bad inputs
        if not np.isfinite(val):
            return default
        return max(0.0, min(1.0, val))
    except Exception:
        return default


def auto_heat_params(df, max_radius=50, max_blur=50):
    n = len(df)
    if n <= 1:
        return max_radius, max_blur
    base = max_radius * math.sqrt(1000 / n)
    radius = int(min(max_radius, max(5, base)))
    blur = int(min(max_blur, max(5, base)))
    return radius, blur

# --- Header canonicalization helper -----------------------------------------
def _canonicalize_columns(df: pd.DataFrame, groups: list[tuple[str, list[str]]]) -> pd.DataFrame:
    """
    Normalize headers so we don't accidentally create duplicate cols like 'Lat' and 'lat '.
    - Strips whitespace and non-breaking spaces
    - Deduplicates by lowercased key (keeps first occurrence)
    - Renames any case/spacing variants to a chosen canonical name per group
    """
    # 1) Clean raw column text
    cleaned = []
    for c in df.columns:
        if isinstance(c, str):
            c2 = c.replace("\u00A0", " ").strip()   # strip NBSP
        else:
            c2 = str(c).strip()
        cleaned.append(c2)
    df.columns = cleaned

    # 2) Collapse duplicates by lowercased key (keep first)
    seen = {}
    keep_idx = []
    for i, c in enumerate(df.columns):
        key = c.lower()
        if key not in seen:
            seen[key] = i
            keep_idx.append(i)
    if len(keep_idx) != len(df.columns):
        df = df.iloc[:, keep_idx]  # drop dup columns

    # 3) Case-insensitive rename to canonical names
    lower_map = {c.lower(): c for c in df.columns}
    rename_map = {}
    for canon, variants in groups:
        # Include the canon itself as a variant
        all_vars = [canon] + variants
        hit = None
        for v in all_vars:
            k = v.lower()
            if k in lower_map:
                hit = lower_map[k]
                break
        if hit is not None and hit != canon:
            rename_map[hit] = canon
    if rename_map:
        df = df.rename(columns=rename_map)

    return df

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
            "Real-world On-scene Time (min)": "45 min",
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
            "Radar Enabled": "Yes",
            "Radar Radius (mi)": 7.0,
            "Radar Yearly Price (USD)": 250000,
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
            "Radar Enabled": "Yes",
            "Radar Radius (mi)": 7.0,
            "Radar Yearly Price (USD)": 250000,
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
            "Radar Enabled": "No",
            "Radar Radius (mi)": "0",
            "Radar Yearly Price (USD)": "0",
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
            "Radar Enabled": "Yes",
            "Radar Radius (mi)": "1.2",
            "Radar Yearly Price (USD)": "78000",
        },
    },
}

COMPETITOR_OPTIONS = [
    "Skydio X10",
    "Brinc Responder",
    "Paladin Dock 3",
    "Dronesense Dock 3",
]

# Optional: competitor coverage padding to close small map gaps (defaults to 0 → no change)
overpack_pct = st.sidebar.slider(
    "Competitor coverage padding (%)",
    min_value=0, max_value=25, value=0, step=1,
    help="Add a few extra competitor sites to close visible gaps caused by rounding/edges."
)

# one-time UI control near top of the Maps section (or wherever you want)
st.sidebar.checkbox("Show FAA grid squares", value=False, key="show_faa")
if "show_faa" not in st.session_state:
    st.session_state["show_faa"] = False


# --- Single global toggle for overlaying our Aerodome coverage on competitor maps ---
st.sidebar.checkbox(
    "Overlay Aerodome coverage on competitor map",
    value=st.session_state.get("show_our_on_comp", False),
    key="show_our_on_comp",
    help="Show our coverage (blue) on the competitor map for visual comparison."
)

# Debug toggle for seeing raw/validity details
st.sidebar.checkbox("Debug raw/validity", value=False, key="debug_validity")

def overlay_our_on_map_if_enabled(m, coords, eff_range_mi, poly_utm=None, inv_calls=None):
    """
    Safely draw our coverage on an existing Folium map *if* the sidebar toggle is on.
    Call this from the competitor-map render code AFTER the Folium map `m` is created.
    - m: folium.Map
    - coords: list[(lat, lon)]
    - eff_range_mi: float (miles)
    - poly_utm: optional polygon in UTM for outlining (if your helper supports it)
    - inv_calls: optional inverse transform / data needed by polygon_outline_latlon
    """
    if not st.session_state.get("show_our_on_comp", False):
        return
    try:
        r_m = float(eff_range_mi) * 1609.34
    except Exception:
        r_m = 0.0
    if r_m <= 0 or not coords:
        return

    try:
        # Circles + launch markers
        for (_la, _lo) in coords:
            folium.Circle(
                location=(_la, _lo), radius=r_m,
                color="#2A6FD3", weight=2, fill=False, opacity=0.8
            ).add_to(m)
            folium.CircleMarker(
                location=(_la, _lo), radius=3,
                color="#2A6FD3", fill=True
            ).add_to(m)

        # Optional outline if your outline helper + data are available
        outline_fn = globals().get("polygon_outline_latlon")
        if callable(outline_fn) and (poly_utm is not None) and (inv_calls is not None):
            try:
                _outline = outline_fn(poly_utm, inv_calls)
                if _outline:
                    folium.PolyLine(
                        locations=[(lt, ln) for lt, ln in _outline],
                        color="#2A6FD3", weight=3, opacity=0.9
                    ).add_to(m)
            except Exception:
                pass
    except Exception:
        # Never break map rendering due to overlay—fail silent
        pass

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
replay_inputs["full_city"] = _find_csv_by_partial("Launch Locations - Full City")


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

#
# Normalize to flexible schema if applicable (non-breaking).
# If the input matches the minimal single-timestamp schema, this returns a normalized df
# with zeros for response metrics; otherwise it passes the original through unchanged.
try:
    df_all = parse_calls_flexible(raw_df)
except Exception:
    df_all = raw_df

# Canonicalize columns/values so downstream logic always sees the same schema
df_all = canonicalize_calls(df_all)

# If `df_all` looks like the minimal-normalized output, mirror key fields back into raw_df
if set(["lat","lon","call_type_up","priority"]).issubset(df_all.columns):
    # Ensure legacy header variants exist in raw_df so older logic doesn't break
    if "Lat" not in raw_df.columns:
        raw_df["Lat"] = pd.to_numeric(df_all["lat"], errors="coerce")
    if "Long" not in raw_df.columns:
        raw_df["Long"] = pd.to_numeric(df_all["lon"], errors="coerce")
    raw_df["Lon"] = raw_df["Long"]  # some paths look for 'Lon'
    raw_df["Call Type"] = df_all["call_type_up"].astype(str)
    pr_num = pd.to_numeric(df_all["priority"], errors="coerce")
    raw_df["Call Priority"] = pr_num.astype("Int64")  # nullable int; preserves <NA>
    if "Call Create" not in raw_df.columns and "call_create" in df_all.columns:
        raw_df["Call Create"] = df_all["call_create"]
    # Also mirror to legacy 'Priority' header for any audit/widgets expecting it
    try:
        raw_df["Priority"] = pr_num.astype("Int64")
    except Exception:
        raw_df["Priority"] = pr_num

# --- Minimal single-timestamp schema detection & backfill ------------------
try:
    _has_min_cols = all(c in df_all.columns for c in ["lat","lon","call_type_up","priority"]) and "call_create" in df_all.columns
    _all_time_zero = (
        pd.to_numeric(df_all.get("patrol_sec", 0), errors="coerce").fillna(0).max() == 0
        and pd.to_numeric(df_all.get("onscene_sec", 0), errors="coerce").fillna(0).max() == 0
    )
    st.session_state["raw_is_minimal"] = bool(_has_min_cols and _all_time_zero)
except Exception:
    st.session_state["raw_is_minimal"] = False

# If we're in the minimal schema case, make sure any later strict checks don't crash
if st.session_state.get("raw_is_minimal", False):
    # Ensure the original raw_df has any columns that strict code expects
    # (we add them as empty so downstream "required" checks pass without blowing up)
    _required_maybe = [
        # common time/phase columns seen across feeds
        "First Dispatch", "First Arrive", "Call Close",
        "Dispatch Time", "Arrive Time", "Close Time",
        # sometimes variants
        "First Dispatch Time", "First Arrive Time", "Call Create",
        # metrics columns some code computes/reads
        "patrol_sec", "drone_eta_sec", "onscene_sec",
        "dispatch_sec", "arrive_sec", "close_sec",
        # common identity columns
        "Call Type", "Call Priority", "Lat", "Lon", "Long",
    ]
    for _col in _required_maybe:
        if _col not in raw_df.columns:
            raw_df[_col] = pd.NA

    # Also make sure df_all has numeric zeros for metric fields (safety)
    for _m in ["patrol_sec","drone_eta_sec","onscene_sec","dispatch_sec","arrive_sec","close_sec"]:
        if _m not in df_all.columns:
            df_all[_m] = 0
        else:
            df_all[_m] = pd.to_numeric(df_all[_m], errors="coerce").fillna(0)

    # Guarantee the count column exists in df_all
    if "count" not in df_all.columns:
        df_all["count"] = 1

    # Mirror canonical fields back to legacy/raw column names for downstream merges
    try:
        if "Lat" in raw_df.columns and "lat" in df_all.columns:
            raw_df["Lat"] = pd.to_numeric(df_all["lat"], errors="coerce")
        if "Long" in raw_df.columns and "lon" in df_all.columns:
            raw_df["Long"] = pd.to_numeric(df_all["lon"], errors="coerce")
            # also provide 'Lon' alias if some code uses it
            raw_df["Lon"] = raw_df["Long"]
        # Provide a numeric 'Priority' column (nullable Int64; no silent zeros)
        _pr_mirror = pd.to_numeric(df_all.get("priority"), errors="coerce")
        try:
            raw_df["Priority"] = _pr_mirror.astype("Int64")
        except Exception:
            raw_df["Priority"] = _pr_mirror
        # Ensure 'Call Type' stays uppercase for the agency mapping
        if "call_type_up" in df_all.columns:
            raw_df["Call Type"] = df_all["call_type_up"].astype(str)
    except Exception:
        pass

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

            # Row 3 = headers, Row 4+ = data
            headers = [h.strip() for h in df0.iloc[2].tolist()]
            data = df0.iloc[3:].copy()
            data.columns = headers
            
            # Ensure Lat/Lon are numeric
            if "Lat" in data.columns and "Lon" in data.columns:
                data["Lat"] = pd.to_numeric(data["Lat"], errors="coerce")
                data["Lon"] = pd.to_numeric(data["Lon"], errors="coerce")
            
            return data.reset_index(drop=True), meta_agency, meta_sqmi

        # fallback: old style CSV
        df_old = pd.read_csv(BytesIO(raw))
        return df_old, None, None

    except Exception:
        # If anything breaks, fallback to old style
        df_old = pd.read_csv(BytesIO(raw))
        return df_old, None, None


# ─── Use loader depending on whether a file was provided ─────────────────

if launch_file is not None:
    st.sidebar.success(
        "Loaded launch locations from saved run." if launch_src == "replay"
        else "Using uploaded launch locations file."
    )
    try:
        launch_file.seek(0)  # reset pointer just in case
    except Exception:
        pass

    # Use the unified loader
    launch_df, _meta_agency, _meta_sqmi = _load_launch_locations_csv(launch_file)

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

# 2b) Normalize headers (case/space-insensitive) and avoid dup columns
launch_df = _canonicalize_columns(
    launch_df,
    groups=[
        ("Location Name", ["Locations", "Location", "Site Name"]),
        ("Address",       ["Site Address", "Location Address"]),
        ("Lat",           ["Latitude", "lat"]),
        ("Lon",           ["Longitude", "Long", "Lng", "lon"]),
        ("Type",          ["Site Type"])
    ],
)
# Only add missing required columns after canonicalization
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

geolocator = Nominatim(user_agent="dfria_app")
geocode = RateLimiter(
    geolocator.geocode,
    min_delay_seconds=1.5,
    max_retries=3,
    error_wait_seconds=2.0
)

# --- Helper: Normalize address for geocoding ---
def normalize_address(addr: str) -> str:
    """
    Basic normalization for geocoding:
    - Strips whitespace and punctuation artifacts
    - Replaces multiple spaces with a single space
    - Ensures consistent formatting
    """
    if not isinstance(addr, str):
        return ""
    a = addr.strip()
    # Replace common artifacts
    a = re.sub(r"\s+", " ", a)  # collapse multiple spaces
    a = a.replace("#", "").replace(";", ",")
    return a

@st.cache_data(show_spinner=False, ttl=600)
def lookup(addr: str):
    """Return (lat, lon) for an address. Tries raw then normalized."""
    if not isinstance(addr, str) or not addr.strip():
        return (None, None)
    a_raw = addr.strip()
    a_norm = normalize_address(a_raw)
    try:
        loc = geocode(a_raw)
        if loc is None and a_norm and a_norm != a_raw:
            loc = geocode(a_norm)
        return (loc.latitude, loc.longitude) if loc else (None, None)
    except Exception:
        return (None, None)

# ---- Apply any persisted manual fixes first (by Address, else Location Name) ----
_launch_overrides = st.session_state.get("launch_overrides", {})  # {key -> (lat, lon)}
if _launch_overrides:
    for i in launch_rows.index:
        key = (str(launch_rows.at[i, "Address"]).strip()
               if str(launch_rows.at[i, "Address"]).strip()
               else str(launch_rows.at[i, "Location Name"]).strip())
        if key in _launch_overrides:
            la, lo = _launch_overrides[key]
            if isinstance(la, (int, float)) and isinstance(lo, (int, float)) and np.isfinite(la) and np.isfinite(lo):
                launch_rows.at[i, "Lat"] = float(la)
                launch_rows.at[i, "Lon"] = float(lo)

# ---- Geocode only rows that still need it (have Address but missing Lat/Lon) ----
to_geocode = launch_rows["Address"].notna() & (
    pd.to_numeric(launch_rows["Lat"], errors="coerce").isna() |
    pd.to_numeric(launch_rows["Lon"], errors="coerce").isna()
)

for idx in launch_rows.loc[to_geocode].index:
    raw_addr = str(launch_rows.at[idx, "Address"]).strip()
    la, lo = lookup(raw_addr)
    if la is not None and lo is not None and np.isfinite(la) and np.isfinite(lo):
        launch_rows.at[idx, "Lat"] = la
        launch_rows.at[idx, "Lon"] = lo

# ---- Preview (helps sanity-check) ----
st.sidebar.subheader("Launch Locations (geocoded)")
st.sidebar.dataframe(
    launch_rows[["Location Name","Address","Lat","Lon"]],
    use_container_width=True
)
if not hotspot_rows.empty:
    st.sidebar.subheader("Hotspot Addresses (from CSV)")
    st.sidebar.dataframe(hotspot_rows[["Location Name","Address"]], use_container_width=True)

# ---- Validate coords; offer manual fix if any remain missing ----
launch_rows["Lat"] = pd.to_numeric(launch_rows["Lat"], errors="coerce")
launch_rows["Lon"] = pd.to_numeric(launch_rows["Lon"], errors="coerce")
valid_coords = launch_rows["Lat"].notna() & launch_rows["Lon"].notna()

if not valid_coords.all():
    # UI to enter manual coordinates for failing rows
    with st.sidebar.expander("Launch Locations — manual coordinates", expanded=True):
        bad_idxs = list(launch_rows.index[~valid_coords])

        def _row_label(i):
            nm = str(launch_rows.at[i, "Location Name"]).strip()
            ad = str(launch_rows.at[i, "Address"]).strip()
            return (nm or ad or f"Row {i}")[:60]

        sel = st.selectbox("Row to fix", options=bad_idxs, format_func=_row_label, key="ll_fix_row")
        lat_in = st.number_input("Manual Latitude", format="%.6f", key="ll_fix_lat")
        lon_in = st.number_input("Manual Longitude", format="%.6f", key="ll_fix_lon")

        if st.button("Apply", key="ll_fix_apply"):
            if np.isfinite(lat_in) and np.isfinite(lon_in):
                # write now
                launch_rows.at[sel, "Lat"] = float(lat_in)
                launch_rows.at[sel, "Lon"] = float(lon_in)

                # persist for future reruns (key by Address, else Location Name)
                _key = (str(launch_rows.at[sel, "Address"]).strip()
                        if str(launch_rows.at[sel, "Address"]).strip()
                        else str(launch_rows.at[sel, "Location Name"]).strip())
                if _key:
                    ov = dict(st.session_state.get("launch_overrides", {}))
                    ov[_key] = (float(lat_in), float(lon_in))
                    st.session_state["launch_overrides"] = ov

                st.success("Saved. Re-running with updated coordinates…")
                st.rerun()
            else:
                st.warning("Please enter valid numeric Latitude and Longitude.")

# Re-check after potential manual fix UI
valid_coords = launch_rows["Lat"].notna() & launch_rows["Lon"].notna()
if not valid_coords.all():
    bad = launch_rows.loc[~valid_coords, ["Location Name","Address","Lat","Lon"]]
    st.sidebar.warning(
        "Some LAUNCH rows still lack valid Lat/Lon. They will be ignored. "
        "Fix them manually in the sidebar or add Lat/Lon in the CSV.\n"
        + bad.to_csv(index=False)
    )

# proceed only with valid rows
launch_rows = launch_rows.loc[valid_coords].copy()

# ─── 2e) Build launch_coords for downstream use (maps/ETA) ─
launch_coords = list(
    launch_rows.loc[valid_coords, ["Lat","Lon"]]
    .itertuples(index=False, name=None)
)
if not launch_coords:
    st.sidebar.error("No valid launch locations available to draw the drone-range circle.")
    st.stop()

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

# New: ALPR clearance rate assumption (default 5%)
alpr_clear_rate = st.sidebar.number_input(
    "ALPR Clearance Rate (0–1)",
    value=float(a.get("alpr_clear_rate", 0.05)),  # default 5%
    step=0.01,
    format="%.2f"
)
# Make available globally for downstream metrics
st.session_state["alpr_clear_rate"] = float(alpr_clear_rate)

# --- Drone selection + speed/range defaults (fallback to Dock 3) ---
def _spec_number(specs: dict, key_candidates: list[str]) -> float | None:
  """Pull a numeric value from specs[key]; handles strings like '47 mph'."""
  if not specs:
    return None
  import re
  for k in key_candidates:
    if k in specs and specs[k] is not None:
      v = specs[k]
      if isinstance(v, (int, float)):
        return float(v)
      m = re.search(r"[-+]?\d*\.?\d+", str(v))
      if m:
        try:
          return float(m.group())
        except Exception:
          pass
  return None

def _dock_label_to_platform(label: str) -> str:
    lab = (label or "").strip().upper().replace("FLOCK AERODOME ", "")
    if lab in {"DOCK 3", "DOCK3"}: return "Flock Aerodome Dock 3"
    if lab in {"ALPHA"}:         return "Flock Aerodome Alpha"
    if lab in {"DELTA"}:         return "Flock Aerodome Delta"
    if lab in {"M350", "M 350"}: return "Flock Aerodome M350"
    # If already a full platform name, pass through
    for k in PLATFORMS.keys():
        if lab == str(k).strip().upper():
            return k
    return "Flock Aerodome Dock 3"

# Determine default selection from launch rows
_default_platform = "Flock Aerodome Dock 3"
try:
    if "Dock Type" in launch_df.columns:
        _dock_mode = (
            launch_df["Dock Type"].astype(str).str.strip().replace({"": None}).dropna().mode()
        )
        if not _dock_mode.empty:
            _default_platform = _dock_label_to_platform(_dock_mode.iat[0])
except Exception:
    pass

FLOCK_PLATFORMS = [k for k in PLATFORMS.keys() if str(k).startswith("Flock ")]

# Global pricing/docks override toggle (show this before platform selection)
override_launch_pricing = st.sidebar.checkbox(
    "Override Launch Locations sheet (pricing/docks)",
    value=st.session_state.get("override_launch_pricing", False),
    help="When enabled, applies the selected price per dock and docks per location to pricing and totals."
)

selected_platform = st.sidebar.selectbox(
    "Drone Platform",
    options=FLOCK_PLATFORMS,
    index=(FLOCK_PLATFORMS.index(_default_platform) if _default_platform in FLOCK_PLATFORMS else 0),
    help="Select the drone platform to use for range, speed, and pricing.",
    disabled=not override_launch_pricing,
)

_specs = (PLATFORMS.get(selected_platform, {}) or {}).get("specs", {}) or {}
DOCK3_SPEED_FALLBACK = 47.0
DOCK3_RANGE_FALLBACK = 3.5
spec_speed = _spec_number(_specs, ["Real-world Speed (MPH)", "Speed (MPH)", "Speed"]) or DOCK3_SPEED_FALLBACK
spec_range = PLATFORMS.get(selected_platform, {}).get("range_mi", DOCK3_RANGE_FALLBACK)

st.sidebar.caption("Selected Drone Specs (configurable)")

# Base defaults from platform
_default_price_per_dock = int(PLATFORMS.get(selected_platform, {}).get('price_per_dock', 0) or 0)
_default_docks_per_loc  = int(PLATFORMS.get(selected_platform, {}).get('docks_per_location', 1) or 1)

# Configurable inputs with platform defaults
drone_range  = st.sidebar.number_input("Range (miles)", value=float(spec_range), step=0.1, disabled=not override_launch_pricing)
drone_speed  = st.sidebar.number_input("Speed (mph)",  value=float(spec_speed), step=1.0, disabled=not override_launch_pricing)
price_per_dock_sel = st.sidebar.number_input(
    "Price per dock (USD)", value=int(_default_price_per_dock), step=1000,
    disabled=not override_launch_pricing
)
docks_per_loc_sel  = st.sidebar.number_input(
    "Docks per location", value=int(_default_docks_per_loc), step=1,
    disabled=not override_launch_pricing
)

# Persist selections for use in downstream pricing functions
st.session_state["selected_platform"] = selected_platform
st.session_state["selected_range_mi"] = float(drone_range)
st.session_state["selected_speed_mph"] = float(drone_speed)
st.session_state["override_launch_pricing"] = bool(override_launch_pricing)
st.session_state["selected_price_per_dock"] = int(price_per_dock_sel)
st.session_state["selected_docks_per_location"] = int(docks_per_loc_sel)

progress.progress(70)

# ─── Agency details (saved with each run) ────────────────────────────────────
with st.sidebar.expander("Agency details", expanded=True):
    analyst_name = st.text_input("Analyst (optional)", value="", key="analyst_name")
    run_notes = st.text_area("Run notes (optional)", height=80, key="run_notes")

# ─── Set Up (PowerPoint layout) ─────────────────────────────────────────────
st.markdown("---")
st.header("Set Up")
c1, c2 = st.columns([2, 1])
with c1:
    st.markdown(f"**Agency Name:** {AGENCY_NAME or '—'}")
    st.markdown(
        "Get agency logo and remove background: "
        "<a href='https://www.remove.bg/' target='_blank'>remove.bg ↗</a>",
        unsafe_allow_html=True,
    )
with c2:
    _deck_type = "Minimal deck template" if st.session_state.get("raw_is_minimal", False) else "Full deck template"
    st.metric("Template", _deck_type)

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

# === Full City Launch Locations ===
st.sidebar.header("7) Full City Launch Locations (optional)")

full_city_src = None
full_city_file = None

if replay_inputs.get("full_city"):
    full_city_file = replay_inputs["full_city"]   # may be bytes or file-like
    full_city_src = "replay"
else:
    full_city_file = st.sidebar.file_uploader("Upload Launch Locations - Full City CSV", type=["csv"])
    full_city_src = "upload" if full_city_file else None

if full_city_src == "replay":
    st.sidebar.success("Loaded Full City CSV from saved run.")
elif full_city_src == "upload":
    st.sidebar.info("Using uploaded Full City CSV.")

# Optional: clear geocoder cache if you had network issues / plane Wi‑Fi
if st.sidebar.button("🔄 Clear geocode cache", help="Forces fresh geocoding next run (useful if some addresses failed on poor Wi‑Fi)"):
    try:
        lookup.clear()  # clear st.cache_data for this function
        st.sidebar.success("Geocode cache cleared. Re-run to fetch fresh coordinates.")
    except Exception:
        st.sidebar.warning("Could not clear cache; continue.")

# Normalize bytes → file-like so your CSV loader works with either source
import io
if isinstance(full_city_file, (bytes, bytearray)):
    _buf_fc = io.BytesIO(full_city_file)
    _buf_fc.name = "Launch Locations - Full City.csv"
    full_city_file = _buf_fc
        
# --- FULL CITY section: robust load + address-first geocode (no duplicate columns) ---
# Helper: extract a float from arbitrary cell text (handles "32.91°", "  -97.32 ", etc.)
def _extract_float(cell):
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return np.nan
    m = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', str(cell))
    try:
        return float(m.group()) if m else np.nan
    except Exception:
        return np.nan

fj_df = None
launch_coords_full = []

if full_city_file is not None:
    try:
        # Ensure we can read from the start
        try:
            full_city_file.seek(0)
        except Exception:
            pass

        # Load CSV (do NOT use data_editor here) — keep original headers
        df_fc = pd.read_csv(full_city_file)

        # Helper: pick a column by tolerant names, returning the original column name
        def pick_fc_col(df, names):
            cmap = {str(c).strip().lower(): c for c in df.columns}
            for n in names:
                k = str(n).strip().lower()
                if k in cmap:
                    return cmap[k]
            return None

        # Identify columns present in the uploaded CSV
        addr_col = pick_fc_col(df_fc, ["Address", "Site Address", "Location Address", "Site"]) 
        lat_col  = pick_fc_col(df_fc, ["Lat", "Latitude"]) 
        lon_col  = pick_fc_col(df_fc, ["Lon", "Long", "Lng", "Longitude"]) 
        type_col = pick_fc_col(df_fc, ["Type", "Site Type"]) 

        # ---- APPLY PERSISTENT MANUAL OVERRIDES (by address) ----
        # Any manual fixes applied in the UI are stored in session_state and re-applied on rerun
        _fc_overrides = st.session_state.get("fc_overrides", {})  # {address_str: (lat, lon)}

        # Ensure Lat/Lon columns exist so we have somewhere to write the overrides
        if lat_col is None:
            df_fc["Lat"] = np.nan
            lat_col = "Lat"
        if lon_col is None:
            df_fc["Lon"] = np.nan
            lon_col = "Lon"

        # If we have an address column and saved overrides, apply them now (don’t overwrite valid coords)
        if addr_col is not None and _fc_overrides:
            for _idx, _addr_val in df_fc[addr_col].items():
                _key = str(_addr_val).strip()
                if _key in _fc_overrides:
                    _la, _lo = _fc_overrides[_key]
                    if _la is not None and _lo is not None and np.isfinite(_la) and np.isfinite(_lo):
                        if pd.isna(df_fc.at[_idx, lat_col]) or pd.isna(df_fc.at[_idx, lon_col]):
                            df_fc.at[_idx, lat_col] = float(_la)
                            df_fc.at[_idx, lon_col] = float(_lo)

        # If a Type column exists, keep only "Launch Location" rows
        if type_col is not None:
            _t = df_fc[type_col].astype(str).str.strip().str.lower()
            df_fc = df_fc.loc[_t.eq("launch location")].copy()

        # Ensure Lat/Lon fields exist (create if missing)
        if lat_col is None:
            df_fc["Lat"] = np.nan
            lat_col = "Lat"
        if lon_col is None:
            df_fc["Lon"] = np.nan
            lon_col = "Lon"

        # Robust parse: do NOT overwrite good values; extract floats from messy strings
        df_fc[lat_col] = df_fc[lat_col].map(_extract_float)
        df_fc[lon_col] = df_fc[lon_col].map(_extract_float)

        # Address-first geocoding with fallback to provided Lat/Lon.
        # We NEVER overwrite a user-provided valid number, and we also
        # treat (0,0) or near-zero pairs as invalid.
        def _is_finite_num(x):
            """True only for real numbers that are finite; safe for None/strings."""
            try:
                # Fast path for numeric types
                if isinstance(x, (int, float, np.floating, np.integer)):
                    return np.isfinite(x)
                # Attempt coercion (handles strings like "32.91", " -97.31 ")
                return np.isfinite(float(x))
            except Exception:
                return False

        def _is_finite_pair(a, b):
            return _is_finite_num(a) and _is_finite_num(b)

        Z_EPS = 1e-6  # treat pairs very close to (0,0) as invalid

        # Normalize address column to string if present
        if addr_col is not None:
            addr_str = df_fc[addr_col].astype(str).str.strip()
        else:
            addr_str = pd.Series([""] * len(df_fc), index=df_fc.index)

        for idx in df_fc.index:
            # current numeric values (after _extract_float above)
            la_existing = df_fc.at[idx, lat_col]
            lo_existing = df_fc.at[idx, lon_col]

            la_out, lo_out = (la_existing, lo_existing)

            # 1) Try to geocode the address first (preferred)
            if addr_str.at[idx]:
                ga, go = lookup(addr_str.at[idx])  # raw then normalized (inside lookup)
                if _is_finite_pair(ga, go):
                    la_out, lo_out = float(ga), float(go)

            # 2) If geocoding failed, but CSV had valid numbers, use them
            if not _is_finite_pair(la_out, lo_out) and _is_finite_pair(la_existing, lo_existing):
                la_out, lo_out = float(la_existing), float(lo_existing)

            # 3) Treat (0,0) (or extremely close) as invalid
            if _is_finite_pair(la_out, lo_out) and (abs(la_out) < Z_EPS and abs(lo_out) < Z_EPS):
                la_out, lo_out = (np.nan, np.nan)

            df_fc.at[idx, lat_col] = la_out
            df_fc.at[idx, lon_col] = lo_out

        # Final numeric coercion to avoid object dtype sneaking through to .abs()
        df_fc[lat_col] = pd.to_numeric(df_fc[lat_col], errors="coerce")
        df_fc[lon_col] = pd.to_numeric(df_fc[lon_col], errors="coerce")
        # Build final coordinate list
        have = (
            df_fc[lat_col].notna() & df_fc[lon_col].notna() &
            ~(df_fc[lat_col].abs() < 1e-6) & ~(df_fc[lon_col].abs() < 1e-6)
        )
        launch_coords_full = [
            (float(la), float(lo))
            for la, lo in df_fc.loc[have, [lat_col, lon_col]].itertuples(index=False, name=None)
            if np.isfinite(la) and np.isfinite(lo)
        ]

        # One-time debug expander: show which rows had user-provided coords prior to any geocoding
        with st.sidebar.expander("Full City — coordinate sources", expanded=False):
            # Show which rows had user-provided coords prior to any geocoding
            _lat_raw = df_fc[lat_col].copy()
            _lon_raw = df_fc[lon_col].copy()
            st.write({
                "rows_in_csv": int(len(df_fc)),
                "rows_with_coords_now": int(have.sum()),
            })
            if (~have).any():
                _show_cols = []
                if "Location Name" in df_fc.columns: _show_cols.append("Location Name")
                if addr_col is not None: _show_cols.append(addr_col)
                _show_cols += [lat_col, lon_col]
                st.caption("Rows still missing coordinates after parsing + geocode:")
                st.dataframe(df_fc.loc[~have, _show_cols], use_container_width=True)

        # Expose fj_df with canonical Lat/Lon names for any downstream code
        fj_df = df_fc.rename(columns={lat_col: "Lat", lon_col: "Lon"})

        # Manual coordinate fix (simple, index-based; no MultiIndex/data_editor)
        if (~have).any():
            with st.sidebar.expander("Full City — manual coordinates", expanded=False):
                bad_idxs = list(df_fc.index[~have])
                def _row_label(i):
                    nm = None
                    if "Location Name" in df_fc.columns:
                        nm = str(df_fc.loc[i, "Location Name"]).strip()
                    ad = None
                    if addr_col is not None:
                        ad = str(df_fc.loc[i, addr_col]).strip()
                    return (nm or ad or f"Row {i}")[:60]

                sel = st.selectbox("Row to fix", options=bad_idxs, format_func=_row_label, key="fc_fix_row2")
                lat_in = st.number_input("Manual Latitude", format="%.6f", key="fc_fix_lat2")
                lon_in = st.number_input("Manual Longitude", format="%.6f", key="fc_fix_lon2")
                if st.button("Apply", key="fc_fix_apply2"):
                    if np.isfinite(lat_in) and np.isfinite(lon_in):
                        # Persist this manual fix so future reruns pick it up before geocoding
                        _addr_key = None
                        if addr_col is not None and sel in df_fc.index:
                            _addr_key = str(df_fc.loc[sel, addr_col]).strip()
                        if _addr_key:
                            _ov = dict(st.session_state.get("fc_overrides", {}))
                            _ov[_addr_key] = (float(lat_in), float(lon_in))
                            st.session_state["fc_overrides"] = _ov

                        # Also write into the working DataFrame right now
                        df_fc.at[sel, lat_col] = float(lat_in)
                        df_fc.at[sel, lon_col] = float(lon_in)

                        # Refresh downstream computed artifacts
                        fj_df = df_fc.rename(columns={lat_col: "Lat", lon_col: "Lon"})
                        have2 = df_fc[lat_col].notna() & df_fc[lon_col].notna()
                        launch_coords_full = [
                            (float(la), float(lo))
                            for la, lo in df_fc.loc[have2, [lat_col, lon_col]].itertuples(index=False, name=None)
                            if np.isfinite(la) and np.isfinite(lo)
                        ]
                        st.success("Applied and saved. Re-running with updated coordinates…")
                        st.rerun()
                    else:
                        st.warning("Please enter valid numeric Latitude and Longitude.")

        if not launch_coords_full:
            st.sidebar.warning(
                "Full City: 0 usable launch coordinates after address-first geocoding. "
                "Check addresses or add Lat/Lon, or enter manually above."
            )

    except Exception as e:
        st.sidebar.error(f"Failed to load/geocode Full City CSV: {e}")
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
    # ── Minimal-aware raw column validation ─────────────────────────────────
    # Choose a dataframe to validate: prefer a local candidate if available
    _df_for_check = None
    for _name in ("df_raw", "raw_df", "raw_calls", "df", "out"):
        if _name in locals():
            _df_for_check = locals()[_name]
            break
    if _df_for_check is None:
        _df_for_check = st.session_state.get("df_raw")

    # If nothing to check, don't block the app
    if isinstance(_df_for_check, pd.DataFrame) and not _df_for_check.empty:
        try:
            # Canonicalize first so defaults are present
            _df_for_check = canonicalize_calls(_df_for_check)
        except Exception:
            pass

        present = {str(c).strip().lower() for c in _df_for_check.columns}

        # Full-schema expectations (legacy)
        required_full = {
            'call_create','first_dispatch','first_arrive','call_close',
            'call_type','call_type_up','priority','lat','lon',
            'patrol_sec','drone_eta_sec','onscene_sec','dispatch_sec','arrive_sec','close_sec','count'
        }
        # Minimal schema expectations (call_create optional)
        required_min = {'call_type','priority','lat','lon'}

        # If minimal mode is active, only enforce the minimal set
        if st.session_state.get('raw_is_minimal', False):
            missing = {c for c in required_min if c not in present}
        else:
            missing = {c for c in required_full if c not in present}

        if missing:
            st.error("Missing required Raw Call Data columns.")
            st.caption(f"Columns missing: {sorted(missing)}")
            st.stop()


# --- Minimal-aware timestamp parsing --------------------------------------
# Always parse Call Create
create_dt = parse_flexible_time(raw_df[col_create])

# For optional times, if the column is missing (None) or not in df, synthesize
def _col_in_df(colname: str | None) -> bool:
    return bool(colname) and (str(colname) in raw_df.columns)

if _col_in_df(col_dispatch):
    dispatch_dt = parse_flexible_time(raw_df[col_dispatch])
else:
    dispatch_dt = pd.to_datetime(create_dt, errors="coerce") + pd.to_timedelta(1, unit="s")

if _col_in_df(col_arrive):
    arrive_dt = parse_flexible_time(raw_df[col_arrive])
else:
    arrive_dt = pd.to_datetime(create_dt, errors="coerce") + pd.to_timedelta(2, unit="s")

if _col_in_df(col_close):
    close_dt = parse_flexible_time(raw_df[col_close])
else:
    close_dt = pd.to_datetime(create_dt, errors="coerce") + pd.to_timedelta(3, unit="s")

response_sec = (arrive_dt - create_dt).dt.total_seconds()

# --- Validity mask ---------------------------------------------------------
_minimal_mode = bool(st.session_state.get("raw_is_minimal", False) or not _col_in_df(col_dispatch) or not _col_in_df(col_arrive) or not _col_in_df(col_close))

if _minimal_mode:
    # Minimal schema: only require a timestamp; do not drop on dispatch/arrive/close
    valid = create_dt.notna()
else:
    # Full schema path: preserve legacy stricter checks
    valid = (
         create_dt.notna()
      &  dispatch_dt.notna()           # drop if blank → self-initiated
      &  arrive_dt.notna()             # drop if blank
      &  (dispatch_dt  > create_dt)    # real dispatch
      &  (arrive_dt    > dispatch_dt)  # real arrival
      &  (arrive_dt    > create_dt)    # no weird negatives
      &  (response_sec > 5)            # ignore <= 5s glitches
    )

# — filter down your DataFrame & all series
raw_df      = raw_df.loc[valid].copy()
create_dt   = create_dt[valid]
dispatch_dt = dispatch_dt[valid]
arrive_dt   = arrive_dt[valid]
close_dt    = close_dt[valid]

# --- Midnight rollover fix for time-of-day inputs ---------------------------
# If any subsequent timestamp is earlier than the previous one, assume it rolled past midnight.
try:
    mask = dispatch_dt.notna() & create_dt.notna() & (dispatch_dt < create_dt)
    if mask.any():
        dispatch_dt.loc[mask] = dispatch_dt.loc[mask] + pd.to_timedelta(1, unit="D")
except Exception:
    pass
try:
    mask = arrive_dt.notna() & dispatch_dt.notna() & (arrive_dt < dispatch_dt)
    if mask.any():
        arrive_dt.loc[mask] = arrive_dt.loc[mask] + pd.to_timedelta(1, unit="D")
except Exception:
    pass
try:
    mask = close_dt.notna() & arrive_dt.notna() & (close_dt < arrive_dt)
    if mask.any():
        close_dt.loc[mask] = close_dt.loc[mask] + pd.to_timedelta(1, unit="D")
except Exception:
    pass

patrol_sec  = (arrive_dt - create_dt).dt.total_seconds()
onscene_sec = (close_dt  - arrive_dt).dt.total_seconds()
lat = pd.to_numeric(raw_df[col_lat], errors="coerce")
lon = pd.to_numeric(raw_df[col_lon], errors="coerce")

progress.progress(85)

# build a proper Lat/Lon array from LAUNCH rows only
try:
    launch_rows["Lat"] = pd.to_numeric(launch_rows["Lat"], errors="coerce")
    launch_rows["Lon"] = pd.to_numeric(launch_rows["Lon"], errors="coerce")
    raw_coords = launch_rows[["Lat", "Lon"]].values

    launch_coords = [
        (float(lat), float(lon))
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
_spd = float(st.session_state.get("selected_speed_mph", drone_speed))
drone_eta_sec = dist_mi / max(_spd,1e-9) * 3600
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

# Ensure only valid coordinates proceed to mapping/metrics
# (Folium HeatMap raises if any lat/lon are NaN.)
df_all = df_all.dropna(subset=["lat", "lon"]).copy()

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

_rng = float(st.session_state.get("selected_range_mi", drone_range))
in_range = dfr_only[
    dfr_only["dist_mi"] <= _rng
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
# Minimal fix: ALPR export now includes a leading 'Org Name' column in col A.
# Drop it so the remaining columns match the original positions used below.
if alpr_df is not None and not alpr_df.empty:
    # Normalize headers (strip NBSP and whitespace)
    alpr_df.columns = [str(c).replace("\u00A0", " ").strip() for c in alpr_df.columns]
    # If the first column is 'Org Name' (case-insensitive), drop it
    if len(alpr_df.columns) > 0 and str(alpr_df.columns[0]).strip().lower() == "org name":
        alpr_df = alpr_df.drop(columns=[alpr_df.columns[0]])
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
    _rng = float(st.session_state.get("selected_range_mi", drone_range))
    in_range_mask = (dist <= _rng) & np.isfinite(dist)

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
    _spd = float(st.session_state.get("selected_speed_mph", drone_speed))
    etas_sec = dist / max(_spd, 1e-9) * 3600
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
        _rng = float(st.session_state.get("selected_range_mi", drone_range))
        in_rng2 = (dist2 <= _rng) & np.isfinite(dist2)

        # ── METRICS (IN-RANGE) ───────────────────────────────────────────────
        # 1) Unique audio locations (addresses) in range
        audio_sites = int(addr_v[in_rng2].nunique())

        # 2) Total hits in range
        audio_hits = int(hits_v[in_rng2].sum())

        # 3) Hits-weighted average ETA in range
        # Use selected speed if provided
        _spd = float(st.session_state.get("selected_speed_mph", drone_speed))
        etas_sec = dist2 / max(_spd, 1e-9) * 3600  # seconds
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
avg_clr      = average(clearable["onscene_sec"]) if "onscene_sec" in clearable.columns else np.nan

# Fallback path: if we are in minimal mode (no real arrive/close)
# OR there are no usable on-scene timestamps, assume 55 minutes.
_AVG_CLR_FALLBACK_SEC = 55 * 60
def _has_col(colname: str | None) -> bool:
    try:
        return bool(colname) and (str(colname) in raw_df.columns)
    except Exception:
        return False

try:
    _no_real_phase_times = bool(
        _minimal_mode or not (_has_col(col_arrive) and _has_col(col_close))
    )
except Exception:
    _no_real_phase_times = True

try:
    _onscene_present = (
        "onscene_sec" in clearable.columns and
        pd.to_numeric(clearable["onscene_sec"], errors="coerce").notna().any()
    )
except Exception:
    _onscene_present = False

_use_clr_fallback = _no_real_phase_times or not _onscene_present or not np.isfinite(avg_clr)
avg_clr_effective = (avg_clr if not _use_clr_fallback else _AVG_CLR_FALLBACK_SEC)

first_on_pct = float(np.mean(in_range["drone_eta_sec"] < in_range["patrol_sec"]) * 100) if in_count else np.nan
pct_dec      = ((avg_in-avg_drone)/avg_in*100) if avg_in>0 else np.nan
alpr_rate   = get_alpr_clear_rate()  # applies to ALPR + Audio hits
exp_cleared = int(round(in_count * cancel_rate + dfr_alpr_audio * alpr_rate))
time_saved   = avg_clr_effective * exp_cleared
officers     = (time_saved/3600)/fte_hours if fte_hours>0 else np.nan
roi          = officers * officer_cost if np.isfinite(officers) else np.nan

# ─── Summary (PowerPoint layout) ─────────────────────────────────────────────
st.markdown("---")
st.header("Summary")
_m1, _m2, _m3 = st.columns(3)
_m1.metric("Total CFS", f"{int(total_cfs):,}")
_m2.metric(
    "Total ALPR + Audio Hits",
    f"{int(total_alpr_audio) if 'total_alpr_audio' in locals() else int((alpr_hits if 'alpr_hits' in locals() else 0) + (audio_hits if 'audio_hits' in locals() else 0)):,}"
)
_m3.metric("DFR Responses to CFS within range", f"{int(in_count):,}")

_m4, _m5, _m6 = st.columns(3)
_m4.metric("DFR Responses to ALPR and Audio within range", f"{int(dfr_alpr_audio):,}")
_m5.metric("Avg Response Time", pretty_value(avg_drone, "mmss"))
_m6.metric("First On Scene %", pretty_value(first_on_pct, "pct"))

_m7, _m8, _m9 = st.columns(3)
_m7.metric("Expected CFS and ALPR + Audio Calls Cleared", f"{int(exp_cleared):,}")
_m8.metric("Force Multiplication", pretty_value(officers, "2dec"))
_m9.metric("ROI", pretty_value(roi, "usd"))

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
    ("Total time spent on Clearable CFS",                   clr_count * avg_clr_effective, "hhmmss"),

    ("Avg Time on Scene – Clearable Calls",                 (avg_clr if not _use_clr_fallback else avg_clr_effective),         "mmss"),
]


# --- Append Comparison & Full-City Comparison metrics to report values ---
# NOTE: we now DEFER building the report table until AFTER the comparison panels
# have populated their metrics into st.session_state. We stash the base `rows`
# and provide a helper to build the DataFrame on demand.

def _safe_pct_fewer(our, comp):
    try:
        our = float(our); comp = float(comp)
        if comp > 0:
            return (1.0 - (our / comp)) * 100.0
    except Exception:
        pass
    return np.nan

def _safe_pct_lower(our_cost, comp_cost):
    try:
        our_cost = float(our_cost); comp_cost = float(comp_cost)
        if comp_cost > 0:
            return (1.0 - (our_cost / comp_cost)) * 100.0
    except Exception:
        pass
    return np.nan

def _as_int_or_none(v):
    try:
        return int(v)
    except Exception:
        return None

def _as_float_or_none(v):
    try:
        return float(v)
    except Exception:
        return None

def _append_cmp_rows(base_rows: list) -> list:
    """Read comparison metrics from session_state and append to a copy of base_rows."""
    # Pull metrics that the comparison panels stash in session_state
    cmp_our_locs      = _as_int_or_none(st.session_state.get("cmp_our_locs"))
    cmp_our_docks     = _as_int_or_none(st.session_state.get("cmp_our_docks"))
    cmp_our_cost_200  = _as_float_or_none(st.session_state.get("cmp_our_cost_200"))
    cmp_our_cost_400  = _as_float_or_none(st.session_state.get("cmp_our_cost_400"))
    cmp_comp_locs     = _as_int_or_none(st.session_state.get("cmp_comp_locs"))
    cmp_comp_docks    = _as_int_or_none(st.session_state.get("cmp_comp_docks"))
    cmp_comp_cost_200 = _as_float_or_none(st.session_state.get("cmp_comp_cost_200"))
    cmp_comp_cost_400 = _as_float_or_none(st.session_state.get("cmp_comp_cost_400"))

    fc_our_locs      = _as_int_or_none(st.session_state.get("fc_our_locs"))
    fc_our_docks     = _as_int_or_none(st.session_state.get("fc_our_docks"))
    fc_our_cost_200  = _as_float_or_none(st.session_state.get("fc_our_cost_200"))
    fc_our_cost_400  = _as_float_or_none(st.session_state.get("fc_our_cost_400"))
    fc_comp_locs     = _as_int_or_none(st.session_state.get("fc_comp_locs"))
    fc_comp_docks    = _as_int_or_none(st.session_state.get("fc_comp_docks"))
    fc_comp_cost_200 = _as_float_or_none(st.session_state.get("fc_comp_cost_200"))
    fc_comp_cost_400 = _as_float_or_none(st.session_state.get("fc_comp_cost_400"))

    # Derived % metrics
    cmp_pct_fewer_locs = _safe_pct_fewer(cmp_our_locs, cmp_comp_locs)
    cmp_pct_lower_cost = _safe_pct_lower(cmp_our_cost_200, cmp_comp_cost_200)
    fc_pct_fewer_locs  = _safe_pct_fewer(fc_our_locs,  fc_comp_locs)
    fc_pct_lower_cost  = _safe_pct_lower(fc_our_cost_200, fc_comp_cost_200)

    out = list(base_rows)  # copy
    out.extend([
        # --- Regular Comparison ---
        ("% Fewer Launch Locations Required in comparison",            cmp_pct_fewer_locs,   "pct"),
        ("Lower cost in comparison (@200 ft)",                        cmp_pct_lower_cost,   "pct"),
        ("# of launch locations in comparison (Aerodome)",            cmp_our_locs,         "int"),
        ("# of total docks in comparison (Aerodome)",                 cmp_our_docks,        "int"),
        ("Yearly cost @200 ft (Aerodome, comparison)",                cmp_our_cost_200,     "usd"),
        ("Yearly cost @400 ft (Aerodome, comparison)",                cmp_our_cost_400,     "usd"),
        ("# of launch locations in comparison (Competitor)",          cmp_comp_locs,        "int"),
        ("# of total docks in comparison (Competitor)",               cmp_comp_docks,       "int"),
        ("Yearly cost @200 ft (Competitor, comparison)",              cmp_comp_cost_200,    "usd"),
        ("Yearly cost @400 ft (Competitor, comparison)",              cmp_comp_cost_400,    "usd"),

        # --- Full City Comparison ---
        ("% Fewer Launch Locations Required in full city comparison", fc_pct_fewer_locs,    "pct"),
        ("Lower cost in full city comparison (@200 ft)",              fc_pct_lower_cost,    "pct"),
        ("# of launch locations in full city comparison (Aerodome)",  fc_our_locs,          "int"),
        ("# of total docks in full city comparison (Aerodome)",       fc_our_docks,         "int"),
        ("Yearly cost @200 ft (Aerodome, full city)",                 fc_our_cost_200,      "usd"),
        ("Yearly cost @400 ft (Aerodome, full city)",                 fc_our_cost_400,      "usd"),
        ("# of launch locations in full city comparison (Competitor)",fc_comp_locs,         "int"),
        ("# of total docks in full city comparison (Competitor)",     fc_comp_docks,        "int"),
        ("Yearly cost @200 ft (Competitor, full city)",               fc_comp_cost_200,     "usd"),
        ("Yearly cost @400 ft (Competitor, full city)",               fc_comp_cost_400,     "usd"),
    ])
    return out

def _build_report_df_from_rows(base_rows: list) -> pd.DataFrame:
    """Build the final report DataFrame using the current session_state for comparison values."""
    _rows_final = _append_cmp_rows(base_rows)
    return pd.DataFrame({
        "Metric": [r[0] for r in _rows_final],
        "Result": [pretty_value(r[1], r[2]) for r in _rows_final],
    })

# Defer building: stash the base rows now; we'll build at the end of the app.
st.session_state["__report_rows_base"] = list(rows)


# =================== PATCH: PERSIST COMPARISON/FULL CITY PANEL METRICS ===================

# --- REGULAR COMPARISON PANEL (Aerodome LEFT) ---
# After rendering Aerodome metrics, persist cmp_our_locs, cmp_our_docks, cmp_our_cost_200, cmp_our_cost_400
# (Insert this block immediately after Aerodome metrics are calculated/rendered in the regular comparison panel)
# Example variable names: required_locs, total_our_docks, our_cost_200, our_cost_400
def persist_cmp_our_metrics(cmp_our_locs, cmp_our_docks, cmp_our_cost_200, cmp_our_cost_400):
    st.session_state["cmp_our_locs"] = cmp_our_locs
    st.session_state["cmp_our_docks"] = cmp_our_docks
    st.session_state["cmp_our_cost_200"] = cmp_our_cost_200
    st.session_state["cmp_our_cost_400"] = cmp_our_cost_400

# --- REGULAR COMPARISON PANEL (Competitor RIGHT) ---
# After rendering competitor metrics, persist cmp_comp_locs, cmp_comp_docks, cmp_comp_cost_200, cmp_comp_cost_400
def persist_cmp_comp_metrics(cmp_comp_locs, cmp_comp_docks, cmp_comp_cost_200, cmp_comp_cost_400):
    st.session_state["cmp_comp_locs"] = cmp_comp_locs
    st.session_state["cmp_comp_docks"] = cmp_comp_docks
    st.session_state["cmp_comp_cost_200"] = cmp_comp_cost_200
    st.session_state["cmp_comp_cost_400"] = cmp_comp_cost_400

# --- FULL CITY COMPARISON PANEL (Aerodome LEFT) ---
# After rendering metrics, persist fc_our_locs, fc_our_docks, fc_our_cost_200, fc_our_cost_400
def persist_fc_our_metrics(fc_our_locs, fc_our_docks, fc_our_cost_200, fc_our_cost_400):
    st.session_state["fc_our_locs"] = fc_our_locs
    st.session_state["fc_our_docks"] = fc_our_docks
    st.session_state["fc_our_cost_200"] = fc_our_cost_200
    st.session_state["fc_our_cost_400"] = fc_our_cost_400

# --- FULL CITY COMPARISON PANEL (Competitor RIGHT) ---
# After rendering metrics, persist fc_comp_locs, fc_comp_docks, fc_comp_cost_200, fc_comp_cost_400
def persist_fc_comp_metrics(fc_comp_locs, fc_comp_docks, fc_comp_cost_200, fc_comp_cost_400):
    st.session_state["fc_comp_locs"] = fc_comp_locs
    st.session_state["fc_comp_docks"] = fc_comp_docks
    st.session_state["fc_comp_cost_200"] = fc_comp_cost_200
    st.session_state["fc_comp_cost_400"] = fc_comp_cost_400

# =================== END PATCH INSERTS ===================


# =================== PATCH: FAA OVERLAY IN COMPETITOR PANELS ===================
# FAA overlay is now handled directly at each map creation site using the
# existing `geojson_overlays=[("FAA Grid", FAA_GEOJSON)]` pattern.
# This placeholder intentionally contains no executable code so that
# there are no out-of-scope references (e.g., `m`).
# =================== END PATCH: FAA OVERLAY IN COMPETITOR PANELS ===================

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
        # Persist effective values so saved runs reflect the assumed 55-min fallback when timestamps are missing
        "avg_clearable_scene_sec": float(avg_clr_effective) if np.isfinite(avg_clr_effective) else None,
        "total_time_on_clearable_sec": float(clr_count * avg_clr_effective) if np.isfinite(avg_clr_effective) else None,
        "alpr_clear_rate": float(get_alpr_clear_rate()),
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
            "alpr_clear_rate": float(get_alpr_clear_rate()),
            "drone_speed_mph": float(st.session_state.get("selected_speed_mph", drone_speed)),
            "drone_range_miles": float(st.session_state.get("selected_range_mi", drone_range)),
            "price_per_dock_usd": int(st.session_state.get("selected_price_per_dock", 0) or 0),
            "docks_per_location": int(st.session_state.get("selected_docks_per_location", 1) or 1),
            "apply_price_override": bool(st.session_state.get("apply_price_override", False)),
            "selected_platform": st.session_state.get("selected_platform", ""),
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


# --- Build & render the report table AFTER comparison panels populated state ---
try:
    if "__report_rows_base" in st.session_state:
        report_df = _build_report_df_from_rows(st.session_state["__report_rows_base"])
        # If some other part of the app renders report_df, just keep it in scope.
        # Otherwise, render a default view here.
        if "rendered_report_table" not in st.session_state:
            st.markdown("### Report Values")
            st.dataframe(report_df, use_container_width=True)
            st.session_state["rendered_report_table"] = True
except Exception as _e:
    st.sidebar.warning(f"Report table build failed: {_e}")

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
        _rng = float(st.session_state.get("selected_range_mi", drone_range))
        _inrng = (_dist <= _rng) & np.isfinite(_dist)
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

# ─── 5.5) TOP SUMMARY (matches PDF headline metrics) ─────────────────────────
## Removed old summary section (replaced by new Summary above)

# ==== FAA overlay data (load once per run) ===================================
_lats = list(df_all["lat"].dropna().astype(float).values)
_lons = list(df_all["lon"].dropna().astype(float).values)
for la, lo in launch_coords:
    _lats.append(float(la))
    _lons.append(float(lo))

FAA_GEOJSON = None
faa_bbox = _data_bbox(_lats, _lons, pad=0.05)
try:
    FAA_GEOJSON = load_esri_geojson(FAA_LAYER_URL, bbox=faa_bbox)
except Exception as _e:
    FAA_GEOJSON = None
    st.sidebar.warning(f"FAA layer failed to load: {_e}")

# --- FAA debug/stats -------------------------------------------------------
try:
    if FAA_GEOJSON and isinstance(FAA_GEOJSON, dict):
        feats = FAA_GEOJSON.get("features") or []
        feat_ct = len(feats)
        ceilings = []
        for f in feats[:5000]:  # cap to avoid huge UI payloads
            try:
                ceilings.append(f.get("properties", {}).get("CEILING"))
            except Exception:
                continue
        uniq_ceil = sorted({c for c in ceilings if c is not None})
        st.session_state["__faa_stats__"] = {
            "bbox": faa_bbox,
            "feature_count": int(feat_ct),
            "unique_ceilings": uniq_ceil[:20],  # show first 20 unique values
        }
    else:
        st.session_state["__faa_stats__"] = {"bbox": faa_bbox, "feature_count": 0, "unique_ceilings": []}
except Exception as _e:
    st.session_state["__faa_stats__"] = {"error": str(_e)}

with st.sidebar.expander("FAA debug", expanded=False):
    stats = st.session_state.get("__faa_stats__", {})
    st.write(stats)

# --- GeoJSON normalizer for Folium ------------------------------------------
def _as_feature_collection(obj):
    """
    Accepts: dict, str (JSON), single Feature/geometry, or list of Features/geoms.
    Returns: FeatureCollection dict or None if it can't be parsed.
    """
    import json

    if obj is None:
        return None

    # If string, try to JSON-decode
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except Exception:
            return None

    # If it's already a dict
    if isinstance(obj, dict):
        t = obj.get("type")
        # Proper FeatureCollection
        if t == "FeatureCollection" and isinstance(obj.get("features"), list):
            return obj
        # Single Feature
        if t == "Feature":
            return {"type": "FeatureCollection", "features": [obj]}
        # Single geometry
        if t in {"Polygon", "MultiPolygon", "LineString", "MultiLineString", "Point", "MultiPoint"}:
            return {
                "type": "FeatureCollection",
                "features": [{"type": "Feature", "properties": {}, "geometry": obj}],
            }
        # Some objects come as {"features": [...] } without explicit type
        if "features" in obj and isinstance(obj["features"], list):
            return {"type": "FeatureCollection", "features": obj["features"]}

        # Not a recognized structure
        return None

    # If it's a list (e.g., list of features or geometries)
    if isinstance(obj, list):
        feats = []
        for g in obj:
            if isinstance(g, str):
                try:
                    import json
                    g = json.loads(g)
                except Exception:
                    continue
            if not isinstance(g, dict):
                continue
            gt = g.get("type")
            if gt == "Feature":
                feats.append(g)
            elif gt in {"Polygon", "MultiPolygon", "LineString", "MultiLineString", "Point", "MultiPoint"}:
                feats.append({"type": "Feature", "properties": {}, "geometry": g})
        if feats:
            return {"type": "FeatureCollection", "features": feats}

    return None

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
    title=None,                # ← default None (not empty string)
    key=None,
    heat_radius=15,
    heat_blur=25,
    show_circle=False,
    launch_coords=None,
    hotspot_center=None,
    hotspot_radius=None,
):
    # Only render a title if one was actually provided
    if title:
        st.subheader(title)

    # drop NaNs so folium never errors
    if {"lat","lon"}.issubset(df_pts.columns):
        df_pts = df_pts.dropna(subset=["lat","lon"]).copy()
        # guard against bogus coords (e.g., 0,0 or out-of-range)
        df_pts = df_pts[
            df_pts["lat"].between(-90, 90)
            & df_pts["lon"].between(-180, 180)
            & ~((df_pts["lat"] == 0) & (df_pts["lon"] == 0))
        ]

    # determine map center
    if show_circle and launch_coords:
        # guard launch_coords too
        _lc = [
            (float(la), float(lo))
            for la, lo in (launch_coords or [])
            if np.isfinite(la) and np.isfinite(lo)
            and -90 <= float(la) <= 90 and -180 <= float(lo) <= 180
            and not (float(la) == 0 and float(lo) == 0)
        ]
        launch_coords = _lc
        center = [float(launch_coords[0][0]), float(launch_coords[0][1])] if launch_coords else [0.0, 0.0]
    elif not df_pts.empty:
        center = [float(df_pts["lat"].mean()), float(df_pts["lon"].mean())]
    else:
        center = [0.0, 0.0]

    m = folium.Map(location=center, zoom_start=10)

    # Persist base zoom for competitor maps
    try:
        if "base_zoom" not in st.session_state:
            z = None
            try:
                z = m.options.get("zoom")
            except Exception:
                pass
            st.session_state["base_zoom"] = z if z is not None else 10
    except Exception:
        pass

    # FAA grid (global toggle)
    try:
        if st.session_state.get("show_faa") and FAA_GEOJSON:
            from folium.features import GeoJsonTooltip
            def _faa_style(feat):
                # lightweight styling; same color, transparent fill
                return {"color": "#3388ff", "weight": 1, "fillColor": "#3388ff", "fillOpacity": 0.18}
            folium.GeoJson(
                FAA_GEOJSON,
                name="FAA Grid",
                style_function=_faa_style,
                tooltip=GeoJsonTooltip(fields=["CEILING"], aliases=["Ceiling (ft)"])
            ).add_to(m)
    except Exception:
        pass

    # blue drone-range circle(s)
    if show_circle and launch_coords:
        for la, lo in launch_coords:
            folium.Circle(
                location=(la, lo),
                radius=float(st.session_state.get("selected_range_mi", drone_range)) * 1609.34,
                color="blue",
                fill=False
            ).add_to(m)

    # red hotspot circle
    if hotspot_center and hotspot_radius:
        folium.Circle(
            location=hotspot_center,
            radius=hotspot_radius * 1609.34,
            color="red",
            weight=3,
            fill=False
        ).add_to(m)

    # heat vs points
    if heat and not df_pts.empty:
        data = (
            df_pts[["lat","lon","count"]].values.tolist()
            if "count" in df_pts.columns
            else df_pts[["lat","lon"]].values.tolist()
        )
        HeatMap(data, radius=heat_radius, blur=heat_blur).add_to(m)
    else:
        for _, r in df_pts.iterrows():
            folium.CircleMarker(
                location=(float(r["lat"]), float(r["lon"])),
                radius=3,
                color="red",
                fill=True,
                fill_opacity=0.6
            ).add_to(m)

    # Layer control only if FAA added (harmless if not)
    try:
        if st.session_state.get("show_faa") and FAA_GEOJSON:
            folium.LayerControl(collapsed=True).add_to(m)
    except Exception:
        pass

    st_folium(m, width=800, height=500, key=key)

# Maps — All DFR Calls
st.markdown("---")
st.header("Maps")
st.subheader("All DFR Calls")
r0, b0 = auto_heat_params(all_dfr)
r_all = st.sidebar.slider("All DFR Calls Heat Radius", 1, 50, value=r0, key="all_r")
b_all = st.sidebar.slider("All DFR Calls Heat Blur",   1, 50, value=b0, key="all_b")
render_map(
    all_dfr,
    heat=True,
    heat_radius=r_all, heat_blur=b_all,
    title=None,
    key="map_all_heat",
    show_circle=True,
    launch_coords=launch_coords
)
metrics_under(
    "",
    ("Total CFS", total_cfs, "int"),
    ("Total DFR CFS", total_dfr, "int"),
    ("Total DFR CFS within Range", in_count, "int"),
    ("Total DFR ALPR Hits within Range", (alpr_hits if 'alpr_hits' in locals() else 0), "int"),
    ("Total DFR CFS (again)", total_dfr, "int"),
    ("Avg Dispatch + Response Time for Patrol (DFR)", avg_patrol, "mmss"),
    ("Avg Time on Scene DFR Calls", avg_scene, "mmss"),
    ("Number of Launch Locations", len(launch_rows), "int"),
    ("Number of Docks", int(pd.to_numeric(launch_rows.get("Number of Docks", 0), errors="coerce").fillna(0).sum()), "int"),
    ("Number of Radar", int(pd.to_numeric(launch_rows.get("Number of Radar", 0), errors="coerce").fillna(0).sum()), "int"),
)

st.markdown("---")

# 6b) Flock Aerodome DFR Coverage Area (circle only)
st.subheader("Flock Aerodome DFR Coverage Area")
render_map(
    pd.DataFrame(),
    heat=False,
    title=None,
    key="map_range_circle",
    show_circle=True,
    launch_coords=launch_coords
)
# no stats row under coverage circle per layout

st.markdown("---")

st.subheader("All DFR Calls within Range (zoom into map)")
# Render in-range map (heat)
r_ir, b_ir = auto_heat_params(in_range)
r_ir = st.sidebar.slider("In-Range Heat Radius", 1, 50, value=r_ir, key="inrng_r")
b_ir = st.sidebar.slider("In-Range Heat Blur",   1, 50, value=b_ir, key="inrng_b")
render_map(
    in_range,
    heat=True,
    heat_radius=r_ir, heat_blur=b_ir,
    title=None,
    key="map_inrange_heat",
    show_circle=True,
    launch_coords=launch_coords
)
metrics_under(
    "",
    ("DFR Avg Response Time", avg_drone, "mmss"),
    ("Expected First On Scene %", first_on_pct, "pct"),
    ("Expected Reduction in Response Times", pct_dec, "pct"),
    ("Number of DFR CFS within Range", in_count, "int"),
    ("Avg Dispatch + Response Time (in-range)", avg_in, "mmss"),
    ("Avg Expected DFR Response Time (in-range)", avg_drone, "mmss"),
)

# 6c) Heatmap: P1 DFR Calls
st.subheader("P1 Calls for Service")
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
    "",
    ("Number of P1 Calls in Range", p1_count, "int"),
    ("Avg Dispatch + Response Time to P1 Calls", avg_p1_pat, "mmss"),
    ("Avg Expected DFR Response Time to P1 Calls", avg_p1_drone, "mmss"),
)

st.markdown("---")

# 6d) Heatmap: All DFR Calls + Hotspot Overlay
st.subheader("Hotspots")
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
        "",
        ("Number of DFR Calls in Hotspot", hotspot_count, "int"),
        ("Avg Dispatch + Response Time (hotspot)", hotspot_avg_patrol, "mmss"),
        ("Avg Expected DFR Response Time (hotspot)", hotspot_avg_drone, "mmss"),
    )
    st.markdown("---")

## Removed headline metrics rows for ALPR/Audio — maps below already carry metrics elsewhere

# 6g) Clearable Calls for Service
## Clearable metrics moved below ALPR/Audio and heatmap re-added there

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
        title="ALPR Heat Map",
        key="map_alpr_heat",
        show_circle=True,
        launch_coords=launch_coords
    )
    metrics_under(
        "",
        ("# of ALPR Locations in Range", (alpr_sites if 'alpr_sites' in locals() else 0), "int"),
        ("# of hits", (alpr_hits if 'alpr_hits' in locals() else 0), "int"),
        ("Average Expected Drone Response Time", (alpr_eta if 'alpr_eta' in locals() else np.nan), "mmss"),
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
        title="Audio Heatmap",
        key="map_audio_heat",
        show_circle=True,
        launch_coords=launch_coords
    )
    metrics_under(
        "",
        ("# of Audio Locations in Range", (audio_sites if 'audio_sites' in locals() else 0), "int"),
        ("# of hits", (audio_hits if 'audio_hits' in locals() else 0), "int"),
        ("Average Expected Drone Response Time", (audio_eta if 'audio_eta' in locals() else np.nan), "mmss"),
    )
else:
    st.sidebar.info("No audio points to display on the heatmap.")

st.markdown("---")
st.subheader("Clearable Calls for Service")
r2, b2 = auto_heat_params(all_clearable)
r_cl = st.sidebar.slider("Clearable Heat Radius", 1, 50, value=r2, key="clr_r")
b_cl = st.sidebar.slider("Clearable Heat Blur",   1, 50, value=b2, key="clr_b")
render_map(
    all_clearable,
    heat=True,
    heat_radius=r_cl, heat_blur=b_cl,
    title=None,
    key="map_clearable_heat",
    show_circle=True,
    launch_coords=launch_coords
)
metrics_under(
    "",
    ("Expected Number of CFS and Alerts Cleared", exp_cleared, "int"),
    ("Number of Officers (Force Multiplication)", officers, "2dec"),
    ("ROI", roi, "usd"),
    ("Full Time Equivalent Cost Used", officer_cost, "usd"),
    ("Call Clearance Rate Used", cancel_rate * 100.0, "pct"),
    ("ALPR Clearance Rate Used", alpr_rate * 100.0, "pct"),
    ("Total Clearable CFS by Drone within Range", clr_count, "int"),
    ("Total Time Spent on Clearable CFS", clr_count * avg_clr_effective, "hhmmss"),
)

# ─── 7) PRICING ───────────────────────────────────────────────────────────────
if False:  # Pricing section hidden (redundant in new layout)
    st.markdown("---")
    st.header("Pricing")

# --- Yearly unit prices (no CapEx) ---
PRICE_PER_DOCK_BY_TYPE = {
    "DOCK 3": 50000,
    "ALPHA": 125000,
    "DELTA": 300000,
}
PRICE_PER_RADAR = 150000  # yearly

def circle_area_sqmi(r_mi: float) -> float:
    import math
    try:
        r = float(r_mi)
    except Exception:
        r = 0.0
    return math.pi * (r ** 2)

def competitor_radar_params(name: str):
    """
    Returns (enabled: bool, radius_mi: float, yearly_price_usd: float) for a competitor,
    reading from PLATFORMS[name]["specs"]. Falls back to disabled/no-cost.
    """
    spec = (PLATFORMS.get(name, {}) or {}).get("specs", {}) or {}
    enabled = str(spec.get("Radar Enabled", "No")).strip().lower() == "yes"
    try:
        radius = float(spec.get("Radar Radius (mi)", 0))
    except Exception:
        radius = 0.0
    try:
        price = float(spec.get("Radar Yearly Price (USD)", 0))
    except Exception:
        price = 0.0
    return enabled, radius, price

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
# Apply global selections only when override is enabled; else use sheet as-is
try:
    if st.session_state.get("override_launch_pricing"):
        sel_price = int(st.session_state.get("selected_price_per_dock", 0) or 0)
        if sel_price > 0:
            _lr["Dock Unit Price"] = int(sel_price)
        sel_docks = int(st.session_state.get("selected_docks_per_location", 1) or 1)
        if sel_docks > 0:
            _lr["Number of Docks"] = sel_docks
except Exception:
    pass
_lr["Dock Yearly Cost"]  = (_lr["Number of Docks"] * _lr["Dock Unit Price"]).astype(int)
_lr["Radar Yearly Cost"] = (_lr["Number of Radar"] * PRICE_PER_RADAR).astype(int)
_lr["Site Yearly Total"] = (_lr["Dock Yearly Cost"] + _lr["Radar Yearly Cost"]).astype(int)

# Totals
total_launch_sites = len(_lr)
total_docks  = int(_lr["Number of Docks"].sum())
total_radars = int(_lr["Number of Radar"].sum())

# 200 ft = NO radar; 400 ft = WITH radar
list_total_200 = int(_lr["Dock Yearly Cost"].sum())
list_total_400 = int((_lr["Dock Yearly Cost"] + _lr["Radar Yearly Cost"]).sum())

# Discount input (sidebar) — apply to *total* shown (simple, same as before)
st.sidebar.header("Pricing Options")
discount_pct = st.sidebar.number_input("Discount (%)", min_value=0.0, max_value=100.0, value=0.0, step=1.0, key="pricing_discount_pct")
discount_rate = float(discount_pct) / 100.0

disc_200 = int(round(list_total_200 * discount_rate))
disc_400 = int(round(list_total_400 * discount_rate))
discounted_200 = int(list_total_200 - disc_200)
discounted_400 = int(list_total_400 - disc_400)

# Recommended dock type(s): based on CSV values present
present_types = [t for t in _lr["Dock Type"].unique().tolist() if t]
if st.session_state.get("override_launch_pricing"):
    # When overriding, recommend the selected platform
    recommended_label = st.session_state.get("selected_platform", present_types[0] if present_types else "—")
else:
    if len(present_types) > 1:
        top_type = (
            _lr.groupby("Dock Type")["Number of Docks"].sum().sort_values(ascending=False).index.tolist() or [""]
        )[0]
        recommended_label = f"{', '.join(present_types)}  (top: {top_type})"
    else:
        recommended_label = present_types[0] if present_types else "—"

# Pricing visuals removed per new layout; calculations retained for downstream use
def _fmt_usd(x): return f"${x:,.0f}"

# ─── 7) COMPARISON (no GeoPandas; city limits from call data) ────────────────
st.markdown("---")
st.header("Comparison")

# Top-of-comparison headline metrics (before maps & competitor selector)
_cmp_m1, _cmp_m2 = st.columns(2)
try:
    # Use session_state values if present; else compute from current cached values
    cmp_our_locs  = int(st.session_state.get("cmp_our_locs")) if st.session_state.get("cmp_our_locs") is not None else None
    cmp_comp_locs = int(st.session_state.get("cmp_comp_locs")) if st.session_state.get("cmp_comp_locs") is not None else None
    cmp_our_cost  = float(st.session_state.get("cmp_our_cost_200")) if st.session_state.get("cmp_our_cost_200") is not None else None
    cmp_comp_cost = float(st.session_state.get("cmp_comp_cost_200")) if st.session_state.get("cmp_comp_cost_200") is not None else None
    pct_fewer = (1.0 - (cmp_our_locs / cmp_comp_locs)) * 100.0 if (cmp_our_locs and cmp_comp_locs and cmp_comp_locs>0) else np.nan
    pct_lower = (1.0 - (cmp_our_cost / cmp_comp_cost)) * 100.0 if (cmp_our_cost and cmp_comp_cost and cmp_comp_cost>0) else np.nan
except Exception:
    pct_fewer, pct_lower = np.nan, np.nan
_cmp_m1.metric("% Fewer Launch Locations Required", pretty_value(pct_fewer, "pct"))
_cmp_m2.metric("% Lower Cost", pretty_value(pct_lower, "pct"))

st.markdown(
    """
    <style>
    /* Make long spec lines wrap consistently in columns */
    .block-container p, .block-container li {
        white-space: normal !important;
        word-break: break-word !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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
import re
# ---------- UTM helpers ----------
def normalize_address(addr: str) -> str:
    """
    Basic address normalizer for geocoding:
    - trims whitespace
    - collapses internal runs of spaces
    - drops trailing ZIP (e.g., ', 12345' or ', 12345-6789')
    - expands a few common abbreviations to help some geocoders
    """
    if not isinstance(addr, str):
        return ""
    s = " ".join(addr.strip().split())
    # Remove trailing zip code (with optional 4-digit extension)
    s = re.sub(r",?\s*\d{5}(-\d{4})?$", "", s)
    # Expand common street abbreviations (simple, non-locale-specific)
    pad = f" {s} "
    replacements = {
        " Rd ": " Road ",
        " St ": " Street ",
        " Ave ": " Avenue ",
        " Hwy ": " Highway ",
        " Blvd ": " Boulevard ",
        " Dr ": " Drive ",
        " Ln ": " Lane ",
        " Pkwy ": " Parkway ",
        " Ct ": " Court ",
        " Cir ": " Circle ",
    }
    for k, v in replacements.items():
        pad = pad.replace(k, v)
    s = pad.strip()
    # Remove any trailing commas introduced by trimming
    s = s.strip(", ")
    return s

SQM_PER_SQMI = 2_589_988.110336

def _extract_full_city_from_any_session_zip():
    """
    Look through st.session_state for any ZIP bytes and try to load
    a CSV whose filename starts with 'Launch Locations - Full City'.
    Returns a DataFrame or None.
    """
    prefix = "launch locations - full city"
    try:
        import streamlit as st  # already imported, but safe
    except Exception:
        return None

    for k, v in list(st.session_state.items()):
        data = None
        # Common patterns: raw bytes, UploadedFile-like, or stored zip bytes
        if isinstance(v, (bytes, bytearray)):
            data = bytes(v)
        elif hasattr(v, "getvalue"):
            try:
                data = v.getvalue()
            except Exception:
                data = None

        if not data:
            continue

        # Try to open as a ZIP and find our CSV by prefix
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                name = next(
                    (
                        n for n in zf.namelist()
                        if n.lower().endswith(".csv")
                        and n.split("/")[-1].lower().startswith(prefix)
                    ),
                    None
                )
                if name:
                    with zf.open(name) as fh:
                        # Reuse your existing CSV loader so columns/cleanups match
                        df, _, _ = _load_launch_locations_csv(fh)
                        return df
        except Exception:
            # Not a ZIP or not the right file; skip
            continue

    return None

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

def _circle_union_area_sqmi_inside(polygon_utm, centers_latlon, radius_mi, fwd: Transformer) -> float:
    if polygon_utm is None or not centers_latlon:
        return 0.0
    r_m = float(radius_mi) * 1609.34
    lats = np.array([la for la, _ in centers_latlon], dtype=float)
    lons = np.array([lo for _, lo in centers_latlon], dtype=float)
    xs, ys = _project_points(fwd, lats, lons)
    circles = [Point(float(x), float(y)).buffer(r_m) for x, y in zip(xs, ys)]
    u = None
    for c in circles:
        u = c if u is None else u.union(c)
    if u is None or u.is_empty:
        return 0.0
    covered = u.intersection(polygon_utm)
    if covered.is_empty:
        return 0.0
    return float(covered.area / SQM_PER_SQMI)

def competitor_plan_constrained(comp_name: str, polygon_utm, fwd: Transformer, inv: Transformer,
                                seed_pts_lat, seed_pts_lon, target_area_sqmi: float, max_iter: int = 8):
    cfg = PLATFORMS[comp_name]
    per_loc_area = math.pi * (cfg["range_mi"] ** 2)
    EFF = 0.88  # packing efficiency guess
    guess = max(1, int(math.ceil((target_area_sqmi / per_loc_area) / EFF)))

    lat = pd.to_numeric(pd.Series(seed_pts_lat), errors="coerce")
    lon = pd.to_numeric(pd.Series(seed_pts_lon), errors="coerce")
    mask = lat.notna() & lon.notna()
    if not mask.any() or polygon_utm is None:
        n = guess
        return {
            "locations": n,
            "yearly_cost": n * cfg["docks_per_location"] * cfg["price_per_dock"],
            "per_location_area_sqmi": per_loc_area,
            "docks_per_location": cfg["docks_per_location"],
            "range_mi": cfg["range_mi"],
        }

    X, Y = _project_points(fwd, lat[mask].values, lon[mask].values)
    pts_in = [(x, y) for x, y in zip(X, Y) if polygon_utm.contains(Point(float(x), float(y)))]
    if not pts_in:
        n = guess
        return {
            "locations": n,
            "yearly_cost": n * cfg["docks_per_location"] * cfg["price_per_dock"],
            "per_location_area_sqmi": per_loc_area,
            "docks_per_location": cfg["docks_per_location"],
            "range_mi": cfg["range_mi"],
        }
    XY = np.asarray(pts_in, dtype=float)

    n = guess
    goal = float(target_area_sqmi)
    for _ in range(max_iter):
        n = max(1, n)
        km = KMeans(n_clusters=min(n, len(XY)), n_init=10, random_state=42).fit(XY)
        cx, cy = km.cluster_centers_[:, 0], km.cluster_centers_[:, 1]
        lat_c, lon_c = _unproject_points(inv, cx, cy)
        covered = _circle_union_area_sqmi_inside(polygon_utm, list(zip(lat_c, lon_c)), cfg["range_mi"], fwd)

        if covered >= goal and n > 1:
            n -= 1
            continue
        if covered < goal:
            n += 1
            if n > guess * 3:
                break
            continue
        break

    n = max(1, n)
    yearly_cost = n * cfg["docks_per_location"] * cfg["price_per_dock"]
    return {
        "locations": n,
        "yearly_cost": yearly_cost,
        "per_location_area_sqmi": per_loc_area,
        "docks_per_location": cfg["docks_per_location"],
        "range_mi": cfg["range_mi"],
    }

# ---------- Outline conversion for folium ----------

def polygon_outline_latlon(poly_utm, inv: Transformer):
    if poly_utm is None or poly_utm.is_empty or inv is None:
        return []
    if poly_utm.geom_type == "MultiPolygon":
        poly_utm = max(poly_utm.geoms, key=lambda p: p.area)
    xs, ys = poly_utm.exterior.coords.xy
    lat, lon = _unproject_points(inv, np.asarray(xs), np.asarray(ys))
    return list(zip(lat.tolist(), lon.tolist()))

# Helper: overlay our Aerodome coverage on a competitor map
def overlay_our_coverage_on(m: folium.Map,
                            our_coords: list[tuple[float, float]],
                            range_mi: float,
                            outline_latlon: list[tuple[float, float]] | None = None):
    """
    Draw our Aerodome coverage (union outline if provided + light blue circles)
    onto an existing Folium map.
    """
    # Optional outline of our union polygon
    try:
        if outline_latlon:
            folium.PolyLine(
                locations=[(lt, ln) for lt, ln in outline_latlon],
                color="blue",
                weight=4,
                opacity=0.6
            ).add_to(m)
    except Exception:
        pass

    # Light blue circles for each of our launch sites
    try:
        r_m = float(range_mi) * 1609.34
    except Exception:
        r_m = 0.0
    if our_coords and r_m > 0:
        for la, lo in our_coords:
            try:
                folium.Circle(
                    location=(float(la), float(lo)),
                    radius=r_m,
                    color="blue",
                    weight=2,
                    fill=True,
                    fill_opacity=0.08,
                    opacity=0.6,
                ).add_to(m)
            except Exception:
                continue

# ---------- Unified target/placement logic for competitor sizing ----------
def compute_competition_target(city_area_sqmi: float,
                               our_sqmi: float,
                               calls_poly_utm,
                               our_poly_utm):
    """
    Returns (TARGET_AREA_SQMI, PLACEMENT_POLY_UTM, mode_str)

    Case A (city <= our): target = city area; place inside city polygon (calls_poly if present)
    Case B (city  > our): target = our area;   place inside our coverage polygon
    """
    city_area_sqmi = float(city_area_sqmi or 0.0)
    our_sqmi       = float(our_sqmi or 0.0)

    if city_area_sqmi > 0 and (our_sqmi <= 0 or city_area_sqmi <= our_sqmi):
        # Case A
        target_area = city_area_sqmi
        placement   = calls_poly_utm or our_poly_utm
        mode        = "CITY"
    else:
        # Case B (includes: city missing/zero → fall back to our coverage)
        target_area = our_sqmi if our_sqmi > 0 else city_area_sqmi
        placement   = our_poly_utm or calls_poly_utm
        mode        = "OUR"

    return float(target_area), placement, mode

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

# ---------- Effective range & area estimates ----------
detected_types_list = sorted(set(_normalized_dock_types()))
is_multi = len(detected_types_list) > 1
aerodome_title = f"Flock Aerodome — {'Multi-platform' if is_multi else detected_types_list[0].split('Flock Aerodome ',1)[-1]}"

our_eff_range = float(st.session_state.get("selected_range_mi", drone_range))
OUR_AREA_SQMI_EST = len(launch_coords) * math.pi * (our_eff_range ** 2)

# ---------- Polygons (for drawing/placement only, not math) ----------
calls_poly_utm, CALLS_AREA_SQMI, (fwd_calls, inv_calls) = calls_concave_hull_utm(
    df_all["lat"].values, df_all["lon"].values, eps_m=None, min_samples=20
)
our_poly_utm, OUR_CIRCLES_AREA_SQMI = union_launch_circles_utm(
    launch_coords, our_eff_range, fwd_calls
)

# ---------- City area comes from CSV (no guessing) ----------
city_area_csv = st.session_state.get("city_area_sqmi") or _meta_sqmi
if city_area_csv is not None:
    try:
        city_area_csv = float(city_area_csv)
    except Exception:
        city_area_csv = None

# Unified sizing/placement (applies to both Comparison & Full City logic)
TARGET_AREA_SQMI, PLACEMENT_POLY_UTM, _cmp_mode = compute_competition_target(
    city_area_sqmi=(city_area_csv if city_area_csv and city_area_csv > 0 else CALLS_AREA_SQMI),
    our_sqmi=OUR_CIRCLES_AREA_SQMI,
    calls_poly_utm=calls_poly_utm,
    our_poly_utm=our_poly_utm
)

# ---------------- Aerodome yearly pricing (no discount UI in Comparison) -----
RADAR_PRICE = 150000
DOCK_PRICES = {"Dock 3": 50000, "Alpha": 125000, "Delta": 300000, "M350": 150000}

def _dock_price_for_row(row):
    v = str(row.get(dock_type_col.name, "") if dock_type_col is not None else "").strip()
    v = v.replace("Flock Aerodome ", "")
    v = (v or "Dock 3").strip()
    return DOCK_PRICES.get(v, DOCK_PRICES["Dock 3"])

def compute_our_yearly_prices_no_discount():
    """Returns (our_200ft_total, our_400ft_total) with our radar price."""
    if launch_rows.empty:
        return 0, 0
    _rows = launch_rows.copy()
    _rows["_docks"]      = pd.to_numeric(docks_col, errors="coerce").fillna(0).astype(int) if docks_col  is not None else 0
    try:
        if st.session_state.get("override_launch_pricing"):
            sel_docks = int(st.session_state.get("selected_docks_per_location", 1) or 1)
            if sel_docks > 0:
                _rows["_docks"] = sel_docks
    except Exception:
        pass
    _rows["_radars"]     = pd.to_numeric(radars_col, errors="coerce").fillna(0).astype(int) if radars_col is not None else 0
    # Apply selected price per dock if provided; else map from type
    try:
        if st.session_state.get("override_launch_pricing"):
            sel_price = int(st.session_state.get("selected_price_per_dock", 0) or 0)
            if sel_price > 0:
                _rows["_dock_price"] = int(sel_price)
            else:
                _rows["_dock_price"] = _rows.apply(_dock_price_for_row, axis=1)
        else:
            _rows["_dock_price"] = _rows.apply(_dock_price_for_row, axis=1)
    except Exception:
        _rows["_dock_price"] = _rows.apply(_dock_price_for_row, axis=1)

    base_docks  = int((_rows["_docks"] * _rows["_dock_price"]).sum())
    base_radars = int(_rows["_radars"].sum() * PRICE_PER_RADAR)

    return int(base_docks), int(base_docks + base_radars)

our_base_200, our_base_400 = compute_our_yearly_prices_no_discount()

# --- Coverage-driven placement (hex grid) -----------------------------------
def place_sites_hex_in_polygon(polygon_utm, n_sites, range_mi, fwd, inv):
    """
    Place coverage circles using a hexagonal grid inside the polygon.
    - Builds a hex lattice of candidate centers spaced by ~1.7×R
    - Picks up to n_sites centers that fall inside polygon
    - Converts back to lat/lon
    """
    if polygon_utm is None or n_sites < 1 or fwd is None or inv is None:
        return []

    r_m = float(range_mi) * 1609.34
    spacing = r_m * 1.7   # tweak 1.6–1.9 to adjust overlap vs. gaps

    minx, miny, maxx, maxy = polygon_utm.bounds
    dx = spacing
    dy = spacing * math.sqrt(3) / 2.0

    candidates = []
    y = miny
    row = 0
    while y <= maxy:
        x = minx + (dx / 2.0 if row % 2 else 0)
        while x <= maxx:
            pt = Point(x, y)
            if polygon_utm.contains(pt):
                candidates.append((x, y))
            x += dx
        y += dy
        row += 1

    if not candidates:
        return []

    # Evenly sample across the candidate list so selection is spread
    if len(candidates) <= n_sites:
        chosen = candidates
    else:
        idxs = np.linspace(0, len(candidates) - 1, num=n_sites, dtype=int)
        chosen = [candidates[i] for i in idxs]

    xs = np.asarray([c[0] for c in chosen], dtype=float)
    ys = np.asarray([c[1] for c in chosen], dtype=float)
    lat_c, lon_c = _unproject_points(inv, xs, ys)
    return list(zip(lat_c.tolist(), lon_c.tolist()))

# ---------------- Competitor math (locations from TARGET_AREA_SQMI) ----------
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

# ---------- Place competitor centers inside polygon (KMeans on calls) ----------
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

# ---------------- Panel renderer (tidy: no duplicate specs, no extra separators) ----------
def panel(title, product_names_list, is_left=True, competitor=None):
    with st.container(border=True):
        if title:
            st.subheader(title)

        # compact, single-use specs renderer (LEFT column only)
        def _render_specs_block(pnames: list[str]):
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
            for pname in pnames:
                specs = PLATFORMS[pname]["specs"]
                st.markdown(f"**{pname}**")
                for r in rows:
                    if r in specs:
                        st.markdown(f"**{r}**: {specs[r]}")

        if is_left:
            # LEFT: our calls heat + blue coverage + FAA
            render_map(
                df_all,
                heat=True,
                heat_radius=8, heat_blur=12,
                key=f"cmp_map_L_{title}",
                show_circle=True,
                launch_coords=launch_coords,
            )

            # Headline metrics (ours)
            _our_locs  = len(launch_rows)
            _our_docks = int(pd.to_numeric(docks_col, errors="coerce").fillna(0).sum()) if docks_col is not None else 0
            try:
                if st.session_state.get("override_launch_pricing"):
                    sel_docks = int(st.session_state.get("selected_docks_per_location", 1) or 1)
                    if sel_docks > 0:
                        _our_docks = len(launch_rows) * sel_docks
            except Exception:
                pass
            c1, c2 = st.columns(2)
            c1.metric("Required Locations", f"{_our_locs:,}")
            c2.metric("Total Docks", f"{_our_docks:,}")

            c4, c5 = st.columns(2)
            c4.metric("Yearly Cost @200 ft", f"${our_base_200:,}")
            c5.metric("Yearly Cost @400 ft", f"${our_base_400:,}")

            # Persist report metrics: regular comparison (Aerodome)
            try:
                st.session_state["cmp_our_locs"] = int(_our_locs)
                st.session_state["cmp_our_docks"] = int(_our_docks)
                st.session_state["cmp_our_cost_200"] = int(our_base_200)
                st.session_state["cmp_our_cost_400"] = int(our_base_400)
            except Exception:
                pass

            # Specs — render ONCE on the left
            if is_multi:
                st.markdown("**Detected Aerodome Platforms:** " + ", ".join(detected_types_list))
                _render_specs_block(detected_types_list)
            elif product_names_list:
                _render_specs_block(product_names_list[:1])

        else:
            # RIGHT: competitor — use unified TARGET_AREA_SQMI + PLACEMENT_POLY_UTM
            plan = competitor_plan(competitor, TARGET_AREA_SQMI)
            required_locs_nom = int(plan["locations"])
            required_locs = int(math.ceil(required_locs_nom * (1.0 + overpack_pct/100.0)))
            # Guardrail: if rounding produced 0 but we had a non-zero nominal, use 1
            try:
                if required_locs < 1 and float(plan.get("locations", 0)) > 0:
                    required_locs = 1
            except Exception:
                if required_locs < 1:
                    required_locs = 1
            comp_range_mi = plan["range_mi"]

            fwd = fwd_calls if fwd_calls else _make_transformers(df_all["lat"].values, df_all["lon"].values)[0]
            inv = inv_calls if inv_calls else _make_transformers(df_all["lat"].values, df_all["lon"].values)[1]

            centers = place_sites_hex_in_polygon(
                PLACEMENT_POLY_UTM, n_sites=required_locs, range_mi=comp_range_mi,
                fwd=fwd, inv=inv
            )
            if not centers:
                centers = place_sites_kmeans_in_polygon(
                    df_all["lat"].values, df_all["lon"].values,
                    PLACEMENT_POLY_UTM, n_sites=required_locs,
                    fwd=fwd, inv=inv
                )

            # map center fallback
            if launch_coords:
                lat0, lon0 = float(launch_coords[0][0]), float(launch_coords[0][1])
            elif len(df_all) > 0:
                lat0, lon0 = float(df_all["lat"].mean()), float(df_all["lon"].mean())
            else:
                lat0, lon0 = 0.0, 0.0

            m = folium.Map(location=[lat0, lon0], zoom_start=st.session_state.get("base_zoom", 11))

            comp_r_m = comp_range_mi * 1609.34
            for (la, lo) in centers:
                folium.Circle(location=(la, lo), radius=comp_r_m, color="red", weight=3, fill=False).add_to(m)
                folium.CircleMarker(location=(la, lo), radius=3, color="red", fill=True).add_to(m)

            outline_latlon = polygon_outline_latlon(PLACEMENT_POLY_UTM, inv)

            # FAA grid (global toggle)
            if st.session_state.get("show_faa") and FAA_GEOJSON:
                from folium.features import GeoJsonTooltip
                def _faa_style(feat):
                    return {"color": "#3388ff", "weight": 1, "fillColor": "#3388ff", "fillOpacity": 0.18}
                folium.GeoJson(
                    FAA_GEOJSON,
                    name="FAA Grid",
                    style_function=_faa_style,
                    tooltip=GeoJsonTooltip(fields=["CEILING"], aliases=["Ceiling (ft)"])
                ).add_to(m)
                folium.LayerControl(collapsed=True).add_to(m)

            # Optional: overlay our Aerodome coverage on the competitor map
            if st.session_state.get("show_our_on_comp"):
                # Prefer our union outline if available
                try:
                    our_outline_cmp = polygon_outline_latlon(our_poly_utm, inv)
                except Exception:
                    our_outline_cmp = None

                # Use our current launch coords and effective range
                _our_coords_cmp = launch_coords if 'launch_coords' in globals() else []
                _our_range_cmp  = our_eff_range if 'our_eff_range' in globals() else drone_range

                if _our_coords_cmp and _our_range_cmp:
                    overlay_our_coverage_on(
                        m,
                        our_coords=_our_coords_cmp,
                        range_mi=_our_range_cmp,
                        outline_latlon=our_outline_cmp
                    )

            st_folium(m, width=800, height=500, key=f"cmp_map_R_{competitor}")

            comp_docks_per_loc = PLATFORMS[competitor]["docks_per_location"]
            total_comp_docks = required_locs * comp_docks_per_loc

            # Read radar parameters from the competitor's specs
            radar_enabled, radar_radius, radar_price = competitor_radar_params(competitor)

            # Base cost (no radar) from the plan
            comp_cost_200 = int(plan["yearly_cost"])

            # 400 ft cost includes radar only if enabled for this competitor
            if radar_enabled and radar_radius > 0:
                area_sqmi = float(TARGET_AREA_SQMI or 0.0)
                radars_needed = math.ceil(area_sqmi / circle_area_sqmi(radar_radius)) if area_sqmi > 0 else 0
                comp_cost_400 = int(comp_cost_200 + radars_needed * radar_price)
            else:
                comp_cost_400 = None  # explicitly mark as N/A

            # show two columns for counts
            cA, cB = st.columns(2)
            cA.metric("Required Locations", f"{required_locs:,}")
            cB.metric("Total Docks", f"{total_comp_docks:,}")

            # show two columns for cost
            cC, cD = st.columns(2)
            cC.metric("Yearly Cost @200 ft", f"${comp_cost_200:,}")
            if comp_cost_400 is not None:
                cD.metric("Yearly Cost @400 ft", f"${comp_cost_400:,}")
            else:
                cD.metric("Yearly Cost @400 ft", "N/A")

            # Persist report metrics: regular comparison (Competitor)
            try:
                st.session_state["cmp_comp_locs"] = int(required_locs)
                st.session_state["cmp_comp_docks"] = int(total_comp_docks)
                st.session_state["cmp_comp_cost_200"] = int(comp_cost_200)
                st.session_state["cmp_comp_cost_400"] = int(comp_cost_400) if radar_enabled else None
            except Exception:
                pass

            # --- Competitor Specs (RIGHT side) ---
            if competitor in PLATFORMS:
                specs = PLATFORMS[competitor].get("specs", {})
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
                if specs:
                    st.markdown("**Specs**")
                    for r in rows:
                        if r in specs:
                            st.markdown(f"**{r}**: {specs[r]}")

# ---- Controls row + two panels (unchanged) ----------------------------------
topL, topR = st.columns([3, 2])
with topR:
    comp_choice = st.selectbox("Compare against", COMPETITOR_OPTIONS, index=0, key="cmp_choice")

L, R = st.columns(2)
with L:
    panel(aerodome_title, detected_types_list if is_multi else detected_types_list[:1], is_left=True)
with R:
    panel(comp_choice, [], is_left=False, competitor=comp_choice)


# ─── Comparison — Full City (only render if CSV present) ────────────────────
if full_city_file:
    st.markdown("---")
    st.header("Comparison — Full City")

    import io
    if isinstance(full_city_file, (bytes, bytearray)):
        buf = io.BytesIO(full_city_file)
        buf.name = "Launch Locations - Full City.csv"
        full_city_file = buf
    else:
        try:
            full_city_file.seek(0)
        except Exception:
            pass

    fj_df, _, _ = _load_launch_locations_csv(full_city_file)
    fj_df.columns = [c.strip() for c in fj_df.columns]

    # ✅ Define FULL_CITY_AREA here so later code can use it
    FULL_CITY_AREA = float(
        st.session_state.get("city_area_sqmi") or 0.0
    ) or float(CALLS_AREA_SQMI or 0.0)

    if "Location Name" not in fj_df.columns and "Locations" in fj_df.columns:
        fj_df.rename(columns={"Locations": "Location Name"}, inplace=True)
    if "Lon" not in fj_df.columns and "Long" in fj_df.columns:
        fj_df.rename(columns={"Long": "Lon"}, inplace=True)
    for col in ["Location Name", "Address", "Lat", "Lon", "Type", "Dock Type", "Number of Docks", "Number of Radar"]:
        if col not in fj_df.columns:
            fj_df[col] = ""

    # --- Select & prepare launch rows (tolerant) ---
    t_raw = fj_df.get("Type")
    if t_raw is None:
        # No Type column? treat all rows as launch rows
        is_launch_mask = pd.Series([True] * len(fj_df), index=fj_df.index)
    else:
        _t = t_raw.astype(str).str.strip().str.lower()
        # accept 'launch', 'launch location', empty, or NaN as launch rows
        is_launch_mask = (
            _t.str.contains("launch", na=False) |
            _t.eq("") |
            _t.isna()
        )

    fj_launch = fj_df.loc[is_launch_mask].copy()

    # ---- Geocode if only address present ----
    fj_launch["Lat"] = pd.to_numeric(fj_launch.get("Lat"), errors="coerce")
    fj_launch["Lon"] = pd.to_numeric(fj_launch.get("Lon"), errors="coerce")
    need_geo = fj_launch["Address"].notna() & (fj_launch["Lat"].isna() | fj_launch["Lon"].isna())

    for idx in fj_launch.loc[need_geo].index:
        addr = normalize_address(fj_launch.at[idx, "Address"])
        la, lo = lookup(addr)
        fj_launch.at[idx, "Lat"] = la
        fj_launch.at[idx, "Lon"] = lo

    # ---- Ingest report (why rows might be missing) ----
    _before = len(fj_df)
    _after_type = len(fj_launch)
    missing_coords = fj_launch[fj_launch["Lat"].isna() | fj_launch["Lon"].isna()][["Location Name","Address","Lat","Lon"]].copy()

    # Drop rows that still lack coordinates
    fj_launch = fj_launch.dropna(subset=["Lat","Lon"]).copy()
    _after_coords = len(fj_launch)

    with st.sidebar.expander("Full City — ingest report", expanded=False):
        st.write(f"Rows in CSV: **{_before}**")
        st.write(f"Rows tagged as Launch: **{_after_type}**")
        st.write(f"Rows kept (have Lat/Lon): **{_after_coords}**")
        if not missing_coords.empty:
            st.caption("Rows dropped due to missing coordinates (fix Type/Address or add Lat/Lon):")
            st.dataframe(missing_coords, use_container_width=True)

    # ---- Build coords list (for map & coverage) ----
    fj_launch["Lat"] = pd.to_numeric(fj_launch["Lat"], errors="coerce")
    fj_launch["Lon"] = pd.to_numeric(fj_launch["Lon"], errors="coerce")
    launch_coords_full = [
        (float(r["Lat"]), float(r["Lon"]))
        for _, r in fj_launch[["Lat","Lon"]].iterrows()
        if np.isfinite(r["Lat"]) and np.isfinite(r["Lon"])
    ]
    if not launch_coords_full:
        st.error("Full Juris CSV has no valid Lat/Lon for launch locations.")
        st.stop()

    # ---- Pricing (mirror main) ----
    fj_launch["Dock Type"]       = fj_launch["Dock Type"].astype(str).str.strip().str.upper()
    fj_launch["Number of Docks"] = pd.to_numeric(fj_launch["Number of Docks"], errors="coerce").fillna(0).astype(int)
    fj_launch["Number of Radar"] = pd.to_numeric(fj_launch["Number of Radar"], errors="coerce").fillna(0).astype(int)

    def _dock_unit_price(dt): return PRICE_PER_DOCK_BY_TYPE.get(str(dt).upper().strip(), 0)
    fj_launch["Dock Unit Price"]   = fj_launch["Dock Type"].map(_dock_unit_price)
    # Apply selected unit price and docks per location only when override is enabled
    try:
        if st.session_state.get("override_launch_pricing"):
            sel_price = int(st.session_state.get("selected_price_per_dock", 0) or 0)
            if sel_price > 0:
                fj_launch["Dock Unit Price"] = int(sel_price)
            sel_docks = int(st.session_state.get("selected_docks_per_location", 1) or 1)
            if sel_docks > 0:
                fj_launch["Number of Docks"] = sel_docks
    except Exception:
        pass
    fj_launch["Dock Yearly Cost"]  = (fj_launch["Number of Docks"] * fj_launch["Dock Unit Price"]).astype(int)
    fj_launch["Radar Yearly Cost"] = (fj_launch["Number of Radar"] * PRICE_PER_RADAR).astype(int)
    # Split Aerodome totals for Full City
    our_full_base_200 = int(fj_launch["Dock Yearly Cost"].sum())
    our_full_base_400 = int((fj_launch["Dock Yearly Cost"] + fj_launch["Radar Yearly Cost"]).sum())

    # Persist report metrics: full city (Aerodome)
    try:
        _fc_our_cost_200 = int(fj_launch["Dock Yearly Cost"].sum())
        _fc_our_cost_400 = int((fj_launch["Dock Yearly Cost"] + fj_launch["Radar Yearly Cost"]).sum())
        _fc_our_locs     = int(len(launch_coords_full))
        _fc_our_docks    = int(fj_launch["Number of Docks"].sum())
        try:
            if st.session_state.get("override_launch_pricing"):
                sel_docks = int(st.session_state.get("selected_docks_per_location", 1) or 1)
                if sel_docks > 0:
                    _fc_our_docks = _fc_our_locs * sel_docks
        except Exception:
            pass

        st.session_state["fc_our_locs"]     = _fc_our_locs
        st.session_state["fc_our_docks"]    = _fc_our_docks
        st.session_state["fc_our_cost_200"] = _fc_our_cost_200
        st.session_state["fc_our_cost_400"] = _fc_our_cost_400
    except Exception:
        pass

    # ---- Build our full-coverage polygon ----
    our_full_poly_utm, OUR_FULL_CIRCLES_AREA_SQMI = union_launch_circles_utm(
        launch_coords_full, our_eff_range, fwd_calls  # reuse transformer if available
    )

    # ---- Unified target/placement logic ----
    TARGET_FC_SQMI, PLACEMENT_FC_POLY_UTM, _fc_mode = compute_competition_target(
        city_area_sqmi=float(FULL_CITY_AREA or 0.0),
        our_sqmi=float(OUR_FULL_CIRCLES_AREA_SQMI or 0.0),
        calls_poly_utm=calls_poly_utm,
        our_poly_utm=our_full_poly_utm
    )

    # ---- Panels ----
    def panel_full_left(title, coords, docks, yearly_cost_200, yearly_cost_400):
        with st.container(border=True):
            st.subheader(title)
            # Create the map object so we can overlay coverage directly
            m = folium.Map(
                location=[float(coords[0][0]), float(coords[0][1])] if coords else [0.0, 0.0],
                zoom_start=st.session_state.get("base_zoom", 11)
            )
            # Render the heatmap of all calls
            if not df_all.empty:
                data = (
                    df_all[["lat","lon","count"]].values.tolist()
                    if "count" in df_all.columns
                    else df_all[["lat","lon"]].values.tolist()
                )
                from folium.plugins import HeatMap
                HeatMap(data, radius=8, blur=12).add_to(m)
            # Draw blue circles for each launch site
            if coords:
                for la, lo in coords:
                    folium.Circle(
                        location=(la, lo),
                        radius=our_eff_range * 1609.34,
                        color="blue",
                        fill=False
                    ).add_to(m)
            # FAA grid (global toggle)
            if st.session_state.get("show_faa") and FAA_GEOJSON:
                from folium.features import GeoJsonTooltip
                def _faa_style(feat):
                    return {"color": "#3388ff", "weight": 1, "fillColor": "#3388ff", "fillOpacity": 0.18}
                folium.GeoJson(
                    FAA_GEOJSON,
                    name="FAA Grid",
                    style_function=_faa_style,
                    tooltip=GeoJsonTooltip(fields=["CEILING"], aliases=["Ceiling (ft)"])
                ).add_to(m)
                folium.LayerControl(collapsed=True).add_to(m)
            st_folium(m, width=800, height=500, key="cmp_full_left_map")

            c1, c2 = st.columns(2)
            c1.metric("Required Locations", f"{len(coords):,}")
            c2.metric("Total Docks", f"{int(docks):,}")
            c3, c4 = st.columns(2)
            c3.metric("Yearly Cost @200 ft", f"${int(yearly_cost_200):,}")
            c4.metric("Yearly Cost @400 ft", f"${int(yearly_cost_400):,}")

            # --- Aerodome Specs (LEFT side, Full City) ---
            def _render_specs_block(pname: str):
                specs = PLATFORMS.get(pname, {}).get("specs", {})
                if not specs:
                    return
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
                st.markdown("**Specs**")
                for r in rows:
                    if r in specs:
                        st.markdown(f"**{r}**: {specs[r]}")

            if is_multi:
                st.markdown("**Detected Aerodome Platforms:** " + ", ".join(detected_types_list))
                for p in detected_types_list:
                    st.markdown(f"**{p}**")
                    _render_specs_block(p)
            else:
                if detected_types_list:
                    _render_specs_block(detected_types_list[0])

    def panel_full_right(competitor):
        with st.container(border=True):
            st.subheader(competitor)
            plan = competitor_plan(competitor, TARGET_FC_SQMI)
            required_locs_nom = int(plan["locations"])
            required_locs = int(math.ceil(required_locs_nom * (1.0 + overpack_pct/100.0)))
            # Guardrail: if rounding produced 0 but we had a non-zero nominal, use 1
            try:
                if required_locs < 1 and float(plan.get("locations", 0)) > 0:
                    required_locs = 1
            except Exception:
                if required_locs < 1:
                    required_locs = 1
            comp_range_mi = plan["range_mi"]
            docks_per_loc = PLATFORMS[competitor]["docks_per_location"]
            yearly_cost   = plan["yearly_cost"]

            if (PLACEMENT_FC_POLY_UTM is None) or (fwd_calls is None) or (inv_calls is None):
                st.error("Not enough geometry to place competitor sites.")
                return

            centers = place_sites_hex_in_polygon(
                PLACEMENT_FC_POLY_UTM, n_sites=required_locs, range_mi=comp_range_mi,
                fwd=fwd_calls, inv=inv_calls
            )
            if not centers:
                centers = place_sites_kmeans_in_polygon(
                    df_all["lat"].values, df_all["lon"].values,
                    PLACEMENT_FC_POLY_UTM, n_sites=required_locs,
                    fwd=fwd_calls, inv=inv_calls
                )

            if launch_coords_full:
                lat0, lon0 = float(launch_coords_full[0][0]), float(launch_coords_full[0][1])
            elif len(df_all):
                lat0, lon0 = float(df_all["lat"].mean()), float(df_all["lon"].mean())
            else:
                lat0, lon0 = 0.0, 0.0

            m = folium.Map(location=[lat0, lon0], zoom_start=st.session_state.get("base_zoom", 11))

            # FAA grid (global toggle)
            if st.session_state.get("show_faa") and FAA_GEOJSON:
                from folium.features import GeoJsonTooltip
                def _faa_style(feat):
                    return {"color": "#3388ff", "weight": 1, "fillColor": "#3388ff", "fillOpacity": 0.18}
                folium.GeoJson(
                    FAA_GEOJSON,
                    name="FAA Grid",
                    style_function=_faa_style,
                    tooltip=GeoJsonTooltip(fields=["CEILING"], aliases=["Ceiling (ft)"])
                ).add_to(m)
                folium.LayerControl(collapsed=True).add_to(m)

            comp_r_m = comp_range_mi * 1609.34
            for (la, lo) in centers:
                folium.Circle(location=(la, lo), radius=comp_r_m, color="red", weight=3, fill=False).add_to(m)
                folium.CircleMarker(location=(la, lo), radius=3, color="red", fill=True).add_to(m)

            outline_latlon = polygon_outline_latlon(PLACEMENT_FC_POLY_UTM, inv_calls)

            # Overlay our Aerodome coverage if toggle is on (single block)
            if st.session_state.get("show_our_on_comp"):
                try:
                    our_outline_fc = polygon_outline_latlon(our_full_poly_utm, inv_calls)
                except Exception:
                    our_outline_fc = None
                _our_coords_fc = launch_coords_full if 'launch_coords_full' in locals() else []
                _our_range_fc  = our_eff_range if 'our_eff_range' in globals() else drone_range
                if _our_coords_fc and _our_range_fc:
                    overlay_our_coverage_on(
                        m,
                        our_coords=_our_coords_fc,
                        range_mi=_our_range_fc,
                        outline_latlon=our_outline_fc
                    )

            st_folium(m, width=800, height=500, key="cmp_full_right_map")

            total_comp_docks = required_locs * docks_per_loc

            # Radar pricing driven by PLATFORMS specs (future-proof)
            radar_enabled, radar_radius_mi, radar_price = competitor_radar_params(competitor)
            if radar_enabled and radar_radius_mi > 0:
                _radar_area = circle_area_sqmi(radar_radius_mi)  # πr² in sq mi
                radars_needed = math.ceil((TARGET_FC_SQMI or 0.0) / max(_radar_area, 1e-9))
            else:
                radars_needed = 0
                radar_price = 0.0

            comp_cost_200 = int(yearly_cost)  # no radar
            comp_cost_400 = int(yearly_cost + radars_needed * radar_price)  # with radar

            # 2x2 layout: [Required Locations | Total Docks], [Yearly Cost @200 ft | Yearly Cost @400 ft]
            c1, c2 = st.columns(2)
            c1.metric("Required Locations", f"{required_locs:,}")
            c2.metric("Total Docks", f"{total_comp_docks:,}")

            c3, c4 = st.columns(2)
            c3.metric("Yearly Cost @200 ft", f"${comp_cost_200:,}")
            if radar_enabled:
                c4.metric("Yearly Cost @400 ft", f"${comp_cost_400:,}")
            else:
                c4.metric("Yearly Cost @400 ft", "N/A")
            # (Intentionally do NOT show radar quantity)

            # Persist report metrics: full city (Competitor)
            try:
                st.session_state["fc_comp_locs"]     = int(required_locs)
                st.session_state["fc_comp_docks"]    = int(total_comp_docks)
                st.session_state["fc_comp_cost_200"] = int(comp_cost_200)
                st.session_state["fc_comp_cost_400"] = int(comp_cost_400) if radar_enabled else None
            except Exception:
                pass

            # --- Competitor Specs (RIGHT side, Full City) ---
            if competitor in PLATFORMS:
                specs = PLATFORMS[competitor].get("specs", {})
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
                if specs:
                    st.markdown("**Specs**")
                    for r in rows:
                        if r in specs:
                            st.markdown(f"**{r}**: {specs[r]}")

    # Put the competitor selector above the two panels
    comp_choice_fc = st.selectbox(
        "Compare against (Full City)",
        COMPETITOR_OPTIONS, index=0, key="cmp_choice_fullcity"
    )
    
    FC_L, FC_R = st.columns(2)
    with FC_L:
        panel_full_left(
            title="Flock Aerodome — Full City",
            coords=launch_coords_full,
            docks=int(fj_launch["Number of Docks"].sum()),
            yearly_cost_200=our_full_base_200,
            yearly_cost_400=our_full_base_400
        )
    with FC_R:
        panel_full_right(competitor=comp_choice_fc)

st.markdown("---")


# ─── REPORT VALUES + EXPORTS (collapsed at bottom) ───────────────────────────
st.header("Report & CSV Exports")
with st.expander("📊 Report Values & Exports", expanded=False):
    st.subheader("Report Values")
    st.dataframe(report_df, use_container_width=True)

    st.subheader("CSV Exports")
    cols = ["lat","lon","patrol_sec","drone_eta_sec","onscene_sec","priority","call_type_up"]
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button("Download DFR Only", to_csv_bytes(dfr_only[cols]), "dfr_only.csv")
    with c2:
        st.download_button("Download In Range", to_csv_bytes(in_range[cols]), "in_range.csv")
    with c3:
        st.download_button("Download Clearable", to_csv_bytes(clearable[cols]), "clearable.csv")




    # --- Debug panel: validity state and DFR-eligible counts ---------------
    with st.expander("🔎 Debug — validity (counts & sample)", expanded=False):
        try:
            _df_all_valid = st.session_state.get("df_all_valid", pd.DataFrame())
            _df_dfr_only  = st.session_state.get("df_dfr_only", pd.DataFrame())
            st.write({
                "raw_is_minimal": bool(st.session_state.get("raw_is_minimal", False)),
                "df_all_rows": int(len(df_all) if isinstance(df_all, pd.DataFrame) else 0),
                "df_all_valid_rows": int(len(_df_all_valid) if isinstance(_df_all_valid, pd.DataFrame) else 0),
                "df_dfr_only_rows": int(len(_df_dfr_only) if isinstance(_df_dfr_only, pd.DataFrame) else 0),
            })
            if isinstance(df_all, pd.DataFrame) and not df_all.empty:
                cols = [c for c in ["call_type_up","priority","lat","lon","create_hour","create_weekday"] if c in df_all.columns]
                if cols:
                    st.caption("First 10 rows used by minimal validity mask:")
                    st.dataframe(df_all[cols].head(10), use_container_width=True)
        except Exception as _e:
            st.write(f"Debug error: {_e}")
