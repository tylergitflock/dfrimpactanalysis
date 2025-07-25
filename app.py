import io, numpy as np, polars as pl, streamlit as st

st.set_page_config(page_title="DFR Impact Analyzer", layout="wide")
st.title("DFR Impact Analyzer")

# ---- Assumptions (left sidebar) ----
with st.sidebar:
    st.header("Assumptions")
    fte_hours    = st.number_input("Full Time Work Year (hours)", value=2080)
    officer_cost = st.number_input("Full Time Equivalent Cost ($)", value=127_940, step=1000)
    cancel_rate  = st.number_input("Drone Clearance Rate (%)", value=11.0, step=0.1) / 100.0
    drone_speed  = st.number_input("Drone Speed (mph)", value=51)
    drone_range  = st.number_input("Drone Range (miles)", value=3.5)

# ---- Uploads ----
st.subheader("1) Upload CSVs")
raw_file    = st.file_uploader("Raw Call Data (.csv)", type=["csv"], key="raw")
agency_file = st.file_uploader("Agency Call Types (.csv)", type=["csv"], key="agency")
launch_file = st.file_uploader("Launch Locations (.csv)", type=["csv"], key="launch")

# Helpers -------------------------------------------------------
def read_csv(file) -> pl.DataFrame:
    """Read CSV safely (works for big files)."""
    b = file.read()
    file.seek(0)
    return pl.read_csv(io.BytesIO(b), infer_schema_length=2000)

def to_seconds(col: pl.Series) -> pl.Series:
    """Convert timestamps to seconds, handling Excel-serials or text timestamps."""
    if pl.datatypes.is_numeric(col.dtype):
        s = col.cast(pl.Float64)
        med = s.drop_nulls().median()
        # Excel day serials often ~ 40k–60k range -> convert to seconds
        if med is not None and 20000 < med < 100000:
            return s * 86400.0
        return s
    try:
        dt = pl.to_datetime(col, strict=False)
        return (dt.cast(pl.Int64) / 1_000_000_000).cast(pl.Float64)
    except Exception:
        return pl.lit(None).alias(col.name).cast(pl.Float64)

def haversine_min(df: pl.DataFrame, launches: pl.DataFrame) -> pl.Series:
    """Minimum distance (miles) from each row's Lat/Lon to any launch."""
    R = 3958.8
    to_rad = np.pi / 180.0
    exprs = []
    for la, lo in launches.select(["Lat","Lon"]).iter_rows():
        exprs.append(
            2*R*pl.arcsin(
                pl.sqrt(
                    (pl.sin(((pl.col("Lat")-la)*to_rad)/2))**2 +
                    pl.cos(la*to_rad)*pl.cos(pl.col("Lat")*to_rad) *
                    (pl.sin(((pl.col("Lon")-lo)*to_rad)/2))**2
                )
            )
        )
    return (pl.min_horizontal(exprs) if exprs else pl.lit(np.inf)).alias("dist_mi")

def build_report(df_raw, df_agency, df_launch,
                 fte_hours, officer_cost, cancel_rate, drone_speed, drone_range):
    # Normalize columns and types
    need = ["Time Call Entered Queue","Time First Unit Assigned","Time First Unit Arrived",
            "Fixed Time Call Closed","Call Type","Call Priority","Lat","Lon"]
    missing = [c for c in need if c not in df_raw.columns]
    if missing:
        raise ValueError("Raw Call Data is missing columns: " + ", ".join(missing))

    df_raw = df_raw.with_columns([
        to_seconds(pl.col("Time Call Entered Queue")).alias("t_create"),
        to_seconds(pl.col("Time First Unit Assigned")).alias("t_dispatch"),
        to_seconds(pl.col("Time First Unit Arrived")).alias("t_arrive"),
        to_seconds(pl.col("Fixed Time Call Closed")).alias("t_close"),
        pl.col("Call Type").cast(pl.Utf8).str.to_uppercase().alias("call_type"),
        pl.col("Call Priority").cast(pl.Utf8).alias("priority"),
        pl.col("Lat").cast(pl.Float64), pl.col("Lon").cast(pl.Float64)
    ])

    need_ag = {"Call Type","DFR Response (Y/N)","Clearable (Y/N)"}
    if not need_ag.issubset(df_agency.columns):
        raise ValueError("Agency Call Types must have: Call Type, DFR Response (Y/N), Clearable (Y/N)")
    df_agency = df_agency.with_columns([
        pl.col("Call Type").cast(pl.Utf8).str.to_uppercase().alias("call_type"),
        pl.col("DFR Response (Y/N)").cast(pl.Utf8).str.to_uppercase().alias("dfr_yn"),
        pl.col("Clearable (Y/N)").cast(pl.Utf8).str.to_uppercase().alias("clr_yn"),
    ])
    dfr_set = set(df_agency.filter(pl.col("dfr_yn")=="Y")["call_type"].to_list())
    clr_set = set(df_agency.filter(pl.col("clr_yn")=="Y")["call_type"].to_list())

    if not {"Lat","Lon"}.issubset(df_launch.columns):
        raise ValueError("Launch Locations must have: Lat, Lon")
    df_launch = df_launch.select([
        pl.col("Lat").cast(pl.Float64).alias("Lat"),
        pl.col("Lon").cast(pl.Float64).alias("Lon")
    ])

    # Distances & ETAs
    df_raw = df_raw.with_columns(haversine_min(df_raw, df_launch))
    df_raw = df_raw.with_columns((pl.col("dist_mi")/drone_speed*3600.0).alias("drone_eta_sec"))
    df_raw = df_raw.with_columns([
        (pl.col("t_arrive") - pl.col("t_dispatch")).alias("patrol_eta_sec"),
        (pl.col("t_arrive") - pl.col("t_create")).alias("create_to_arrive_sec"),
        (pl.col("t_close")  - pl.col("t_arrive")).alias("on_scene_sec"),
    ])

    # Filters
    df_dfr = df_raw.filter(
        (pl.col("t_dispatch") > 0) &
        (pl.col("t_arrive")   > 0) &
        (pl.col("t_arrive")   != pl.col("t_create")) &  # remove self-initiated
        (pl.col("call_type").is_in(dfr_set))
    )
    df_in  = df_dfr.filter(pl.col("dist_mi") <= drone_range)
    df_clr = df_in.filter(pl.col("call_type").is_in(clr_set))

    # Metrics
    total_cfs        = df_raw.height
    total_potential  = df_dfr.height
    in_range_count   = df_in.height
    clearable_count  = df_clr.height

    avg_disp_patrol  = df_dfr["patrol_eta_sec"].mean()
    avg_scene_all    = df_dfr["on_scene_sec"].mean()
    avg_patrol_in    = df_in["create_to_arrive_sec"].mean()
    avg_drone_resp   = df_in["drone_eta_sec"].mean()
    first_on_pct     = (df_in.filter(pl.col("drone_eta_sec") < pl.col("create_to_arrive_sec")).height
                        / max(in_range_count,1))

    pct_decrease     = ((avg_patrol_in - avg_drone_resp)/avg_patrol_in) if (avg_patrol_in and avg_patrol_in>0) else 0.0
    avg_scene_clear  = df_clr["on_scene_sec"].mean()
    expected_cleared = round(in_range_count * cancel_rate)
    total_time_clear = float(df_clr["on_scene_sec"].sum() or 0.0)
    officers_fte     = ((avg_scene_clear or 0.0) * expected_cleared / 3600.0) / max(fte_hours,1)
    roi_savings      = officers_fte * officer_cost

    # Seconds → fractional days (for durations)
    def sec_to_days(x): return x/86400.0 if x is not None else None

    rows = [
        ("DFR Responses within Range", in_range_count, "#,##0"),
        ("Expected DFR Drone Response Times by Location", sec_to_days(avg_drone_resp), "[m]:ss"),
        ("Expected First on Scene %", first_on_pct, "0%"),
        ("Expected CFS Cleared", expected_cleared, "#,##0"),
        ("Number of Officers - Force Multiplication", officers_fte, "0.00"),
        ("ROI from Potential Calls Cleared", roi_savings, "$#,##0"),
        ("Total CFS", total_cfs, "#,##0"),
        ("Total Potential DFR Calls", total_potential, "#,##0"),
        ("Avg Disp + Patrol Response Time to DFR Calls", sec_to_days(avg_disp_patrol), "[m]:ss"),
        ("Avg Time on Scene ALL DFR Calls", sec_to_days(avg_scene_all), "[m]:ss"),
        ("Avg Disp + Patrol Response Time to In-Range Calls", sec_to_days(avg_patrol_in), "[m]:ss"),
        ("Expected Decrease in Response Times", pct_decrease, "0%"),
        ("Total clearable CFS within range", clearable_count, "#,##0"),
        ("Total time spent on Clearable CFS", sec_to_days(total_time_clear), "[h]:mm:ss"),
        ("Avg Time on Scene – Clearable Calls", sec_to_days(avg_scene_clear), "[m]:ss"),
    ]
    return pl.DataFrame({"Metric":[r[0] for r in rows],
                         "Value":[r[1] for r in rows],
                         "Format":[r[2] for r in rows]})

# ---- Run button ----
can_run = bool(raw_file and agency_file and launch_file)
if st.button("Run Analysis", type="primary", disabled=not can_run):
    with st.spinner("Crunching…"):
        df_raw    = read_csv(raw_file)
        df_agency = read_csv(agency_file)
        df_launch = read_csv(launch_file)
        report    = build_report(df_raw, df_agency, df_launch,
                                 fte_hours, officer_cost, cancel_rate, drone_speed, drone_range)
    st.success("Done.")
    st.subheader("Report Values")
    st.dataframe(report)

    st.download_button("Download Report Values (CSV)",
        data=report.write_csv().encode("utf-8"),
        file_name="report_values.csv", mime="text/csv")
