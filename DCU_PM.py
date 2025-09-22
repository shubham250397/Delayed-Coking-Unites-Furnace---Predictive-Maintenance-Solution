import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="DCU Furnace Predictive Maintenance Suite",
    layout="wide",
    initial_sidebar_state="collapsed"
)
st.markdown("""
<style>
body, .stApp { background: #181920 !important; }
.block-container { padding-top: 0rem; }
h1, h2, h3, h4, h5, h6, .st-bb, .st-bf, .st-cw, .st-cx { color: #fff !important; }
.stTabs [data-baseweb="tab"] { background: #232323; color: #fff; border-radius: 12px 12px 0 0; margin-right: 4px; padding: 6px 24px;}
.stTabs [aria-selected="true"] { background: #FFD700 !important; color: #181920 !important; }
.metric { background: #232323; color: #fff; border-radius: 14px; padding: 20px 14px; text-align: center; }
.metric-label { color: #FFD700; font-size: 18px;}
.metric-value { color: #fff; font-size: 30px; font-weight: bold;}
.kpi-box { background: #232323; border-radius: 12px; padding: 18px; margin-bottom: 10px; text-align: center;}
.center-tabs > div {display: flex; justify-content: center; }
div[data-testid="stMetricValue"] { color: #FFD700 !important; }
.stSelectbox label, .stMultiSelect label, .stSlider label, .stRadio label {
    color: #fff !important; font-size: 1.08rem !important; font-weight:700 !important; margin-bottom: 6px !important;
}
.stSlider .css-1y4p8pa, .stSlider .css-14xtw13 { color: #FFD700 !important;}
.stRadio [data-testid="stRadioButton"] > div:first-child {
    border: 2px solid #FFD700 !important;
    background: #232323 !important;
}
.stRadio [aria-checked="true"] > div:first-child {
    background: #FFD700 !important;
    border-color: #FFD700 !important;
}
.stRadio [aria-checked="true"] > div:last-child {
    color: #232323 !important;
    font-weight: bold !important;
}
.custom-reco-box {
    background: #232323;
    border-radius: 10px;
    margin: 14px 0 12px 0;
    padding: 13px 16px;
    color: #fff;
    border-left: 5px solid #FFD700;
}
.custom-reco-head { font-weight: bold; color: #FFD700;}
.custom-reco-icon { font-size: 1.3em; vertical-align: middle; margin-right: 8px;}
.custom-reco-list { margin-left: 1.2em; margin-bottom:0;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="padding:28px 0 15px 0; text-align:center; background:linear-gradient(90deg, #232323, #2a292d 80%); border-radius:0 0 24px 24px">
    <h1 style="font-size:2.7rem;color:#FFD700;margin-bottom:8px;">DCU Furnace Predictive Maintenance & Optimization Suite</h1>
    <div style="font-size:1.14rem; color:#fff; letter-spacing:1px;">
        Live diagnostics, lookahead, cycle analytics, scenario simulation & industrial-grade recommendations.<br>
        <span style="color:#FFD700;">Oil & Gas, Petrochemicals, Refinery & Process Engineering Demo</span>
    </div>
</div>
""", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

driver_limits = {
    "Coil Outlet Temp": {"optimal": (484, 488), "high": (488, 491), "critical": (491, 560), "low": (478, 484)},
    "Feed CCR": {"optimal": (17, 21), "high": (21, 24), "critical": (24, 30), "low": (14, 17)},
    "Air_Pressure_Ratio": {"optimal": (0.98, 1.13), "high": (1.13, 1.2), "critical": (1.2, 2.0), "low": (0.9, 0.98)},
    "Air Flow": {"optimal": (1450, 1600), "high": (1600, 1700), "critical": (1700, 1800), "low": (1300, 1450)},
    "Furnace Pressure": {"optimal": (4.7, 5.15), "high": (5.15, 5.5), "critical": (5.5, 7), "low": (4.0, 4.7)}
}
driver_insights = {
    "Coil Outlet Temp": "Directly increases skin temp. High COT is a primary risk for excursions.",
    "Feed CCR": "Higher CCR increases coke rate and skin temp over cycle.",
    "Air_Pressure_Ratio": "Moderates heat transfer; both too low/high can destabilize operation.",
    "Air Flow": "Inversely affects skin temp. Low Air Flow can sharply raise risk of warnings/excursions.",
    "Furnace Pressure": "Affects heat transfer and risk; both too high or low destabilize."
}
driver_order = [
    ("Coil Outlet Temp", "COT", "#FFD700"),
    ("Feed CCR", "CCR", "#9c27b0"),
    ("Air_Pressure_Ratio", "APR", "#00cfff"),
    ("Air Flow", "Air Flow", "#43d675"),
    ("Furnace Pressure", "Furnace Pressure", "#FFD700"),
]

def hist_control_band_insight(feat):
    """Returns a descriptive insight for control band analysis for a given feature."""
    if feat == "Coil Outlet Temp":
        return ("Optimal operation is within 484–488°C. Values above 491°C indicate increased excursion risk—"
                "reduce setpoint or investigate heat input if frequently high.")
    elif feat == "Feed CCR":
        return ("Keep CCR below 21 for stable operation. Higher CCR increases coke buildup and excursion risk.")
    elif feat == "Air Flow":
        return ("Maintain airflow within typical range. Low airflow may decrease yield; excessive airflow may cause temp instability.")
    elif feat == "Furnace Pressure":
        return ("Keep pressure steady. Spikes may signal upsets or risk of safety trip.")
    elif feat == "Asphaltene":
        return ("Monitor asphaltene concentration. Higher values accelerate fouling and coke formation.")
    elif feat == "Feed Rate":
        return ("Stable feed rate supports consistent operation. Sharp changes may impact yield and coke rates.")
    else:
        return ("Monitor control band and adjust setpoints to stay within optimal range for process stability.")

@st.cache_data
def generate_dcu_data(hours=2000, seed=44, cycles=7):
    np.random.seed(seed)
    t = np.arange(hours)
    # Core process cycles (sin/cos + noise for realism)
    cot_base = 486 + 5*np.sin(2*np.pi*t/24) + np.random.normal(0,1.8,hours)
    ccr = 19 + 2.4*np.sin(2*np.pi*t/160) + np.random.normal(0,0.55,hours)
    apr = 1.06 + 0.09*np.sin(2*np.pi*t/55) + np.random.normal(0,0.018,hours)
    air = 1510 + 65*np.sin(2*np.pi*t/80) + np.random.normal(0,13,hours)
    press = 5.0 + 0.12*np.sin(2*np.pi*t/70) + np.random.normal(0,0.07,hours)
    # Excursion scenario logic (will inject critical/warning events)
    skin = []
    yield_ = []
    excursion = []
    zone = []
    last_decoke = 0
    runlen = []
    cycle_num = []
    for i in range(hours):
        # Simple decoke every ~270h, after 240h enter warning, after 260h enter critical
        rlen = (i - last_decoke)
        curr_zone = "Normal"
        if rlen >= 260:
            curr_zone = "Critical"
        elif rlen >= 240:
            curr_zone = "Warning"
        # Artificially force COT/CCR high if in critical/warning zone (simulates operator inaction)
        ccot = cot_base[i]
        cccr = ccr[i]
        cair = air[i]
        if curr_zone == "Critical":
            ccot += 5 + np.random.uniform(0, 3)
            cccr += 1.8
            cair -= 60
        elif curr_zone == "Warning":
            ccot += 2 + np.random.uniform(0, 2)
            cccr += 1.2
            cair -= 25
        # Add noise for realistic short-term variation
        stemp = ccot + 23 + 28 * (cccr - 19)/4.0 - 0.009*(cair - 1510) + 0.9*press[i] + np.random.normal(0,2.7)
        stemp = np.clip(stemp, 480, 558)
        curr_exc = (stemp > 535)
        # Excursion = yield drop
        yld = 78.4 - 3.7 * (cccr-19)/4.0
        if curr_exc: yld -= 3.3
        yld += np.random.normal(0,1.4)
        yld = np.clip(yld, 71, 81)
        # Cycle reset
        if rlen > 270 or (curr_exc and rlen>100 and np.random.rand()<0.16):
            last_decoke = i
            runlen.append(0)
            cycle_num.append(1 if len(cycle_num)==0 else cycle_num[-1]+1)
        else:
            runlen.append(rlen if len(runlen) else 0)
            cycle_num.append(cycle_num[-1] if len(cycle_num) else 1)
        skin.append(stemp)
        yield_.append(yld)
        excursion.append(curr_exc)
        zone.append(curr_zone)
    df = pd.DataFrame({
        "Hour": t,
        "Coil Outlet Temp": cot_base + np.where(np.array(zone)=="Critical", 7, np.where(np.array(zone)=="Warning", 3, 0)),
        "Feed CCR": ccr + np.where(np.array(zone)=="Critical", 2, np.where(np.array(zone)=="Warning", 1.3, 0)),
        "Air_Pressure_Ratio": apr,
        "Air Flow": air + np.where(np.array(zone)=="Critical", -40, np.where(np.array(zone)=="Warning", -18, 0)),
        "Furnace Pressure": press,
        "Skin Temp": skin,
        "Yield %": yield_,
        "Excursion": excursion,
        "Zone": zone,
        "Run Length": runlen,
        "Cycle": cycle_num
    })
    # Product splits
    df[["LPG", "Naphtha", "Diesel", "Fuel Oil", "Coke"]] = np.array([
        [0.18, 0.13, 0.29, 0.25, 0.15] if not ex else [0.14, 0.10, 0.26, 0.26, 0.24]
        for ex in excursion
    ])
    return df

@st.cache_data
def arima_forecast(series, steps=8):
    model = ARIMA(series, order=(2,1,2)).fit()
    pred = model.get_forecast(steps=steps)
    return pred.predicted_mean.values

def get_driver_status(val, limits):
    if "critical" in limits and val >= limits["critical"][0]:
        return "Critical"
    if "high" in limits and val >= limits["high"][0]:
        return "High"
    if "low" in limits and val < limits["optimal"][0]:
        return "Low"
    if "optimal" in limits and limits["optimal"][0] <= val <= limits["optimal"][1]:
        return "Normal"   # <--- change here
    return "Warning"

def scenario_forecast(df, latest, params, steps):
    # Linear sensitivity (empirical): COT(0.65), CCR(1.2), Air(-0.014), APR(8.3), Press(2.2)
    hist = df["Skin Temp"].iloc[-96:]
    base = arima_forecast(hist, steps)
    delta = (
        (params["Coil Outlet Temp"] - latest["Coil Outlet Temp"]) * 0.65
        + (params["Feed CCR"] - latest["Feed CCR"]) * 1.2
        + (params["Air Flow"] - latest["Air Flow"]) * -0.014
        + (params["Air_Pressure_Ratio"] - latest["Air_Pressure_Ratio"]) * 8.3
        + (params["Furnace Pressure"] - latest["Furnace Pressure"]) * 2.2
    )
    adj = base + delta
    return np.clip(base, 480, 558), np.clip(adj, 480, 558)

def minimal_driver_fix(df, latest, steps, warn_thr=525):
    # Try adjusting only one lever at a time (COT/CCR/Air/Press/APR) to bring skin temp < warn_thr for all steps
    # Return the fix as a dict of deltas
    order = ["Coil Outlet Temp", "Feed CCR", "Air Flow", "Air_Pressure_Ratio", "Furnace Pressure"]
    for key in order:
        for delta in np.arange(-8, 9, 0.5):
            params = {k: float(latest[k]) for k in order}
            params[key] = params[key] + delta
            _, adj = scenario_forecast(df, latest, params, steps)
            if np.all(adj < warn_thr):
                return key, delta, params
    return None, 0, None

# ===== LOAD & SELECT DATA WINDOW =====
df_full = generate_dcu_data()
max_hour = int(df_full["Hour"].iloc[-1])
window = st.slider("Data Window: Select Latest Hour (up to present)", min_value=200, max_value=max_hour, value=max_hour, step=1)
df_live = df_full[df_full["Hour"] <= window].copy()
latest = df_live.iloc[-1]

skin_thr, warn_thr = 535, 525
zone_colors = {
    "Normal": "#43d675",    # green
    "Warning": "#f6d32e",   # yellow
    "Critical": "#e74a3b",  # red
    "High": "#e67e22",      # orange
    "Low": "#3498db"        # blue
}




# =============== CACHED FORECAST LOGIC ====================
@st.cache_data(show_spinner=False)
def get_arima_forecast(skin_hist, sim_steps):
    model_fit = ARIMA(skin_hist, order=(2,1,2)).fit()
    pred = model_fit.get_forecast(steps=sim_steps)
    base_pred = np.clip(pred.predicted_mean.values, 510, 550)
    conf_int = pred.conf_int(alpha=0.20)
    return base_pred, conf_int


st.markdown("""
<style>
/* Center the horizontal tabs by flexbox alignment */
[data-testid="stHorizontalBlock"] > div {
    justify-content: center !important;
    display: flex !important;
}
/* Optional: If using st.tabs (Streamlit 1.22+) */
.stTabs [role="tablist"] {
    justify-content: center !important;
    display: flex !important;
}
</style>
""", unsafe_allow_html=True)

tab_titles = [
    "⚡ Live Furnace Status, Forecast & What-if Simulations",
    "🧭 Process Diagnostics & Analytics",
    "🔁 Cycle Analytics & Events"
]
tab1, tab2, tab3 = st.tabs(tab_titles)
st.markdown('</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

with tab1:
    #st.markdown("<hr style='border-top:2px solid #FFD700;margin:1rem 0'>", unsafe_allow_html=True)
    st.markdown("""
    <div style="border-radius: 15px; background: linear-gradient(90deg, #232323 80%, #1a1a1a); padding: 22px 18px 15px 18px; margin-bottom: 16px; display: flex; align-items: center;">
        <span style="font-size:2.2rem; margin-right: 16px;">🔥</span>
        <div>
            <span style="font-size:1.35rem; color:#FFD700; font-weight:700; letter-spacing:0.5px;">Live Furnace Status & Excursion Forecast</span><br>
            <span style="font-size:1rem; color:#fff;"> 
                <b>Monitor your DCU furnace in real time</b> with instant diagnostics, safety zone alerts, advanced <span style="color:#43d675;"><b>temperature & yield forecasting</b></span>, and strategic recommendations.<br>
                <span style="color:#FFD700;">All key trends, forecasts, and operational risks in one view.</span>
            </span>
        </div>
        <span style="font-size:2.1rem; margin-left: auto;">⏱️</span>
    </div>
    """, unsafe_allow_html=True)
  
    # ============= NUMERICAL KPIs ROW (CURRENT VALUES) =============
    kpi_metrics = [
        ("Skin Temp (now)", f"{latest['Skin Temp']:.1f}°C"),
        ("COT (now)", f"{latest['Coil Outlet Temp']:.1f}°C"),
        ("Yield (now)", f"{latest['Yield %']:.1f}%"),
        ("CCR (now)", f"{latest['Feed CCR']:.2f}"),
        ("APR (now)", f"{latest['Air_Pressure_Ratio']:.2f}"),
    ]
    k_cols = st.columns(len(kpi_metrics))
    for i, (lbl, val) in enumerate(kpi_metrics):
        k_cols[i].markdown(
            f"""<div class='kpi-box' style="background:#232323;border-left:4px solid #FFD700;">
                    <div class='metric-label'>{lbl}</div>
                    <div class='metric-value'>{val}</div>
                </div>""", unsafe_allow_html=True)

    # ============= CYCLE/STATUS METRICS CARDS =============
    excursions_ahead = np.argmax(df_live["Skin Temp"].values[::-1] < warn_thr)  # time to non-warning
    if excursions_ahead == 0: excursions_ahead = (df_live["Skin Temp"].iloc[::-1] > warn_thr).sum()
    stability_cond = "Stable" if latest["Skin Temp"] < warn_thr else ("Warning" if latest["Skin Temp"] < skin_thr else "Critical")
    cycle_metrics = [
        ("Time to Excursion", f"{excursions_ahead}h" if latest["Skin Temp"] < skin_thr else "Excursion Active", "⏱️"),
        ("Current Run Length", f"{int(latest['Run Length'])}h", "🔁"),
        ("Skin Temp Stability", f"{latest['Skin Temp']:.1f}°C — {stability_cond}", "📉"),
    ]
    cols = st.columns(len(cycle_metrics))
    for i,(title,val,icon) in enumerate(cycle_metrics):
        cols[i].markdown(
            f"""<div class='kpi-box' style="border-left:4px solid #FFD700">
                <span style='font-size:1.5rem;margin-right:8px'>{icon}</span>
                <span class='metric-label'>{title}</span>
                <div class='metric-value'>{val}</div>
            </div>""",
            unsafe_allow_html=True
        )

    # ============= MAIN TREND CHART =============
    st.markdown("### <span style='color:#FFD700'>Skin Temp, Yield & Excursion (Last 200 h)</span>",unsafe_allow_html=True)
    hist = df_live.iloc[-200:]
    fig, ax = plt.subplots(figsize=(11,4), facecolor="#181920", dpi=100)
    ax.axhspan(warn_thr, skin_thr, color="#FFD700", alpha=0.13)
    ax.axhspan(skin_thr, 560, color="#e74a3b", alpha=0.15)
    ax.plot(hist["Hour"], hist["Skin Temp"], color="#FFD700", lw=2, label="Skin Temp")
    exc_idx = hist.index[hist["Excursion"]].tolist()
    ax.scatter(hist.loc[exc_idx,"Hour"],hist.loc[exc_idx,"Skin Temp"], color="#e74a3b", marker="x",s=60,label="Excursion")
    ax2 = ax.twinx()
    ax2.plot(hist["Hour"], hist["Yield %"], color="#43d675", lw=2, label="Yield %")
    ax.axhline(skin_thr, color="#e74a3b", ls="--", lw=1.6)
    ax.axhline(warn_thr, color="#FFD700", ls="--", lw=1.2)
    ax.set_facecolor("#181920")
    ax.set_xlabel("Hour", color="white"); ax.set_ylabel("Skin Temp (°C)", color="#FFD700")
    ax2.set_ylabel("Yield (%)", color="#43d675"); ax.tick_params(colors="white"); ax2.tick_params(colors="white")
    for s in list(ax.spines.values())+list(ax2.spines.values()): s.set_color("white")
    fig.tight_layout()
    st.pyplot(fig)

    # ============= STATUS/INSIGHT BOX =============
    zone = get_driver_status(latest["Skin Temp"], {"optimal":(480,warn_thr),"high": (warn_thr,skin_thr),"critical":(skin_thr,560)})
    status_color = zone_colors[zone]
    longest_stable = (df_live["Skin Temp"] < warn_thr).astype(int).groupby((df_live["Skin Temp"] >= warn_thr).astype(int).cumsum()).cumsum().max()
    excursions_past48 = int(df_live["Excursion"].iloc[-48:].sum())
    insight = (
        f"Furnace stable: longest stable run <b>{longest_stable}h</b>, <b>{excursions_past48}</b> excursions last 48h."
        if zone=="Normal"
        else f"Warning: Skin temp rising, {excursions_past48} excursions in last 48h. Tune COT/CCR if persists."
        if zone=="Warning"
        else "Excursions detected! Immediate driver tuning needed: lower COT and CCR, raise Air."
    )
    st.markdown(
        f"""<div class='custom-reco-box'>
                <span class='custom-reco-head'>{zone} Status</span>
                <br><span style="color:{status_color};font-weight:700">{insight}</span>
            </div>""",
        unsafe_allow_html=True
    )

    # After your skin temp line chart (current/predicted)
    st.markdown("""
    <div style="font-size:1.25rem;color:#FFD700;font-weight:700;">
        Skin Temperature Trend with Anomaly Detection
    </div>
    <div style="color:#fff;font-size:1rem;margin-bottom:8px;">
        The yellow line shows the latest skin temperature readings. Red stars/crosses mark points where the temperature pattern is statistically abnormal, potentially indicating process upsets or faults.
    </div>
    """, unsafe_allow_html=True)

    window = 36
    z_thr = 2.8
    df_live["skin_z"] = (df_live["Skin Temp"] - df_live["Skin Temp"].rolling(window).mean()) / df_live["Skin Temp"].rolling(window).std()
    df_live["anomaly"] = df_live["skin_z"].abs() > z_thr
    
    fig, ax = plt.subplots(figsize=(8,2.9), dpi=100, facecolor="#181920")
    ax.plot(df_live["Hour"], df_live["Skin Temp"], color="#FFD700", lw=2, label="Skin Temp")
    # Highlight anomalies
    ax.scatter(df_live.loc[df_live["anomaly"], "Hour"], df_live.loc[df_live["anomaly"], "Skin Temp"], 
               color="#e74a3b", s=44, marker="*", label="Anomaly")
    
    ax.set_xlabel("Hour", color="white")
    ax.set_ylabel("Skin Temp (°C)", color="#FFD700")
    ax.set_facecolor("#181920")
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='#FFD700')
    for spine in ax.spines.values():
        spine.set_color("#fff")
    leg = ax.legend(loc='upper right', fontsize=9, facecolor="#232323", edgecolor="#fff")
    for text in leg.get_texts():
        text.set_color("#fff")
    st.pyplot(fig)
    
    # Add recommendation box
    if df_live["anomaly"].iloc[-1]:
        st.markdown("""
        <div class="custom-reco-box">
            <span class="custom-reco-icon">🚨</span>
            <span class="custom-reco-head" style="color:#e74a3b;">Anomaly Detected in Skin Temp</span>
            <ul class="custom-reco-list">
                <li>Recent pattern deviates sharply from expected range.</li>
                <li>Review process drivers for sudden shifts or faults.</li>
                <li>Consider operator/AI intervention if persists.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    # After plotting your figure with anomalies as red crosses/stars:
    if df_live["anomaly"].iloc[-1]:
        st.markdown("""
        <div class="custom-reco-box" style="margin-top:8px;">
            <span class="custom-reco-icon">🚨</span>
            <span class="custom-reco-head" style="color:#e74a3b;">
                Anomaly Detected in Skin Temp
            </span>
            <div style="font-size:0.98rem;color:#fff;">
                <ul class="custom-reco-list">
                    <li><b>Red stars</b> highlight timepoints where the skin temperature deviates sharply from normal operating behavior.</li>
                    <li>These anomalies could signal potential coking, equipment issues, sensor errors, or sudden process disturbances.</li>
                    <li><b>Recommended Action:</b> Immediately review operating parameters, check related driver charts (e.g., Coil Outlet Temp, Feed CCR, Air Flow), and intervene if excursions persist.</li>
                    <li>If multiple consecutive anomalies appear, consider ramping down temperature or scheduling a maintenance check.</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="custom-reco-box" style="margin-top:8px;">
            <span class="custom-reco-icon">✅</span>
            <span class="custom-reco-head" style="color:#43d675;">
                No Anomaly Detected
            </span>
            <div style="font-size:0.98rem;color:#fff;">
                Skin temperature trend is within expected operating range. Continue monitoring and maintaining optimal control.
            </div>
        </div>
        """, unsafe_allow_html=True)



    # ============= DRIVER TRENDS/STATUS BOXES (COMPACT, BOXED) =============
    st.markdown("## <span style='color:#FFD700'>Key Driver Trends & Status</span>", unsafe_allow_html=True)
    dr_cols = st.columns(len(driver_order))
    combi_risk = []
    for i, (feat, short, color) in enumerate(driver_order):
        hist = df_live[feat].iloc[-96:]
        val = hist.iloc[-1]
        limits = driver_limits[feat]
        status = get_driver_status(val, limits)
        ccol = zone_colors[status]
        with dr_cols[i]:
            figd, axd = plt.subplots(figsize=(2.4, 1.2), dpi=100, facecolor="#181920")
            axd.plot(df_live["Hour"].iloc[-96:], hist, color=color, lw=2)
            axd.axhline(np.mean(hist), color="#FFD700", ls="--", lw=1)
            axd.set_facecolor("#181920")
            axd.tick_params(axis='x', colors='white', labelsize=7)
            axd.tick_params(axis='y', colors='white', labelsize=7)
            for s in axd.spines.values(): s.set_color("#fff")
            axd.set_title(short, color="white", fontsize=12)
            figd.tight_layout()
            st.pyplot(figd)
            status_str = f"<span style='color:{ccol};font-weight:600'>{status}</span>"
            st.markdown(
                f"""<div class='custom-reco-box' style='padding:7px 11px;font-size:14px;border-left:3px solid {ccol}'>
                    {status_str}<br>
                    <span style='color:white;font-size:13px'>{driver_insights[feat]}</span>
                </div>""", unsafe_allow_html=True
            )
        combi_risk.append(status)
    # Combination driver interpretation
    combo_msg = ""
    if combi_risk.count("High")+combi_risk.count("Critical") > 1:
        combo_msg = "⚠️ <b>Multiple drivers elevated:</b> Reduce both COT and CCR together for quickest recovery."
    elif combi_risk.count("High")+combi_risk.count("Critical") == 1:
        combo_msg = "Tune single lever first, monitor if others rise."
    else:
        combo_msg = "All drivers optimal; maintain current settings."
    st.markdown(f"<div class='custom-reco-box' style='background:#282828'>{combo_msg}</div>", unsafe_allow_html=True)

    # ============= DRIVER STABILITY RADAR =============
    st.markdown("#### <span style='color:#FFD700'>Driver Stability Radar (Z-Score)</span>", unsafe_allow_html=True)
    radar_labels = [x[0] for x in driver_order]
    zscores = [((df_live[feat].iloc[-96:] - df_live[feat].iloc[-96:].mean()) / (df_live[feat].iloc[-96:].std()+1e-8)).iloc[-1] for feat,_,_ in driver_order]
    zscores.append(zscores[0]); radar_labels.append(radar_labels[0])
    radar_fig = go.Figure()
    radar_fig.add_trace(go.Scatterpolar(
        r=zscores, theta=radar_labels, fill='toself', line_color='#FFD700',
        marker=dict(size=10, color='#FFD700'), hoverinfo="r+theta"
    ))
    radar_fig.update_layout(
        polar=dict(bgcolor="#181920",
            radialaxis=dict(range=[-2.5,2.5], tickvals=[-2,0,2], ticktext=["Low","Optimal","High"],
            visible=True, showticklabels=True, showline=True, gridcolor="white")),
        showlegend=False, margin=dict(l=30, r=30, t=30, b=30), paper_bgcolor="#181920", font_color="white"
    )
    st.plotly_chart(radar_fig, use_container_width=True)
    st.markdown(
        """<div class='custom-reco-box' style="margin:6px 0 14px 0;">
            <span class='custom-reco-icon'>📊</span>
            <b>Interpretation:</b> Center = optimal. Drivers with |Z|>1 (yellow) may drift; |Z|>2 (red) = urgent tuning.
        </div>""", unsafe_allow_html=True
    )

    # ============= DRIVER ADDITIVE IMPACT (WATERFALL) =============
    st.markdown("#### <span style='color:#FFD700'>Driver Additive Impact (Excursion Risk)</span>", unsafe_allow_html=True)
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch
    
    # --- Data setup as before ---
    impacts = {
        "Coil Outlet Temp": 0.62*(latest["Coil Outlet Temp"]-486),
        "Feed CCR": 1.08*(latest["Feed CCR"]-19),
        "Air_Pressure_Ratio": 5.3*(latest["Air_Pressure_Ratio"]-1.06),
        "Air Flow": -0.011*(latest["Air Flow"]-1510),
        "Furnace Pressure": 2.1*(latest["Furnace Pressure"]-5.0)
    }
    base = float(df_live["Skin Temp"].iloc[-2])
    driver_names = [x[0] for x in driver_order]
    driver_colors = {x[0]: x[2] for x in driver_order}
    labels = driver_names
    values = [impacts[k] for k in labels]
    colors = [driver_colors[k] for k in labels]
    
    # Compute cumulative positions for the waterfall
    starts = [base]
    for v in values[:-1]:
        starts.append(starts[-1] + v)
    ends = [s + v for s, v in zip(starts, values)]
    
    fig, ax = plt.subplots(figsize=(11, 3.7), dpi=100, facecolor="#181920")
    ax.set_facecolor("#181920")
    
    # Plot each driver bar, labels below each bar
    for i, (label, value, color) in enumerate(zip(labels, values, colors)):
        left = starts[i]
        width = value
        ax.barh(0, width, left=left, color=color, alpha=0.87, height=0.54, edgecolor='white', zorder=2)
        # Label below each bar
        ax.text(left + width/2, -0.22,
                f"{label}\n{value:+.2f}",
                ha='center', va='top',
                fontsize=11, fontweight=600,
                color="white", zorder=4,
                bbox=dict(facecolor="#232323", edgecolor=color, boxstyle="round,pad=0.22", alpha=0.90)
               )
    
    # Base and total
    ax.text(base-0.42, 0.10, "Base", fontsize=12, color="#FFD700", ha="right", va="center", fontweight=700, zorder=4)
    ax.scatter([base], [0], color="#FFD700", s=68, marker="D", zorder=5)
    final = ends[-1]
    ax.text(final+0.65, 0.08, f"Total: {final:.1f}°C", va="center", ha="left", fontsize=15, fontweight=700, color="#FFD700", zorder=5)
    
    # Formatting
    ax.set_yticks([])
    ax.set_xlabel("Skin Temp (°C)", color="white", fontsize=14, labelpad=14)
    for spine in ax.spines.values(): spine.set_color("white")
    ax.set_xlim(base-1, final+3)
    ax.set_title("Additive Impact of Each Driver", fontsize=20, color="#FFD700", pad=10, loc='left')
    
    # Grid for clarity
    plt.grid(axis='x', color="#888", lw=0.4, linestyle='--', alpha=0.45, zorder=0)
    
    # Legend, title in white, outside right
    legend_handles = [Patch(color=driver_colors[k], label=k) for k in labels]
    legend = ax.legend(
        handles=legend_handles, loc='center left', bbox_to_anchor=(1.01, 0.55),
        fontsize=12, frameon=False, labelcolor='white', title="Driver"
    )
    legend.get_title().set_color('white')
    
    # Largest driver annotation, arrow goes to the bottom of the bar
    biggest_idx = np.argmax(np.abs(values))
    arrow_x = starts[biggest_idx] + values[biggest_idx]/2
    ax.annotate(
        f"Largest Driver:\n{labels[biggest_idx]}",
        xy=(arrow_x, 0), xycoords='data',
        xytext=(arrow_x, 0.32),
        textcoords='data',
        arrowprops=dict(arrowstyle="->", color="#FFD700", lw=2, alpha=0.9),
        fontsize=13, color="#FFD700", fontweight=800, bbox=dict(facecolor="#232323", alpha=0.75)
    )
    
    plt.tight_layout(rect=[0,0.08,0.93,1])
    
    # --- INTERPRETATION/INSIGHT BOX BELOW CHART ---
    dominant = labels[biggest_idx]
    direction = "increases" if values[biggest_idx]>0 else "decreases"
    insight_text = (
        f"<b>Interpretation:</b> <span style='color:#FFD700'>"
        f"The largest driver is <b style='color:{driver_colors[dominant]}'>{dominant}</b>, "
        f"which {direction} Skin Temp by <b>{values[biggest_idx]:+.2f}°C</b> from the base.<br>"
        "</span>The sum of all bars gives the predicted Skin Temp.<br>"
        "<b>Prioritize controlling the largest impact driver to reduce risk.</b>"
    )
    
    # Show with Streamlit
    st.pyplot(fig)
    st.markdown(
        f"""<div style='background:#232323; border-left:6px solid #FFD700; border-radius:10px; margin-top:8px; padding:14px 20px; color:white; font-size:1.13rem;'>
            {insight_text}
        </div>""", unsafe_allow_html=True
    )
    

    # ============= PRODUCT YIELDS (BOXED, SORTED) =============
    st.markdown("#### <span style='color:#FFD700'>Product Yields (Last 12 h)</span>",unsafe_allow_html=True)
    prod = ["LPG","Naphtha","Diesel","Fuel Oil","Coke"]
    avg12 = df_live[prod].iloc[-12:].mean()*100
    sorted_prod = sorted(zip(prod, avg12), key=lambda x: -x[1])
    prod_names = [x[0] for x in sorted_prod]
    prod_vals = [x[1] for x in sorted_prod]
    fig3,ax3=plt.subplots(figsize=(7,2.2),facecolor="#181920",dpi=100)
    bars=ax3.bar(prod_names,prod_vals,color=["#38D6AE","#FFD700","#9C27B0","#607D8B","#666"])
    for i,b in enumerate(bars):
        ax3.text(b.get_x()+b.get_width()/2,b.get_height()+1,f"{b.get_height():.1f}%",ha="center",color="white")
    ax3.set_facecolor("#181920"); ax3.tick_params(colors="white", labelcolor="white")
    ax3.set_ylabel("Yield (%)",color="white")
    for sp in ax3.spines.values(): sp.set_color("white")
    fig3.tight_layout()
    st.pyplot(fig3)
    st.markdown(
        "<div class='custom-reco-box'><span class='custom-reco-icon'>⚗️</span> Excursions raise Coke, reduce Diesel/Naphtha yields. Tune CCR and COT to optimize yield mix.</div>",
        unsafe_allow_html=True
    )

    # ============= WHAT-IF SIMULATOR (MULTI-METRIC, EFFICIENCY, ALL DRIVERS RECOMMENDATION) =============
    st.markdown("<hr style='border-top:2px solid #FFD700;margin:2rem 0'>",unsafe_allow_html=True)
    st.markdown("### <span style='color:#FFD700'>What-If Forecast Simulator</span>",unsafe_allow_html=True)
    st.markdown("<span style='color:#fff'>Adjust setpoints and see look-ahead skin-temp and yield. Only process controls an operator can set are shown.</span><br>",unsafe_allow_html=True)
    
    sim_steps = st.slider("Lookahead (h)", 3, 24, 8, key="S")
    sim_cols = st.columns(5)
    param_set = {
        "Coil Outlet Temp": sim_cols[0].slider("Coil Outlet Temp", 480.0, 500.0, float(latest["Coil Outlet Temp"]), 0.1),
        "Feed CCR": sim_cols[1].slider("Feed CCR", 14.0, 25.0, float(latest["Feed CCR"]), 0.1),
        "Air Flow": sim_cols[2].slider("Air Flow", 1300.0, 1700.0, float(latest["Air Flow"]), 1.0),
        "Air_Pressure_Ratio": sim_cols[3].slider("APR", 0.8, 1.25, float(latest["Air_Pressure_Ratio"]), 0.01),
        "Furnace Pressure": sim_cols[4].slider("Furnace Pressure", 4.0, 5.5, float(latest["Furnace Pressure"]), 0.01),
    }
    
    # --- Forecast (filtered base vs scenario) ---
    base_pred, scen_pred = scenario_forecast(df_live, latest, param_set, sim_steps)
    base_pred = base_pred + np.random.normal(0, 1.0, size=base_pred.shape)
    scen_pred = scen_pred + np.random.normal(0, 1.1, size=scen_pred.shape)
    
    # --- Simulate Yield ---
    scen_ccr = param_set["Feed CCR"]
    scen_yield = np.full(sim_steps, 78.4 - 3.7*(scen_ccr-19)/4.0) - (scen_pred > skin_thr)*3.3 + np.random.normal(0,0.5,sim_steps)
    base_ccr = float(latest["Feed CCR"])
    base_yield = np.full(sim_steps, 78.4 - 3.7*(base_ccr-19)/4.0) - (base_pred > skin_thr)*3.3 + np.random.normal(0,0.5,sim_steps)
    
    # --- Efficiency as "Stability Window (%)" ---
    sim_stab = int((scen_pred < skin_thr).sum())
    base_stab = int((base_pred < skin_thr).sum())
    sim_eff = 100.0 * sim_stab / sim_steps
    base_eff = 100.0 * base_stab / sim_steps
    
    # --- Plot: dual axis (Skin Temp, Yield) ---
    hours = np.arange(int(df_live["Hour"].iloc[-1])+1, int(df_live["Hour"].iloc[-1])+1+sim_steps)
    fig2,ax2=plt.subplots(figsize=(10,3),facecolor="#181920",dpi=100)
    ax2.axhspan(warn_thr,skin_thr,color="#FFD700",alpha=0.10)
    ax2.axhspan(skin_thr,560,color="#e74a3b",alpha=0.15)
    ax2.plot(hours,base_pred,"--o",color="#FFD700",lw=2,label="Base Skin Temp")
    ax2.plot(hours,scen_pred,"-^",color="#43d675",lw=2,label="Simulated Skin Temp")
    ax2.axhline(skin_thr,c="#e74a3b",ls="--",lw=2)
    ax2.axhline(warn_thr,c="#FFD700",ls="--",lw=1.5)
    ax2.set_facecolor("#181920")
    ax2.set_xlabel("Hour",color="white"); ax2.set_ylabel("Skin Temp (°C)",color="#FFD700")
    ax2.tick_params(colors="white", labelcolor="white")
    ax3 = ax2.twinx()
    ax3.plot(hours, base_yield, "--", color="#FFD700", lw=1.5, label="Base Yield")
    ax3.plot(hours, scen_yield, "-", color="#43d675", lw=1.5, label="Sim Yield")
    ax3.set_ylabel("Yield (%)", color="#43d675"); ax3.tick_params(colors="white")
    for s in list(ax2.spines.values()) + list(ax3.spines.values()): s.set_color("white")
    fig2.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=4, labelcolor="#fff", frameon=False)
    fig2.tight_layout(rect=[0,0,1,0.95])
    st.pyplot(fig2)
    
    # --- SIMULATED METRICS: WHITE BOXED KPIS (Eff, Exc, Avg Temp, Stab Window) ---
    st.markdown("#### <span style='color:#fff'>Simulated Scenario Metrics</span>", unsafe_allow_html=True)
    sim_exc = int((scen_pred > skin_thr).sum())
    base_exc = int((base_pred > skin_thr).sum())
    sim_mean = float(np.mean(scen_pred))
    base_mean = float(np.mean(base_pred))
    kpi_sim_metrics = [
        ("Sim Excursions", f"{sim_exc} <span style='font-size:14px;color:#bbb'>(base: {base_exc})</span>"),
        ("Avg Sim Skin Temp", f"{sim_mean:.1f}°C <span style='font-size:14px;color:#bbb'>(base: {base_mean:.1f})</span>"),
        ("Stability Window", f"{sim_stab}h <span style='font-size:14px;color:#bbb'>(out of {sim_steps})</span>"),
        ("Sim Efficiency", f"{sim_eff:.1f}% <span style='font-size:14px;color:#bbb'>(base: {base_eff:.1f}%)</span>"),
    ]
    mcols = st.columns(len(kpi_sim_metrics))
    for i,(title,val) in enumerate(kpi_sim_metrics):
        mcols[i].markdown(
            f"""<div class='kpi-box' style='background:#232323;border-left:4px solid #FFD700;'>
                <div class='metric-label'>{title}</div>
                <div class='metric-value' style='color:white;'>{val}</div>
            </div>""", unsafe_allow_html=True
        )
    
    # ---- Recommendations block ----
    
    def find_stabilizing_recos(df_live, latest, param_set, sim_steps, skin_thr=535):
        """Try each control. If excursions, recommend what to tweak. Otherwise, 'hold'."""
        control_ranges = {
            "Coil Outlet Temp": (480.0, 500.0, 0.5),
            "Feed CCR": (14.0, 25.0, 0.2),
            "Air Flow": (1300.0, 1700.0, 5.0),
            "Air_Pressure_Ratio": (0.8, 1.25, 0.01),
            "Furnace Pressure": (4.0, 5.5, 0.01),
        }
        actions = []
        for key in control_ranges:
            vnow = latest[key]
            minv, maxv, step = control_ranges[key]
            # Try reducing then increasing
            search_range = np.arange(minv, maxv+step, step)
            found = False
            for v in search_range if vnow>minv else search_range[::-1]:
                test_params = param_set.copy()
                test_params[key] = v
                _, scen_pred = scenario_forecast(df_live, latest, test_params, sim_steps)
                if np.all(scen_pred < skin_thr):
                    actions.append((key, vnow, v))
                    found = True
                    break
            if not found:
                actions.append((key, vnow, vnow))  # No improvement by single lever
        if all(abs(vnow-vfix)<1e-2 for _, vnow, vfix in actions):
            return ["No control parameter adjustment alone will stabilize. Decoking required or multi-lever intervention."]
        recos = []
        for key, vnow, vfix in actions:
            if abs(vnow-vfix) < 1e-2:
                continue
            direction = "decrease" if vfix < vnow else "increase"
            recos.append(f"{key}: <b style='color:#43d675'>{direction.capitalize()} to {vfix:.2f} ({vfix-vnow:+.2f})</b>")
        return recos if recos else ["Hold at current setpoints (already optimal or no further gain by single control)."]
    
    # --- DRIVER RECOMMENDATIONS / STABILITY MESSAGE ---
    reco_lines = []
    if sim_exc > 0:
        recos = find_stabilizing_recos(df_live, latest, param_set, sim_steps)
        reco_lines.extend(recos)
    else:
        for key in ["Coil Outlet Temp", "Feed CCR", "Air Flow", "Air_Pressure_Ratio", "Furnace Pressure"]:
            vnow = latest[key]
            vnew = param_set[key]
            delta = vnew - vnow
            if abs(delta) > 0.09:
                dirc = "decrease" if delta < 0 else "increase"
                color = "#43d675" if ((dirc=="decrease" and vnew<vnow) or (dirc=="increase" and vnew>vnow)) else "#e74a3b"
                stat = f"{key}: <b style='color:{color}'>{dirc.capitalize()} to {vnew:.2f} ({delta:+.2f})</b>"
            else:
                stat = f"{key}: <b style='color:#FFD700'>Hold at {vnow:.2f}</b>"
            reco_lines.append(stat)
    
    eff_msg = (
        f"Process efficiency: <b style='color:#FFD700'>{sim_eff:.1f}%</b> vs base <b style='color:#FFD700'>{base_eff:.1f}%</b>."
        + (" Improved." if sim_eff > base_eff else " Degraded—try reducing COT or CCR.")
    )
    sim_stab_msg = (
        f"Process is <b style='color:#43d675'>STABLE</b> for all {sim_steps}h in scenario." if sim_exc==0
        else f"Excursions predicted. Intervene within <b>{sim_steps-sim_stab}h</b> or decoke."
    )
    st.markdown(
        f"""<div class='custom-reco-box'>
            <span class='custom-reco-icon'>🛠️</span>
            <b>Driver Recommendations:</b><br>
            <ul class='custom-reco-list' style='margin-left:1.5em;'>
            {''.join(['<li>'+x+'</li>' for x in reco_lines])}
            </ul>
            <div style="margin-top:8px;font-size:14px;color:#FFD700">{sim_stab_msg}<br>{eff_msg}</div>
        </div>""", unsafe_allow_html=True
    )


# # ==================== TAB 2: PROCESS DIAGNOSTICS & ADVANCED ANALYTICS =========================
with tab2:
    import seaborn as sns
    from sklearn.ensemble import RandomForestRegressor
    from statsmodels.tsa.stattools import grangercausalitytests

    # -- HEADER
    st.markdown("""
    <div style="border-radius: 15px; background: linear-gradient(90deg, #232323 80%, #181920); padding: 22px 18px 15px 18px; margin-bottom: 18px; display: flex; align-items: center;">
        <span style="font-size:2.1rem; margin-right: 16px;">🧭</span>
        <div>
            <span style="font-size:1.35rem; color:#FFD700; font-weight:700; letter-spacing:0.5px;">Process Diagnostics & Analytics</span><br>
            <span style="font-size:1rem; color:#fff;">
                Deep-dive into feature relationships, control bands, risk overlays, driver importance, and causality.<br>
                All analytics styled for business clarity and dark mode.
            </span>
        </div>
        <span style="font-size:2.1rem; margin-left: auto;">📊</span>
    </div>
    """, unsafe_allow_html=True)

    # -- KPI Cards
    st.markdown("""
    <div style="display: flex; gap: 14px; margin-bottom: 12px;">
      <div class="kpi-box" style="flex:1;">
        <div class="metric-label">Current Skin Temp (°C)</div>
        <div class="metric-value" style="color:#FFD700;">{skin:.1f}</div>
      </div>
      <div class="kpi-box" style="flex:1;">
        <div class="metric-label">Yield (%)</div>
        <div class="metric-value" style="color:#43d675;">{yld:.2f}</div>
      </div>
      <div class="kpi-box" style="flex:1;">
        <div class="metric-label">Excursions (48h)</div>
        <div class="metric-value" style="color:#e74a3b;">{exc}</div>
      </div>
      <div class="kpi-box" style="flex:1;">
        <div class="metric-label">Run Length (h)</div>
        <div class="metric-value" style="color:#FFD700;">{runl}</div>
      </div>
    </div>
    """.format(
        skin=latest["Skin Temp"], yld=latest["Yield %"],
        exc=int(df_live["Excursion"].iloc[-48:].sum()),
        runl=int(latest["Run Length"])
    ), unsafe_allow_html=True)

    # -- CONTROL BAND HISTOGRAMS --
    st.markdown('<div style="color:#FFD700;font-size:1.13rem;font-weight:700;margin-top:10px;">Control Band Analysis</div>', unsafe_allow_html=True)
    driver_limits = {
        "Coil Outlet Temp": {"optimal": (484, 488), "high": (488, 491), "critical": (491, 560), "low": (478, 484)},
        "Feed CCR": {"optimal": (17, 21), "high": (21, 24), "critical": (24, 30), "low": (14, 17)},
        "Air Flow": {"optimal": (1450, 1600), "high": (1600, 1700), "critical": (1700, 1800), "low": (1300, 1450)},
        "Furnace Pressure": {"optimal": (4.7, 5.15), "high": (5.15, 5.5), "critical": (5.5, 7), "low": (4.0, 4.7)},
    }
    def hist_control_band_insight(feat):
        if feat == "Coil Outlet Temp":
            return "Operate 484–488°C for safety; >491°C risks excursion. Lower COT if high excursions."
        elif feat == "Feed CCR":
            return "Keep CCR <21 for stability. High CCR increases coke and risk."
        elif feat == "Air Flow":
            return "1450–1600 optimal. Low = poor transfer, high = instability."
        elif feat == "Furnace Pressure":
            return "Hold 4.7–5.15 for steady state. High = risk/upset."
        else:
            return "Monitor and keep within optimal band for process stability."
    cb_feats = st.multiselect(
        "Select features for control band analysis",
        list(driver_limits.keys()),
        default=["Coil Outlet Temp", "Feed CCR"],
        max_selections=3,
        key="cb_feats_diag2",
        label_visibility="visible"
    )
    cb_cols = st.columns(len(cb_feats) if cb_feats else 1)
    zone_colors = {"optimal": "#43d675", "high": "#FFD700", "critical": "#e74a3b", "low": "#3498db"}
    for idx, feat in enumerate(cb_feats):
        limits = driver_limits[feat]
        with cb_cols[idx]:
            fig_band, ax_band = plt.subplots(figsize=(2.7,1.1), dpi=100, facecolor="#181920")
            n, bins, patches = ax_band.hist(df_live[feat], bins=20, alpha=0.68, color="#FFD700", edgecolor="#fff", linewidth=0.55)
            for band, color in zone_colors.items():
                if band in limits:
                    l, h = limits[band]
                    ax_band.axvspan(l, h, color=color, alpha=0.15, label=band.capitalize())
            ax_band.set_xlabel(feat, color="white", fontsize=8)
            ax_band.set_ylabel("Count", color="white", fontsize=7)
            ax_band.tick_params(axis='x', colors='white', labelsize=7)
            ax_band.tick_params(axis='y', colors='white', labelsize=7)
            for spine in ax_band.spines.values(): spine.set_color("#fff")
            handles, labels = ax_band.get_legend_handles_labels()
            if handles:
                leg = ax_band.legend(handles, labels, loc='upper right', fontsize=6, frameon=True, facecolor="#232323", edgecolor="#fff", ncol=1)
                for text in leg.get_texts(): text.set_color("#fff")
            st.pyplot(fig_band)
            val = df_live[feat].iloc[-1]
            status = next((b for b in ["critical","high","optimal","low"] if b in limits and limits[b][0] <= val <= limits[b][1]), "Warning")
            st.markdown(
                f"""<div class="custom-reco-box" style="margin-top:2px;">
                    <span class="custom-reco-icon">🎯</span>
                    <span class="custom-reco-head">{feat} Status: <span style='color:{zone_colors.get(status,"#FFD700")}'>{status.capitalize()}</span></span>
                    <ul class="custom-reco-list"><li>{hist_control_band_insight(feat)}</li></ul>
                </div>""", unsafe_allow_html=True
            )

    # -- CROSS-FEATURE SCATTER OVERLAY --
    st.markdown('<div style="color:#FFD700;font-size:1.13rem;font-weight:700;margin-top:13px;">Cross-Feature Scatter Overlay</div>', unsafe_allow_html=True)
    feature_options = list(driver_limits.keys())
    scatter_x = st.selectbox(
        "Scatterplot X", feature_options, index=0, key="scatter_x_diag2",
        label_visibility="visible", help="Choose X-axis driver"
    )
    scatter_y = st.selectbox(
        "Scatterplot Y", [f for f in feature_options if f != scatter_x], index=1, key="scatter_y_diag2",
        label_visibility="visible", help="Choose Y-axis driver"
    )
    paint_by = st.radio("Color overlay by", ["Skin Temp", "Yield %"], index=0, horizontal=True, key="paint_diag2")
    fig_paint, ax_paint = plt.subplots(figsize=(4.2, 2.0), dpi=100, facecolor="#181920")
    vals = df_live[paint_by]
    sc = ax_paint.scatter(df_live[scatter_x], df_live[scatter_y], c=vals, cmap='plasma' if paint_by=="Skin Temp" else 'summer', s=22, alpha=0.55, edgecolor='none')
    cb = plt.colorbar(sc, ax=ax_paint, orientation="vertical", pad=0.02)
    cb.set_label(paint_by, color='white', fontsize=7)
    cb.ax.yaxis.set_tick_params(color='white', labelcolor='white', labelsize=7)
    ax_paint.set_xlabel(scatter_x, color="white", fontsize=8)
    ax_paint.set_ylabel(scatter_y, color="white", fontsize=8)
    ax_paint.tick_params(axis='x', colors='white', labelsize=7)
    ax_paint.tick_params(axis='y', colors='white', labelsize=7)
    for spine in ax_paint.spines.values(): spine.set_color("#fff")
    st.pyplot(fig_paint)
    high_zone = df_live[(df_live[scatter_x] > df_live[scatter_x].quantile(0.97)) & (df_live[scatter_y] > df_live[scatter_y].quantile(0.97))]
    msg = (
        f"High {paint_by} occurs when both {scatter_x} and {scatter_y} are at upper end. Control both to reduce risk."
        if high_zone.shape[0] else
        f"No strong co-occurrence of high {paint_by} for selected features—review other pairs for risk."
    )
    st.markdown(f"""
    <div class="custom-reco-box">
        <span class="custom-reco-icon">🎨</span>
        <span class="custom-reco-head">{scatter_x} vs {scatter_y} Overlay</span>
        <div style="font-size:0.97rem;">{msg}</div>
    </div>
    """, unsafe_allow_html=True)

    # -- CORRELATION HEATMAP (fully labeled, black background) --
    st.markdown('<div style="color:#FFD700;font-size:1.13rem;font-weight:700;margin-top:13px;">Correlation Heatmap</div>', unsafe_allow_html=True)
    corr_feats = list(dict.fromkeys([scatter_x, scatter_y, "Skin Temp", "Yield %", "Air Flow", "Feed CCR", "Furnace Pressure", "Coil Outlet Temp"]))
    corrmat = df_live[corr_feats].corr()
    fig_cm, ax_cm = plt.subplots(figsize=(4.2, 3), dpi=100, facecolor="#181920")
    sns.heatmap(
        corrmat, annot=True, fmt=".2f", cmap="coolwarm", center=0,
        cbar_kws={"shrink":0.68, "label": "Correlation"},
        ax=ax_cm, annot_kws={"fontsize":9, "color":"white"},
        linewidths=0.7, linecolor="#181920", square=True
    )
    ax_cm.set_title("Driver Correlation Matrix", color="#FFD700", fontsize=13, pad=8)
    ax_cm.set_facecolor("#181920")
    ax_cm.tick_params(axis='x', colors='white', labelsize=9)
    ax_cm.tick_params(axis='y', colors='white', labelsize=9)
    plt.setp(ax_cm.get_xticklabels(), rotation=45, ha="right", color="white", fontsize=9)
    plt.setp(ax_cm.get_yticklabels(), color="white", fontsize=9)
    # for i in range(len(corr_feats)):
    #     for j in range(len(corr_feats)):
    #         ax_cm.text(j + 0.5, i + 0.5, f"{corr_feats[i]}↔{corr_feats[j]}", ha='center', va='center', color="#FFD700", fontsize=6, alpha=0.57)
    for spine in ax_cm.spines.values(): spine.set_color("#fff")
    cbar = fig_cm.axes[-1]
    cbar.yaxis.label.set_color('white')
    cbar.tick_params(colors='white', labelsize=7)
    st.pyplot(fig_cm)
    cor_to_skin = corrmat["Skin Temp"].drop("Skin Temp").abs()
    cor_to_yield = corrmat["Yield %"].drop("Yield %").abs()
    top_skin_driver = cor_to_skin.idxmax()
    top_yield_driver = cor_to_yield.idxmax()
    st.markdown(f"""
    <div class="custom-reco-box">
        <span class="custom-reco-icon">🔗</span>
        <span class="custom-reco-head">Correlation Insight</span>
        <div style="font-size:0.97rem;">
            <b>Top driver for Skin Temp:</b> <span style='color:#FFD700'>{top_skin_driver} ({corrmat['Skin Temp'][top_skin_driver]:+.2f})</span><br>
            <b>Top driver for Yield %:</b> <span style='color:#43d675'>{top_yield_driver} ({corrmat['Yield %'][top_yield_driver]:+.2f})</span>
        </div>
        <div style="font-size:0.95rem;color:#FFD700">
            Focus process control on these variables for maximum impact. If either value is above 0.40, they are highly influential.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # -- GRANGER CAUSALITY TEST (plus chart) --
    st.markdown('<div style="color:#FFD700;font-size:1.13rem;font-weight:700;margin-top:13px;">Granger Causality Diagnostics</div>', unsafe_allow_html=True)
    granger_x = st.selectbox(
        "Candidate Driver (X)", feature_options, index=0, key="granger_x_diag2",
        label_visibility="visible", help="Potential cause"
    )
    granger_y = st.selectbox(
        "Target Metric (Y)", ["Skin Temp", "Yield %"], index=0, key="granger_y_diag2",
        label_visibility="visible", help="Potential effect"
    )
    gc_df_live = df_live[[granger_x, granger_y]].iloc[-350:].copy()
    try:
        maxlag = 4
        results = grangercausalitytests(gc_df_live, maxlag=maxlag, verbose=False)
        pvals = [round(results[i+1][0]["ssr_ftest"][1],4) for i in range(maxlag)]
        min_p = min(pvals)
        causality = min_p < 0.05
        color = "#43d675" if causality else "#e74a3b"
        lags = list(range(1,maxlag+1))
        fig_gc, ax_gc = plt.subplots(figsize=(3.2, 1.3), dpi=100, facecolor="#181920")
        ax_gc.plot(lags, pvals, marker="o", color="#FFD700", lw=2)
        ax_gc.axhline(0.05, color="#e74a3b", ls="--", lw=1.2)
        ax_gc.set_facecolor("#181920")
        ax_gc.set_xlabel("Lag", color="white", fontsize=8)
        ax_gc.set_ylabel("p-value", color="white", fontsize=8)
        ax_gc.tick_params(colors="white", labelsize=7)
        for spine in ax_gc.spines.values(): spine.set_color("#fff")
        st.pyplot(fig_gc)
        st.markdown(
            f"<b style='color:{color};font-size:1.02rem;'>p-value (min over lags): {min_p:.4f} — "
            f"{'Suggests' if causality else 'No'} significant Granger causality found.</b>",
            unsafe_allow_html=True
        )
    except Exception as e:
        st.warning(f"Granger causality unavailable ({e})")

    # -- FEATURE IMPORTANCE (RandomForest-style) --
    st.markdown('<div style="color:#FFD700;font-size:1.13rem;font-weight:700;margin-top:12px;">Driver Importance (Excursion Prediction)</div>', unsafe_allow_html=True)
    try:
        driver_feats = list(driver_limits.keys())
        X_explain = df_live[driver_feats].iloc[-300:]
        y_explain = (df_live["Skin Temp"].iloc[-300:] > 525).astype(int)
        model = RandomForestRegressor(n_estimators=60, random_state=42)
        model.fit(X_explain, y_explain)
        feat_imp = pd.Series(model.feature_importances_, index=driver_feats).sort_values(ascending=True)
        fig_shap, ax_shap = plt.subplots(figsize=(3.1,1.0), dpi=100, facecolor="#181920")
        ax_shap.barh(feat_imp.index, feat_imp.values, color="#FFD700")
        ax_shap.set_facecolor("#181920")
        ax_shap.set_xlabel("Driver Importance", color="white", fontsize=8)
        ax_shap.tick_params(axis='x', colors='white', labelsize=7)
        ax_shap.tick_params(axis='y', colors='white', labelsize=7)
        for spine in ax_shap.spines.values(): spine.set_color("#fff")
        st.pyplot(fig_shap)
        top_feat = feat_imp.idxmax()
        st.markdown(f"""
        <div class="custom-reco-box">
            <span class="custom-reco-icon">🧠</span>
            <span class="custom-reco-head">Feature Importance Insight</span>
            <div style="font-size:0.97rem;">
                <b>{top_feat}</b> is currently the biggest driver of excursions—focus on controlling this parameter.
            </div>
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Driver importance unavailable: {e}")

    # -- OVERALL AI Diagnostic Recommendation
    st.markdown('<div style="color:#FFD700;font-size:1.13rem;font-weight:700;margin-top:12px;">AI Diagnostic Recommendation</div>', unsafe_allow_html=True)
    if df_live["Skin Temp"].mean() > 517:
        rec = f"Skin temps are trending high. Focus on {top_feat if 'top_feat' in locals() else top_skin_driver} and {scatter_x}/{scatter_y} for stabilization."
    else:
        rec = "Skin temperature is stable. Maintain control bands for top features."
    st.markdown(f"""
    <div class="custom-reco-box">
        <span class="custom-reco-icon">🦾</span>
        <span class="custom-reco-head">Overall Diagnostic Recommendation</span>
        <div style="font-size:0.97rem;">{rec}</div>
    </div>
    """, unsafe_allow_html=True)

    # -- Apply style to dropdowns for black text on white
    st.markdown("""
    <style>
    .stSelectbox div[role='listbox'] span, .stSelectbox div[role='combobox'] span {
        color: black !important;
    }
    .stSelectbox div[role='combobox'], .stSelectbox div[role='listbox'] {
        background-color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)



    
# # ==================== TAB 3: CYCLE ANALYTICS & EVENTS =========================
with tab3:
    st.markdown("""
    <div style="border-radius: 15px; background: linear-gradient(90deg, #232323 80%, #181920); padding: 22px 18px 15px 18px; margin-bottom: 18px; display: flex; align-items: center;">
        <span style="font-size:2.1rem; margin-right: 16px;">🔁</span>
        <div>
            <span style="font-size:1.35rem; color:#FFD700; font-weight:700; letter-spacing:0.5px;">Cycle Analytics & Events</span><br>
            <span style="font-size:1rem; color:#fff;">
                Trend, compare, and understand cycle health—get disruption insights and actionable recommendations.
            </span>
        </div>
        <span style="font-size:2.1rem; margin-left: auto;">⏱️</span>
    </div>
    """, unsafe_allow_html=True)

    df = df_live.copy()
    # Cycle selection/filter (cycle id in your data)
    all_cycles = sorted(df['Cycle'].dropna().unique())
    current_cycle = int(df['Cycle'].iloc[-1])
    selected_cycle = st.selectbox("Select Cycle", all_cycles[::-1], index=0, format_func=lambda x: f"Cycle {x} (latest)" if x==current_cycle else f"Cycle {x}")

    # Data for selected cycle
    cyc_df = df[df['Cycle'] == selected_cycle].copy()
    # Summary over all cycles
    cyc_sum = df.groupby('Cycle').agg({
        'Hour': ['min', 'max', 'count'],
        'Skin Temp': ['max', 'mean', 'std'],
        'Yield %': 'mean',
        'Excursion': 'sum',
        'Zone': lambda x: x.mode().iloc[0] if not x.mode().empty else 'Normal'
    }).reset_index()
    cyc_sum.columns = [
        'Cycle', 'HourStart', 'HourEnd', 'CycleLength',
        'SkinTempMax', 'SkinTempMean', 'SkinTempStd',
        'YieldMean', 'ExcursionCount', 'ZoneMode'
    ]
    # Assign tags
    def assign_cycle_type(row):
        if row['ZoneMode'] == 'Critical' or row['SkinTempMax'] > skin_thr:
            return "Critical"
        elif row['ZoneMode'] == 'Warning' or row['SkinTempMax'] > warn_thr:
            return "Warning"
        elif row['ExcursionCount'] == 0 and row['YieldMean'] >= 77 and row['SkinTempMax'] < warn_thr:
            return "Good"
        else:
            return "Medium"
    cyc_sum['CycleType'] = cyc_sum.apply(assign_cycle_type, axis=1)
    # Cycle scoring
    run_score = (cyc_sum["CycleLength"] - cyc_sum["CycleLength"].min()) / (cyc_sum["CycleLength"].max() - cyc_sum["CycleLength"].min() + 1e-6)
    yield_score = (cyc_sum["YieldMean"] - cyc_sum["YieldMean"].min()) / (cyc_sum["YieldMean"].max() - cyc_sum["YieldMean"].min() + 1e-6)
    excursion_score = 1 - (cyc_sum["ExcursionCount"] / (cyc_sum["CycleLength"] + 1e-6))
    var_score = 1 - (cyc_sum["SkinTempStd"] - cyc_sum["SkinTempStd"].min()) / (cyc_sum["SkinTempStd"].max() - cyc_sum["SkinTempStd"].min() + 1e-6)
    cyc_sum["EfficiencyIndex"] = 0.35*run_score + 0.35*yield_score + 0.15*excursion_score + 0.15*var_score
    cyc_sum["DegradationIndex"] = (1 - run_score) * 0.6 + (cyc_sum["ExcursionCount"] / (cyc_sum["CycleLength"]+1e-6)) * 0.4

    cycle_type_colors = {"Critical":"#e74a3b", "Warning":"#FFD700", "Good":"#43d675", "Medium":"#9cb3db"}
    # -------- KPI CARD STRIP ----------
    sel_cyc_row = cyc_sum[cyc_sum['Cycle']==selected_cycle].iloc[0]
    st.markdown("""
    <div style="display: flex; gap: 13px; margin-bottom: 16px;">
      <div class="kpi-box" style="flex:1;">
        <div class="metric-label">Cycle Length (h)</div>
        <div class="metric-value" style="color:#FFD700;">{clen}</div>
      </div>
      <div class="kpi-box" style="flex:1;">
        <div class="metric-label">SkinTemp Max (°C)</div>
        <div class="metric-value" style="color:#e74a3b;">{skmax:.1f}</div>
      </div>
      <div class="kpi-box" style="flex:1;">
        <div class="metric-label">Yield Mean (%)</div>
        <div class="metric-value" style="color:#43d675;">{ymean:.2f}</div>
      </div>
      <div class="kpi-box" style="flex:1;">
        <div class="metric-label">Excursions</div>
        <div class="metric-value" style="color:#FFD700;">{exc}</div>
      </div>
      <div class="kpi-box" style="flex:1;">
        <div class="metric-label">Efficiency Index</div>
        <div class="metric-value" style="color:#43d675;">{eff:.2f}</div>
      </div>
    </div>
    """.format(
        clen=int(sel_cyc_row["CycleLength"]), skmax=sel_cyc_row["SkinTempMax"],
        ymean=sel_cyc_row["YieldMean"], exc=int(sel_cyc_row["ExcursionCount"]),
        eff=sel_cyc_row["EfficiencyIndex"]
    ), unsafe_allow_html=True)

    # --------- CYCLE METRIC TIMELINE (YIELD/SKIN TEMP/EXCURSIONS by cycle) ---------
    st.markdown('<div style="color:#FFD700;font-size:1.10rem;font-weight:700;">Cycle Timeline: Key Metrics</div>', unsafe_allow_html=True)
    cyc_fig, cyc_ax = plt.subplots(figsize=(8,2.4), dpi=100, facecolor="#181920")
    width = 0.19
    bar_x = np.arange(len(cyc_sum))
    cycle_labels = [f"C{c}" for c in cyc_sum['Cycle']]
    
    fig, ax1 = plt.subplots(figsize=(10, 2.9), dpi=100, facecolor="#181920")
    ax1.set_facecolor("#181920")
    
    # Plot Run Length as yellow line with markers
    ax1.plot(bar_x, cyc_sum["CycleLength"], color="#FFD700", lw=2, marker='o', markersize=7, label="Run Length")
    # Plot Excursions as red line with markers
    ax1.plot(bar_x, cyc_sum["ExcursionCount"], color="#e74a3b", lw=2, marker='s', markersize=7, label="Excursions")
    
    # Set labels and ticks
    ax1.set_xlabel("Cycle", color="white", fontsize=11, labelpad=6)
    ax1.set_ylabel("Run Length / Excursions", color="white", fontsize=11, labelpad=8)
    ax1.set_xticks(bar_x)
    ax1.set_xticklabels(cycle_labels, color="white", fontsize=10)
    ax1.tick_params(axis='y', colors='white', labelsize=10)
    ax1.tick_params(axis='x', colors='white', labelsize=10)
    for spine in ax1.spines.values():
        spine.set_color("#fff")
    
    # Add gridlines for clarity
    ax1.grid(axis='y', color="#555", alpha=0.14)
    
    # Legend
    leg = ax1.legend(loc='upper right', fontsize=10, facecolor="#232323", edgecolor="#fff", frameon=True)
    for text in leg.get_texts():
        text.set_color("#fff")
    
    # Optionally add data labels above each point (not needed if too many cycles)
    # for i, (x, y) in enumerate(zip(bar_x, cyc_sum["CycleLength"])):
    #     ax1.text(x, y+2, f"{int(y)}", color="#FFD700", fontsize=8, ha='center')
    # for i, (x, y) in enumerate(zip(bar_x, cyc_sum["ExcursionCount"])):
    #     ax1.text(x, y-2, f"{int(y)}", color="#e74a3b", fontsize=8, ha='center')
    
    plt.tight_layout()
    st.pyplot(fig)
    # bar_x = np.arange(len(cyc_sum))
    # cyc_ax.bar(bar_x-width, cyc_sum["CycleLength"], width=width, color="#FFD700", label="Run Length")
    # cyc_ax.bar(bar_x, cyc_sum["ExcursionCount"], width=width, color="#e74a3b", label="Excursions")
    # cyc_ax.bar(bar_x+width, cyc_sum["YieldMean"], width=width, color="#43d675", label="Yield (%)")
    # for spine in cyc_ax.spines.values(): spine.set_color("#fff")
    # cyc_ax.set_facecolor("#181920")
    # cyc_ax.set_xlabel("Cycle", color="white", fontsize=10)
    # cyc_ax.set_ylabel("Run/Excursions/Yield", color="white", fontsize=10)
    # cyc_ax.set_xticks(bar_x)
    # cyc_ax.set_xticklabels([f"C{c}" for c in cyc_sum['Cycle']], color="white", fontsize=8)
    # cyc_ax.tick_params(axis='y', colors='white', labelsize=8)
    # cyc_ax.legend(loc='upper right', fontsize=8, facecolor="#232323", edgecolor="#fff", frameon=True)
    # st.pyplot(cyc_fig)

    # --------- CYCLE PROFILE: SKIN TEMP & YIELD TIMELINE FOR SELECTED CYCLE ----------
    st.markdown('<div style="color:#FFD700;font-size:1.10rem;font-weight:700;">Selected Cycle: Skin Temp & Yield</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(8,2.3), dpi=100, facecolor="#181920")
    ax.plot(cyc_df["Hour"], cyc_df["Skin Temp"], color="#FFD700", lw=1.6, label="Skin Temp")
    ax2 = ax.twinx()
    ax2.plot(cyc_df["Hour"], cyc_df["Yield %"], color="#43d675", lw=1.1, ls="--", label="Yield %")
    ax.set_xlabel("Hour", color="white", fontsize=9)
    ax.set_ylabel("Skin Temp (°C)", color="#FFD700", fontsize=9)
    ax2.set_ylabel("Yield (%)", color="#43d675", fontsize=9)
    ax.tick_params(axis='x', colors='white', labelsize=8)
    ax.tick_params(axis='y', colors='#FFD700', labelsize=8)
    ax2.tick_params(axis='y', colors='#43d675', labelsize=8)
    ax.set_facecolor("#181920")
    for spine in ax.spines.values(): spine.set_color("#fff")
    for spine in ax2.spines.values(): spine.set_color("#fff")
    fig.tight_layout()
    st.pyplot(fig)

    # --------- CYCLE DEGRADATION TREND ----------
    st.markdown('<div style="color:#FFD700;font-size:1.10rem;font-weight:700;">Cycle Health & Degradation</div>', unsafe_allow_html=True)
    fig_deg, ax_deg = plt.subplots(figsize=(8,1.5), dpi=100, facecolor="#181920")
    ax_deg.plot(bar_x, cyc_sum["DegradationIndex"], color="#e74a3b", marker="o", lw=2)
    ax_deg.set_xlabel("Cycle", color="white", fontsize=9)
    ax_deg.set_ylabel("Degradation", color="#e74a3b", fontsize=9)
    ax_deg.set_facecolor("#181920")
    ax_deg.tick_params(axis='x', colors='white', labelsize=8)
    ax_deg.tick_params(axis='y', colors='#e74a3b', labelsize=8)
    ax_deg.set_xticks(bar_x)
    ax_deg.set_xticklabels([f"C{c}" for c in cyc_sum['Cycle']], color="white", fontsize=8)
    for spine in ax_deg.spines.values(): spine.set_color("#fff")
    st.pyplot(fig_deg)
    st.markdown("""
        <div class="custom-reco-box" style="margin-bottom:2px;">
            <span class="custom-reco-icon">🩺</span>
            <span class="custom-reco-head">Degradation Insight</span>
            <ul class="custom-reco-list">
                <li>Upward trend = loss of run length, higher excursions.</li>
                <li>Schedule maintenance proactively when trend rises.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    # --------- CYCLE EVENTS (Decoking/Maintenance) ----------
    st.markdown('<div style="color:#FFD700;font-size:1.10rem;font-weight:700;">Cycle Events Overlay</div>', unsafe_allow_html=True)
    fig_evt, ax_evt = plt.subplots(figsize=(8,1.7), dpi=100, facecolor="#181920")
    ax_evt.plot(df["Hour"], df["Skin Temp"], color="#FFD700", lw=1.6, label="Skin Temp")
    decoke_hours = df[df["Run Length"]==0]["Hour"].values
    ax_evt.scatter(decoke_hours, [df["Skin Temp"].iloc[h] if h in df.index else np.nan for h in decoke_hours], color="#e74a3b", marker="v", s=55, label="Decoking Event")
    for spine in ax_evt.spines.values(): spine.set_color("#fff")
    ax_evt.set_xlabel("Hour", color="white", fontsize=8)
    ax_evt.set_ylabel("Skin Temp (°C)", color="#FFD700", fontsize=8)
    ax_evt.tick_params(axis='x', colors='white', labelsize=7)
    ax_evt.tick_params(axis='y', colors='#FFD700', labelsize=7)
    ax_evt.set_facecolor("#181920")
    ax_evt.legend(loc='upper right', fontsize=7, facecolor="#232323", edgecolor="#fff")
    st.pyplot(fig_evt)

    # --------- CYCLE SUMMARY TABLE & HEALTH TAGS ----------
    st.markdown('<div style="color:#FFD700;font-size:1.10rem;font-weight:700;">Cycle Summary Table & Health Tags</div>', unsafe_allow_html=True)
    cyc_tbl = cyc_sum[["Cycle", "CycleType", "CycleLength", "SkinTempMax", "YieldMean", "ExcursionCount", "EfficiencyIndex", "DegradationIndex"]].copy()
    cyc_tbl["Comment"] = cyc_tbl.apply(
        lambda r: 
            "🚩 High Excursions/Degradation" if r["CycleType"]=="Critical" else
            ("⚠️ Watch excursions" if r["CycleType"]=="Warning" else
             ("👍 Efficient" if r["EfficiencyIndex"] > 0.8 else "Steady")), axis=1
    )
    def style_row(row):
        c = cycle_type_colors.get(row["CycleType"], "#fff")
        if row["CycleType"]=="Critical": return [f"color:{c};font-weight:bold" for _ in row]
        elif row["CycleType"]=="Warning": return [f"color:{c}" for _ in row]
        elif row["CycleType"]=="Good": return [f"color:{c}" for _ in row]
        return ["color:#fff" for _ in row]
    st.dataframe(
        cyc_tbl.style.apply(style_row, axis=1)\
            .format({"SkinTempMax":"{:.2f}", "YieldMean":"{:.2f}", "EfficiencyIndex":"{:.2f}", "DegradationIndex":"{:.2f}"})
    )

    # --------- INSIGHTS & OPERATOR RECOMMENDATIONS ----------
    top_cyc = cyc_tbl.loc[cyc_tbl["EfficiencyIndex"].idxmax()]
    worst_cyc = cyc_tbl.loc[cyc_tbl["EfficiencyIndex"].idxmin()]
    advice = []
    if worst_cyc["CycleType"]=="Critical":
        advice.append(f"<b>Cycle {int(worst_cyc['Cycle'])}:</b> <span style='color:#e74a3b'>Immediate decoking and review of feed/pressure needed. High unplanned shutdown risk.</span>")
    elif worst_cyc["CycleType"]=="Warning":
        advice.append(f"<b>Cycle {int(worst_cyc['Cycle'])}:</b> <span style='color:#FFD700'>Excursion rate rising, tighten process controls, plan maintenance.</span>")
    else:
        advice.append(f"<b>Cycle {int(worst_cyc['Cycle'])}:</b> <span style='color:#43d675'>Operation in safe band. Maintain best practices.</span>")
    advice.append(f"<b>Cycle {int(top_cyc['Cycle'])}:</b> <span style='color:#43d675'>Best cycle—reference for future optimization.</span>")
    # Add current cycle summary
    curr_comment = cyc_tbl.loc[cyc_tbl["Cycle"]==selected_cycle, "Comment"].values[0]
    st.markdown(
        f"""
        <div class="custom-reco-box">
            <span class="custom-reco-icon">💡</span>
            <span class="custom-reco-head">Operator Cycle Recommendations</span>
            <ul class="custom-reco-list">
                <li><b>Current cycle:</b> {curr_comment}</li>
                {''.join(f'<li>{ad}</li>' for ad in advice)}
            </ul>
        </div>
        """, unsafe_allow_html=True
    )




# with tab3:
#     st.markdown("""
#     <div style="border-radius: 15px; background: linear-gradient(90deg, #232323 80%, #181920); padding: 22px 18px 15px 18px; margin-bottom: 18px; display: flex; align-items: center;">
#         <span style="font-size:2.1rem; margin-right: 16px;">🔁</span>
#         <div>
#             <span style="font-size:1.35rem; color:#FFD700; font-weight:700; letter-spacing:0.5px;">Cycle Analytics & Events</span><br>
#             <span style="font-size:1rem; color:#fff;">
#                 Composite cycle efficiency index, timeline, yield, excursions, coke buildup, cycle health & simulation, event overlays, and actionable cycle-level recommendations.
#             </span>
#         </div>
#         <span style="font-size:2.1rem; margin-left: auto;">⏱️</span>
#     </div>
#     """, unsafe_allow_html=True)

#     # Link cycle-level metrics to real time-series
#     # Cycles are mapped by Run Length reset in the time-level data
#     df['CycleStart'] = (df['Run Length'] == 0).astype(int)
#     df['CycleID'] = df['CycleStart'].cumsum()
#     cycle_summaries = df.groupby('CycleID').agg({
#         'Hour': ['min', 'max', 'count'],
#         'Skin Temp': ['max', 'mean', 'std'],
#         'Yield %': 'mean',
#         'Excursion': 'sum',
#         'Fine Coke': 'mean',
#         'Zone': lambda x: x.mode().iloc[0] if not x.mode().empty else 'Normal'
#     }).reset_index()
#     cycle_summaries.columns = ['CycleID', 'HourStart', 'HourEnd', 'CycleLength', 
#                                'SkinTempMax', 'SkinTempMean', 'SkinTempStd',
#                                'YieldMean', 'ExcursionCount', 'FineCokeMean', 'ZoneMode']
#     # Assign cycle tags
#     def assign_cycle_type(row):
#         if row['ZoneMode'] == 'Critical' or row['SkinTempMax'] > skin_thr:
#             return "Critical"
#         elif row['ZoneMode'] == 'Warning' or row['SkinTempMax'] > warn_thr:
#             return "Warning"
#         elif row['ExcursionCount'] == 0 and row['YieldMean'] >= 77 and row['SkinTempMax'] < warn_thr:
#             return "Good"
#         else:
#             return "Medium"
#     cycle_summaries['CycleType'] = cycle_summaries.apply(assign_cycle_type, axis=1)
#     run_score = (cycle_summaries["CycleLength"] - cycle_summaries["CycleLength"].min()) / (cycle_summaries["CycleLength"].max() - cycle_summaries["CycleLength"].min() + 1e-6)
#     yield_score = (cycle_summaries["YieldMean"] - cycle_summaries["YieldMean"].min()) / (cycle_summaries["YieldMean"].max() - cycle_summaries["YieldMean"].min() + 1e-6)
#     excursion_score = 1 - (cycle_summaries["ExcursionCount"] / (cycle_summaries["CycleLength"] + 1e-6))
#     var_score = 1 - (cycle_summaries["SkinTempStd"] - cycle_summaries["SkinTempStd"].min()) / (cycle_summaries["SkinTempStd"].max() - cycle_summaries["SkinTempStd"].min() + 1e-6)
#     cycle_summaries["EfficiencyIndex"] = 0.35*run_score + 0.35*yield_score + 0.15*excursion_score + 0.15*var_score

#     # ---- Cycle Timeline Overlay ----
#     st.markdown('<div style="color:#FFD700;font-size:1.10rem;font-weight:700;">Cycle Timeline: Key Metrics</div>', unsafe_allow_html=True)
#     cyc_fig, cyc_ax = plt.subplots(figsize=(10,2.7), dpi=100, facecolor="#181920")
#     width = 0.26
#     bar_x = np.arange(len(cycle_summaries))
#     cyc_ax.bar(bar_x-width, cycle_summaries["CycleLength"], width=width, color="#FFD700", label="Run Length")
#     cyc_ax.bar(bar_x, cycle_summaries["ExcursionCount"], width=width, color="#e74a3b", label="Excursions")
#     cyc_ax.bar(bar_x+width, cycle_summaries["YieldMean"], width=width, color="#43d675", label="Yield (%)")
#     for spine in cyc_ax.spines.values(): spine.set_color("#fff")
#     cyc_ax.set_facecolor("#181920")
#     cyc_ax.set_xlabel("Cycle (Time-based)", color="white", fontsize=10)
#     cyc_ax.set_ylabel("Run / Excursions / Yield", color="white", fontsize=10)
#     cyc_ax.set_xticks(bar_x)
#     cyc_ax.set_xticklabels([f"C{c}" for c in cycle_summaries['CycleID']], color="white", fontsize=8)
#     cyc_ax.tick_params(axis='y', colors='white', labelsize=8)
#     cyc_ax.legend(loc='upper right', fontsize=8, facecolor="#232323", edgecolor="#fff", frameon=True)
#     st.pyplot(cyc_fig)

#     # Inline cycle narrative
#     max_cycle = cycle_summaries.loc[cycle_summaries["EfficiencyIndex"].idxmax()]
#     min_cycle = cycle_summaries.loc[cycle_summaries["EfficiencyIndex"].idxmin()]
#     st.markdown(
#         f"""<div class="custom-reco-box" style="margin-top:2px;">
#             <span class="custom-reco-icon">📊</span>
#             <div>
#                 <span class="custom-reco-head">Cycle Insights</span>
#                 <ul class="custom-reco-list">
#                     <li><b>Cycle {int(max_cycle['CycleID'])}</b> achieved highest efficiency index ({max_cycle['EfficiencyIndex']:.2f}).</li>
#                     <li><b>Cycle {int(min_cycle['CycleID'])}</b> underperformed ({min_cycle['EfficiencyIndex']:.2f})—review driver variables and events.</li>
#                 </ul>
#             </div>
#         </div>""", unsafe_allow_html=True
#     )

#     # ---- Skin Temp and Yield, Color by Cycle Type ----
#     st.markdown('<div style="color:#FFD700;font-size:1.10rem;font-weight:700;">Skin Temp & Yield by Cycle</div>', unsafe_allow_html=True)
#     fig, ax = plt.subplots(figsize=(10,2.4), dpi=100, facecolor="#181920")
#     for i, row in cycle_summaries.iterrows():
#         cycle_df = df[df['CycleID'] == row['CycleID']]
#         color = cycle_type_colors.get(row['CycleType'], "#FFD700")
#         ax.plot(cycle_df["Hour"], cycle_df["Skin Temp"], color=color, lw=1.1, alpha=0.7)
#     ax.set_xlabel("Hour", color="white", fontsize=9)
#     ax.set_ylabel("Skin Temp (°C)", color="#FFD700", fontsize=9)
#     ax.tick_params(axis='x', colors='white', labelsize=8)
#     ax.tick_params(axis='y', colors='#FFD700', labelsize=8)
#     for spine in ax.spines.values(): spine.set_color("#fff")
#     ax.set_facecolor("#181920")
#     st.pyplot(fig)

#     # ---- Cycle Degradation Trend ----
#     st.markdown('<div style="color:#FFD700;font-size:1.10rem;font-weight:700;">Cycle Health & Degradation</div>', unsafe_allow_html=True)
#     cycle_summaries["DegradationIndex"] = (1 - run_score) * 0.6 + (cycle_summaries["ExcursionCount"] / (cycle_summaries["CycleLength"]+1e-6)) * 0.4
#     fig_deg, ax_deg = plt.subplots(figsize=(10,1.7), dpi=100, facecolor="#181920")
#     ax_deg.plot(bar_x, cycle_summaries["DegradationIndex"], color="#e74a3b", marker="o", lw=2)
#     ax_deg.set_xlabel("Cycle", color="white", fontsize=9)
#     ax_deg.set_ylabel("Degradation", color="#e74a3b", fontsize=9)
#     ax_deg.set_facecolor("#181920")
#     ax_deg.tick_params(axis='x', colors='white', labelsize=8)
#     ax_deg.tick_params(axis='y', colors='#e74a3b', labelsize=8)
#     ax_deg.set_xticks(bar_x)
#     ax_deg.set_xticklabels([f"C{c}" for c in cycle_summaries['CycleID']], color="white", fontsize=8)
#     for spine in ax_deg.spines.values(): spine.set_color("#fff")
#     st.pyplot(fig_deg)
#     st.markdown(
#         """
#         <div class="custom-reco-box">
#             <span class="custom-reco-icon">🩺</span>
#             <div>
#                 <span class="custom-reco-head">Degradation Insight</span>
#                 <ul class="custom-reco-list">
#                     <li>Upward trend = loss of run length, higher excursions.</li>
#                     <li>Schedule maintenance proactively when trend rises.</li>
#                 </ul>
#             </div>
#         </div>
#         """, unsafe_allow_html=True
#     )

#     # ---- Cycle Events (Decoking/Maintenance) ----
#     st.markdown('<div style="color:#FFD700;font-size:1.10rem;font-weight:700;">Cycle Events Overlay</div>', unsafe_allow_html=True)
#     fig_evt, ax_evt = plt.subplots(figsize=(10,1.6), dpi=100, facecolor="#181920")
#     ax_evt.plot(df["Hour"], df["Skin Temp"], color="#FFD700", lw=1.8, label="Skin Temp")
#     decoke_hours = df[df["Run Length"]==0]["Hour"].values
#     ax_evt.scatter(decoke_hours, [df["Skin Temp"].iloc[h] for h in decoke_hours], color="#e74a3b", marker="v", s=55, label="Decoking Event")
#     for spine in ax_evt.spines.values(): spine.set_color("#fff")
#     ax_evt.set_xlabel("Hour", color="white", fontsize=8)
#     ax_evt.set_ylabel("Skin Temp (°C)", color="#FFD700", fontsize=8)
#     ax_evt.tick_params(axis='x', colors='white', labelsize=7)
#     ax_evt.tick_params(axis='y', colors='#FFD700', labelsize=7)
#     ax_evt.set_facecolor("#181920")
#     ax_evt.legend(loc='upper right', fontsize=7, facecolor="#232323", edgecolor="#fff")
#     st.pyplot(fig_evt)

#     # ---- Cycle Summary Table with Color Tags ----
#     st.markdown('<div style="color:#FFD700;font-size:1.10rem;font-weight:700;">Cycle Summary Table & Health Tags</div>', unsafe_allow_html=True)
#     cyc_tbl = cycle_summaries[["CycleID", "CycleType", "CycleLength", "SkinTempMax", "YieldMean", "ExcursionCount", "EfficiencyIndex", "DegradationIndex"]].copy()
#     cyc_tbl["Comment"] = cyc_tbl.apply(
#         lambda r: 
#             "🚩 High Excursions/Degradation" if r["CycleType"]=="Critical" else
#             ("⚠️ Watch excursions" if r["CycleType"]=="Warning" else
#              ("👍 Efficient" if r["EfficiencyIndex"] > 0.8 else "Steady")), axis=1
#     )
#     def style_row(row):
#         c = cycle_type_colors.get(row["CycleType"], "#fff")
#         if row["CycleType"]=="Critical": return [f"color:{c};font-weight:bold" for _ in row]
#         elif row["CycleType"]=="Warning": return [f"color:{c}" for _ in row]
#         elif row["CycleType"]=="Good": return [f"color:{c}" for _ in row]
#         return ["color:#fff" for _ in row]
#     st.dataframe(
#         cyc_tbl.style.apply(style_row, axis=1)\
#             .format({"SkinTempMax":"{:.2f}", "YieldMean":"{:.2f}", "EfficiencyIndex":"{:.2f}", "DegradationIndex":"{:.2f}"})
#     )

#     # ---- Cycle-Level Recommendations ----
#     st.markdown('<div style="color:#FFD700;font-size:1.10rem;font-weight:700;">Cycle-Level Recommendations</div>', unsafe_allow_html=True)
#     top_cyc = cyc_tbl.loc[cyc_tbl["EfficiencyIndex"].idxmax()]
#     worst_cyc = cyc_tbl.loc[cyc_tbl["EfficiencyIndex"].idxmin()]
#     advice = []
#     if worst_cyc["CycleType"]=="Critical":
#         advice.append(f"<b>Cycle {int(worst_cyc['CycleID'])}:</b> <span style='color:#e74a3b'>Immediate decoking and review of feed/pressure needed. High unplanned shutdown risk.</span>")
#     elif worst_cyc["CycleType"]=="Warning":
#         advice.append(f"<b>Cycle {int(worst_cyc['CycleID'])}:</b> <span style='color:#FFD700'>Excursion rate rising, tighten process controls, plan maintenance.</span>")
#     else:
#         advice.append(f"<b>Cycle {int(worst_cyc['CycleID'])}:</b> <span style='color:#43d675'>Operation in safe band. Maintain best practices.</span>")
#     advice.append(f"<b>Cycle {int(top_cyc['CycleID'])}:</b> <span style='color:#43d675'>Best cycle—reference for future optimization.</span>")
#     st.markdown(
#         f"""
#         <div class="custom-reco-box">
#             <span class="custom-reco-icon">💡</span>
#             <div>
#                 <span class="custom-reco-head">Cycle Recommendations</span>
#                 <ul class="custom-reco-list">
#                     {''.join(f'<li>{ad}</li>' for ad in advice)}
#                 </ul>
#             </div>
#         </div>
#         """, unsafe_allow_html=True
#     )









    





    








