# DCU Furnace Predictive Maintenance & Optimization Suite

An industrial-style Streamlit app for delayed coker unit (DCU) furnace monitoring, diagnostics, and scenario simulation.  
It provides live KPIs, forecasted skin temperature/yield, driver analytics, cycle health, and operator-ready recommendations.

## ✨ Features

- **Tab 1 – Live Furnace Status & What-If**
  - Real-time KPIs (Skin Temp, COT, Yield, CCR, APR)
  - Anomaly detection on skin temperature (rolling Z-score)
  - Look-ahead **ARIMA forecasting** with scenario controls (COT/CCR/Air/APR/Pressure)
  - Efficiency window, excursions, and comparative scenario KPIs
  - Driver stability radar and additive-impact waterfall (risk attribution)
  - Product yield summary (last 12h)

- **Tab 2 – Process Diagnostics & Analytics**
  - Control band histograms with **optimal/high/critical/low** overlays + status cards
  - Cross-feature scatter overlay colored by Skin Temp or Yield
  - Correlation heatmap with labeled matrix
  - **Granger causality** diagnostics
  - Random-Forest driver importance for excursion risk
  - AI-style summary recommendations

- **Tab 3 – Cycle Analytics & Events**
  - Cycle KPIs (length, SkinTempMax, YieldMean, Excursions, Efficiency Index)
  - Timeline of cycle run length & excursions
  - Selected cycle profile (Skin Temp & Yield)
  - Degradation index trend and decoking event overlays
  - Cycle summary table with health tags & operator recommendations

> A synthetic data generator is included to run the app end-to-end without external data.

---

## 🚀 Quickstart

### 1) Clone & install

```bash
git clone <your-repo-url>
cd <your-repo-folder>

# (optional) create a virtual env
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
