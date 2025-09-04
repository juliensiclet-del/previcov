# PRÉVI-COV — ETP (prévisions prudentes, historique réaliste)
# - Prévisions centrées sur moyenne mobile 12m + pente limitée (±0,05/mois)
# - Nettoyage historique (winsorisation + médiane glissante + bornes DSO/DPO)
# - Scénarios ETP symétriques (matières, retards, météo, taux, DSO≤60 / DPO≥60)
# - Règles métier : cap post-scénarios + override “situation idéale” + cohérence Dette/EBITDA→risque
# - UI 100% française

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import plotly.express as px

st.set_page_config(page_title="PRÉVI-COV — ETP (prévisions prudentes)", layout="wide")
st.title("PRÉVI-COV — ETP — Prévisions prudentes")

st.caption(
    "Prévisions à 12 mois **stabilisées** : moyenne mobile 12 mois + pente limitée. "
    "Objectif : des trajectoires crédibles, sans envolées artificielles."
)

# ----------------- Upload & démo -----------------
c1, c2 = st.columns(2)
with c1:
    fin_file = st.file_uploader("📥 Importer finance.csv", type=["csv"])
with c2:
    prj_file = st.file_uploader("📥 Importer projects.csv", type=["csv"])

if fin_file is None or prj_file is None:
    st.info("Mode démo (données réalistes). Chargez les vôtres pour vos simulations.")
    # Démo réaliste (alignée covenant : levier ≤ 3)
    dates = pd.date_range("2023-01-01", periods=36, freq="MS")
    rng = np.random.default_rng(42)
    net_debt = 3000 + rng.normal(0, 50, len(dates))
    ebitda   = 1200 + rng.normal(0, 30, len(dates))
    equity   = 4500 + rng.normal(0, 80, len(dates))
    debt_ebitda = np.clip(net_debt / np.maximum(ebitda, 1), 2.2, 2.9)
    gearing     = np.clip(net_debt / np.maximum(equity, 1), 1.2, 1.8)
    finance = pd.DataFrame({
        "date": dates,
        "net_debt": net_debt, "ebitda": ebitda, "equity": equity,
        "debt_ebitda": debt_ebitda, "gearing": gearing,
        "cash_net": 600 + rng.normal(0, 20, len(dates)),
        "dso": 55 + rng.normal(0, 5, len(dates)),
        "dpo": 65 + rng.normal(0, 4, len(dates)),
        "steel_index": 100 + rng.normal(0.4, 1.5, len(dates)).cumsum(),
        "rate": 1.5 + rng.normal(0, 0.05, len(dates)),
    })
    pr = []
    for d in dates:
        for pid in ["A12","B07","C03","D15"]:
            rp = rng.integers(300, 900)
            planned_margin = 12 + rng.normal(0, 1.0)
            actual_margin  = planned_margin + rng.normal(-0.3, 0.6)
            delay_weeks    = max(0, int(rng.normal(2, 1.5)))
            weather_days   = max(0, int(rng.normal(2, 1.5)))
            pr.append([d, pid, rp, planned_margin, actual_margin, delay_weeks, weather_days])
    projects = pd.DataFrame(pr, columns=[
        "date","project_id","rp","planned_margin_pct","actual_margin_pct","delay_weeks","weather_days"
    ])
else:
    finance = pd.read_csv(fin_file, parse_dates=["date"])
    projects = pd.read_csv(prj_file, parse_dates=["date"])

finance = finance.sort_values("date")
projects = projects.sort_values(["date","project_id"])

st.subheader("Aperçu des données")
st.dataframe(finance.tail(8), use_container_width=True)

# ----------------- Paramètres & scénarios -----------------
st.sidebar.header("⚙️ Paramètres covenants")
gearing_threshold = st.sidebar.number_input("Seuil Gearing (max)", 0.5, 5.0, 2.0, step=0.1, format="%.1f")
debt_ebitda_threshold = st.sidebar.number_input("Seuil Dette/EBITDA (max)", 1.0, 8.0, 3.0, step=0.1, format="%.1f")

st.sidebar.header("🎛️ Scénarios (métier ETP)")
shock_cost = st.sidebar.slider("Matières : variation des coûts [%]", -20, 50, 0, step=1)
shock_delay = st.sidebar.slider("Retards chantier [semaines]", -4, 26, 0, step=1)
shock_weather = st.sidebar.slider("Météo défavorable [jours/mois]", -5, 20, 0, step=1)
shock_rate = st.sidebar.slider("Variation des taux [points]", -2, 4, 0, step=1)
client_terms = st.sidebar.slider("Délai règlements clients (DSO) [jours] — viser ≤ 60", 30, 120, 60, step=5)
supplier_terms = st.sidebar.slider("Délai paiements fournisseurs (DPO) [jours] — viser ≥ 60", 30, 120, 60, step=5)

# ----------------- Utilitaires -----------------
def ensure_ratios(fin: pd.DataFrame) -> pd.DataFrame:
    out = fin.copy()
    if "gearing" not in out or out["gearing"].isna().all():
        if {"net_debt","equity"}.issubset(out.columns):
            out["gearing"] = out["net_debt"] / out["equity"].replace(0, np.nan)
        else:
            approx_equity = np.maximum(out.get("net_debt", pd.Series([2000]*len(out))), 1) / 1.6
            out["gearing"] = out.get("net_debt", pd.Series([2000]*len(out))) / approx_equity
    if "debt_ebitda" not in out or out["debt_ebitda"].isna().all():
        if {"net_debt","ebitda"}.issubset(out.columns):
            out["debt_ebitda"] = out["net_debt"] / out["ebitda"].replace(0, np.nan)
    return out

def winsorize(s: pd.Series, lower_q=0.02, upper_q=0.98) -> pd.Series:
    if s.dropna().empty:
        return s
    lo, hi = s.quantile(lower_q), s.quantile(upper_q)
    return s.clip(lower=lo, upper=hi)

def smooth_rolling_median(s: pd.Series, window=3) -> pd.Series:
    return s.rolling(window, min_periods=1, center=True).median()

def clean_finance(fin: pd.DataFrame) -> pd.DataFrame:
    df = fin.copy().sort_values("date")
    # Borne pédagogique : levier ≤ 3 en historique (cohérence covenant)
    if "debt_ebitda" in df:
        df["debt_ebitda"] = df["debt_ebitda"].clip(0.5, 3.0)
    # Plages métier + interpolation douce DSO/DPO
    for col in ["dso","dpo"]:
        if col in df:
            df[col] = df[col].clip(30, 120)
            jump = df[col].diff().abs() > 40
            df.loc[jump, col] = np.nan
            df[col] = df[col].interpolate(limit_direction="both")
    # Lissage & winsorisation sur ratios
    for col in ["debt_ebitda","gearing"]:
        if col in df:
            df[col] = smooth_rolling_median(df[col])
            df[col] = winsorize(df[col], 0.02, 0.98)
    return df

def conservative_forecast(series: pd.Series, periods: int = 12,
                          ma_window: int = 12, slope_cap: float = 0.05,
                          blend: float = 0.5) -> np.ndarray:
    s = series.dropna()
    if s.empty:
        return np.zeros(periods)
    anchor = float(s.tail(ma_window).mean()) if len(s) >= ma_window else float(s.tail(min(6, len(s))).median())
    last = float(s.iloc[-1])
    y = s.tail(min(ma_window, len(s))).values
    x = np.arange(len(y))
    if len(y) >= 2:
        a, b = np.polyfit(x, y, 1)
        slope = np.clip(a, -slope_cap, slope_cap)
    else:
        slope = 0.0
    base = last + slope * np.arange(1, periods+1)
    blended = blend * base + (1 - blend) * anchor
    return blended

def apply_etp_scenarios(gearing_vals, de_vals,
                        shock_cost, shock_delay_weeks, shock_weather_days, shock_rate_pts,
                        client_terms_days, supplier_terms_days):
    g_adj = (gearing_vals
             + 0.008*shock_cost
             + 0.03*(shock_delay_weeks/4)
             + 0.015*(shock_weather_days/5)
             + 0.03*shock_rate_pts)
    de_adj = (de_vals
              + 0.012*shock_cost
              + 0.025*(shock_delay_weeks/4)
              + 0.015*(shock_weather_days/5)
              + 0.025*shock_rate_pts)
    delta_dso = client_terms_days - 60
    delta_dpo = supplier_terms_days - 60
    g_adj = g_adj + 0.003*delta_dso - 0.003*delta_dpo
    de_adj = de_adj + 0.010*(delta_dso/10) - 0.010*(delta_dpo/10)
    return np.maximum(g_adj, 0.05), np.maximum(de_adj, 0.3)

# ----------------- Préparation -----------------
finance = ensure_ratios(finance)
finance = clean_finance(finance)

last_date = finance["date"].max()
h = 12
future_dates = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=h, freq="MS")

# ----------------- Prévisions prudentes -----------------
fcst = pd.DataFrame({"date": future_dates})
gearing_raw = conservative_forecast(finance.set_index("date")["gearing"], periods=h, slope_cap=0.03, blend=0.5)
de_raw      = conservative_forecast(finance.set_index("date")["debt_ebitda"], periods=h, slope_cap=0.05, blend=0.5)

fcst["gearing_fcst"]      = np.clip(gearing_raw, 0.0, gearing_threshold * 1.4)
fcst["debt_ebitda_fcst"]  = np.clip(de_raw,      0.0, debt_ebitda_threshold * 1.4)

# ----------------- Scénarios + cap post-scénarios -----------------
fcst["gearing_scn"], fcst["de_scn"] = apply_etp_scenarios(
    fcst["gearing_fcst"].values, fcst["debt_ebitda_fcst"].values,
    shock_cost, shock_delay, shock_weather, shock_rate, client_terms, supplier_terms
)
fcst["gearing_scn"] = np.clip(fcst["gearing_scn"], 0.0, gearing_threshold * 1.5)
fcst["de_scn"]      = np.clip(fcst["de_scn"],      0.0, debt_ebitda_threshold * 1.5)

# ----------------- Probabilité de bris (12 mois) -----------------
tmp = finance.copy()
tmp["breach_gearing"]      = (tmp["gearing"].rolling(12, min_periods=1).max() > gearing_threshold).shift(-11).fillna(False)
tmp["breach_debt_ebitda"]  = (tmp["debt_ebitda"].rolling(12, min_periods=1).max() > debt_ebitda_threshold).shift(-11).fillna(False)
tmp["breach_any"]          = (tmp["breach_gearing"] | tmp["breach_debt_ebitda"]).astype(int)

feat = tmp[["gearing","debt_ebitda","cash_net","dso","dpo","rate"]].copy()
for col in ["gearing","debt_ebitda","cash_net"]:
    feat[f"{col}_chg_3m"] = feat[col].diff(3)
    feat[f"{col}_chg_6m"] = feat[col].diff(6)
feat = feat.fillna(0.0)
X = feat.values
y = tmp["breach_any"].values

if y.sum() > 3 and len(finance) >= 30:
    split_idx = len(X) - 6
    X_train, y_train = X[:split_idx], y[:split_idx]
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("gb", GradientBoostingClassifier(n_estimators=250, learning_rate=0.05, max_depth=3, random_state=42))
    ])
    model.fit(X_train, y_train)
    cal = CalibratedClassifierCV(model, method="isotonic", cv=3)
    cal.fit(X_train, y_train)
    prob_reference = float(cal.predict_proba(feat.iloc[-1:].values)[:,1])
else:
    prob_reference = 0.10  # référence si peu d'historique

# Uplift (points de probabilité) + DSO/DPO
uplift = (
    0.005*shock_cost +
    0.008*(shock_delay/4) +
    0.006*(shock_weather/5) +
    0.025*shock_rate
)
uplift += 0.0025*(client_terms - 60) - 0.0025*(supplier_terms - 60)

prob_scenario = float(np.clip(prob_reference + uplift, 0.0, 0.99))

# Override “situation idéale”
ideal = (shock_cost <= 0 and shock_delay <= 0 and shock_weather <= 0 and shock_rate <= 0
         and client_terms <= 60 and supplier_terms >= 60)
within_buffers = (
    (float(np.nanmax(fcst["gearing_scn"])) <= gearing_threshold * 0.95) and
    (float(np.nanmax(fcst["de_scn"]))     <= debt_ebitda_threshold * 0.95)
)
if ideal and within_buffers:
    prob_scenario = min(prob_scenario, 0.05)

# Cohérence métier Dette/EBITDA
max_de = float(np.nanmax(fcst["de_scn"]))
if max_de > debt_ebitda_threshold:
    prob_scenario = min(0.99, prob_scenario + 0.12)
elif max_de < debt_ebitda_threshold * 0.8:
    prob_scenario = max(0.01, prob_scenario - 0.08)

# ----------------- Graphiques -----------------
col1, col2 = st.columns(2)
with col1:
    st.subheader("Gearing — Historique & Prévision (scénario)")
    hist = finance[["date","gearing"]].rename(columns={"gearing":"value"}); hist["type"]="Historique"
    fut  = fcst[["date","gearing_scn"]].rename(columns={"gearing_scn":"value"}); fut["type"]="Prévision (scénario)"
    fig = px.line(pd.concat([hist, fut]), x="date", y="value", color="type")
    fig.add_hline(y=gearing_threshold)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Dette/EBITDA — Historique & Prévision (scénario)")
    hist = finance[["date","debt_ebitda"]].rename(columns={"debt_ebitda":"value"}); hist["type"]="Historique"
    fut  = fcst[["date","de_scn"]].rename(columns={"de_scn":"value"}); fut["type"]="Prévision (scénario)"
    fig = px.line(pd.concat([hist, fut]), x="date", y="value", color="type")
    fig.add_hline(y=debt_ebitda_threshold)
    st.plotly_chart(fig, use_container_width=True)

st.caption("Prévisions **prudemment ancrées** (moyenne mobile 12m) avec **pente limitée** ; cap léger post-scénarios. "
           "Historique borné à ≤ 3 pour respecter le covenant de levier.")

st.markdown("---")
st.subheader("🛑 Probabilité de bris (12 mois)")
c1, c2, c3 = st.columns(3)
c1.metric("Probabilité (référence)", f"{prob_reference:.0%}")
c2.metric("Probabilité (scénario)", f"{prob_scenario:.0%}", delta=f"{(prob_scenario-prob_reference):+.0%}")
statut = "🟢 Sûr" if prob_scenario < 0.30 else ("🟠 Sous surveillance" if prob_scenario < 0.60 else "🔴 Risque élevé")
c3.metric("Statut", statut)

st.info(
"Astuce : si vos courbes historiques oscillent dans une fourchette serrée, ces prévisions resteront **stables et crédibles**. "
"Pour des scénarios extrêmes, ajustez la **pente max** (`slope_cap`) ou le **cap** post-scénarios."
)

st.caption("Dépendances : streamlit, pandas, numpy, scikit-learn, plotly.")
