# PRÉVI-COV — ETP (version avancée : Prophet -> SARIMAX -> Linéaire, risque calibré, scénarios ETP complets)
# - Ratios suivis : Gearing (max 2,0) et Dette/EBITDA (max 3,0)
# - Prévisions : Prophet (priorité), fallback SARIMAX, fallback linéaire local
# - Modèle de risque : Gradient Boosting + calibration (si historique suffisant), sinon probabilité de référence
# - Scénarios symétriques : matières, retards, météo, taux, DSO (viser ≤60), DPO (viser ≥60)
# - Override "situation idéale" : si aucun choc et DSO<=60 / DPO>=60 et ratios < seuils (coussin), risque capé à 5%
# - UI en français

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss
import plotly.express as px

st.set_page_config(page_title="PRÉVI-COV — ETP (avancé)", layout="wide")
st.title("PRÉVI-COV — ETP (Console IA de gestion prédictive des covenants) — Version avancée")

st.caption(
    "Importez vos **données financières mensuelles** (finance.csv) et votre **suivi chantiers** (projects.csv). "
    "Prévisions avancées **Prophet → SARIMAX → Linéaire**, scénarios ETP symétriques, probabilité de bris calibrée."
)

with st.expander("📄 Formats attendus", expanded=False):
    st.markdown("""
**1) finance.csv (mensuel, 1 ligne / mois)**  
- `date` (YYYY-MM-01)  
- `gearing` **ou** `net_debt` & `equity`  
- `debt_ebitda` **ou** `net_debt` & `ebitda`  
- `ebitda`, `net_debt`, `equity` (recommandé), `cash_net`, `dso`, `dpo`, `steel_index` (optionnel), `rate` (taux moyen)

**2) projects.csv (suivi chantiers, 1 ligne / chantier / mois)**  
- `date`, `project_id`, `rp` (reste à produire k€),  
- `planned_margin_pct`, `actual_margin_pct` (si dispo),  
- `delay_weeks`, `weather_days` (jours perdus/mois, optionnel)
""")

# ---------- Upload ----------
c1, c2 = st.columns(2)
with c1:
    fin_file = st.file_uploader("📥 Importer finance.csv", type=["csv"], key="fin")
with c2:
    prj_file = st.file_uploader("📥 Importer projects.csv", type=["csv"], key="prj")

# Mode démo si fichiers absents
if fin_file is None or prj_file is None:
    st.info("Aucun fichier fourni : **mode démo**.")
    dates = pd.date_range("2023-01-01", periods=36, freq="MS")
    rng = np.random.default_rng(42)
    equity = 4000 + rng.normal(0, 60, len(dates)).cumsum()
    net_debt = 3500 + rng.normal(0, 70, len(dates))
    ebitda = 1000 + rng.normal(0, 40, len(dates)).cumsum()
    rate = 1.5 + rng.normal(0, 0.05, len(dates))
    steel_index = 100 + rng.normal(0.4, 1.5, len(dates)).cumsum()
    debt_ebitda = np.clip(net_debt / np.maximum(ebitda, 1), 0.8, 6.0)
    gearing = np.clip(net_debt / np.maximum(equity, 1), 0.2, 4.0)
    dso = 60 + rng.normal(0, 5, len(dates))
    dpo = 60 + rng.normal(0, 5, len(dates))
    finance = pd.DataFrame({
        "date": dates, "equity": equity, "net_debt": net_debt, "ebitda": ebitda,
        "gearing": gearing, "debt_ebitda": debt_ebitda,
        "cash_net": 500 + rng.normal(0, 25, len(dates)),
        "dso": dso, "dpo": dpo, "steel_index": steel_index, "rate": rate
    })
    pr = []
    for d in dates:
        for pid in ["A12","B07","C03","D15"]:
            rp = rng.integers(300, 900)
            planned_margin = 12 + rng.normal(0, 1.2)
            actual_margin = planned_margin + rng.normal(-0.4, 0.7)
            delay_weeks = max(0, int(rng.normal(3, 2)))
            weather_days = max(0, int(rng.normal(2, 2)))
            pr.append([d, pid, rp, planned_margin, actual_margin, delay_weeks, weather_days])
    projects = pd.DataFrame(pr, columns=["date","project_id","rp","planned_margin_pct","actual_margin_pct","delay_weeks","weather_days"])
else:
    finance = pd.read_csv(fin_file, parse_dates=["date"])
    projects = pd.read_csv(prj_file, parse_dates=["date"])

finance = finance.sort_values("date")
projects = projects.sort_values(["date","project_id"])

st.subheader("Aperçu des données")
st.dataframe(finance.tail(6), use_container_width=True)
st.dataframe(projects.tail(6), use_container_width=True)

# ---------- Paramètres covenants ----------
st.sidebar.header("⚙️ Paramètres covenants")
gearing_threshold = st.sidebar.number_input("Seuil Gearing (max)", 0.5, 5.0, 2.0, step=0.1, format="%.1f")
debt_ebitda_threshold = st.sidebar.number_input("Seuil Dette/EBITDA (max)", 1.0, 8.0, 3.0, step=0.1, format="%.1f")

# ---------- Scénarios ----------
st.sidebar.header("🎛️ Scénarios (métier ETP)")
shock_cost = st.sidebar.slider("Matières : variation des coûts [%]", -20, 50, 0, step=1)
shock_delay = st.sidebar.slider("Retards chantier [semaines]", -4, 26, 0, step=1)
shock_weather = st.sidebar.slider("Météo défavorable [jours/mois]", -5, 20, 0, step=1)
shock_rate = st.sidebar.slider("Variation des taux [points]", -2, 4, 0, step=1)

client_terms = st.sidebar.slider("Délai règlements clients (DSO) [jours] — viser ≤ 60", 30, 120, 60, step=5)
supplier_terms = st.sidebar.slider("Délai paiements fournisseurs (DPO) [jours] — viser ≥ 60", 30, 120, 60, step=5)

# ---------- Utilitaires ----------
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

def forecast_with_fallback(series: pd.Series, periods: int = 12) -> np.ndarray:
    """
    1) Prophet (saisonnalité annuelle) -> 2) SARIMAX (m=12) -> 3) Linéaire local / moyenne récente
    """
    s = series.copy().dropna()
    s.index = pd.DatetimeIndex(s.index)  # assure un index temporel
    # Prophet
    try:
        from prophet import Prophet
        if len(s) >= 6:
            dfp = pd.DataFrame({"ds": s.index, "y": s.values})
            m = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=True)
            m.fit(dfp)
            fut = m.make_future_dataframe(periods=periods, freq="MS")
            fc = m.predict(fut)
            return fc["yhat"].iloc[-periods:].values
    except Exception:
        pass
    # SARIMAX
    try:
        import statsmodels.api as sm
        if len(s) >= 18:
            model = sm.tsa.statespace.SARIMAX(s, order=(1,1,1), seasonal_order=(1,1,1,12), enforce_stationarity=False, enforce_invertibility=False)
            res = model.fit(disp=False)
            fc = res.get_forecast(steps=periods).predicted_mean.values
            return fc
    except Exception:
        pass
    # Linéaire / moyenne
    if len(s) < 18:
        last = float(s.tail(6).mean()) if len(s) >= 6 else float(s.tail(1))
        return np.repeat(last, periods)
    y = s.values[-12:]
    x = np.arange(len(y))
    coef = np.polyfit(x, y, 1)
    return coef[0]*np.arange(len(y), len(y)+periods) + coef[1]

def project_weighted_margin(projects_month: pd.DataFrame) -> float:
    df = projects_month.copy()
    if df.empty: return np.nan
    w = df["rp"].clip(lower=0.0)
    base_margin = np.where(df["actual_margin_pct"].notna(), df["actual_margin_pct"], df["planned_margin_pct"])
    return float((w * base_margin).sum() / max(w.sum(), 1e-6))

# Élasticités équilibrées + effets symétriques (valeurs pédagogiques)
def apply_etp_scenarios(gearing_vals, de_vals, portfolio_margin,
                        shock_cost, shock_delay_weeks, shock_weather_days, shock_rate_pts,
                        client_terms_days, supplier_terms_days):
    g_adj = (gearing_vals
             + 0.01*shock_cost
             + 0.04*(shock_delay_weeks/4)
             + 0.02*(shock_weather_days/5)
             + 0.04*shock_rate_pts)
    de_adj = (de_vals
              + 0.015*shock_cost
              + 0.03*(shock_delay_weeks/4)
              + 0.02*(shock_weather_days/5)
              + 0.03*shock_rate_pts)
    # DSO/DPO par rapport à 60 (DSO>60 dégrade ; DPO<60 dégrade)
    delta_dso = client_terms_days - 60
    delta_dpo = supplier_terms_days - 60
    g_adj = g_adj + 0.003*delta_dso - 0.003*delta_dpo
    de_adj = de_adj + 0.010*(delta_dso/10) - 0.010*(delta_dpo/10)
    # Marge indicative
    margin_adj = portfolio_margin - (1.0*(shock_cost/10.0) + 0.4*(shock_delay_weeks/4.0) + 0.2*(shock_weather_days/5.0))
    return np.maximum(g_adj, 0.05), np.maximum(de_adj, 0.3), margin_adj

# ---------- Préparation des ratios ----------
finance = ensure_ratios(finance)
last_date = finance["date"].max()
h = 12
future_dates = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=h, freq="MS")

# Prévisions avancées
fcst = pd.DataFrame({"date": future_dates})
fcst["gearing_fcst"] = forecast_with_fallback(finance.set_index("date")["gearing"], periods=h)
fcst["debt_ebitda_fcst"] = forecast_with_fallback(finance.set_index("date")["debt_ebitda"], periods=h)

current_month = projects[projects["date"] == projects["date"].max()]
port_margin_now = project_weighted_margin(current_month)

# Application scénarios
fcst["gearing_scn"], fcst["de_scn"], fcst["margin_portfolio_scn"] = apply_etp_scenarios(
    gearing_vals=fcst["gearing_fcst"].values,
    de_vals=fcst["debt_ebitda_fcst"].values,
    portfolio_margin=port_margin_now if not np.isnan(port_margin_now) else 12.0,
    shock_cost=shock_cost, shock_delay_weeks=shock_delay, shock_weather_days=shock_weather, shock_rate_pts=shock_rate,
    client_terms_days=client_terms, supplier_terms_days=supplier_terms
)

# ---------- Probabilité de bris à 12 mois ----------
tmp = finance.copy()
tmp["breach_gearing"] = (tmp["gearing"].rolling(12, min_periods=1).max() > gearing_threshold).shift(-11).fillna(False)
tmp["breach_debt_ebitda"] = (tmp["debt_ebitda"].rolling(12, min_periods=1).max() > debt_ebitda_threshold).shift(-11).fillna(False)
tmp["breach_any"] = (tmp["breach_gearing"] | tmp["breach_debt_ebitda"]).astype(int)

# Features
feat = tmp[["gearing","debt_ebitda","cash_net","dso","dpo","rate"]].copy()
for col in ["gearing","debt_ebitda","cash_net"]:
    feat[f"{col}_chg_3m"] = feat[col].diff(3)
    feat[f"{col}_chg_6m"] = feat[col].diff(6)
feat = feat.fillna(0.0)
X = feat.values
y = tmp["breach_any"].values

metrics_text = ""
if y.sum() > 3 and len(finance) >= 30:
    # Split temporel simple (derniers 6 points = test)
    split_idx = len(X) - 6
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test,  y_test  = X[split_idx:], y[split_idx:]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("gb", GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=3))
    ])
    model.fit(X_train, y_train)

    # Calibration isotone sur l'ensemble (simple pour démo), sinon sur validation
    cal = CalibratedClassifierCV(model, method="isotonic", cv=3)
    cal.fit(X_train, y_train)

    # perfs
    try:
        proba_test = cal.predict_proba(X_test)[:,1]
        auc = roc_auc_score(y_test, proba_test)
        brier = brier_score_loss(y_test, proba_test)
        metrics_text = f"**AUC**≈{auc:.2f} • **Brier**≈{brier:.2f}"
    except Exception:
        metrics_text = ""

    prob_reference = float(cal.predict_proba(feat.iloc[-1:].values)[:,1])
else:
    prob_reference = 0.10  # référence si pas assez d'historique
    metrics_text = "_Historique insuffisant pour évaluer AUC/Brier_"

# Uplift symétrique + DSO/DPO
uplift = (
    0.006*shock_cost +
    0.010*(shock_delay/4) +
    0.007*(shock_weather/5) +
    0.030*shock_rate
)
delta_dso = client_terms - 60
delta_dpo = supplier_terms - 60
uplift += 0.0025*delta_dso - 0.0025*delta_dpo

prob_scenario = float(np.clip(prob_reference + uplift, 0.0, 0.99))

# Override “situation idéale”
ideal = (shock_cost <= 0 and shock_delay <= 0 and shock_weather <= 0 and shock_rate <= 0
         and client_terms <= 60 and supplier_terms >= 60)
within_buffers = (
    (np.nanmax(fcst["gearing_scn"].values) <= gearing_threshold * 0.95) and
    (np.nanmax(fcst["de_scn"].values)     <= debt_ebitda_threshold * 0.95)
)
if ideal and within_buffers:
    prob_scenario = min(prob_scenario, 0.05)

# ---------- Graphiques ----------
col1, col2 = st.columns(2)
with col1:
    st.subheader("Gearing — Historique & Prévision (scénario)")
    hist = finance[["date","gearing"]].rename(columns={"gearing":"value"}); hist["type"]="Historique"
    fut = fcst[["date","gearing_scn"]].rename(columns={"gearing_scn":"value"}); fut["type"]="Prévision (scénario)"
    fig = px.line(pd.concat([hist, fut]), x="date", y="value", color="type")
    fig.add_hline(y=gearing_threshold)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Dette/EBITDA — Historique & Prévision (scénario)")
    hist = finance[["date","debt_ebitda"]].rename(columns={"debt_ebitda":"value"}); hist["type"]="Historique"
    fut = fcst[["date","de_scn"]].rename(columns={"de_scn":"value"}); fut["type"]="Prévision (scénario)"
    fig = px.line(pd.concat([hist, fut]), x="date", y="value", color="type")
    fig.add_hline(y=debt_ebitda_threshold)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.subheader("🛑 Probabilité de bris (12 mois)")
c1, c2, c3 = st.columns(3)
c1.metric("Probabilité (référence)", f"{prob_reference:.0%}")
c2.metric("Probabilité (scénario)", f"{prob_scenario:.0%}", delta=f"{(prob_scenario-prob_reference):+.0%}")
statut = "🟢 Sûr" if prob_scenario < 0.30 else ("🟠 Sous surveillance" if prob_scenario < 0.60 else "🔴 Risque élevé")
c3.metric("Statut", statut)

if metrics_text:
    st.caption("Qualité modèle : " + metrics_text)

st.markdown("### 🧭 Lecture & recommandations")
st.markdown(
f"""
- **Scénario** : matières **{shock_cost:+d}%**, retards **{shock_delay:+d} sem**, météo **{shock_weather:+d} j/mois**, taux **{shock_rate:+d} pt** ; DSO **{client_terms} j** (≤ 60), DPO **{supplier_terms} j** (≥ 60).  
- **Logique** : effets **symétriques** (un scénario favorable réduit le risque). DSO>60 et DPO<60 pénalisent la probabilité.  
- **Surcharge 'situation idéale'** : si aucun choc et 60/60 et ratios sous seuils (coussin 5 %), la probabilité est **capée à 5 %**.
"""
)

st.info(
"**Bonnes pratiques** : viser **DSO ≤ 60 j** et **DPO ≥ 60 j** ; prioriser chantiers à forte marge RàP ; "
"plan de trésorerie ; relances clients ; négociation fournisseurs ; "
"si 🟠/🔴 durable, anticiper un échange bancaire avec stress tests."
)

st.caption("Version avancée : Prophet/SARIMAX + calibration. À affiner en production avec données ETP réelles (inclure 'equity' pour un gearing exact).")
