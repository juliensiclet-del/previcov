import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import plotly.express as px

st.set_page_config(page_title="PRÉVI-COV — Demo", layout="wide")

st.title("PRÉVI-COV — Console IA de prévision des covenants (démo)")
st.caption("Uploadez vos données mensuelles (CSV) et testez les prévisions + risque de bris (3/6/12 mois) avec scénarios.")

with st.expander("📄 Format attendu du CSV", expanded=False):
    st.markdown("""
    Colonnes recommandées (mensuelles) : `date, dscr, debt_ebitda, margin_project, dso, dpo, cash_net, steel_index, rate`.
    - `date` au format ISO (YYYY-MM-01).
    - Valeurs numériques (décimales avec point).
    """)

uploaded = st.file_uploader("Importer un CSV", type=["csv"])
if uploaded is None:
    st.info("Aucun fichier chargé — une **donnée d'exemple** est utilisée.")
    df = pd.read_csv("sample_data_previcov.csv", parse_dates=["date"])
else:
    df = pd.read_csv(uploaded, parse_dates=["date"])

df = df.sort_values("date")
st.dataframe(df.tail(12), use_container_width=True)

# --------------------------------------------------
# PARAMÈTRES
# --------------------------------------------------
st.sidebar.header("⚙️ Paramètres")
dscr_threshold = st.sidebar.number_input("Seuil covenant DSCR (min)", 1.2, 2.0, 1.3, step=0.1, format="%.2f")
debt_ebitda_threshold = st.sidebar.number_input("Seuil Dette/EBITDA (max)", 1.0, 6.0, 3.5, step=0.1, format="%.1f")

st.sidebar.subheader("🎛️ Scénarios (deltas)")
shock_cost = st.sidebar.slider("Choc coût matières [%]", -20, 40, 10, step=1)
shock_delay = st.sidebar.slider("Retard chantiers [semaines]", 0, 26, 6, step=1)
shock_rate = st.sidebar.slider("Hausse du taux d'intérêt [points]", -2, 4, 1, step=1)

# --------------------------------------------------
# PRÉVISIONS SIMPLES À 12 MOIS
# --------------------------------------------------
def simple_forecast(series, periods=12):
    if len(series) < 18:
        last = series.iloc[-6:].mean()
        return np.repeat(last, periods)
    y = series.values[-12:]
    x = np.arange(len(y))
    coef = np.polyfit(x, y, 1)
    x_future = np.arange(len(y), len(y)+periods)
    return coef[0]*x_future + coef[1]

horizon = 12
fcst_dscr = simple_forecast(df["dscr"], periods=horizon)
fcst_debt_ebitda = simple_forecast(df["debt_ebitda"], periods=horizon)

future_dates = pd.date_range(df["date"].iloc[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
fcst = pd.DataFrame({"date": future_dates, "dscr_fcst": fcst_dscr, "debt_ebitda_fcst": fcst_debt_ebitda})

# --------------------------------------------------
# SCÉNARIOS — ÉLASTICITÉS PLUS RÉALISTES
# --------------------------------------------------
# Hypothèses "métier" (effets absolus moyens) :
# - Choc matières : +10% => DSCR -0.20 ; Dette/EBITDA +0.30
# - Retard chantiers : +4 semaines => DSCR -0.08 ; Dette/EBITDA +0.05
# - Hausse de taux : +1 point => DSCR -0.06 ; Dette/EBITDA +0.05
def apply_shocks_to_series(dscr, de, shock_cost, shock_delay_weeks, shock_rate_pts):
    dscr_adj = dscr - 0.02 * shock_cost - 0.08 * (shock_delay_weeks/4) - 0.06 * shock_rate_pts
    de_adj   = de   + 0.03 * shock_cost + 0.05 * (shock_delay_weeks/4) + 0.05 * shock_rate_pts
    return np.maximum(dscr_adj, 0.3), np.maximum(de_adj, 0.5)

fcst["dscr_fcst_scn"], fcst["debt_ebitda_fcst_scn"] = apply_shocks_to_series(
    fcst["dscr_fcst"].values,
    fcst["debt_ebitda_fcst"].values,
    shock_cost=shock_cost,
    shock_delay_weeks=shock_delay,
    shock_rate_pts=shock_rate
)

# --------------------------------------------------
# CIBLE SUPERVISÉE : "BRIS DANS LES 12 PROCHAINS MOIS"
# --------------------------------------------------
tmp = df.copy()
tmp["breach_dscr"] = (tmp["dscr"].rolling(12, min_periods=1).min() < dscr_threshold).shift(-11).fillna(False)
tmp["breach_debt_ebitda"] = (tmp["debt_ebitda"].rolling(12, min_periods=1).max() > debt_ebitda_threshold).shift(-11).fillna(False)
tmp["breach_any"] = (tmp["breach_dscr"] | tmp["breach_debt_ebitda"]).astype(int)

# Features (niveaux + deltas)
features = tmp[["dscr","debt_ebitda","margin_project","dso","dpo","cash_net","steel_index","rate"]].copy()
for col in ["dscr","debt_ebitda","margin_project","cash_net"]:
    features[f"{col}_chg_3m"] = features[col].diff(3)
    features[f"{col}_chg_6m"] = features[col].diff(6)
features = features.fillna(0.0)

X = features.values
y = tmp["breach_any"].values

# Entraînement (si assez de positifs)
if y.sum() > 1 and len(df) > 24:
    model = Pipeline([("scaler", StandardScaler()),
                      ("clf", CalibratedClassifierCV(LogisticRegression(max_iter=200), method="isotonic", cv=3))])
    model.fit(X, y)
    x_last = features.iloc[-1:].values
    prob_base = float(model.predict_proba(x_last)[0,1])
else:
    model = None
    prob_base = 0.15  # baseline par défaut si peu de données

# --------------------------------------------------
# RISQUE DE BRIS — UPLIFT LIÉ AUX SCÉNARIOS (PLUS SENSIBLE)
# --------------------------------------------------
# Pondérations cohérentes avec les effets ci-dessus (en points de probabilité)
uplift = (
    0.012 * max(shock_cost, 0) +      # +1,2 pp par +1% coût matières
    0.02  * (shock_delay/4)   +       # +2 pp par 4 semaines de retard
    0.06  * max(shock_rate, 0)        # +6 pp par +1 point de taux
)
prob_scn = float(np.clip(prob_base + uplift, 0.0, 0.99))

# --------------------------------------------------
# GRAPHIQUES
# --------------------------------------------------
col1, col2 = st.columns(2)
with col1:
    st.subheader("Prévision DSCR")
    hist = df[["date","dscr"]].rename(columns={"dscr":"value"}); hist["type"]="Historique"
    fut = fcst[["date","dscr_fcst_scn"]].rename(columns={"dscr_fcst_scn":"value"}); fut["type"]="Prévision (scénario)"
    plot_df = pd.concat([hist, fut])
    fig = px.line(plot_df, x="date", y="value", color="type")
    fig.add_hline(y=dscr_threshold)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Prévision Dette/EBITDA")
    hist = df[["date","debt_ebitda"]].rename(columns={"debt_ebitda":"value"}); hist["type"]="Historique"
    fut = fcst[["date","debt_ebitda_fcst_scn"]].rename(columns={"debt_ebitda_fcst_scn":"value"}); fut["type"]="Prévision (scénario)"
    plot_df = pd.concat([hist, fut])
    fig = px.line(plot_df, x="date", y="value", color="type")
    fig.add_hline(y=debt_ebitda_threshold)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.subheader("🛑 Risque de bris (12 mois)")
c1, c2, c3 = st.columns(3)
c1.metric("Probabilité (baseline)", f"{prob_base:.0%}")
c2.metric("Probabilité (scénario)", f"{prob_scn:.0%}", delta=f"{(prob_scn-prob_base):+.0%}")
status = "🟢 Sûr" if prob_scn < 0.2 else ("🟠 Sous surveillance" if prob_scn < 0.4 else "🔴 Risque élevé")
c3.metric("Statut", status)

st.markdown(
    "**Explications** — Les élasticités de scénario sont calibrées 'métier' : "
    "+10% matières ≈ DSCR -0,20 & Dette/EBITDA +0,30 ; +4 semaines retard ≈ DSCR -0,08 ; "
    "+1 point de taux ≈ DSCR -0,06. Ces hypothèses rendent l'impact visible sur les courbes et la probabilité."
)

st.success("Flux démontré : Données → Prévisions KPI → Probabilité de bris → Scénarios → Alertes/Statut.")
