
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

st.sidebar.header("⚙️ Paramètres")
dscr_threshold = st.sidebar.number_input("Seuil covenant DSCR (min)", 1.2, 2.0, 1.3, step=0.1, format="%.2f")
debt_ebitda_threshold = st.sidebar.number_input("Seuil Dette/EBITDA (max)", 1.0, 6.0, 3.5, step=0.1, format="%.1f")

st.sidebar.subheader("🎛️ Scénarios (deltas)")
shock_cost = st.sidebar.slider("Choc coût matières (impact indirect sur marge) [%]", -20, 40, 10, step=1)
shock_delay = st.sidebar.slider("Retard chantiers (impact indirect sur DSCR) [semaines]", 0, 26, 6, step=1)
shock_rate = st.sidebar.slider("Hausse du taux d'intérêt [points]", -2, 4, 1, step=1)

# --- simple time-series forecast by local linear trend (fallback to moving avg if not enough data) ---
def simple_forecast(series, periods=12):
    if len(series) < 18:
        last = series.iloc[-6:].mean()
        return np.repeat(last, periods)
    # local linear extrapolation on last 12 months
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

# --- Apply scenario shocks (very simple elasticities for demo) ---
fcst["dscr_fcst_scn"] = fcst["dscr_fcst"] * (1 - 0.006*shock_delay) * (1 - 0.01*shock_cost) * (1 - 0.05*shock_rate)
fcst["debt_ebitda_fcst_scn"] = fcst["debt_ebitda_fcst"] * (1 + 0.003*shock_delay) * (1 + 0.01*shock_cost) * (1 + 0.05*shock_rate)

# --- Build a simple supervised target from history: "breach within 12m ahead" ---
tmp = df.copy()
tmp["breach_dscr"] = (tmp["dscr"].rolling(12, min_periods=1).min() < dscr_threshold).shift(-11).fillna(False)
tmp["breach_debt_ebitda"] = (tmp["debt_ebitda"].rolling(12, min_periods=1).max() > debt_ebitda_threshold).shift(-11).fillna(False)
tmp["breach_any"] = (tmp["breach_dscr"] | tmp["breach_debt_ebitda"]).astype(int)

# features: last values + deltas
features = tmp[["dscr","debt_ebitda","margin_project","dso","dpo","cash_net","steel_index","rate"]].copy()
for col in ["dscr","debt_ebitda","margin_project","cash_net"]:
    features[f"{col}_chg_3m"] = features[col].diff(3)
    features[f"{col}_chg_6m"] = features[col].diff(6)
features = features.fillna(0.0)

X = features.values
y = tmp["breach_any"].values

# only train if enough positives
if y.sum() > 1 and len(df) > 24:
    model = Pipeline([("scaler", StandardScaler()),
                      ("clf", CalibratedClassifierCV(LogisticRegression(max_iter=200), method="isotonic", cv=3))])
    model.fit(X, y)
    # construct last row for forecasting risk baseline
    x_last = features.iloc[-1:].values
    prob_base = float(model.predict_proba(x_last)[0,1])
else:
    model = None
    prob_base = 0.15  # default baseline

# translate scenario into a risk uplift/downgrade
uplift = 0.0 + 0.005*shock_delay + 0.01*max(shock_cost,0) + 0.03*max(shock_rate,0)
prob_scn = min(max(prob_base + uplift, 0.0), 0.99)

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

st.markdown("**Explications (démo)** — Les variations récentes des KPIs (Δ3m/Δ6m) et les chocs de scénario modulent la probabilité. Pour une version production, remplacer les élasticités par un moteur de scénarios relié aux drivers métiers et des modèles de séries temporelles robustes (SARIMAX/Prophet).")

st.success("Cette démo illustre le flux : Données → Prévisions KPI → Probabilité de bris → Scénarios → Alertes/Statut.")
