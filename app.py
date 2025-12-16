# PRÉVI-COV — ETP
# Pilotage prédictif des covenants bancaires
# Ratio suivi : Dette / EBITDA (levier uniquement)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

# --------------------------------------------------
# CONFIGURATION PAGE
# --------------------------------------------------
st.set_page_config(page_title="PRÉVI-COV — Levier", layout="wide")
st.title("PRÉVI-COV — Gestion prédictive du levier (Dette / EBITDA)")
st.caption(
    "Outil pédagogique destiné aux entreprises de travaux publics (ETP). "
    "Objectif : anticiper le risque de bris de covenant sur le ratio de levier."
)

# --------------------------------------------------
# IMPORT DES DONNÉES
# --------------------------------------------------
uploaded = st.file_uploader("📥 Importer le fichier finance.csv", type=["csv"])

if uploaded is None:
    st.info("Aucun fichier importé – mode démonstration activé.")
    dates = pd.date_range("2023-01-01", periods=36, freq="MS")
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "date": dates,
        "debt_ebitda": np.clip(2.7 + rng.normal(0, 0.15, len(dates)), 2.2, 3.4),
        "cash_net": 3_500_000 + rng.normal(0, 150_000, len(dates)),
        "dso": 67 + rng.normal(0, 3, len(dates)),
        "dpo": 48 + rng.normal(0, 3, len(dates)),
        "rate": 2.5 + rng.normal(0, 0.15, len(dates)),
    })
else:
    df = pd.read_csv(uploaded, parse_dates=["date"])

df = df.sort_values("date").reset_index(drop=True)

st.subheader("Aperçu des données")
st.dataframe(df.tail(12), use_container_width=True)

# --------------------------------------------------
# PARAMÈTRES COVENANT
# --------------------------------------------------
st.sidebar.header("⚙️ Paramètres du covenant")
levier_max = st.sidebar.number_input(
    "Seuil Dette / EBITDA (max)",
    min_value=1.0,
    max_value=8.0,
    value=3.0,
    step=0.1
)

# --------------------------------------------------
# SCÉNARIOS
# --------------------------------------------------
st.sidebar.header("🎛️ Scénarios ETP")
shock_cost = st.sidebar.slider("Hausse des coûts matières (%)", -20, 50, 0)
shock_delay = st.sidebar.slider("Retards chantiers (semaines)", 0, 26, 0)
shock_weather = st.sidebar.slider("Intempéries (jours/mois)", 0, 20, 0)
shock_rate = st.sidebar.slider("Hausse des taux (points)", -2, 4, 0)

dso_target = st.sidebar.slider("Délai clients (DSO) – viser ≤ 60 j", 30, 120, 60)
dpo_target = st.sidebar.slider("Délai fournisseurs (DPO) – viser ≥ 60 j", 30, 120, 60)

# --------------------------------------------------
# PRÉVISION PRUDENTE
# --------------------------------------------------
def forecast_prudent(series, horizon=12):
    base = series.rolling(12, min_periods=6).mean().iloc[-1]
    slope = np.clip(series.diff().mean(), -0.05, 0.05)
    return np.array([base + slope * (i + 1) for i in range(horizon)])

horizon = 12
future_dates = pd.date_range(df["date"].iloc[-1] + pd.offsets.MonthBegin(1),
                             periods=horizon, freq="MS")

base_fcst = forecast_prudent(df["debt_ebitda"], horizon)

# --------------------------------------------------
# IMPACT DES SCÉNARIOS
# --------------------------------------------------
levier_scn = base_fcst.copy()
levier_scn += 0.01 * shock_cost
levier_scn += 0.03 * (shock_delay / 4)
levier_scn += 0.02 * (shock_weather / 5)
levier_scn += 0.03 * shock_rate
levier_scn += 0.01 * ((dso_target - 60) / 10)
levier_scn -= 0.01 * ((dpo_target - 60) / 10)

levier_scn = np.clip(levier_scn, 0.5, 10.0)

fcst = pd.DataFrame({
    "date": future_dates,
    "levier": levier_scn
})

# --------------------------------------------------
# MODÈLE DE RISQUE (BRIS À 12 MOIS)
# --------------------------------------------------
tmp = df.copy()
tmp["breach"] = (
    tmp["debt_ebitda"].rolling(12, min_periods=1).max()
    > levier_max
).shift(-11).fillna(False).astype(int)

features = tmp[["debt_ebitda", "cash_net", "dso", "dpo", "rate"]].fillna(0)
X = features.values
y = tmp["breach"].values

if y.sum() > 3 and len(df) > 24:
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(n_estimators=200, random_state=42))
    ])
    model.fit(X, y)
    cal = CalibratedClassifierCV(model, cv=3, method="isotonic")
    cal.fit(X, y)
    risque_actuel = cal.predict_proba(features.iloc[-1:].values)[0, 1]
else:
    risque_actuel = 0.12

# Ajustement scénario
risque_projete = np.clip(
    risque_actuel
    + 0.02 * (shock_cost / 10)
    + 0.05 * (shock_delay / 10)
    + 0.08 * shock_rate,
    0, 0.99
)

if levier_scn.max() < levier_max * 0.9:
    risque_projete = max(0.03, risque_projete - 0.1)

# --------------------------------------------------
# AFFICHAGE
# --------------------------------------------------
st.subheader("📈 Prévision du levier Dette / EBITDA")

hist = df[["date", "debt_ebitda"]].rename(columns={"debt_ebitda": "valeur"})
hist["type"] = "Historique"
fut = fcst.rename(columns={"levier": "valeur"})
fut["type"] = "Prévision (scénario)"

fig = px.line(pd.concat([hist, fut]), x="date", y="valeur", color="type")
fig.add_hline(y=levier_max)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.subheader("🛑 Risque de bris de covenant (12 mois)")

c1, c2, c3 = st.columns(3)
c1.metric("Risque actuel", f"{risque_actuel:.0%}")
c2.metric("Risque projeté", f"{risque_projete:.0%}",
          delta=f"{(risque_projete - risque_actuel):+.0%}")

if risque_projete < 0.20:
    statut = "🟢 Sûr"
elif risque_projete < 0.40:
    statut = "🟠 Sous surveillance"
else:
    statut = "🔴 Risque élevé"

c3.metric("Statut", statut)
