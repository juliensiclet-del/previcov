import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

st.set_page_config(page_title="PRÉVI-COV — Levier (ETP)", layout="wide")
st.title("PRÉVI-COV — Pilotage prédictif du levier (Dette / EBITDA) — ETP")
st.caption(
    "Version ETP : levier calculé en **glissant 12 mois (LTM)** pour refléter la saisonnalité (hiver/août). "
    "Prévisions + scénarios + risque de bris à 12 mois."
)

# =========================
# Import données
# =========================
uploaded = st.file_uploader("📥 Importer finance.csv", type=["csv"])

if uploaded is None:
    st.info("Aucun fichier importé — mode démo.")
    dates = pd.date_range("2023-01-01", periods=48, freq="MS")
    rng = np.random.default_rng(42)

    # Démo : EBITDA mensuel avec creux août
    weights = np.array([0.09,0.09,0.09,0.09,0.08,0.08,0.10,0.03,0.10,0.10,0.08,0.07])
    weights = weights / weights.sum()
    annual_ebitda = 5_200_000
    ebitda = np.array([annual_ebitda*w for w in np.tile(weights, 4)])[:len(dates)]
    ebitda = ebitda * (1 + rng.normal(0, 0.03, len(dates)))  # bruit léger

    net_debt = np.linspace(15_900_000, 13_250_000, len(dates)) + rng.normal(0, 120_000, len(dates))
    equity = np.linspace(8_700_000, 12_450_000, len(dates)) + rng.normal(0, 80_000, len(dates))

    df = pd.DataFrame({
        "date": dates,
        "net_debt": net_debt,
        "ebitda": ebitda,
        "equity": equity,
        "cash_net": 3_500_000 + rng.normal(0, 150_000, len(dates)),
        "dso": 67 + rng.normal(0, 3, len(dates)),
        "dpo": 48 + rng.normal(0, 3, len(dates)),
        "steel_index": 100 + np.maximum(0, rng.normal(0.7, 1.0, len(dates))).cumsum(),
        "rate": 2.5 + rng.normal(0, 0.15, len(dates)),
    })
else:
    df = pd.read_csv(uploaded, parse_dates=["date"])

df = df.sort_values("date").reset_index(drop=True)

st.subheader("Aperçu des données")
st.dataframe(df.tail(12), use_container_width=True)

# =========================
# Paramètres covenant
# =========================
st.sidebar.header("⚙️ Covenant")
levier_max = st.sidebar.number_input("Seuil Dette / EBITDA (max)", 1.0, 10.0, 3.0, 0.1)

# =========================
# Scénarios ETP
# =========================
st.sidebar.header("🎛️ Scénarios (ETP)")
shock_cost = st.sidebar.slider("Hausse des coûts matières (%)", -20, 50, 0, 1)
shock_delay = st.sidebar.slider("Retards chantiers (semaines)", 0, 26, 0, 1)
shock_weather = st.sidebar.slider("Intempéries (jours/mois)", 0, 20, 0, 1)
shock_rate = st.sidebar.slider("Hausse des taux (points)", -2, 4, 0, 1)

dso_target = st.sidebar.slider("Délai clients (DSO) — viser ≤ 60 j", 30, 120, 60, 5)
dpo_target = st.sidebar.slider("Délai fournisseurs (DPO) — viser ≥ 60 j", 30, 120, 60, 5)

# =========================
# Saisonnalité (clé ETP)
# =========================
st.sidebar.header("📅 Saisonnalité")
use_seasonality = st.sidebar.checkbox("Activer la saisonnalité ETP", value=True)
seasonality_strength = st.sidebar.slider("Intensité saisonnière", 0.0, 1.5, 1.0, 0.1)

# Clé ETP fournie (9/9/9/9/8/8/10/3/10/10/8/7)
ETP_WEIGHTS = np.array([0.09,0.09,0.09,0.09,0.08,0.08,0.10,0.03,0.10,0.10,0.08,0.07])
ETP_WEIGHTS = ETP_WEIGHTS / ETP_WEIGHTS.sum()

# =========================
# Nettoyage minimal
# =========================
def to_num(col, default=0.0):
    if col not in df.columns:
        return pd.Series([default]*len(df))
    s = pd.to_numeric(df[col], errors="coerce")
    return s.interpolate(limit_direction="both").fillna(default)

df["net_debt"] = to_num("net_debt", np.nan)
df["ebitda"] = to_num("ebitda", np.nan)
df["cash_net"] = to_num("cash_net", 0.0)
df["dso"] = to_num("dso", 60.0).clip(20, 180)
df["dpo"] = to_num("dpo", 60.0).clip(20, 180)
df["rate"] = to_num("rate", 2.0).clip(0, 25)

# =========================
# Levier covenant "réaliste" : LTM (glissant 12 mois)
# =========================
use_ltm = st.sidebar.checkbox("Calculer le levier en glissant 12 mois (recommandé)", value=True)

if use_ltm and df["net_debt"].notna().any() and df["ebitda"].notna().any():
    df["ebitda_ltm"] = df["ebitda"].rolling(12, min_periods=6).sum()
    df["levier"] = df["net_debt"] / df["ebitda_ltm"].replace(0, np.nan)
else:
    # fallback : utiliser debt_ebitda si présent
    if "debt_ebitda" not in df.columns:
        st.error("Il manque les colonnes nécessaires : soit (net_debt + ebitda), soit debt_ebitda.")
        st.stop()
    df["levier"] = to_num("debt_ebitda", np.nan)

df["levier"] = df["levier"].replace([np.inf, -np.inf], np.nan).interpolate(limit_direction="both")
df["levier"] = df["levier"].clip(0.5, 12.0)

# =========================
# Prévision : net_debt + ebitda mensuel (avec saisonnalité) -> levier LTM futur
# =========================
horizon = 12
future_dates = pd.date_range(df["date"].iloc[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")

def forecast_net_debt(series, h=12):
    # tendance douce sur 24 derniers mois
    s = series.dropna()
    if len(s) < 6:
        return np.repeat(float(s.iloc[-1]), h)
    y = s.tail(min(24, len(s))).values
    x = np.arange(len(y))
    a, b = np.polyfit(x, y, 1)
    # limiter la pente à ±1%/mois
    slope_cap = 0.01 * np.nanmean(y)
    a = np.clip(a, -slope_cap, slope_cap)
    x_f = np.arange(len(y), len(y)+h)
    return a * x_f + b

def forecast_annual_ebitda_from_ltm(ebitda_monthly):
    # Annualisation simple à partir des 12 derniers mois observés
    s = ebitda_monthly.dropna()
    if len(s) < 12:
        return float(s.tail(min(6, len(s))).sum()) * (12/max(1, min(6, len(s))))
    return float(s.tail(12).sum())

def forecast_ebitda_monthly(annual_ebitda, future_dates, strength=1.0):
    # applique la clé ETP par défaut + intensité
    base = annual_ebitda * ETP_WEIGHTS
    out = []
    for d in future_dates:
        m = d.month - 1
        # intensité : 0 => uniforme ; 1 => clé brute ; >1 => plus marqué
        w = (1-strength)*(1/12) + strength*ETP_WEIGHTS[m]
        out.append(annual_ebitda * w)
    return np.array(out)

net_debt_fcst = forecast_net_debt(df["net_debt"], horizon)

annual_ebitda_est = forecast_annual_ebitda_from_ltm(df["ebitda"])
if use_seasonality:
    ebitda_fcst = forecast_ebitda_monthly(annual_ebitda_est, future_dates, strength=seasonality_strength)
else:
    ebitda_fcst = np.repeat(annual_ebitda_est/12, horizon)

# =========================
# Scénarios : impact sur EBITDA (matières, retards, météo, taux) + BFR via DSO/DPO
# =========================
# approche pédagogique : scénarios dégradent l'EBITDA futur (donc levier ↑)
ebitda_impact = 1.0
ebitda_impact -= 0.002 * shock_cost            # +10% matières => -2% EBITDA (ordre de grandeur)
ebitda_impact -= 0.010 * (shock_delay / 4)     # +4 semaines => -1%
ebitda_impact -= 0.008 * (shock_weather / 5)   # +5 jours => -0.8%
ebitda_impact = np.clip(ebitda_impact, 0.75, 1.10)

# DSO/DPO : on n'a pas de modèle cash complet, donc on traduit en "stress" levier
stress_bfr = 0.0
stress_bfr += 0.010 * ((dso_target - 60) / 10)   # DSO>60 => levier ↑
stress_bfr += -0.010 * ((dpo_target - 60) / 10)  # DPO<60 => levier ↑ (delta négatif)

# Taux : peut dégrader la capacité (via charges financières), donc stress
stress_rate = 0.020 * shock_rate

ebitda_fcst_scn = np.clip(ebitda_fcst * ebitda_impact, 1.0, None)

# construire ebitda_ltm futur en concaténant historique + futur
ebitda_all = pd.concat([df["ebitda"], pd.Series(ebitda_fcst_scn)], ignore_index=True)
netdebt_all = pd.concat([df["net_debt"], pd.Series(net_debt_fcst)], ignore_index=True)

# LTM futur : rolling 12 sur l'ensemble
ebitda_ltm_all = ebitda_all.rolling(12, min_periods=6).sum()
levier_all = netdebt_all / ebitda_ltm_all.replace(0, np.nan)

# extraire futur levier
levier_scn = levier_all.iloc[len(df):len(df)+horizon].to_numpy()
levier_scn = np.clip(levier_scn + stress_bfr + stress_rate, 0.5, 12.0)

fcst = pd.DataFrame({"date": future_dates, "levier_scn": levier_scn})

# =========================
# Modèle de risque (supervisé simple) : bris levier dans 12 mois
# =========================
tmp = df.copy()
tmp["breach"] = (
    tmp["levier"].rolling(12, min_periods=1).max() > levier_max
).shift(-11).fillna(False).astype(int)

features = tmp[["levier","cash_net","dso","dpo","rate"]].fillna(0.0)
X = features.values
y = tmp["breach"].values

if y.sum() > 3 and len(tmp) >= 30:
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(n_estimators=250, learning_rate=0.05, max_depth=3, random_state=42))
    ])
    model.fit(X, y)
    cal = CalibratedClassifierCV(model, cv=3, method="isotonic")
    cal.fit(X, y)
    risque_actuel = float(cal.predict_proba(features.iloc[-1:].values)[0, 1])
else:
    risque_actuel = 0.12

# risque projeté : basé sur dépassement futur + stress scénarios
risque_projete = np.clip(risque_actuel + 0.10 * (fcst["levier_scn"].max() > levier_max) + 0.02*max(shock_rate,0), 0, 0.99)

# situation idéale -> risque bas
ideal = (shock_cost <= 0 and shock_delay == 0 and shock_weather == 0 and shock_rate <= 0 and dso_target <= 60 and dpo_target >= 60
         and float(fcst["levier_scn"].max()) <= levier_max * 0.95)
if ideal:
    risque_projete = min(risque_projete, 0.06)

# =========================
# Affichage
# =========================
st.markdown("---")
st.subheader("📈 Prévision du levier Dette / EBITDA")

hist = df[["date","levier"]].rename(columns={"levier":"valeur"})
hist["type"] = "Historique (levier LTM)" if use_ltm else "Historique"

fut = fcst.rename(columns={"levier_scn":"valeur"})
fut["type"] = "Prévision (scénario)"

plot_df = pd.concat([hist, fut], ignore_index=True)
fig = px.line(plot_df, x="date", y="valeur", color="type")
fig.add_hline(y=levier_max)
st.plotly_chart(fig, use_container_width=True)

st.caption(
    "La prévision n'est pas une courbe “lissée” arbitrairement : elle est reconstruite à partir de la dette nette et "
    "d’un EBITDA mensuel saisonnalisé, puis convertie en levier **glissant 12 mois** (logique covenant)."
)

st.markdown("---")
st.subheader("🛑 Risque de bris de covenant (12 mois)")
c1, c2, c3 = st.columns(3)
c1.metric("Risque actuel", f"{risque_actuel:.0%}")
c2.metric("Risque projeté", f"{risque_projete:.0%}", delta=f"{(risque_projete-risque_actuel):+.0%}")

if risque_projete < 0.20:
    statut = "🟢 Sûr"
elif risque_projete < 0.40:
    statut = "🟠 Sous surveillance"
else:
    statut = "🔴 Risque élevé"

c3.metric("Statut", statut)
