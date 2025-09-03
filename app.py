# PRÉVI-COV — ETP Edition
# Streamlit app focalisée TP : 2 jeux de données, scénarios métier, impacts pondérés par chantiers.

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import plotly.express as px

st.set_page_config(page_title="PRÉVI-COV — ETP", layout="wide")
st.title("PRÉVI-COV — ETP (Console IA de gestion prédictive des covenants)")

st.caption(
    "Importez vos **données financières mensuelles** et votre **suivi chantiers**. "
    "Simulez des scénarios (matières, retards, météo, taux) et mesurez l'impact sur DSCR / Dette/EBITDA et le risque de bris."
)

with st.expander("📄 Formats attendus", expanded=False):
    st.markdown("""
**1) finance.csv (mensuel, 1 ligne / mois)**  
Colonnes minimales :  
- `date` (YYYY-MM-01)  
- `dscr` (si connu) **ou** `ebitda`, `interest_paid` (intérêts 12m), `debt_service` (annuité 12m)  
- `debt_ebitda` (si connu) **ou** `net_debt`, `ebitda`  
- `cash_net`, `dso`, `dpo`, `steel_index` (optionnel), `rate` (taux moyen)  

**2) projects.csv (suivi chantiers, 1 ligne / chantier / mois)**  
- `date` (YYYY-MM-01), `project_id`, `rp` (reste à produire en k€),  
- `planned_margin_pct` (marge budgétée %), `actual_margin_pct` (si dispo),  
- `delay_weeks` (cumul de retard), `weather_days` (jours perdus, optionnel).
""")

# --------------------------
# UPLOAD DATA
# --------------------------
c1, c2 = st.columns(2)
with c1:
    fin_file = st.file_uploader("📥 Importer finance.csv", type=["csv"], key="fin")
with c2:
    prj_file = st.file_uploader("📥 Importer projects.csv", type=["csv"], key="prj")

def read_csv_default(p, parse_dates=None):
    try:
        return pd.read_csv(p, parse_dates=parse_dates)
    except Exception:
        return None

if fin_file is None or prj_file is None:
    st.info("Aucun fichier fourni : **mode démo** (synthetic).")
    # --- DEMO synthetic ---
    dates = pd.date_range("2023-01-01", periods=30, freq="MS")
    rng = np.random.default_rng(42)
    ebitda = 1000 + rng.normal(0, 60, len(dates)).cumsum()
    net_debt = 3500 + rng.normal(0, 50, len(dates))
    rate = 1.4 + rng.normal(0, 0.06, len(dates))
    steel_index = 100 + rng.normal(0.8, 2.5, len(dates)).cumsum()
    interest_paid = np.maximum(0.012*net_debt, 100)
    debt_service = 400 + 15*(rate-1.4)
    dscr = np.clip((ebitda / np.maximum(debt_service, 1)), 0.4, 3.5)
    debt_ebitda = np.clip(net_debt / np.maximum(ebitda, 1), 1.0, 6.0)
    finance = pd.DataFrame({
        "date": dates, "ebitda": ebitda, "interest_paid": interest_paid, "debt_service": debt_service,
        "net_debt": net_debt, "steel_index": steel_index, "rate": rate,
        "cash_net": 500 + rng.normal(0, 30, len(dates)), "dso": 60 + rng.normal(0, 4, len(dates)),
        "dpo": 55 + rng.normal(0, 3, len(dates)), "dscr": dscr, "debt_ebitda": debt_ebitda
    })
    # projets
    pr = []
    for d in dates:
        for pid in ["A12", "B07", "C03"]:
            rp = rng.integers(300, 900)
            planned_margin = 12 + rng.normal(0, 1.5)
            delay_weeks = max(0, int(rng.normal(4, 2)))
            weather_days = max(0, int(rng.normal(3, 2)))
            pr.append([d, pid, rp, planned_margin, np.nan, delay_weeks, weather_days])
    projects = pd.DataFrame(pr, columns=["date","project_id","rp","planned_margin_pct","actual_margin_pct","delay_weeks","weather_days"])
else:
    finance = pd.read_csv(fin_file, parse_dates=["date"])
    projects = pd.read_csv(prj_file, parse_dates=["date"])

finance = finance.sort_values("date")
projects = projects.sort_values(["date","project_id"])

st.subheader("Aperçu des données")
st.dataframe(finance.tail(6), use_container_width=True)
st.dataframe(projects.tail(6), use_container_width=True)

# --------------------------
# PARAMÈTRES COVENANTS & SCÉNARIOS
# --------------------------
st.sidebar.header("⚙️ Paramètres covenants")
dscr_threshold = st.sidebar.number_input("Seuil DSCR (min)", 1.2, 2.5, 1.3, step=0.1, format="%.2f")
debt_ebitda_threshold = st.sidebar.number_input("Seuil Dette/EBITDA (max)", 1.0, 6.0, 3.5, step=0.1, format="%.1f")

st.sidebar.header("🎛️ Scénarios (métier ETP)")
shock_cost = st.sidebar.slider("Matières : hausse des coûts [%]", -20, 50, 15, step=1)
shock_delay = st.sidebar.slider("Retards chantier [semaines]", 0, 26, 12, step=1)
shock_weather = st.sidebar.slider("Météo défavorable [jours/mois]", 0, 20, 5, step=1)
shock_rate = st.sidebar.slider("Hausse des taux [points]", -1, 4, 2, step=1)

# --------------------------
# OUTILS
# --------------------------
def simple_forecast(series, periods=12):
    """Projection rapide (tendance locale)"""
    if len(series) < 18:
        last = float(series.tail(6).mean())
        return np.repeat(last, periods)
    y = series.values[-12:]
    x = np.arange(len(y))
    coef = np.polyfit(x, y, 1)
    return coef[0]*np.arange(12,12+periods) + coef[1]

def ensure_ratios(fin):
    """Si dscr / debt_ebitda manquants, les approxime à partir des colonnes disponibles."""
    out = fin.copy()
    if "dscr" not in out or out["dscr"].isna().all():
        if {"ebitda","debt_service"}.issubset(out.columns):
            out["dscr"] = out["ebitda"] / out["debt_service"].clip(lower=1e-6)
    if "debt_ebitda" not in out or out["debt_ebitda"].isna().all():
        if {"net_debt","ebitda"}.issubset(out.columns):
            out["debt_ebitda"] = out["net_debt"] / out["ebitda"].clip(lower=1e-6)
    return out

def project_weighted_margin(projects_month):
    """Calcule la marge 'pondérée reste à produire' du portefeuille."""
    df = projects_month.copy()
    if df.empty: 
        return np.nan
    w = df["rp"].clip(lower=0.0)
    base_margin = np.where(df["actual_margin_pct"].notna(), df["actual_margin_pct"], df["planned_margin_pct"])
    return float((w * base_margin).sum() / max(w.sum(), 1e-6))

def apply_etp_scenarios(dscr, de, portfolio_margin, steel_idx, rate, 
                        shock_cost, shock_delay_weeks, shock_weather_days, shock_rate_pts):
    """
    Élasticités calibrées 'métier TP' (ordres de grandeur pédagogiques) :
    - Matières +10%  => marge portefeuille -2.0 pts,  DSCR -0.20, Debt/EBITDA +0.30
    - Retard +4 sem  => marge -0.8 pt,                DSCR -0.08, Debt/EBITDA +0.05
    - Météo +5 jours => marge -0.5 pt,                DSCR -0.05, Debt/EBITDA +0.03
    - Taux +1 pt     => (intérêts ↑)                  DSCR -0.06, Debt/EBITDA +0.05
    """
    dscr_adj = dscr - 0.02*shock_cost - 0.08*(shock_delay_weeks/4) - 0.05*(shock_weather_days/5) - 0.06*shock_rate_pts
    de_adj   = de   + 0.03*shock_cost + 0.05*(shock_delay_weeks/4) + 0.03*(shock_weather_days/5) + 0.05*shock_rate_pts
    margin_adj = portfolio_margin - (2.0*(shock_cost/10.0) + 0.8*(shock_delay_weeks/4.0) + 0.5*(shock_weather_days/5.0))
    return np.maximum(dscr_adj, 0.3), np.maximum(de_adj, 0.5), margin_adj

# --------------------------
# PRÉPARATION
# --------------------------
finance = ensure_ratios(finance)
last_date = finance["date"].max()
h = 12
future_dates = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=h, freq="MS")

# Prévisions brutes
fcst = pd.DataFrame({"date": future_dates})
fcst["dscr_fcst"] = simple_forecast(finance["dscr"], periods=h)
fcst["debt_ebitda_fcst"] = simple_forecast(finance["debt_ebitda"], periods=h)

# Marge portefeuille actuelle et future (constante hors scénario)
current_month = projects[projects["date"]==projects["date"].max()]
port_margin_now = project_weighted_margin(current_month)

# Application scénarios
fcst["dscr_scn"], fcst["de_scn"], fcst["margin_portfolio_scn"] = apply_etp_scenarios(
    dscr=fcst["dscr_fcst"].values,
    de=fcst["debt_ebitda_fcst"].values,
    portfolio_margin=port_margin_now if not np.isnan(port_margin_now) else 12.0,
    steel_idx=finance.get("steel_index", pd.Series(index=finance.index, data=np.nan)).iloc[-1] if "steel_index" in finance else 100.0,
    rate=finance.get("rate", pd.Series(index=finance.index, data=np.nan)).iloc[-1] if "rate" in finance else 1.5,
    shock_cost=shock_cost, shock_delay_weeks=shock_delay, shock_weather_days=shock_weather, shock_rate_pts=shock_rate
)

# --------------------------
# RISQUE DE BRIS 12M
# --------------------------
tmp = finance.copy()
tmp["breach_dscr"] = (tmp["dscr"].rolling(12, min_periods=1).min() < dscr_threshold).shift(-11).fillna(False)
tmp["breach_debt_ebitda"] = (tmp["debt_ebitda"].rolling(12, min_periods=1).max() > debt_ebitda_threshold).shift(-11).fillna(False)
tmp["breach_any"] = (tmp["breach_dscr"] | tmp["breach_debt_ebitda"]).astype(int)

# features : niveaux + deltas
feat = tmp[["dscr","debt_ebitda","cash_net","dso","dpo","rate"]].copy()
for col in ["dscr","debt_ebitda","cash_net"]:
    feat[f"{col}_chg_3m"] = feat[col].diff(3)
    feat[f"{col}_chg_6m"] = feat[col].diff(6)
feat = feat.fillna(0.0)

X = feat.values
y = tmp["breach_any"].values

if y.sum() > 1 and len(finance) > 24:
    model = Pipeline([("scaler", StandardScaler()),
                      ("clf", CalibratedClassifierCV(LogisticRegression(max_iter=200), method="isotonic", cv=3))])
    model.fit(X, y)
    x_last = feat.iloc[-1:].values
    prob_base = float(model.predict_proba(x_last)[0,1])
else:
    prob_base = 0.18  # baseline pédagogique si peu d'historique

# Uplift de risque cohérent avec scénarios ETP
uplift = (
    0.012*max(shock_cost,0) +       # +1,2 pp par +1% matières
    0.02*(shock_delay/4) +          # +2 pp par 4 semaines
    0.015*(shock_weather/5) +       # +1,5 pp par 5 jours
    0.06*max(shock_rate,0)          # +6 pp par +1 point de taux
)
prob_scn = float(np.clip(prob_base + uplift, 0.0, 0.99))

# --------------------------
# GRAPHIQUES
# --------------------------
col1, col2 = st.columns(2)
with col1:
    st.subheader("DSCR — Historique & Prévision (scénario)")
    hist = finance[["date","dscr"]].rename(columns={"dscr":"value"}); hist["type"]="Historique"
    fut = fcst[["date","dscr_scn"]].rename(columns={"dscr_scn":"value"}); fut["type"]="Prévision (scénario)"
    plot_df = pd.concat([hist, fut])
    fig = px.line(plot_df, x="date", y="value", color="type")
    fig.add_hline(y=dscr_threshold)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Dette/EBITDA — Historique & Prévision (scénario)")
    hist = finance[["date","debt_ebitda"]].rename(columns={"debt_ebitda":"value"}); hist["type"]="Historique"
    fut = fcst[["date","de_scn"]].rename(columns={"de_scn":"value"}); fut["type"]="Prévision (scénario)"
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

st.markdown("### 🧭 Lecture & recommandations")
st.markdown(
f"""
- **Hypothèses de scénario appliquées** : matières **+{shock_cost}%**, retards **{shock_delay} sem**, météo défavorable **{shock_weather} j/mois**, taux **+{shock_rate} pt**.  
- **Effets calibrés 'métier TP'** (ordres de grandeur) : +10% matières → DSCR **-0,20** / Dette/EBITDA **+0,30** ; +4 sem de retard → DSCR **-0,08** ; +5 jours météo → DSCR **-0,05** ; +1 pt de taux → DSCR **-0,06**.  
- **Marge portefeuille pondérée (RàP)** : ~ **{fcst['margin_portfolio_scn'].iloc[0]:.1f} %** sous scénario.
"""
)

st.info(
"**Actions possibles** : (1) Prioriser les chantiers à forte marge RàP ; "
"(2) Discuter un aménagement d'échéancier si statut 🔴 ; "
"(3) Renforcer le recouvrement (DSO) et lisser le CAPEX ; "
"(4) Formaliser un addendum bancaire avec stress tests en annexe."
)

st.caption("Outil pédagogique : à calibrer sur données réelles ETP pour décision.")
