# PRÉVI-COV — ETP Edition (gearing + DSO/DPO centrés 60j + UI FR)
# - Gearing (Net Debt / Equity), seuil max = 2.0
# - Dette/EBITDA, seuil max = 3.0
# - Scénarios symétriques dont DSO/DPO centrés à 60j (DSO>60 = risque↑ ; DPO<60 = risque↑)
# - Tous les libellés visibles en français (baseline -> référence)

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
    "Importez vos **données financières mensuelles** (finance.csv) et votre **suivi chantiers** (projects.csv). "
    "Simulez des scénarios (matières, retards, météo, taux, règlements clients, paiements fournisseurs) et mesurez l'impact "
    "sur **Gearing** et **Dette/EBITDA**, ainsi que la **probabilité de bris**."
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
    dates = pd.date_range("2023-01-01", periods=30, freq="MS")
    rng = np.random.default_rng(42)
    equity = 4000 + rng.normal(0, 80, len(dates)).cumsum()
    net_debt = 3500 + rng.normal(0, 80, len(dates))
    ebitda = 1000 + rng.normal(0, 50, len(dates)).cumsum()
    rate = 1.5 + rng.normal(0, 0.05, len(dates))
    steel_index = 100 + rng.normal(0.5, 1.8, len(dates)).cumsum()
    debt_ebitda = np.clip(net_debt / np.maximum(ebitda, 1), 0.8, 6.0)
    gearing = np.clip(net_debt / np.maximum(equity, 1), 0.2, 4.0)
    finance = pd.DataFrame({
        "date": dates, "equity": equity, "net_debt": net_debt, "ebitda": ebitda,
        "gearing": gearing, "debt_ebitda": debt_ebitda,
        "cash_net": 500 + rng.normal(0, 30, len(dates)),
        "dso": 60 + rng.normal(0, 5, len(dates)),
        "dpo": 60 + rng.normal(0, 5, len(dates)),
        "steel_index": steel_index, "rate": rate
    })
    pr = []
    for d in dates:
        for pid in ["A12","B07","C03"]:
            rp = rng.integers(300, 900)
            planned_margin = 12 + rng.normal(0, 1.5)
            actual_margin = planned_margin + rng.normal(-0.4, 0.8)
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

# DSO/DPO centrés à 60j
client_terms = st.sidebar.slider("Délai règlements clients (DSO) [jours] — viser ≤ 60", 30, 120, 60, step=5)
supplier_terms = st.sidebar.slider("Délai paiements fournisseurs (DPO) [jours] — viser ≥ 60", 30, 120, 60, step=5)

# ---------- Outils ----------
def simple_forecast(series, periods=12):
    if len(series) < 18:
        last = float(series.tail(6).mean())
        return np.repeat(last, periods)
    y = series.values[-12:]
    x = np.arange(len(y))
    coef = np.polyfit(x, y, 1)
    return coef[0]*np.arange(12,12+periods) + coef[1]

def ensure_ratios(fin):
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

def project_weighted_margin(projects_month):
    df = projects_month.copy()
    if df.empty: return np.nan
    w = df["rp"].clip(lower=0.0)
    base_margin = np.where(df["actual_margin_pct"].notna(), df["actual_margin_pct"], df["planned_margin_pct"])
    return float((w * base_margin).sum() / max(w.sum(), 1e-6))

# Élasticités équilibrées + effets symétriques (valeurs pédagogiques)
def apply_etp_scenarios(gearing_vals, de_vals, portfolio_margin,
                        shock_cost, shock_delay_weeks, shock_weather_days, shock_rate_pts,
                        client_terms_days, supplier_terms_days):
    # Impacts "core" symétriques
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
    # DSO/DPO par rapport à 60j (effet sur ratios : DSO>60 dégrade, DSO<60 améliore ; DPO<60 dégrade, DPO>60 améliore)
    delta_dso = client_terms_days - 60
    delta_dpo = supplier_terms_days - 60
    g_adj = g_adj + 0.003*delta_dso - 0.003*delta_dpo
    de_adj = de_adj + 0.010*(delta_dso/10) - 0.010*(delta_dpo/10)
    # marge portefeuille indicative
    margin_adj = portfolio_margin - (1.0*(shock_cost/10.0) + 0.4*(shock_delay_weeks/4.0) + 0.2*(shock_weather_days/5.0))
    return np.maximum(g_adj, 0.05), np.maximum(de_adj, 0.3), margin_adj

# ---------- Préparation ----------
finance = ensure_ratios(finance)
last_date = finance["date"].max()
h = 12
future_dates = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=h, freq="MS")

fcst = pd.DataFrame({"date": future_dates})
fcst["gearing_fcst"] = simple_forecast(finance["gearing"], periods=h)
fcst["debt_ebitda_fcst"] = simple_forecast(finance["debt_ebitda"], periods=h)

current_month = projects[projects["date"]==projects["date"].max()]
port_margin_now = project_weighted_margin(current_month)

fcst["gearing_scn"], fcst["de_scn"], fcst["margin_portfolio_scn"] = apply_etp_scenarios(
    gearing_vals=fcst["gearing_fcst"].values,
    de_vals=fcst["debt_ebitda_fcst"].values,
    portfolio_margin=port_margin_now if not np.isnan(port_margin_now) else 12.0,
    shock_cost=shock_cost,
    shock_delay_weeks=shock_delay,
    shock_weather_days=shock_weather,
    shock_rate_pts=shock_rate,
    client_terms_days=client_terms,
    supplier_terms_days=supplier_terms
)

# ---------- Probabilité de bris à 12 mois ----------
tmp = finance.copy()
tmp["breach_gearing"] = (tmp["gearing"].rolling(12, min_periods=1).max() > gearing_threshold).shift(-11).fillna(False)
tmp["breach_debt_ebitda"] = (tmp["debt_ebitda"].rolling(12, min_periods=1).max() > debt_ebitda_threshold).shift(-11).fillna(False)
tmp["breach_any"] = (tmp["breach_gearing"] | tmp["breach_debt_ebitda"]).astype(int)

feat = tmp[["gearing","debt_ebitda","cash_net","dso","dpo","rate"]].copy()
for col in ["gearing","debt_ebitda","cash_net"]:
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
    prob_reference = float(model.predict_proba(x_last)[0,1])
else:
    prob_reference = 0.10  # probabilité de référence si peu d'historique

# Uplift symétrique (en points de probabilité) + DSO/DPO centrés à 60j
uplift = (
    0.006*shock_cost +            # ±0,6 pp par ±1% matières
    0.010*(shock_delay/4) +       # ±1,0 pp par 4 semaines
    0.007*(shock_weather/5) +     # ±0,7 pp par 5 jours
    0.030*shock_rate              # ±3,0 pp par 1 point de taux
)
# Effet DSO/DPO sur la probabilité : DSO>60 pénalise, DPO<60 pénalise
delta_dso = client_terms - 60       # >0 si >60 (pénalise), <0 si <60 (bénéfice)
delta_dpo = supplier_terms - 60     # >0 si >60 (bénéfice), <0 si <60 (pénalise)
uplift += 0.0025*delta_dso - 0.0025*delta_dpo

prob_scenario = float(np.clip(prob_reference + uplift, 0.0, 0.99))

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

st.markdown("### 🧭 Lecture & recommandations")
st.markdown(
f"""
- **Hypothèses de scénario** : matières **{shock_cost:+d}%**, retards **{shock_delay:+d} sem**, météo **{shock_weather:+d} j/mois**, taux **{shock_rate:+d} pt** ; DSO **{client_terms} j** (viser ≤ 60), DPO **{supplier_terms} j** (viser ≥ 60).  
- **Règle DSO/DPO** : **DSO > 60** ⟶ probabilité ↑ (dégradation) ; **DPO < 60** ⟶ probabilité ↑ (dégradation). Mouvement inverse ⟶ probabilité ↓ (amélioration).  
- **Marge portefeuille pondérée (RàP)** (indicative) : ~ **{fcst['margin_portfolio_scn'].iloc[0]:.1f}%**.
"""
)

st.info(
"**Bonnes pratiques** : viser **DSO ≤ 60 j** et **DPO ≥ 60 j** ; prioriser les chantiers à forte marge RàP ; "
"préparer un plan de trésorerie ; activer les relances clients ; négocier des conditions fournisseurs ; "
"si 🟠/🔴 durable, anticiper un échange bancaire avec stress tests en annexe."
)

st.caption("Outil pédagogique : à calibrer sur données réelles ETP (inclure idéalement une colonne 'equity' pour un gearing exact).")
