# PRÉVI-COV — ETP (version avancée & robuste)
# - Prévisions : Prophet → SARIMAX → Linéaire (fallback)
# - Nettoyage historique : winsorisation, médiane glissante, rebaselining (optionnel), bornes DSO/DPO
# - Exclusion manuelle de mois
# - Scénarios ETP symétriques (matières, retards, météo, taux, DSO/DPO)
# - Garde-fous métier (ancrage + plafond) sur les prévisions
# - Override "situation idéale" + règle de cohérence Dette/EBITDA → Risque
# - UI en français (probabilité de référence vs probabilité scénario)

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
    "Prévisions avancées (Prophet → SARIMAX → Linéaire), nettoyage de l’historique, scénarios ETP symétriques, "
    "probabilité de bris calibrée, et garde-fous métier."
)

with st.expander("📄 Formats attendus", expanded=False):
    st.markdown("""
**1) finance.csv (mensuel, 1 ligne / mois)**  
- `date` (YYYY-MM-01)  
- `gearing` **ou** `net_debt` & `equity`  
- `debt_ebitda` **ou** `net_debt` & `ebitda`  
- Recommandé : `ebitda`, `net_debt`, `equity`, `cash_net`, `dso`, `dpo`, `steel_index` (optionnel), `rate`

**2) projects.csv (1 ligne / chantier / mois)**  
- `date`, `project_id`, `rp` (reste à produire k€),  
- `planned_margin_pct`, `actual_margin_pct` (si dispo),  
- `delay_weeks`, `weather_days` (jours perdus/mois, optionnel).
""")

# ---------- Upload ----------
c1, c2 = st.columns(2)
with c1:
    fin_file = st.file_uploader("📥 Importer finance.csv", type=["csv"], key="fin")
with c2:
    prj_file = st.file_uploader("📥 Importer projects.csv", type=["csv"], key="prj")

# ---------- Démo si fichiers absents ----------
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

# ---------- Paramètres & Nettoyage ----------
st.sidebar.header("⚙️ Paramètres covenants")
gearing_threshold = st.sidebar.number_input("Seuil Gearing (max)", 0.5, 5.0, 2.0, step=0.1, format="%.1f")
debt_ebitda_threshold = st.sidebar.number_input("Seuil Dette/EBITDA (max)", 1.0, 8.0, 3.0, step=0.1, format="%.1f")

st.sidebar.header("🧹 Nettoyage de l'historique")
enable_clean = st.sidebar.checkbox("Activer le nettoyage (recommandé)", value=True)
allow_rebase = st.sidebar.checkbox("Rebaseliner après rupture (optionnel)", value=False)

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

def winsorize(s: pd.Series, lower_q=0.02, upper_q=0.98) -> pd.Series:
    if s.dropna().empty:
        return s
    lo, hi = s.quantile(lower_q), s.quantile(upper_q)
    return s.clip(lower=lo, upper=hi)

def smooth_rolling_median(s: pd.Series, window=3) -> pd.Series:
    return s.rolling(window, min_periods=1, center=True).median()

def clean_finance(fin: pd.DataFrame, allow_rebase: bool=False) -> pd.DataFrame:
    df = fin.copy().sort_values("date")
    # Plages métier DSO/DPO + correction de sauts aberrants
    if "dso" in df: 
        df["dso"] = df["dso"].clip(30, 120)
        jump = df["dso"].diff().abs() > 40
        df.loc[jump, "dso"] = np.nan
        df["dso"] = df["dso"].interpolate(limit_direction="both")
    if "dpo" in df:
        df["dpo"] = df["dpo"].clip(30, 120)
        jump = df["dpo"].diff().abs() > 40
        df.loc[jump, "dpo"] = np.nan
        df["dpo"] = df["dpo"].interpolate(limit_direction="both")

    # Recalcul “propre” de debt_ebitda si possible en ignorant EBITDA ≤ 0
    if {"net_debt","ebitda"}.issubset(df.columns):
        eb = df["ebitda"].where(df["ebitda"] > 0)
        df["debt_ebitda"] = df["net_debt"] / eb

    # Lissage & winsorisation sur ratios
    for col in ["debt_ebitda","gearing"]:
        if col in df:
            df[col] = smooth_rolling_median(df[col])
            df[col] = winsorize(df[col], 0.02, 0.98)

    # Rebaselining optionnel (rupture forte)
    if allow_rebase:
        for col in ["debt_ebitda","gearing"]:
            if col in df and df[col].notna().sum() >= 18:
                s = df[col]
                m6a = s.rolling(6).median().shift(6)
                m6b = s.rolling(6).median()
                rupt = ((m6b - m6a).abs() / (m6a.abs() + 1e-6)) > 0.35
                if rupt.any():
                    median12 = s.rolling(12, min_periods=6).median()
                    df[col] = 0.5*s + 0.5*median12
    return df

def forecast_with_fallback(series: pd.Series, periods: int = 12) -> np.ndarray:
    """1) Prophet (saisonnalité annuelle) → 2) SARIMAX (m=12) → 3) Linéaire / moyenne récente."""
    s = series.copy().dropna()
    s.index = pd.DatetimeIndex(s.index)
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
            model = sm.tsa.statespace.SARIMAX(s, order=(1,1,1), seasonal_order=(1,1,1,12),
                                              enforce_stationarity=False, enforce_invertibility=False)
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

def apply_etp_scenarios(gearing_vals, de_vals, portfolio_margin,
                        shock_cost, shock_delay_weeks, shock_weather_days, shock_rate_pts,
                        client_terms_days, supplier_terms_days):
    # Impacts symétriques (valeurs pédagogiques)
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
    # DSO/DPO vs 60j : DSO>60 dégrade ; DPO<60 dégrade
    delta_dso = client_terms_days - 60
    delta_dpo = supplier_terms_days - 60
    g_adj = g_adj + 0.003*delta_dso - 0.003*delta_dpo
    de_adj = de_adj + 0.010*(delta_dso/10) - 0.010*(delta_dpo/10)
    # Marge indicative
    margin_adj = portfolio_margin - (1.0*(shock_cost/10.0) + 0.4*(shock_delay_weeks/4.0) + 0.2*(shock_weather_days/5.0))
    return np.maximum(g_adj, 0.05), np.maximum(de_adj, 0.3), margin_adj

def damp_and_clip(series_hist: pd.Series, fcst_vals: np.ndarray,
                  threshold: float, alpha: float = 0.6, cap_mult: float = 1.5) -> np.ndarray:
    """
    Amortit la tendance (ancrage sur médiane 12 mois) et plafonne la prévision pour cohérence métier.
    - alpha : poids du modèle vs. ancrage historique (0.6 = 60% modèle, 40% ancrage)
    - cap_mult : plafond = threshold * cap_mult (ex. 3.0 * 1.5 = 4.5)
    """
    anchor = float(series_hist.tail(12).median()) if len(series_hist) >= 3 else float(series_hist.tail(1))
    blended = alpha * fcst_vals + (1 - alpha) * anchor
    cap = threshold * cap_mult
    return np.clip(blended, 0.0, cap)

# ---------- Préparation & Nettoyage ----------
finance = ensure_ratios(finance)

# Exclusion manuelle de mois (optionnelle)
mois = finance["date"].dt.strftime("%Y-%m").tolist()
to_drop = st.multiselect("Exclure certains mois de l'historique (optionnel)", mois, default=[])
if to_drop:
    mask = ~finance["date"].dt.strftime("%Y-%m").isin(to_drop)
    finance = finance.loc[mask].copy()

if enable_clean:
    finance = clean_finance(finance, allow_rebase=allow_rebase)

last_date = finance["date"].max()
h = 12
future_dates = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=h, freq="MS")

# ---------- Prévisions avancées + garde-fous ----------
fcst = pd.DataFrame({"date": future_dates})

# Gearing
gearing_raw = forecast_with_fallback(finance.set_index("date")["gearing"], periods=h)
fcst["gearing_fcst"] = damp_and_clip(
    finance.set_index("date")["gearing"], np.asarray(gearing_raw),
    threshold=gearing_threshold, alpha=0.6, cap_mult=1.5
)

# Dette/EBITDA (✱ point sensible → amorti + plafond)
de_raw = forecast_with_fallback(finance.set_index("date")["debt_ebitda"], periods=h)
fcst["debt_ebitda_fcst"] = damp_and_clip(
    finance.set_index("date")["debt_ebitda"], np.asarray(de_raw),
    threshold=debt_ebitda_threshold, alpha=0.6, cap_mult=1.5
)

# ---------- Application scénarios + clip post-scénarios ----------
current_month = projects[projects["date"] == projects["date"].max()]
port_margin_now = project_weighted_margin(current_month)

fcst["gearing_scn"], fcst["de_scn"], fcst["margin_portfolio_scn"] = apply_etp_scenarios(
    gearing_vals=fcst["gearing_fcst"].values,
    de_vals=fcst["debt_ebitda_fcst"].values,
    portfolio_margin=port_margin_now if not np.isnan(port_margin_now) else 12.0,
    shock_cost=shock_cost, shock_delay_weeks=shock_delay, shock_weather_days=shock_weather, shock_rate_pts=shock_rate,
    client_terms_days=client_terms, supplier_terms_days=supplier_terms
)

# Contrainte métier post-scénarios (éviter envolées / négatifs)
fcst["de_scn"] = np.clip(fcst["de_scn"], 0.0, debt_ebitda_threshold * 1.7)
fcst["gearing_scn"] = np.clip(fcst["gearing_scn"], 0.0, gearing_threshold * 1.7)

# ---------- Probabilité de bris (12 mois) ----------
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

metrics_text = ""
if y.sum() > 3 and len(finance) >= 30:
    # Split temporel : derniers 6 points = test
    split_idx = len(X) - 6
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test,  y_test  = X[split_idx:], y[split_idx:]

    base_model = Pipeline([
        ("scaler", StandardScaler()),
        ("gb", GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42))
    ])
    base_model.fit(X_train, y_train)

    cal = CalibratedClassifierCV(base_model, method="isotonic", cv=3)
    cal.fit(X_train, y_train)

    try:
        proba_test = cal.predict_proba(X_test)[:,1]
        auc = roc_auc_score(y_test, proba_test)
        brier = brier_score_loss(y_test, proba_test)
        metrics_text = f"**AUC**≈{auc:.2f} • **Brier**≈{brier:.2f}"
    except Exception:
        metrics_text = "_Métriques indisponibles_"

    prob_reference = float(cal.predict_proba(feat.iloc[-1:].values)[:,1])
else:
    prob_reference = 0.10  # probabilité de référence si historique insuffisant
    metrics_text = "_Historique insuffisant pour évaluer AUC/Brier_"

# Uplift symétrique (en points de probabilité) + DSO/DPO
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

# Override “situation idéale” : aucun choc + DSO<=60 + DPO>=60 + ratios sous seuils (coussin 5%)
ideal = (shock_cost <= 0 and shock_delay <= 0 and shock_weather <= 0 and shock_rate <= 0
         and client_terms <= 60 and supplier_terms >= 60)
within_buffers = (
    (np.nanmax(fcst["gearing_scn"].values) <= gearing_threshold * 0.95) and
    (np.nanmax(fcst["de_scn"].values)     <= debt_ebitda_threshold * 0.95)
)
if ideal and within_buffers:
    prob_scenario = min(prob_scenario, 0.05)

# Règle métier explicite : plus Dette/EBITDA est haut, plus le risque ↑ ; si très bas, risque ↓
max_debt_ebitda = float(np.nanmax(fcst["de_scn"].values))
if max_debt_ebitda > debt_ebitda_threshold:
    prob_scenario = min(0.99, prob_scenario + 0.15)
elif max_debt_ebitda < debt_ebitda_threshold * 0.8:
    prob_scenario = max(0.01, prob_scenario - 0.10)

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

st.caption("Nota : les prévisions sont **amorties et plafonnées** pour rester réalistes vis-à-vis des covenants "
           "(ancrage médiane 12 mois, plafond = 1,5× le seuil ; re-clip après scénarios).")

st.markdown("---")
st.subheader("🛑 Probabilité de bris (12 mois)")
c1, c2, c3 = st.columns(3)
c1.metric("Probabilité (référence)", f"{prob_reference:.0%}")
c2.metric("Probabilité (scénario)", f"{prob_scenario:.0%}", delta=f"{(prob_scenario-prob_reference):+.0%}")
statut = "🟢 Sûr" if prob_scenario < 0.30 else ("🟠 Sous surveillance" if prob_scenario < 0.60 else "🔴 Risque élevé")
c3.metric("Statut", statut)

if metrics_text:
    st.caption("Qualité du modèle : " + metrics_text)

st.markdown("### 🧭 Lecture & recommandations")
st.markdown(
f"""
- **Scénario** : matières **{shock_cost:+d}%**, retards **{shock_delay:+d} sem**, météo **{shock_weather:+d} j/mois**, taux **{shock_rate:+d} pt** ; DSO **{client_terms} j** (≤ 60), DPO **{supplier_terms} j** (≥ 60).  
- Effets **symétriques** : un scénario favorable réduit le risque ; défavorable l’augmente.  
- **Règles de cohérence** : plafonds métier sur les prévisions ; renforcement du risque si Dette/EBITDA > seuil ; override vert en situation idéale.
"""
)

st.info(
"Bonnes pratiques : viser **DSO ≤ 60 j** et **DPO ≥ 60 j** ; prioriser chantiers à forte marge RàP ; "
"plan de trésorerie et relances clients ; négociation fournisseurs ; si 🟠/🔴 durable, anticiper un échange bancaire (stress tests)."
)

st.caption("Dépendances : streamlit, pandas, numpy, scikit-learn, plotly, prophet, statsmodels.")
