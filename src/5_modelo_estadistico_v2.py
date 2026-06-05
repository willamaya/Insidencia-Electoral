"""
5_modelo_estadistico_v2.py — Fase 1: HLM + WAPE + De Finetti
=============================================================
Mejoras sobre el OLS original (5_modelo_estadistico.py):

  1. HLM (Hierarchical Linear Model) con pendiente aleatoria por municipio
     votos_SE ~ (beta_p + u_m) * votos_CA + gamma_m + epsilon
     donde u_m ~ N(0, tau²) — el arrastre varía por municipio

  2. WAPE (Weighted Absolute Percentage Error) como métrica principal
     WAPE = Σ|pred - real| * real / Σreal

  3. Test formal de sobredispersión (régimen de De Finetti)
     H0: votantes independientes (Binomial)
     H1: cohesión social (sobredispersión, régimen supercrítico)

Genera:
  outputs/hlm_resultados.csv         — coeficientes HLM por colectividad
  outputs/hlm_vs_ols.csv             — comparativa R² OLS vs HLM
  outputs/definetti_test.csv         — test de sobredispersión
  outputs/metricas_v2.csv            — WAPE + MAE + RMSE
"""
import logging
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_PROC, DATA_INTERIM, OUTPUTS

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

try:
    import statsmodels.formula.api as smf
    import statsmodels.api as sm
    HAS_SM = True
except ImportError:
    HAS_SM = False
    log.error("statsmodels no instalado")

COLECTIVIDADES = [
    "ALIANZA VERDE", "CENTRO DEMOCRATICO", "CONSERVADOR-SALV NACIONAL",
    "CR-NUEVO LIBERALISMO", "PARTIDO DE LA U", "PARTIDO LIBERAL",
]


# ── 1. HLM con pendiente aleatoria ───────────────────────────────────────────

def ajustar_hlm(df: pd.DataFrame, colectividad: str) -> dict:
    """
    Modelo mixto:
      votos_SE ~ votos_CA_cand + (votos_CA_cand | municipio)

    Equivalente statsmodels:
      MixedLM(endog=SE, exog=[CA], groups=municipio, exog_re=[CA])

    Retorna dict con coeficientes, tau², sigma², comparativa con OLS.
    """
    sub = df[df["colectividad"] == colectividad].dropna(
        subset=["votos","votos_se_colectividad_mesa"]).copy()
    sub = sub[sub["votos"] > 0].copy()
    sub.rename(columns={"votos":"votos_ca","votos_se_colectividad_mesa":"votos_se"}, inplace=True)

    if len(sub) < 50 or sub["amb_municipio"].nunique() < 5:
        log.warning(f"  {colectividad}: datos insuficientes para HLM")
        return None

    n  = len(sub)
    n_m = sub["amb_municipio"].nunique()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            # ── HLM: pendiente aleatoria por municipio ─────────────────────
            model_hlm = smf.mixedlm(
                "votos_se ~ votos_ca",
                data=sub,
                groups=sub["amb_municipio"],
                re_formula="~votos_ca"          # pendiente aleatoria
            )
            res_hlm = model_hlm.fit(reml=True, method="lbfgs")

            # ── OLS (sin efectos fijos, para comparativa justa) ────────────
            model_ols = smf.ols("votos_se ~ votos_ca + C(amb_municipio)", data=sub)
            res_ols   = model_ols.fit()

            # ── OLS simple (sin controles) ─────────────────────────────────
            model_ols_simple = smf.ols("votos_se ~ votos_ca", data=sub)
            res_ols_simple   = model_ols_simple.fit()

            # Extraer parámetros HLM
            beta_hlm  = res_hlm.fe_params.get("votos_ca", np.nan)
            intercept = res_hlm.fe_params.get("Intercept", np.nan)
            sigma2    = res_hlm.scale                          # varianza residual
            cov_re    = res_hlm.cov_re                         # var-cov efectos aleatorios
            tau2_int  = float(cov_re.iloc[0,0]) if cov_re.shape[0] > 0 else np.nan
            tau2_slp  = float(cov_re.iloc[1,1]) if cov_re.shape[0] > 1 else np.nan

            # R² HLM aproximado (correlación pred-obs)
            pred_hlm  = res_hlm.fittedvalues
            ss_res    = np.sum((sub["votos_se"].values - pred_hlm)**2)
            ss_tot    = np.sum((sub["votos_se"].values - sub["votos_se"].mean())**2)
            r2_hlm    = 1 - ss_res/ss_tot

            # ICC (Intra-class Correlation Coefficient)
            # ICC = tau²_intercept / (tau²_intercept + sigma²)
            icc = tau2_int / (tau2_int + sigma2) if (tau2_int + sigma2) > 0 else np.nan

            # WAPE por modelo
            def wape(y_true, y_pred):
                return (np.abs(y_true - y_pred) * y_true).sum() / y_true.sum()

            wape_hlm = wape(sub["votos_se"].values, pred_hlm)
            wape_ols = wape(sub["votos_se"].values, res_ols.fittedvalues)
            wape_det = wape(sub["votos_se"].values, sub["votos_ca"].values)  # modelo actual

            log.info(f"  {colectividad}:")
            log.info(f"    β_HLM={beta_hlm:.4f}  τ²_slope={tau2_slp:.4f}  σ²={sigma2:.4f}")
            log.info(f"    ICC={icc:.3f}  R²_HLM={r2_hlm:.4f}  R²_OLS={res_ols.rsquared:.4f}")
            log.info(f"    WAPE: det={wape_det:.3f}  OLS={wape_ols:.3f}  HLM={wape_hlm:.3f}")

            return {
                "colectividad"       : colectividad,
                "n_obs"              : n,
                "n_municipios"       : n_m,
                "beta_hlm"           : round(beta_hlm, 4),
                "beta_ols_fe"        : round(res_ols.params.get("votos_ca",np.nan), 4),
                "beta_ols_simple"    : round(res_ols_simple.params.get("votos_ca",np.nan), 4),
                "ic95_low_hlm"       : round(res_hlm.conf_int().loc["votos_ca",0], 4),
                "ic95_high_hlm"      : round(res_hlm.conf_int().loc["votos_ca",1], 4),
                "tau2_intercept"     : round(tau2_int, 4),
                "tau2_slope"         : round(tau2_slp, 6) if not np.isnan(tau2_slp) else np.nan,
                "sigma2_residual"    : round(sigma2, 4),
                "icc"                : round(icc, 4),
                "r2_hlm_aprox"       : round(r2_hlm, 4),
                "r2_ols_fe"          : round(res_ols.rsquared, 4),
                "r2_ols_simple"      : round(res_ols_simple.rsquared, 4),
                "wape_deterministico": round(wape_det, 4),
                "wape_ols_fe"        : round(wape_ols, 4),
                "wape_hlm"           : round(wape_hlm, 4),
                "mejora_wape_pct"    : round((wape_ols - wape_hlm) / wape_ols * 100, 2),
                "aic_hlm"            : round(res_hlm.aic, 2),
                "bic_hlm"            : round(res_hlm.bic, 2),
                "convergencia"       : "OK",
            }

        except Exception as e:
            log.warning(f"  HLM falló para {colectividad}: {e}")
            return {"colectividad": colectividad, "convergencia": f"FALLO: {e}"}


# ── 2. Test de sobredispersión — Régimen de De Finetti ────────────────────────

def test_definetti(df: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada colectividad, testea sobredispersión de votos/potencial
    respecto al modelo Binomial.

    H0: Var(C_ij/N_i) = p(1-p)/N_i   (votantes independientes)
    H1: Var observada >> esperada     (cohesión social → régimen supercrítico)

    Estadístico: φ = Var_obs / Var_Binomial
    φ ≈ 1     → régimen subcrítico (independencia)
    φ >> 1    → régimen supercrítico (cohesión, correlación positiva)

    Estimación del exponente α de decaimiento n^{-α}:
    Si Var ~ n^{-α}: α = -log(Var_obs/Var_ref) / log(n/n_ref)
    """
    resultados = []
    for col in COLECTIVIDADES:
        sub = df[df["colectividad"] == col].dropna(
            subset=["votos","potencial_electoral"]).copy()
        sub = sub[sub["potencial_electoral"] > 0].copy()
        if len(sub) < 30:
            continue

        # Proporción observada por mesa
        p_hat = sub["votos"] / sub["potencial_electoral"]
        p_bar = p_hat.mean()
        n_bar = sub["potencial_electoral"].mean()

        var_obs = p_hat.var()
        var_bin = p_bar * (1 - p_bar) / n_bar   # varianza Binomial esperada

        phi = var_obs / var_bin if var_bin > 0 else np.nan

        # Test chi² de sobredispersión (Pearson)
        # X² = Σ (O-E)² / E con E = N_i * p_bar
        E = sub["potencial_electoral"] * p_bar
        O = sub["votos"]
        chi2_stat = ((O - E)**2 / E.replace(0, np.nan)).sum()
        df_chi = len(sub) - 1
        p_value = 1 - stats.chi2.cdf(chi2_stat, df_chi)

        # Estimación exponente α por cuantiles de tamaño de mesa
        mediana_n = sub["potencial_electoral"].median()
        grandes  = sub[sub["potencial_electoral"] > mediana_n]["p_hat" if "p_hat" in sub.columns else p_hat.index] if False else p_hat[sub["potencial_electoral"] > mediana_n]
        pequeñas = p_hat[sub["potencial_electoral"] <= mediana_n]
        var_G = grandes.var()
        var_P = pequeñas.var()
        n_G   = sub[sub["potencial_electoral"] > mediana_n]["potencial_electoral"].mean()
        n_P   = sub[sub["potencial_electoral"] <= mediana_n]["potencial_electoral"].mean()
        if var_G > 0 and var_P > 0 and n_G != n_P:
            alpha_est = -np.log(var_G/var_P) / np.log(n_G/n_P)
        else:
            alpha_est = np.nan

        if 0 < alpha_est < 1:
            regime = f"SUBCRÍTICO (α={alpha_est:.2f})"
        elif alpha_est >= 1:
            regime = f"CRÍTICO/NEUTRO (α={alpha_est:.2f})"
        else:
            regime = f"SUPERCRÍTICO (φ={phi:.1f})"

        log.info(f"  [{col}] φ={phi:.2f}  p-val_chi2={p_value:.2e}  régimen={regime}")

        resultados.append({
            "colectividad"      : col,
            "n_mesas"           : len(sub),
            "p_bar"             : round(p_bar, 4),
            "n_bar_potencial"   : round(n_bar, 1),
            "var_observada"     : round(var_obs, 8),
            "var_binomial"      : round(var_bin, 8),
            "phi_sobredispersion": round(phi, 2),
            "chi2_stat"         : round(chi2_stat, 1),
            "p_valor_chi2"      : p_value,
            "sobredispersion"   : phi > 2,
            "alpha_decaimiento" : round(alpha_est, 3) if not np.isnan(alpha_est) else np.nan,
            "regimen_definetti" : regime,
            "interpretacion"    : (
                "Cohesión social alta: los errores estándar OLS están subestimados. "
                "El HLM con RE y corrección de sobredispersión es necesario."
                if phi > 2 else
                "Comportamiento cercano a independencia: OLS es adecuado."
            )
        })

    return pd.DataFrame(resultados)


# ── 3. Métricas WAPE + MAE + RMSE comparativas ───────────────────────────────

def calcular_metricas(df: pd.DataFrame, hlm_results: list) -> pd.DataFrame:
    """Tabla comparativa de métricas entre modelo determinístico, OLS y HLM."""
    rows = []
    hlm_map = {r["colectividad"]: r for r in hlm_results if r and "wape_hlm" in r}

    for col in COLECTIVIDADES:
        sub = df[df["colectividad"] == col].dropna(
            subset=["votos","votos_se_colectividad_mesa"]).copy()
        sub = sub[sub["votos"] > 0]
        if len(sub) < 10:
            continue

        y    = sub["votos_se_colectividad_mesa"].values
        pred = sub["dominancia_mesa"].values * y  # modelo determinístico

        mae  = mean_absolute_error(y, pred)
        rmse = np.sqrt(mean_squared_error(y, pred))
        wape = (np.abs(y - pred) * y).sum() / y.sum()
        r2   = 1 - np.sum((y-pred)**2)/np.sum((y-y.mean())**2)

        hlm = hlm_map.get(col, {})
        row = {
            "colectividad"         : col,
            "n"                    : len(sub),
            "MAE_deterministico"   : round(mae, 3),
            "RMSE_deterministico"  : round(rmse, 3),
            "WAPE_deterministico"  : round(wape, 4),
            "R2_deterministico"    : round(r2, 4),
            "WAPE_OLS_FE"          : hlm.get("wape_ols_fe", np.nan),
            "WAPE_HLM"             : hlm.get("wape_hlm", np.nan),
            "R2_OLS_FE"            : hlm.get("r2_ols_fe", np.nan),
            "R2_HLM"               : hlm.get("r2_hlm_aprox", np.nan),
            "mejora_HLM_vs_OLS_%"  : hlm.get("mejora_wape_pct", np.nan),
            "ICC"                  : hlm.get("icc", np.nan),
            "tau2_slope"           : hlm.get("tau2_slope", np.nan),
        }
        rows.append(row)
    return pd.DataFrame(rows)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    if not HAS_SM:
        log.error("statsmodels requerido. pip install statsmodels")
        return

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATA_PROC / "mesa_consolidada.parquet")
    log.info(f"Datos cargados: {len(df):,} filas")

    # ── 1. HLM ──────────────────────────────────────────────────────────────
    log.info("\n=== FASE 1A: HLM con pendiente aleatoria ===")
    hlm_results = []
    for col in COLECTIVIDADES:
        log.info(f"Ajustando HLM: {col}")
        res = ajustar_hlm(df, col)
        if res:
            hlm_results.append(res)

    df_hlm = pd.DataFrame(hlm_results)
    df_hlm.to_csv(OUTPUTS / "hlm_resultados.csv", index=False)
    log.info(f"HLM guardado: {OUTPUTS / 'hlm_resultados.csv'}")

    # ── 2. Test De Finetti ────────────────────────────────────────────────
    log.info("\n=== FASE 1B: Test Sobredispersión (De Finetti) ===")
    df_def = test_definetti(df)
    df_def.to_csv(OUTPUTS / "definetti_test.csv", index=False)
    log.info(f"De Finetti guardado: {OUTPUTS / 'definetti_test.csv'}")

    # ── 3. Métricas comparativas ──────────────────────────────────────────
    log.info("\n=== FASE 1C: Métricas comparativas ===")
    df_met = calcular_metricas(df, hlm_results)
    df_met.to_csv(OUTPUTS / "metricas_v2.csv", index=False)
    log.info(f"Métricas guardadas: {OUTPUTS / 'metricas_v2.csv'}")

    # ── 4. Resumen en consola ─────────────────────────────────────────────
    log.info("\n" + "="*70)
    log.info("RESUMEN COMPARATIVO OLS vs HLM")
    log.info("="*70)

    cols_show = ["colectividad","beta_hlm","beta_ols_fe","tau2_slope",
                 "icc","r2_hlm_aprox","r2_ols_fe","wape_hlm","wape_ols_fe","mejora_wape_pct"]
    cols_ok = [c for c in cols_show if c in df_hlm.columns]
    if cols_ok:
        log.info("\n" + df_hlm[cols_ok].to_string(index=False))

    log.info("\n" + "="*70)
    log.info("RÉGIMEN DE DE FINETTI")
    log.info("="*70)
    cols_def = ["colectividad","phi_sobredispersion","p_valor_chi2","regimen_definetti"]
    cols_ok_def = [c for c in cols_def if c in df_def.columns]
    if cols_ok_def:
        log.info("\n" + df_def[cols_ok_def].to_string(index=False))

    log.info("\n" + "="*70)
    log.info("MÉTRICAS WAPE/R² COMPARATIVAS")
    log.info("="*70)
    cols_met = ["colectividad","WAPE_deterministico","WAPE_OLS_FE","WAPE_HLM",
                "R2_OLS_FE","R2_HLM","mejora_HLM_vs_OLS_%","ICC"]
    cols_ok_met = [c for c in cols_met if c in df_met.columns]
    if cols_ok_met:
        log.info("\n" + df_met[cols_ok_met].to_string(index=False))

    return df_hlm, df_def, df_met


if __name__ == "__main__":
    main()
