"""
5_modelo_fase2.py — Fase 2: CLR + Bootstrap sesgo general
==========================================================
Mejoras sobre la Fase 1 (HLM):

  1. Transformación CLR (Centered Log-Ratio) sobre dominancias
     Convierte el simplex Δ^{J-1} al espacio euclidiano ℝ^{J-1}
     Permite OLS/HLM sin violar la restricción de suma

  2. Bootstrap de mesas para:
     a) IC más robustos bajo sobredispersión (vs IC OLS normales)
     b) Estimación del sesgo general de elección b:
        b = sqrt(Var_bootstrap_total - Var_modelo)
     c) Distribución empírica del coeficiente β_p

  3. Comparativa CLR vs dominancia directa en R²/WAPE

Genera:
  outputs/clr_resultados.csv       — modelo con CLR por colectividad
  outputs/bootstrap_beta.csv       — distribución bootstrap de β
  outputs/sesgo_general.csv        — estimación del sesgo b por colectividad
  outputs/comparativa_fases.csv    — OLS / HLM / CLR+HLM lado a lado
"""
import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_PROC, OUTPUTS

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

import statsmodels.formula.api as smf

COLECTIVIDADES = [
    "ALIANZA VERDE", "CENTRO DEMOCRATICO", "CONSERVADOR-SALV NACIONAL",
    "CR-NUEVO LIBERALISMO", "PARTIDO DE LA U", "PARTIDO LIBERAL",
]
B_BOOTSTRAP = 100   # réplicas bootstrap
SEED        = 2026


# ── 1. Transformación CLR ─────────────────────────────────────────────────────

def clr_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula clr(d_ij) para cada candidato j en cada mesa i.

    clr(d_ij) = log(d_ij + ε) − (1/J_i) Σ_j log(d_ij + ε)

    donde:
      J_i = número de candidatos con votos > 0 en la mesa i
      ε   = pseudo-count para estabilidad numérica (no hay ceros en nuestros datos)

    El clr captura la posición relativa del candidato dentro de la composición
    en escala log, preservando las proporciones pero en espacio euclidiano.
    """
    EPS = 1e-9
    out = df.copy()

    # Media geométrica de log(d) por mesa (denominador CLR)
    log_d = np.log(out["dominancia_mesa"].clip(lower=EPS))
    geo_mean_log = log_d.groupby(out["amb_mesa"]).transform("mean")
    out["clr_dominancia"] = log_d - geo_mean_log

    # Estadísticos del CLR para diagnóstico
    log.info(f"  CLR calculado: media={out['clr_dominancia'].mean():.4f}  "
             f"sd={out['clr_dominancia'].std():.4f}  "
             f"min={out['clr_dominancia'].min():.3f}  max={out['clr_dominancia'].max():.3f}")
    return out


# ── 2. Modelo HLM con predictor CLR ──────────────────────────────────────────

def ajustar_hlm_clr(df: pd.DataFrame, colectividad: str) -> dict:
    """
    HLM con predictor CLR en lugar de dominancia directa.

    Modelo A (referencia): votos_SE ~ dominancia * votos_SE_partido + (1|muni)
    Modelo B (CLR):        votos_SE ~ clr(dominancia) + (clr|muni)

    El predictor CLR es el candidato en escala log-relativa.
    La variable dependiente es votos_SE_atribuidos = dominancia × votos_SE_total.
    """
    sub = df[df["colectividad"] == colectividad].dropna(
        subset=["clr_dominancia", "votos", "votos_se_colectividad_mesa"]).copy()
    sub = sub[sub["votos"] > 0].copy()

    if len(sub) < 50 or sub["amb_municipio"].nunique() < 5:
        return None

    # Variable dependiente: log(votos_SE_atribuidos + 1) — más estable
    sub["se_atrib"]      = sub["dominancia_mesa"] * sub["votos_se_colectividad_mesa"]
    sub["log_se_atrib"]  = np.log1p(sub["se_atrib"])
    sub["log_se_total"]  = np.log1p(sub["votos_se_colectividad_mesa"])

    results = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            # Modelo A: dominancia directa → log(SE_atrib)
            mA = smf.mixedlm("log_se_atrib ~ dominancia_mesa",
                             data=sub, groups=sub["amb_municipio"]).fit(reml=True)

            # Modelo B: CLR → log(SE_total) — solo intercepto aleatorio (más estable)
            mB = smf.mixedlm("log_se_total ~ clr_dominancia",
                             data=sub,
                             groups=sub["amb_municipio"]).fit(reml=True)

            def r2_approx(model, y):
                res = y - model.fittedvalues
                return 1 - np.sum(res**2) / np.sum((y - y.mean())**2)

            def wape(y, yhat):
                return (np.abs(y - yhat) * y).sum() / y.sum() if y.sum() > 0 else np.nan

            r2A = r2_approx(mA, sub["log_se_atrib"])
            r2B = r2_approx(mB, sub["log_se_total"])

            # WAPE en escala original
            pred_A_orig = np.expm1(mA.fittedvalues)
            pred_B_orig = np.expm1(mB.fittedvalues)
            wA = wape(sub["se_atrib"].values, pred_A_orig)
            wB = wape(sub["votos_se_colectividad_mesa"].values, pred_B_orig)

            beta_clr = mB.fe_params.get("clr_dominancia", np.nan)
            cov_re = mB.cov_re
            tau2_slp_clr = float(cov_re.iloc[1,1]) if cov_re.shape[0] > 1 else np.nan

            log.info(f"  [{colectividad}] β_CLR={beta_clr:.4f}  "
                     f"R²_A(log_dom)={r2A:.4f}  R²_B(CLR)={r2B:.4f}  "
                     f"WAPE_A={wA:.3f}  WAPE_B={wB:.3f}")

            results = {
                "colectividad"    : colectividad,
                "n"               : len(sub),
                "beta_clr"        : round(beta_clr, 4),
                "tau2_slope_clr"  : round(tau2_slp_clr, 6) if not np.isnan(tau2_slp_clr) else np.nan,
                "r2_hlm_dom"      : round(r2A, 4),
                "r2_hlm_clr"      : round(r2B, 4),
                "wape_hlm_dom"    : round(wA, 4),
                "wape_hlm_clr"    : round(wB, 4),
                "mejora_clr_pct"  : round((wA - wB) / wA * 100, 2) if wA > 0 else np.nan,
                "aic_clr"         : round(mB.aic, 2),
                "bic_clr"         : round(mB.bic, 2),
            }
        except Exception as e:
            log.warning(f"  CLR HLM falló para {colectividad}: {e}")
            results = {"colectividad": colectividad, "error": str(e)}

    return results


# ── 3. Bootstrap por mesas ────────────────────────────────────────────────────

def bootstrap_beta(df: pd.DataFrame, colectividad: str,
                   B: int = B_BOOTSTRAP) -> dict:
    """
    Bootstrap por mesas (cluster bootstrap):
      - Unidad de remuestreo = mesa completa (preserva correlación intra-mesa)
      - En cada réplica: estimar β_OLS simple (votos_SE ~ votos_CA)
      - Distribución de B estimaciones → IC bootstrap + estimación de sesgo

    Sesgo general de elección b:
      b = sqrt(max(0, Var_bootstrap - Var_modelo))
    Representa la variabilidad adicional no capturable con un solo año electoral.
    """
    rng = np.random.default_rng(SEED)
    sub = df[df["colectividad"] == colectividad].dropna(
        subset=["votos","votos_se_colectividad_mesa"]).copy()
    sub = sub[sub["votos"] > 0].copy()
    if len(sub) < 30:
        return None

    mesas = sub["amb_mesa"].unique()
    betas = []

    for _ in range(B):
        # Remuestrear mesas con reemplazo
        sampled = rng.choice(mesas, size=len(mesas), replace=True)
        boot = pd.concat([sub[sub["amb_mesa"] == m] for m in sampled], ignore_index=True)
        try:
            # Regresión matricial directa — 50x más rápido que smf.ols
            X = np.column_stack([np.ones(len(boot)), boot["votos"].values])
            y = boot["votos_se_colectividad_mesa"].values
            beta = np.linalg.lstsq(X, y, rcond=None)[0][1]
            betas.append(beta)
        except Exception:
            continue

    betas = np.array([b for b in betas if not np.isnan(b)])
    if len(betas) < 10:
        return None

    # OLS puntual sobre datos completos
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m_full = smf.ols("votos_se_colectividad_mesa ~ votos", data=sub).fit()

    beta_ols    = m_full.params.get("votos", np.nan)
    se_ols      = m_full.bse.get("votos", np.nan)
    se_boot     = betas.std()
    var_ols     = se_ols**2
    var_boot    = se_boot**2
    # Sesgo general b = varianza extra no explicada por el modelo
    b_sesgo     = np.sqrt(max(0, var_boot - var_ols))

    # IC bootstrap (percentil)
    ic_low  = np.percentile(betas, 2.5)
    ic_high = np.percentile(betas, 97.5)

    log.info(f"  [{colectividad}] β={beta_ols:.4f}  "
             f"SE_OLS={se_ols:.4f}  SE_boot={se_boot:.4f}  "
             f"b_sesgo={b_sesgo:.4f}  "
             f"IC_boot=[{ic_low:.4f}, {ic_high:.4f}]")

    return {
        "colectividad"   : colectividad,
        "beta_ols"       : round(beta_ols, 4),
        "se_ols"         : round(se_ols, 4),
        "se_bootstrap"   : round(se_boot, 4),
        "ratio_se"       : round(se_boot / se_ols, 2) if se_ols > 0 else np.nan,
        "b_sesgo_general": round(b_sesgo, 4),
        "ic_ols_low"     : round(beta_ols - 1.96*se_ols, 4),
        "ic_ols_high"    : round(beta_ols + 1.96*se_ols, 4),
        "ic_boot_low"    : round(ic_low, 4),
        "ic_boot_high"   : round(ic_high, 4),
        "ic_amplitud_ols" : round(1.96*se_ols*2, 4),
        "ic_amplitud_boot": round(ic_high - ic_low, 4),
        "n_replicas"     : len(betas),
        "beta_dist_mean" : round(betas.mean(), 4),
        "beta_dist_sd"   : round(betas.std(), 4),
        "beta_dist_skew" : round(float(stats.skew(betas)), 3),
    }


# ── 4. Tabla comparativa global ───────────────────────────────────────────────

def tabla_comparativa(hlm_v1, clr_results, boot_results):
    """Consolida OLS (Fase 0) / HLM Fase 1 / CLR+HLM Fase 2 / Bootstrap."""
    hlm1 = pd.read_csv(OUTPUTS / "hlm_resultados.csv") if (OUTPUTS / "hlm_resultados.csv").exists() else pd.DataFrame()
    ols  = pd.read_csv(OUTPUTS / "regresion_resultados.csv") if (OUTPUTS / "regresion_resultados.csv").exists() else pd.DataFrame()

    clr_df  = pd.DataFrame([r for r in clr_results  if r and "beta_clr"    in r])
    boot_df = pd.DataFrame([r for r in boot_results if r and "se_bootstrap" in r])

    merged = ols[["colectividad","coef_votos_ca","r2","r2_adj"]].rename(
        columns={"coef_votos_ca":"beta_OLS","r2":"R2_OLS","r2_adj":"R2adj_OLS"})

    if not hlm1.empty:
        merged = merged.merge(
            hlm1[["colectividad","beta_hlm","r2_hlm_aprox","wape_hlm","icc","tau2_slope"]].rename(
                columns={"r2_hlm_aprox":"R2_HLM","wape_hlm":"WAPE_HLM"}),
            on="colectividad", how="left")

    if not clr_df.empty:
        merged = merged.merge(
            clr_df[["colectividad","beta_clr","r2_hlm_clr","wape_hlm_clr","mejora_clr_pct"]].rename(
                columns={"r2_hlm_clr":"R2_CLR","wape_hlm_clr":"WAPE_CLR"}),
            on="colectividad", how="left")

    if not boot_df.empty:
        merged = merged.merge(
            boot_df[["colectividad","se_bootstrap","ratio_se","b_sesgo_general",
                      "ic_boot_low","ic_boot_high"]],
            on="colectividad", how="left")

    return merged


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    df_raw = pd.read_parquet(DATA_PROC / "mesa_consolidada.parquet")
    log.info(f"Datos: {len(df_raw):,} filas")

    # ── CLR transformation ───────────────────────────────────────────────────
    log.info("\n=== FASE 2A: Transformación CLR ===")
    df = clr_transform(df_raw)

    # ── HLM con CLR ─────────────────────────────────────────────────────────
    log.info("\n=== FASE 2B: HLM con predictor CLR ===")
    clr_results = []
    for col in COLECTIVIDADES:
        log.info(f"Ajustando CLR-HLM: {col}")
        res = ajustar_hlm_clr(df, col)
        if res:
            clr_results.append(res)

    df_clr = pd.DataFrame(clr_results)
    df_clr.to_csv(OUTPUTS / "clr_resultados.csv", index=False)

    # ── Bootstrap ────────────────────────────────────────────────────────────
    log.info(f"\n=== FASE 2C: Bootstrap ({B_BOOTSTRAP} réplicas por colectividad) ===")
    boot_results = []
    for col in COLECTIVIDADES:
        log.info(f"Bootstrap: {col}")
        res = bootstrap_beta(df, col)
        if res:
            boot_results.append(res)

    df_boot = pd.DataFrame(boot_results)
    df_boot.to_csv(OUTPUTS / "bootstrap_beta.csv", index=False)

    # Sesgo general
    df_sesgo = df_boot[["colectividad","b_sesgo_general","ratio_se",
                         "ic_amplitud_ols","ic_amplitud_boot"]].copy()
    df_sesgo.to_csv(OUTPUTS / "sesgo_general.csv", index=False)

    # ── Tabla comparativa ────────────────────────────────────────────────────
    log.info("\n=== TABLA COMPARATIVA FASE 0→1→2 ===")
    df_comp = tabla_comparativa(None, clr_results, boot_results)
    df_comp.to_csv(OUTPUTS / "comparativa_fases.csv", index=False)

    # ── Resumen consola ──────────────────────────────────────────────────────
    log.info("\n" + "="*75)
    log.info("FASE 2B — CLR: β_CLR  y mejora sobre HLM con dominancia directa")
    log.info("="*75)
    if not df_clr.empty:
        cols = ["colectividad","beta_clr","r2_hlm_dom","r2_hlm_clr","wape_hlm_dom","wape_hlm_clr","mejora_clr_pct"]
        log.info("\n" + df_clr[[c for c in cols if c in df_clr.columns]].to_string(index=False))

    log.info("\n" + "="*75)
    log.info("FASE 2C — Bootstrap: SE_bootstrap vs SE_OLS y sesgo general b")
    log.info("="*75)
    if not df_boot.empty:
        cols = ["colectividad","beta_ols","se_ols","se_bootstrap","ratio_se",
                "b_sesgo_general","ic_amplitud_ols","ic_amplitud_boot"]
        log.info("\n" + df_boot[[c for c in cols if c in df_boot.columns]].to_string(index=False))

    log.info("\n" + "="*75)
    log.info("COMPARATIVA COMPLETA: OLS → HLM → CLR+HLM")
    log.info("="*75)
    if not df_comp.empty:
        log.info("\n" + df_comp.to_string(index=False))

    return df_clr, df_boot, df_comp


if __name__ == "__main__":
    main()
