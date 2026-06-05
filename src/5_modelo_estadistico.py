"""
5_modelo_estadistico.py — Correlaciones y regresión OLS con efectos fijos
Genera:
  outputs/correlaciones.csv
  outputs/regresion_resultados.csv
  outputs/outliers_mesas.csv
"""
import logging
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_PROC, OUTPUTS

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

try:
    import statsmodels.formula.api as smf
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    log.warning("statsmodels no instalado — regresión OLS deshabilitada. "
                "Instala con: pip install statsmodels")


# ── Correlaciones por colectividad ────────────────────────────────────────────

def calcular_correlaciones(mesa: pd.DataFrame) -> pd.DataFrame:
    """
    Por colectividad: Pearson y Spearman entre votos CA candidato y votos SE
    a nivel mesa, municipio y departamento.
    """
    resultados = []
    for col, grp in mesa.groupby("colectividad"):
        grp = grp.dropna(subset=["votos", "votos_se_colectividad_mesa"])
        grp = grp[(grp["votos"] > 0) & (grp["votos_se_colectividad_mesa"] > 0)]
        if len(grp) < 10:
            continue

        x = grp["votos"].values
        y = grp["votos_se_colectividad_mesa"].values

        r_p, p_p = stats.pearsonr(x, y)
        r_s, p_s = stats.spearmanr(x, y)

        # Por municipio (agregado)
        muni = grp.groupby("amb_municipio")[["votos", "votos_se_colectividad_mesa"]].sum()
        r_m, p_m = stats.pearsonr(muni["votos"], muni["votos_se_colectividad_mesa"]) \
            if len(muni) > 3 else (np.nan, np.nan)

        resultados.append({
            "colectividad"    : col,
            "n_mesas"         : len(grp),
            "pearson_mesa"    : round(r_p, 4),
            "p_pearson_mesa"  : round(p_p, 4),
            "spearman_mesa"   : round(r_s, 4),
            "p_spearman_mesa" : round(p_s, 4),
            "pearson_municipio": round(r_m, 4) if not np.isnan(r_m) else None,
            "n_municipios"    : len(muni),
        })

    return pd.DataFrame(resultados).sort_values("pearson_mesa", ascending=False)


# ── Regresión OLS con efectos fijos por municipio ────────────────────────────

def regresion_ols(mesa: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada colectividad:
      votos_se ~ votos_ca + C(municipio)
    Reporta coeficiente de votos_ca, R², p-valor e IC 95%.

    ADVERTENCIA: inferencia ecológica — los coeficientes describen
    relaciones agregadas por mesa, NO comportamiento individual de votantes.
    """
    if not HAS_STATSMODELS:
        log.warning("statsmodels no disponible — saltando regresión OLS")
        return pd.DataFrame()

    resultados = []
    for col, grp in mesa.groupby("colectividad"):
        grp = grp.dropna(subset=["votos", "votos_se_colectividad_mesa"])
        grp = grp[(grp["votos"] > 0)].copy()
        if len(grp) < 30:
            continue

        grp["votos_ca_cand"]  = grp["votos"]
        grp["votos_se_colec"] = grp["votos_se_colectividad_mesa"]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                modelo = smf.ols(
                    "votos_se_colec ~ votos_ca_cand + C(amb_municipio)",
                    data=grp
                ).fit()

                coef  = modelo.params.get("votos_ca_cand", np.nan)
                ci    = modelo.conf_int().loc["votos_ca_cand"] if "votos_ca_cand" in modelo.params else [np.nan, np.nan]
                pval  = modelo.pvalues.get("votos_ca_cand", np.nan)

                resultados.append({
                    "colectividad"        : col,
                    "n_obs"               : len(grp),
                    "n_municipios"        : grp["amb_municipio"].nunique(),
                    "coef_votos_ca"       : round(coef, 4),
                    "ic_95_low"           : round(ci[0], 4),
                    "ic_95_high"          : round(ci[1], 4),
                    "p_valor"             : round(pval, 4),
                    "r2"                  : round(modelo.rsquared, 4),
                    "r2_adj"              : round(modelo.rsquared_adj, 4),
                    "significativo_5pct"  : pval < 0.05,
                    "advertencia"         : "Inferencia ecológica: datos agregados por mesa, NO inferencia individual",
                })

                # Guardar residuos para outliers
                grp["residuo"] = modelo.resid
                grp["colectividad"] = col
                grp.to_parquet(
                    Path(DATA_PROC) / f"residuos_{col.replace(' ','_')}.parquet",
                    index=False
                )
            except Exception as e:
                log.warning(f"OLS falló para {col}: {e}")

    return pd.DataFrame(resultados)


# ── Detección de mesas anómalas ───────────────────────────────────────────────

def detectar_outliers(mesa: pd.DataFrame) -> pd.DataFrame:
    """
    Mesas donde ratio_arrastre_mesa se aleja >3 desviaciones estándar
    del promedio de su colectividad × municipio.
    """
    df = mesa.dropna(subset=["ratio_arrastre_mesa"]).copy()
    df["z_ratio"] = df.groupby(["colectividad", "amb_municipio"])[
        "ratio_arrastre_mesa"
    ].transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))

    outliers = df[df["z_ratio"].abs() > 3][
        ["amb_municipio", "municipio", "amb_puesto", "puesto", "num_mesa",
         "cedula", "candidato", "colectividad",
         "votos", "votos_se_colectividad_mesa", "ratio_arrastre_mesa", "z_ratio"]
    ].sort_values("z_ratio", key=abs, ascending=False)
    return outliers


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    mesa = pd.read_parquet(DATA_PROC / "mesa_consolidada.parquet")
    log.info(f"Mesa consolidada cargada: {len(mesa):,} filas")

    log.info("Calculando correlaciones...")
    corr = calcular_correlaciones(mesa)
    corr.to_csv(OUTPUTS / "correlaciones.csv", index=False)
    log.info(f"\n{corr.to_string(index=False)}")

    log.info("\nEjecutando regresión OLS con efectos fijos...")
    reg = regresion_ols(mesa)
    if not reg.empty:
        reg.to_csv(OUTPUTS / "regresion_resultados.csv", index=False)
        log.info(f"\n{reg[['colectividad','coef_votos_ca','p_valor','r2','significativo_5pct']].to_string(index=False)}")
        log.info("\n⚠ ADVERTENCIA METODOLÓGICA: Los coeficientes de regresión reflejan "
                 "asociaciones entre agregados por mesa. No implican causalidad ni "
                 "comportamiento individual del votante (falacia ecológica).")

    log.info("\nDetectando mesas anómalas...")
    out = detectar_outliers(mesa)
    out.to_csv(OUTPUTS / "outliers_mesas.csv", index=False)
    log.info(f"  {len(out)} mesas anómalas (|z|>3) guardadas")

    return corr, reg, out


if __name__ == "__main__":
    main()
