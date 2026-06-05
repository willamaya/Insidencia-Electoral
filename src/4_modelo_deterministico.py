"""
4_modelo_deterministico.py — Modelo de arrastre (coattail) mesa por mesa
Genera:
  data/processed/mesa_consolidada.parquet    ← CA + SE lado a lado por mesa
  data/processed/arrastre_candidato.parquet  ← índices por candidato CA
  outputs/ranking_candidatos_incidencia.csv
  outputs/ranking_municipios_incidencia.csv
"""
import logging
import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_INTERIM, DATA_PROC, OUTPUTS, MESA_KEY, CONFIG

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ── Paso 1: votos de Cámara por candidato × mesa ──────────────────────────────

def tabla_candidatos_camara(df_ca: pd.DataFrame) -> pd.DataFrame:
    """
    Una fila = candidato × mesa.
    Solo candidatos individuales (tipo_voto='preferente').
    Incluye: votos del candidato en esa mesa y votos totales del partido en esa mesa.
    Añade dominancia = votos_candidato / votos_partido_mesa.
    """
    pref = df_ca[df_ca["tipo_voto"] == "preferente"].copy()

    # votos del partido por mesa (incluye lista + preferente)
    vp_mesa = (df_ca.groupby(MESA_KEY + ["codpar"])["votos"]
               .sum().reset_index()
               .rename(columns={"votos": "votos_partido_ca_mesa"}))

    pref = pref.merge(vp_mesa, on=MESA_KEY + ["codpar"], how="left")
    pref["dominancia_mesa"] = np.where(
        pref["votos_partido_ca_mesa"] > 0,
        pref["votos"] / pref["votos_partido_ca_mesa"],
        0.0
    )
    return pref


# ── Paso 2: votos de Senado por colectividad × mesa ───────────────────────────

def tabla_senado_por_colectividad(df_se: pd.DataFrame,
                                   hom: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega los votos de Senado por colectividad y mesa,
    aplicando el peso de homologación para coaliciones partidas (ej: codpar=121 → 50%+50%).
    """
    # votos totales senado por codpar × mesa
    se_mesa = (df_se.groupby(MESA_KEY + ["codpar", "colectividad"])["votos"]
               .sum().reset_index()
               .rename(columns={"votos": "votos_se_codpar_mesa"}))

    # Cruzar con pesos del homologación
    hom_sub = hom[["se_codpar", "colectividad", "peso"]].copy()
    hom_sub["se_codpar"] = hom_sub["se_codpar"].astype(str)
    se_mesa["codpar"]    = se_mesa["codpar"].astype(str)

    se_mesa = se_mesa.merge(hom_sub, left_on=["codpar", "colectividad"],
                            right_on=["se_codpar", "colectividad"], how="left")
    se_mesa["peso"] = se_mesa["peso"].fillna(1.0)
    se_mesa["votos_se_ponderado"] = se_mesa["votos_se_codpar_mesa"] * se_mesa["peso"]

    # Consolidar por colectividad × mesa (suma de pesos aplicados)
    se_col = (se_mesa.groupby(MESA_KEY + ["colectividad"])["votos_se_ponderado"]
              .sum().reset_index()
              .rename(columns={"votos_se_ponderado": "votos_se_colectividad_mesa"}))
    return se_col


# ── Paso 3: tabla consolidada CA + SE por mesa ────────────────────────────────

def consolidar_mesa(cands: pd.DataFrame,
                    se_col: pd.DataFrame) -> pd.DataFrame:
    """
    Une candidatos CA con los votos SE de su colectividad en la misma mesa.
    Calcula:
      ratio_arrastre_mesa  = votos_se_colectividad / votos_partido_ca_mesa
      incidencia_bruta     = votos_se / votos_candidato  (por mesa)
      incidencia_ponderada = (votos_se × dominancia) / votos_candidato
    """
    merged = cands.merge(se_col, on=MESA_KEY + ["colectividad"], how="left")
    merged["votos_se_colectividad_mesa"] = (
        merged["votos_se_colectividad_mesa"].fillna(0))

    merged["ratio_arrastre_mesa"] = np.where(
        merged["votos_partido_ca_mesa"] > 0,
        merged["votos_se_colectividad_mesa"] / merged["votos_partido_ca_mesa"],
        np.nan
    )
    merged["incidencia_bruta_mesa"] = np.where(
        merged["votos"] > 0,
        merged["votos_se_colectividad_mesa"] / merged["votos"],
        np.nan
    )
    merged["incidencia_ponderada_mesa"] = np.where(
        merged["votos"] > 0,
        (merged["votos_se_colectividad_mesa"] * merged["dominancia_mesa"]) / merged["votos"],
        np.nan
    )
    return merged


# ── Paso 4: índice de incidencia por candidato ────────────────────────────────

def indice_por_candidato(mesa: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega la tabla de mesa consolidada al nivel candidato.

    indice_incidencia       = Σ votos_se_colectividad (mesas con votos CA)
                              / Σ votos_candidato_ca
    indice_incidencia_pond  = Σ (votos_se × dominancia) / Σ votos_candidato_ca
    ratio_arrastre_global   = votos_se_colectividad_total / votos_partido_ca_total
    cobertura_territorial   = nº mesas con votos CA / nº mesas totales del municipio (aprox)
    """
    grp = mesa.groupby(
        ["cedula", "candidato", "codpar", "nombre_partido", "colectividad",
         "amb_depto", "depto"]
    ).agg(
        municipios          = ("amb_municipio", "nunique"),
        puestos             = ("amb_puesto",    "nunique"),
        mesas_con_votos_ca  = ("amb_mesa",      "nunique"),
        votos_ca_total      = ("votos",          "sum"),
        votos_partido_ca_total = ("votos_partido_ca_mesa", "sum"),
        votos_se_bruto_total   = ("votos_se_colectividad_mesa", "sum"),
        votos_se_ponderado_total = (
            "votos_se_colectividad_mesa",
            lambda x: (x * mesa.loc[x.index, "dominancia_mesa"]).sum()
        ),
    ).reset_index()

    grp["indice_incidencia"] = np.where(
        grp["votos_ca_total"] > 0,
        grp["votos_se_bruto_total"] / grp["votos_ca_total"],
        np.nan
    )
    grp["indice_incidencia_pond"] = np.where(
        grp["votos_ca_total"] > 0,
        grp["votos_se_ponderado_total"] / grp["votos_ca_total"],
        np.nan
    )
    grp["ratio_arrastre_global"] = np.where(
        grp["votos_partido_ca_total"] > 0,
        grp["votos_se_bruto_total"] / grp["votos_partido_ca_total"],
        np.nan
    )
    return grp.sort_values("indice_incidencia_pond", ascending=False)


# ── Paso 5: índice por municipio ──────────────────────────────────────────────

def indice_por_municipio(mesa: pd.DataFrame) -> pd.DataFrame:
    grp = mesa.groupby(
        ["amb_municipio", "municipio", "colectividad", "codpar", "nombre_partido"]
    ).agg(
        votos_ca_partido    = ("votos_partido_ca_mesa", lambda x: x.groupby(
                                mesa.loc[x.index, "amb_mesa"]).first().sum()),
        votos_ca_candidatos = ("votos",                 "sum"),
        votos_se            = ("votos_se_colectividad_mesa", lambda x: x.groupby(
                                mesa.loc[x.index, "amb_mesa"]).first().sum()),
    ).reset_index()

    grp["ratio_arrastre"] = np.where(
        grp["votos_ca_partido"] > 0,
        grp["votos_se"] / grp["votos_ca_partido"],
        np.nan
    )
    return grp.sort_values(["colectividad", "ratio_arrastre"], ascending=[True, False])


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    DATA_PROC.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    df_ca = pd.read_parquet(DATA_INTERIM / "votos_camara_mesa.parquet")
    df_se = pd.read_parquet(DATA_INTERIM / "votos_senado_mesa.parquet")
    hom   = pd.read_csv(CONFIG / "homologacion_partidos.csv", dtype=str)
    hom["peso"] = hom["peso"].astype(float)

    log.info("Paso 1: candidatos Cámara × mesa...")
    cands = tabla_candidatos_camara(df_ca)
    log.info(f"  {len(cands):,} filas (candidato×mesa)")

    log.info("Paso 2: Senado por colectividad × mesa...")
    se_col = tabla_senado_por_colectividad(df_se, hom)
    log.info(f"  {len(se_col):,} filas (colectividad×mesa)")

    log.info("Paso 3: consolidando CA + SE por mesa...")
    mesa = consolidar_mesa(cands, se_col)
    out_mesa = DATA_PROC / "mesa_consolidada.parquet"
    mesa.to_parquet(out_mesa, index=False)
    log.info(f"  Guardado: {out_mesa}  ({len(mesa):,} filas)")

    log.info("Paso 4: índice de incidencia por candidato...")
    idx_cand = indice_por_candidato(mesa)
    out_cand = DATA_PROC / "arrastre_candidato.parquet"
    idx_cand.to_parquet(out_cand, index=False)
    idx_cand.to_csv(OUTPUTS / "ranking_candidatos_incidencia.csv", index=False)
    log.info(f"  Guardado: {OUTPUTS/'ranking_candidatos_incidencia.csv'}  ({len(idx_cand)} candidatos)")

    log.info("Paso 5: índice por municipio...")
    idx_muni = indice_por_municipio(mesa)
    idx_muni.to_csv(OUTPUTS / "ranking_municipios_incidencia.csv", index=False)
    log.info(f"  Guardado: {OUTPUTS/'ranking_municipios_incidencia.csv'}")

    # Resumen en consola
    log.info("\n=== TOP 10 CANDIDATOS POR ÍNDICE DE INCIDENCIA PONDERADO ===")
    top = idx_cand[["candidato", "nombre_partido", "votos_ca_total",
                     "votos_se_bruto_total", "indice_incidencia",
                     "indice_incidencia_pond"]].head(10)
    log.info("\n" + top.to_string(index=False))

    return mesa, idx_cand, idx_muni


if __name__ == "__main__":
    main()
