"""
2_parseo.py — Normaliza la DB cruda → esquema tidy por mesa
Genera:
  data/interim/votos_camara_mesa.parquet
  data/interim/votos_senado_mesa.parquet
"""
import logging
import sqlite3
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DB_PATH, DATA_INTERIM, DEPTO, CAM_CA, CAM_SE

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def derivar_tipo_voto(df: pd.DataFrame) -> pd.Series:
    """
    Devuelve una Serie con el tipo de voto por fila:
      'preferente' → candidato individual con voto nominado
      'lista'      → SOLO POR LA LISTA (codcan=0)
    """
    return df["codcan"].apply(
        lambda c: "lista" if str(c).strip() in ("0", "") else "preferente"
    )


def cargar_tabla(con: sqlite3.Connection, tabla: str, corporacion: str,
                 cam: str, hom: pd.DataFrame) -> pd.DataFrame:
    """
    Lee la tabla de votos, aplica filtros, añade columnas derivadas
    y retorna un DataFrame tidy.
    """
    log.info(f"Cargando {tabla} (cam={cam}, depto={DEPTO})...")
    df = pd.read_sql_query(f"""
        SELECT
            v.amb_depto, v.depto,
            v.amb_municipio, v.municipio,
            v.amb_zona, v.zona,
            v.amb_puesto, v.puesto,
            v.num_mesa, v.amb_mesa,
            v.tipo_circunsc,
            v.codpar,
            p.nombre  AS nombre_partido,
            v.codcan, v.cedula, v.candidato,
            v.votos_candidato   AS votos,
            v.votos_partido     AS votos_partido_mesa,
            v.potencial_electoral,
            v.votantes,
            v.votos_validos
        FROM {tabla} v
        LEFT JOIN partidos p ON v.codpar = p.codpar
        WHERE v.cam = ? AND v.amb_depto = ?
    """, con, params=(cam, DEPTO))

    log.info(f"  {len(df):,} filas cargadas")

    # Columnas derivadas
    df["corporacion"] = corporacion
    df["tipo_voto"]   = derivar_tipo_voto(df)

    # Homologación → colectividad
    hom_map = hom.set_index("ca_codpar" if corporacion == "Camara" else "se_codpar")
    col_key  = "ca_codpar" if corporacion == "Camara" else "se_codpar"
    df["codpar"] = df["codpar"].astype(str)
    hom_sub = hom[[col_key, "colectividad"]].drop_duplicates(subset=col_key)
    hom_sub[col_key] = hom_sub[col_key].astype(str)
    df = df.merge(hom_sub, left_on="codpar", right_on=col_key, how="left")
    df["colectividad"] = df["colectividad"].fillna("SIN_HOMOLOGACION")
    df.drop(columns=[col_key], errors="ignore", inplace=True)

    # Normalizar tipos
    df["num_mesa"]            = df["num_mesa"].astype(int)
    df["votos"]               = pd.to_numeric(df["votos"], errors="coerce").fillna(0).astype(int)
    df["votos_partido_mesa"]  = pd.to_numeric(df["votos_partido_mesa"], errors="coerce").fillna(0).astype(int)
    df["potencial_electoral"] = pd.to_numeric(df["potencial_electoral"], errors="coerce").fillna(0).astype(int)
    df["votantes"]            = pd.to_numeric(df["votantes"], errors="coerce").fillna(0).astype(int)
    df["votos_validos"]       = pd.to_numeric(df["votos_validos"], errors="coerce").fillna(0).astype(int)

    return df


def main():
    DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(DB_PATH))

    # Tabla homologación
    hom = pd.read_csv(Path(__file__).resolve().parent.parent / "config" / "homologacion_partidos.csv",
                      dtype=str)
    hom["ca_codpar"] = hom["ca_codpar"].astype(str)
    hom["se_codpar"] = hom["se_codpar"].astype(str)
    hom["peso"]      = hom["peso"].astype(float)

    # Cámara
    df_ca = cargar_tabla(con, "votos_mesa_camara", "Camara", CAM_CA, hom)
    out_ca = DATA_INTERIM / "votos_camara_mesa.parquet"
    df_ca.to_parquet(out_ca, index=False)
    log.info(f"Guardado: {out_ca}  ({len(df_ca):,} filas)")

    # Senado
    df_se = cargar_tabla(con, "votos_mesa_senado", "Senado", CAM_SE, hom)
    out_se = DATA_INTERIM / "votos_senado_mesa.parquet"
    df_se.to_parquet(out_se, index=False)
    log.info(f"Guardado: {out_se}  ({len(df_se):,} filas)")

    con.close()

    # Resumen rápido
    log.info("=== RESUMEN PARSEO ===")
    for label, df in [("Cámara", df_ca), ("Senado", df_se)]:
        pref = (df["tipo_voto"] == "preferente").sum()
        lista = (df["tipo_voto"] == "lista").sum()
        sin_hom = (df["colectividad"] == "SIN_HOMOLOGACION")
        log.info(f"  {label}: preferente={pref:,}  lista={lista:,}  "
                 f"sin_homologacion={sin_hom.sum():,} votos "
                 f"({df.loc[sin_hom,'votos'].sum():,} votos)")


if __name__ == "__main__":
    main()
