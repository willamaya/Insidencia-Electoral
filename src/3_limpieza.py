"""
3_limpieza.py — Validaciones de integridad y control de calidad
Genera: outputs/calidad_datos.md
"""
import logging
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_INTERIM, OUTPUTS, MESA_KEY

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

MESAS_ESPERADAS = {"0700": 3282}   # Boyacá


def check_votos_vs_potencial(df: pd.DataFrame, label: str) -> list:
    """
    Validación correcta: votos_validos <= votantes <= potencial_electoral.
    (No comparar suma de candidatos vs potencial: un elector vota UN candidato
    de UN partido; la suma cross-partido ≈ votos_validos, no votos totales.)
    """
    errores = []
    check = (df.groupby(MESA_KEY + ["potencial_electoral", "votantes", "votos_validos"])
             .size().reset_index())
    bad_vv = check[check["votos_validos"] > check["votantes"]]
    bad_vt = check[check["votantes"]      > check["potencial_electoral"]]
    if len(bad_vv):
        errores.append(f"[{label}] {len(bad_vv)} mesas con votos_validos > votantes")
        log.warning(errores[-1])
    if len(bad_vt):
        errores.append(f"[{label}] {len(bad_vt)} mesas con votantes > potencial_electoral")
        log.warning(errores[-1])
    return errores


def check_duplicados(df: pd.DataFrame, label: str) -> list:
    """No debe haber filas duplicadas con misma clave mesa+candidato."""
    errores = []
    dup_key = MESA_KEY + ["codpar", "cedula"]
    dupes = df.duplicated(subset=dup_key)
    if dupes.any():
        errores.append(f"[{label}] {dupes.sum()} filas duplicadas en clave mesa+partido+cedula")
        log.warning(errores[-1])
    return errores


def check_cobertura(df: pd.DataFrame, label: str) -> list:
    """% de mesas descargadas vs esperadas por departamento."""
    errores = []
    for depto, esperadas in MESAS_ESPERADAS.items():
        descargadas = df[df["amb_depto"] == depto]["amb_mesa"].nunique()
        pct = descargadas / esperadas * 100
        msg = f"[{label}] Depto {depto}: {descargadas}/{esperadas} mesas ({pct:.1f}%)"
        if pct < 99:
            errores.append(f"⚠ {msg}")
            log.warning(f"Cobertura baja: {msg}")
        else:
            log.info(f"Cobertura OK: {msg}")
    return errores


def check_votos_partido_vs_candidatos(df: pd.DataFrame, label: str) -> list:
    """
    Para cada (mesa, codpar): suma(votos candidatos) ≈ votos_partido_mesa.
    Se permite pequeña diferencia por redondeos.
    """
    errores = []
    agg = df.groupby(MESA_KEY + ["codpar", "votos_partido_mesa"]).agg(
        suma_cands=("votos", "sum")).reset_index()
    agg["diff"] = agg["suma_cands"] - agg["votos_partido_mesa"]
    mask = agg["diff"].abs() > 2
    if mask.any():
        n = mask.sum()
        errores.append(f"[{label}] {n} combos (mesa,codpar) con |diff votos_partido - suma_candidatos| > 2")
        log.warning(errores[-1])
    return errores


def generar_reporte(errores: list, df_ca: pd.DataFrame, df_se: pd.DataFrame) -> str:
    lines = ["# Reporte de calidad de datos — Boyacá 2026\n"]

    lines.append("## Cobertura")
    for label, df in [("Cámara", df_ca), ("Senado", df_se)]:
        munis  = df["amb_municipio"].nunique()
        puestos= df["amb_puesto"].nunique()
        mesas  = df["amb_mesa"].nunique()
        filas  = len(df)
        lines.append(f"- **{label}**: {munis} municipios | {puestos} puestos | "
                     f"{mesas} mesas | {filas:,} filas")

    lines.append("\n## Homologación")
    for label, df in [("Cámara", df_ca), ("Senado", df_se)]:
        sin = (df["colectividad"] == "SIN_HOMOLOGACION")
        votos_sin = df.loc[sin, "votos"].sum()
        lines.append(f"- **{label}** sin homologación: {sin.sum():,} filas | {votos_sin:,} votos")

    lines.append("\n## Tipo de voto")
    for label, df in [("Cámara", df_ca), ("Senado", df_se)]:
        t = df.groupby("tipo_voto")["votos"].sum()
        lines.append(f"- **{label}**: " + "  |  ".join(f"{k}={v:,}" for k, v in t.items()))

    lines.append("\n## Validaciones")
    if errores:
        for e in errores:
            lines.append(f"- {e}")
    else:
        lines.append("- ✓ Todas las validaciones pasaron")

    return "\n".join(lines)


def main():
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    df_ca = pd.read_parquet(DATA_INTERIM / "votos_camara_mesa.parquet")
    df_se = pd.read_parquet(DATA_INTERIM / "votos_senado_mesa.parquet")

    errores = []
    for label, df in [("Cámara", df_ca), ("Senado", df_se)]:
        errores += check_votos_vs_potencial(df, label)
        errores += check_duplicados(df, label)
        errores += check_cobertura(df, label)
        errores += check_votos_partido_vs_candidatos(df, label)

    reporte = generar_reporte(errores, df_ca, df_se)
    out = OUTPUTS / "calidad_datos.md"
    out.write_text(reporte, encoding="utf-8")
    log.info(f"Reporte guardado: {out}")

    n_err = len([e for e in errores if "⚠" in e or "ERROR" in e.upper()])
    log.info(f"Total alertas: {n_err}")
    return errores


if __name__ == "__main__":
    main()
