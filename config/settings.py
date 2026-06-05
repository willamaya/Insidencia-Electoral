"""
settings.py — Configuración central del proyecto electoral 2026
"""
from pathlib import Path

# ── Rutas ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
# DB dentro del proyecto (para despliegue en Streamlit Cloud)
DB_PATH     = ROOT / "data" / "puestos_2026.db"
DATA_RAW    = ROOT / "data" / "raw"
DATA_INTERIM= ROOT / "data" / "interim"
DATA_PROC   = ROOT / "data" / "processed"
OUTPUTS     = ROOT / "outputs"
CONFIG      = ROOT / "config"

# ── Filtros ────────────────────────────────────────────────────────────────────
DEPTO       = "0700"          # Boyacá
CAM_CA      = "1"             # Cámara territorial
CAM_SE      = "0"             # Senado

# ── Columnas geográficas (clave única de mesa) ─────────────────────────────────
MESA_KEY    = ["amb_depto", "amb_municipio", "amb_zona", "amb_puesto", "num_mesa"]

# ── Columnas de salida tidy ────────────────────────────────────────────────────
COLS_TIDY = [
    "amb_depto", "depto",
    "amb_municipio", "municipio",
    "amb_zona", "zona",
    "amb_puesto", "puesto",
    "num_mesa", "amb_mesa",
    "corporacion",
    "tipo_circunsc",
    "codpar", "nombre_partido", "colectividad",
    "cedula", "candidato",
    "tipo_voto",
    "votos",
    "votos_partido_mesa",
    "potencial_electoral", "votantes", "votos_validos",
]
