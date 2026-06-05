"""Colores oficiales de colectividades para visualizaciones."""

COLORES = {
    "ALIANZA VERDE"            : "#007C34",
    "PACTO HISTORICO"          : "#8B1A4A",
    "CENTRO DEMOCRATICO"       : "#1E477D",
    "PARTIDO LIBERAL"          : "#E30716",
    "CONSERVADOR-SALV NACIONAL": "#0867B1",
    "CR-NUEVO LIBERALISMO"     : "#F95846",
    "PARTIDO DE LA U"          : "#48AB38",
    "SIN_HOMOLOGACION"         : "#AAAAAA",
    "ALMA-OXIGENO"             : "#FF8C00",
}

COLOR_DEFAULT = "#888888"

def color(colectividad: str) -> str:
    return COLORES.get(colectividad, COLOR_DEFAULT)
