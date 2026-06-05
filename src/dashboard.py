"""
dashboard.py — Dashboard Electoral Interactivo 2026
Análisis de arrastre Cámara ↔ Senado, Boyacá

Ejecutar:
    cd proyecto-electoral-2026
    streamlit run src/dashboard.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy import stats
from scipy.stats import gaussian_kde

import sqlite3
from config.settings import DATA_PROC, DATA_INTERIM, OUTPUTS, DB_PATH
from config.colores_partidos import COLORES, color

# Mapeo colectividad CA → codpar SE
COLEC_TO_SE_CODPAR = {
    "ALIANZA VERDE"            : "57",
    "PACTO HISTORICO"          : "92",
    "CENTRO DEMOCRATICO"       : "10",
    "PARTIDO LIBERAL"          : "2",
    "CONSERVADOR-SALV NACIONAL": ["3", "17"],
    "CR-NUEVO LIBERALISMO"     : "44",
    "PARTIDO DE LA U"          : "9",
}

# ── Configuración de página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Arrastre Electoral Boyacá 2026",
    page_icon="🗳️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Forzar modo claro siempre
st.markdown("""
<style>
    html, body, [data-testid="stAppViewContainer"],
    [data-testid="stHeader"], [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        color: #111111 !important;
    }
    [data-testid="stSidebar"] { background-color: #f7f7f7 !important; }
    .stDataFrame, .stTable { background-color: #ffffff !important; }
    .stMarkdown, .stCaption, label, p, h1, h2, h3 { color: #111111 !important; }
    .stMetric { background-color: #f0f4ff !important; border-radius: 8px; padding: 8px; }
    .stExpander { border: 1px solid #e0e0e0 !important; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)

COLECTIVIDADES_VALIDAS = [
    "ALIANZA VERDE", "PACTO HISTORICO", "CENTRO DEMOCRATICO",
    "PARTIDO LIBERAL", "CONSERVADOR-SALV NACIONAL",
    "CR-NUEVO LIBERALISMO", "PARTIDO DE LA U",
]


# ── Carga de datos (cacheada) ─────────────────────────────────────────────────
@st.cache_data
def cargar_datos():
    mesa     = pd.read_parquet(DATA_PROC / "mesa_consolidada.parquet")
    ca       = pd.read_parquet(DATA_INTERIM / "votos_camara_mesa.parquet")
    se       = pd.read_parquet(DATA_INTERIM / "votos_senado_mesa.parquet")
    idx      = pd.read_parquet(DATA_PROC / "arrastre_candidato.parquet")
    corr     = pd.read_csv(OUTPUTS / "correlaciones.csv")
    reg      = pd.read_csv(OUTPUTS / "regresion_resultados.csv") if (OUTPUTS / "regresion_resultados.csv").exists() else pd.DataFrame()
    outliers = pd.read_csv(OUTPUTS / "outliers_mesas.csv") if (OUTPUTS / "outliers_mesas.csv").exists() else pd.DataFrame()
    return mesa, ca, se, idx, corr, reg, outliers

@st.cache_data
def cargar_modelos_v2():
    """Carga los resultados de las fases 1 y 2 del modelo estadístico avanzado."""
    def load(name):
        p = OUTPUTS / name
        return pd.read_csv(p) if p.exists() else pd.DataFrame()
    return {
        "hlm"        : load("hlm_resultados.csv"),
        "definetti"  : load("definetti_test.csv"),
        "metricas"   : load("metricas_v2.csv"),
        "bootstrap"  : load("bootstrap_beta.csv"),
        "clr"        : load("clr_resultados.csv"),
        "comparativa": load("comparativa_fases.csv"),
    }

@st.cache_data
def candidatos_senado(se_df):
    """Candidatos SE con voto preferente, ordenados por votos."""
    return (se_df[se_df["tipo_voto"] == "preferente"]
            .groupby(["cedula", "candidato", "colectividad"])["votos"]
            .sum().reset_index()
            .sort_values("votos", ascending=False))

@st.cache_data
def senadores_por_colectividad(colectividad: str, mesas: tuple) -> pd.DataFrame:
    """
    Candidatos SE de la colectividad con votos en las mesas dadas.
    mesas debe ser una tuple ORDENADA para que el cache funcione correctamente.
    Si mesas tiene > 500 elementos se filtra solo por codpar (sin restricción de mesa).
    """
    codpar_raw = COLEC_TO_SE_CODPAR.get(colectividad)
    if codpar_raw is None:
        return pd.DataFrame()
    codpars = codpar_raw if isinstance(codpar_raw, list) else [codpar_raw]
    ph = ",".join("?" * len(codpars))
    con = sqlite3.connect(str(DB_PATH))
    if len(mesas) > 500:
        # Sin restricción de mesas: devuelve todos los candidatos SE del partido
        df = pd.read_sql_query(f"""
            SELECT cedula, candidato, amb_mesa, num_mesa,
                   SUM(votos_candidato) votos
            FROM votos_mesa_senado
            WHERE codpar IN ({ph}) AND cam='0'
              AND cedula != '' AND cedula != '0'
            GROUP BY cedula, candidato, amb_mesa, num_mesa
            ORDER BY SUM(votos_candidato) DESC
        """, con, params=codpars)
    else:
        mesas_ph = ",".join("?" * len(mesas))
        df = pd.read_sql_query(f"""
            SELECT cedula, candidato, amb_mesa, num_mesa,
                   SUM(votos_candidato) votos
            FROM votos_mesa_senado
            WHERE codpar IN ({ph}) AND cam='0'
              AND amb_mesa IN ({mesas_ph})
              AND cedula != '' AND cedula != '0'
            GROUP BY cedula, candidato, amb_mesa, num_mesa
            ORDER BY SUM(votos_candidato) DESC
        """, con, params=codpars + list(mesas))
    con.close()
    return df

@st.cache_data
def votos_senador_por_mesa(cedula_senador: str, colectividad: str, mesas: tuple) -> pd.DataFrame:
    """Votos del senador específico por mesa (mesas ordenada para cache)."""
    codpar_raw = COLEC_TO_SE_CODPAR.get(colectividad)
    if codpar_raw is None:
        return pd.DataFrame()
    codpars = codpar_raw if isinstance(codpar_raw, list) else [codpar_raw]
    ph = ",".join("?" * len(codpars))
    con = sqlite3.connect(str(DB_PATH))
    if len(mesas) > 500:
        df = pd.read_sql_query(f"""
            SELECT amb_mesa, num_mesa, municipio,
                   SUM(votos_candidato) v_senador
            FROM votos_mesa_senado
            WHERE cedula=? AND codpar IN ({ph}) AND cam='0'
            GROUP BY amb_mesa, num_mesa, municipio
        """, con, params=[cedula_senador] + codpars)
    else:
        mesas_ph = ",".join("?" * len(mesas))
        df = pd.read_sql_query(f"""
            SELECT amb_mesa, num_mesa, municipio,
                   SUM(votos_candidato) v_senador
            FROM votos_mesa_senado
            WHERE cedula=? AND codpar IN ({ph}) AND cam='0'
              AND amb_mesa IN ({mesas_ph})
            GROUP BY amb_mesa, num_mesa, municipio
        """, con, params=[cedula_senador] + codpars + list(mesas))
    con.close()
    return df[df["v_senador"] > 0]

@st.cache_data
def cedula_de_senador(nombre_senador: str, colectividad: str) -> str:
    """Obtiene la cédula a partir del nombre del senador."""
    codpar_raw = COLEC_TO_SE_CODPAR.get(colectividad)
    if codpar_raw is None:
        return ""
    codpars = codpar_raw if isinstance(codpar_raw, list) else [codpar_raw]
    ph = ",".join("?" * len(codpars))
    con = sqlite3.connect(str(DB_PATH))
    row = con.execute(
        f"SELECT cedula FROM votos_mesa_senado WHERE candidato=? AND codpar IN ({ph}) AND cam='0' LIMIT 1",
        [nombre_senador] + codpars
    ).fetchone()
    con.close()
    return row[0] if row else ""

@st.cache_data
def se_partido_por_mesa(colectividad: str, mesas: tuple) -> pd.DataFrame:
    """
    Votos totales del partido SE (incluyendo lista) por mesa.
    mesas debe ser una tuple ORDENADA.
    """
    codpar_raw = COLEC_TO_SE_CODPAR.get(colectividad)
    if codpar_raw is None:
        return pd.DataFrame()
    codpars = codpar_raw if isinstance(codpar_raw, list) else [codpar_raw]
    ph = ",".join("?" * len(codpars))
    con = sqlite3.connect(str(DB_PATH))
    if len(mesas) > 500:
        df = pd.read_sql_query(f"""
            SELECT amb_mesa, num_mesa, SUM(votos_candidato) v_se_partido
            FROM votos_mesa_senado
            WHERE codpar IN ({ph}) AND cam='0'
            GROUP BY amb_mesa, num_mesa
        """, con, params=codpars)
    else:
        mesas_ph = ",".join("?" * len(mesas))
        df = pd.read_sql_query(f"""
            SELECT amb_mesa, num_mesa, SUM(votos_candidato) v_se_partido
            FROM votos_mesa_senado
            WHERE codpar IN ({ph}) AND cam='0'
              AND amb_mesa IN ({mesas_ph})
            GROUP BY amb_mesa, num_mesa
        """, con, params=codpars + list(mesas))
    con.close()
    return df


# ── Helpers ───────────────────────────────────────────────────────────────────
def fmt(n, decimals=0):
    if pd.isna(n): return "—"
    return f"{n:,.{decimals}f}".replace(",", ".")

def delta_arrow(val, ref):
    if pd.isna(val) or pd.isna(ref) or ref == 0: return ""
    d = (val - ref) / ref * 100
    return f"{'▲' if d > 0 else '▼'} {abs(d):.1f}% vs dpto."

def get_reg(reg_df, colectividad):
    if reg_df.empty: return None
    r = reg_df[reg_df["colectividad"] == colectividad]
    return r.iloc[0] if not r.empty else None

def get_corr(corr_df, colectividad):
    if corr_df.empty: return None
    r = corr_df[corr_df["colectividad"] == colectividad]
    return r.iloc[0] if not r.empty else None


# ── Filtros en cascada ────────────────────────────────────────────────────────
def render_sidebar(mesa, ca, se, idx, cands_se):
    st.sidebar.title("🗳️ Filtros")

    # Modo bidireccional
    modo = st.sidebar.radio("Modo de análisis", ["CA → SE", "SE → CA"],
                            horizontal=True,
                            help="CA→SE: partido Cámara incide en Senado. SE→CA: inverso.")
    st.sidebar.markdown("---")

    # Modelo visual
    modelo = st.sidebar.multiselect(
        "Modelos a mostrar",
        ["Determinístico", "Estadístico", "Probabilístico"],
        default=["Determinístico", "Estadístico"],
    )
    st.sidebar.markdown("---")

    if modo == "CA → SE":
        # devuelve: df, colectividad, candidato_ca, modo, modelo, senador_sel
        return _filtros_ca_se(mesa, idx, modelo, modo)
    else:
        # devuelve: df, colectividad, senador, modo, modelo, candidato_ca_filtro
        return _filtros_se_ca(mesa, se, ca, cands_se, modelo, modo)


def _filtros_ca_se(mesa, idx, modelo, modo):
    col_opts = sorted([c for c in mesa["colectividad"].unique() if c in COLECTIVIDADES_VALIDAS])
    colectividad = st.sidebar.selectbox("Colectividad (Cámara)", col_opts)

    cands = sorted(mesa[mesa["colectividad"] == colectividad]["candidato"].unique())
    candidato = st.sidebar.selectbox("Candidato (Cámara)", ["Todos"] + cands)

    df = mesa[(mesa["colectividad"] == colectividad) &
              (mesa["tipo_voto"] == "preferente")]
    if candidato != "Todos":
        df = df[df["candidato"] == candidato]

    df = _filtros_geo(df, st.sidebar)

    # ── BUG FIX 1: selector de senador VA AQUÍ, después de los filtros geo ────
    # Solo habilitado cuando hay un candidato CA seleccionado
    senador_sel = None
    if candidato != "Todos":
        # BUG FIX 2: tuple ordenada para cache determinista
        mesas_ord = tuple(sorted(df["amb_mesa"].unique()))
        senador_sel = render_selector_senador(colectividad, mesas_ord)

    return df, colectividad, candidato, modo, modelo, senador_sel


def render_atribucion_inversa(df_ca_filt, colectividad, candidato_se, candidato_ca_filtro):
    """
    SE→CA: Para un senador dado, muestra mesa a mesa cuánto
    contribuyó cada candidato a Cámara a su votación.

    df_ca_filt : mesa_consolidada filtrado a la colectividad + geo
    candidato_se : nombre del senador seleccionado
    candidato_ca_filtro : candidato CA específico para destacar (o None = todos)
    """
    if not candidato_se or candidato_se == "Todos":
        st.info("👆 Selecciona un **senador** en el sidebar para ver la atribución.")
        return

    mesas_ord = tuple(sorted(df_ca_filt["amb_mesa"].unique()))
    if not mesas_ord:
        st.warning("Sin mesas disponibles.")
        return

    # Cédula del senador
    cedula_se = cedula_de_senador(candidato_se, colectividad)
    if not cedula_se:
        st.warning(f"No se encontró cédula para {candidato_se}.")
        return

    # Votos del senador por mesa
    sen_mesa = votos_senador_por_mesa(cedula_se, colectividad, mesas_ord)
    if sen_mesa.empty:
        st.warning("El senador no tiene votos en las mesas seleccionadas.")
        return

    # CA partido por mesa (votos totales y por candidato)
    ca_por_mesa = (df_ca_filt[df_ca_filt["tipo_voto"] == "preferente"]
                   .groupby(["amb_mesa", "num_mesa", "municipio", "puesto",
                              "cedula", "candidato"])
                   .agg(v_ca=("votos","sum"),
                        v_partido_ca=("votos_partido_mesa","first"))
                   .reset_index())

    if candidato_ca_filtro:
        ca_por_mesa = ca_por_mesa[ca_por_mesa["candidato"] == candidato_ca_filtro]

    # Merge con votos del senador
    tbl = ca_por_mesa.merge(sen_mesa[["amb_mesa","v_senador"]], on="amb_mesa", how="inner")
    if tbl.empty:
        st.warning("Sin coincidencia de mesas entre CA y el senador seleccionado.")
        return

    tbl["dominancia"]  = tbl["v_ca"] / tbl["v_partido_ca"].replace(0, np.nan)
    tbl["sen_atrib"]   = tbl["dominancia"] * tbl["v_senador"]
    tbl["pct_apoyo"]   = tbl["dominancia"] * 100
    tbl = tbl.sort_values(["num_mesa", "v_ca"], ascending=[True, False])

    # ── Totales por candidato CA ─────────────────────────────────────────────
    resumen = (tbl.groupby(["cedula","candidato"])
               .agg(mesas=("amb_mesa","nunique"),
                    v_ca_total=("v_ca","sum"),
                    sen_atrib_total=("sen_atrib","sum"))
               .reset_index()
               .sort_values("v_ca_total", ascending=False))
    total_atrib = resumen["sen_atrib_total"].sum()
    resumen["pct_total"] = resumen["sen_atrib_total"] / total_atrib * 100 if total_atrib > 0 else 0

    total_votos_sen = sen_mesa["v_senador"].sum()

    st.markdown(f"**Senador:** `{candidato_se}` — **{total_votos_sen:,}** votos en "
                f"{len(sen_mesa)} mesas &nbsp;|&nbsp; Colectividad: `{colectividad}`")

    # ── KPIs ─────────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Votos senador (filtro)", f"{total_votos_sen:,}")
    k2.metric("Mesas con votos", f"{len(sen_mesa)}")
    k3.metric("Candidatos CA en esas mesas", f"{tbl['cedula'].nunique()}")
    k4.metric("Votos CA partido (mismas mesas)",
              f"{tbl.groupby('amb_mesa')['v_partido_ca'].first().sum():,.0f}")

    # ── Tabla resumen por candidato CA ───────────────────────────────────────
    st.markdown("**Resumen: contribución de cada candidato CA al senador**")
    res_disp = resumen[["candidato","mesas","v_ca_total","sen_atrib_total","pct_total"]].copy()
    res_disp.columns = ["Candidato CA","Mesas","V.CA total","SE atrib. total","% apoyo senador"]
    res_disp["SE atrib. total"] = res_disp["SE atrib. total"].round(1)
    res_disp["% apoyo senador"] = res_disp["% apoyo senador"].round(1)
    total_row_r = pd.DataFrame([[
        "TOTAL", len(sen_mesa),
        resumen["v_ca_total"].sum(),
        round(total_atrib,1), 100.0
    ]], columns=res_disp.columns)
    st.dataframe(
        pd.concat([res_disp, total_row_r], ignore_index=True),
        use_container_width=True, height=220,
        column_config={
            "% apoyo senador": st.column_config.ProgressColumn(
                "% apoyo senador", min_value=0, max_value=100, format="%.1f%%")
        }
    )

    # ── Gráfico barras candidatos CA ─────────────────────────────────────────
    fig_res = px.bar(
        resumen.sort_values("sen_atrib_total", ascending=True),
        x="sen_atrib_total", y="candidato", orientation="h",
        color="pct_total",
        color_continuous_scale=[[0,"#1E477D"],[0.5,color(colectividad)],[1,"#FF8C00"]],
        labels={"sen_atrib_total":"Votos senador atribuidos","candidato":"Candidato CA",
                "pct_total":"% apoyo"},
        title=f"Contribución de candidatos CA al senador {candidato_se[:30]}",
        text=resumen.sort_values("sen_atrib_total",ascending=True)["pct_total"].apply(lambda v: f"{v:.1f}%")
    )
    fig_res.update_traces(textposition="outside")
    fig_res.update_layout(height=max(250, len(resumen)*45+80),
                          template="plotly_dark", showlegend=False,
                          coloraxis_showscale=False)
    st.plotly_chart(fig_res, use_container_width=True)

    # ── Tabla detalle mesa × candidato CA ────────────────────────────────────
    with st.expander("📋 Detalle mesa × candidato CA", expanded=False):
        det = tbl[["num_mesa","municipio","puesto","candidato",
                    "v_ca","v_partido_ca","v_senador","sen_atrib","pct_apoyo"]].copy()
        det.columns = ["Mesa","Municipio","Puesto","Candidato CA",
                       "V.CA cand","V.CA partido","V.Senador","SE atrib.","% apoyo"]
        det["SE atrib."] = det["SE atrib."].round(1)
        det["% apoyo"]   = det["% apoyo"].round(1)
        st.dataframe(det, use_container_width=True, height=380,
                     column_config={
                         "% apoyo": st.column_config.ProgressColumn(
                             "% apoyo", min_value=0, max_value=100, format="%.1f%%")
                     })
        csv = det.to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇ Descargar detalle",csv,
                           f"se_ca_{candidato_se[:20]}.csv","text/csv")


def _filtros_se_ca(mesa, se, ca, cands_se, modelo, modo):
    col_opts = sorted([c for c in cands_se["colectividad"].unique() if c in COLECTIVIDADES_VALIDAS])
    colectividad = st.sidebar.selectbox("Colectividad (Senado)", col_opts)

    cands = cands_se[cands_se["colectividad"] == colectividad]["candidato"].tolist()
    senador = st.sidebar.selectbox("Senador", ["Todos"] + cands)

    # Filtrar mesas donde ese senador tuvo votos
    df = mesa[(mesa["colectividad"] == colectividad) &
              (mesa["tipo_voto"] == "preferente")]
    if senador != "Todos":
        mesas_se = se[(se["candidato"] == senador) & (se["votos"] > 0)]["amb_mesa"].unique()
        df = df[df["amb_mesa"].isin(mesas_se)]

    df = _filtros_geo(df, st.sidebar)

    # Selector opcional de candidato CA (para enfocar la atribución)
    candidato_ca_filtro = None
    if senador != "Todos":
        st.sidebar.markdown("---")
        st.sidebar.markdown("**🔎 Filtrar candidato CA**")
        cands_ca = sorted(df["candidato"].unique())
        sel_ca = st.sidebar.selectbox("Candidato Cámara (opcional)", ["Todos"] + cands_ca,
                                      help="Filtra la tabla de atribución a un candidato CA específico")
        candidato_ca_filtro = None if sel_ca == "Todos" else sel_ca

    return df, colectividad, senador, modo, modelo, candidato_ca_filtro


def render_selector_senador(colectividad: str, mesas_ordenadas: tuple) -> str:
    """
    Desplegable de candidatos SE para la sección de atribución.
    Se llama desde _filtros_ca_se para que aparezca en el lugar correcto del sidebar.
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown("**📌 Atribución → Senador**")
    df_sen = senadores_por_colectividad(colectividad, mesas_ordenadas)
    if df_sen.empty:
        st.sidebar.caption("Sin candidatos SE. Selecciona un municipio primero.")
        return None
    totales = (df_sen.groupby(["cedula","candidato"])["votos"]
               .sum().reset_index().sort_values("votos", ascending=False))
    opts = ["(todos)"] + totales["candidato"].tolist()
    sel = st.sidebar.selectbox(
        "Candidato Senado", opts,
        help="Elige un senador para ver cuántos de sus votos vienen de este candidato CA"
    )
    return None if sel == "(todos)" else sel


def _filtros_geo(df, sidebar):
    munis = sorted(df["municipio"].unique())
    sel_munis = sidebar.multiselect("Municipio", munis, default=[])
    if sel_munis:
        df = df[df["municipio"].isin(sel_munis)]

    zonas = sorted(df["zona"].unique())
    sel_zonas = sidebar.multiselect("Localidad / Zona", zonas, default=[])
    if sel_zonas:
        df = df[df["zona"].isin(sel_zonas)]

    puestos = sorted(df["puesto"].unique())
    sel_puestos = sidebar.multiselect("Puesto de votación", puestos, default=[])
    if sel_puestos:
        df = df[df["puesto"].isin(sel_puestos)]

    if df["num_mesa"].nunique() > 1:
        min_m, max_m = int(df["num_mesa"].min()), int(df["num_mesa"].max())
        rng = sidebar.slider("Mesa (rango)", min_m, max_m, (min_m, max_m))
        df = df[(df["num_mesa"] >= rng[0]) & (df["num_mesa"] <= rng[1])]

    sidebar.markdown("---")
    sidebar.caption("⚠️ **Advertencia metodológica**: datos agregados por mesa. "
                    "Los índices NO implican causalidad individual (falacia ecológica).")
    return df


# ── KPIs ──────────────────────────────────────────────────────────────────────
def render_kpis(df, mesa_full, colectividad, modo):
    total_ca   = df["votos"].sum()
    total_se   = df["votos_se_colectividad_mesa"].sum()
    ratio      = total_se / total_ca if total_ca > 0 else float("nan")
    idx_pond   = (df["indice_incidencia_pond"].median() if "indice_incidencia_pond" in df.columns
                  else float("nan"))

    # Ratio de referencia correcto: SE total partido / CA total partido (sin duplicar filas)
    # votos_se_colectividad_mesa se repite por candidato → usamos una sola fila por mesa
    mesa_ref = (mesa_full[mesa_full["colectividad"] == colectividad]
                .groupby("amb_mesa")
                .agg(votos_ca_mesa=("votos", "sum"),
                     votos_se_mesa=("votos_se_colectividad_mesa", "first"))
                .reset_index())
    ref_ca  = mesa_ref["votos_ca_mesa"].sum()
    ref_se  = mesa_ref["votos_se_mesa"].sum()
    ref_rat = ref_se / ref_ca if ref_ca > 0 else float("nan")

    c1, c2, c3, c4 = st.columns(4)
    lab = "CA" if "CA" in modo else "SE"
    c1.metric(f"Votos {lab} (filtro)", fmt(total_ca),
              delta=delta_arrow(total_ca, ref_ca))
    c2.metric("Votos SE colectividad (mismas mesas)", fmt(total_se),
              delta=delta_arrow(total_se, ref_se))
    c3.metric("Ratio arrastre (SE/CA partido)", fmt(ratio, 3),
              delta=delta_arrow(ratio, ref_rat))
    c4.metric("Mesas incluidas", f"{df['amb_mesa'].nunique():,} / {mesa_full['amb_mesa'].nunique():,}")


# ── Scatter CA vs SE ──────────────────────────────────────────────────────────
def render_scatter(df, colectividad, modelo, reg_df, corr_df):
    if len(df) < 3:
        st.warning("Pocas mesas para scatter (mínimo 3).")
        return

    c = color(colectividad)
    fig = go.Figure()

    x = df["votos"].values
    y = df["votos_se_colectividad_mesa"].values

    # Determinístico — puntos coloreados por municipio
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers",
        marker=dict(color=df["municipio"].astype("category").cat.codes,
                    colorscale="Viridis", size=5, opacity=0.6),
        text=df.apply(lambda r: f"{r['municipio']}<br>{r['puesto']}<br>Mesa {r['num_mesa']}"
                                f"<br>CA: {r['votos']}  SE: {r['votos_se_colectividad_mesa']:.0f}", axis=1),
        hovertemplate="%{text}<extra></extra>",
        name="Mesas"
    ))

    x_range = np.linspace(0, x.max() * 1.05, 200)

    # Estadístico — OLS
    if "Estadístico" in modelo and len(x) > 5:
        reg = get_reg(reg_df, colectividad)
        corr = get_corr(corr_df, colectividad)
        if reg is not None:
            coef = reg["coef_votos_ca"]
            intercept = np.mean(y) - coef * np.mean(x)
            y_ols = coef * x_range + intercept

            # IC confianza 95%
            n, se_resid = len(x), np.std(y - (coef * x + intercept))
            margin = 1.96 * se_resid * np.sqrt(1/n + (x_range - np.mean(x))**2 / np.sum((x - np.mean(x))**2))

            fig.add_trace(go.Scatter(
                x=np.concatenate([x_range, x_range[::-1]]),
                y=np.concatenate([y_ols + margin, (y_ols - margin)[::-1]]),
                fill="toself", fillcolor=f"rgba(100,100,200,0.15)",
                line=dict(color="rgba(0,0,0,0)"), name="IC 95% OLS", showlegend=True
            ))
            fig.add_trace(go.Scatter(
                x=x_range, y=y_ols, mode="lines",
                line=dict(color="#3366CC", width=2, dash="solid"),
                name=f"OLS (coef={coef:.3f}, R²={reg['r2']:.3f})"
            ))
            if corr is not None:
                fig.add_annotation(
                    x=0.02, y=0.97, xref="paper", yref="paper",
                    text=f"Pearson={corr['pearson_mesa']:.3f}  Spearman={corr['spearman_mesa']:.3f}",
                    showarrow=False, bgcolor="white", bordercolor="gray"
                )

    # Probabilístico — bandas de predicción
    if "Probabilístico" in modelo and len(x) > 10:
        residuos_path = DATA_PROC / f"residuos_{colectividad.replace(' ','_').replace('-','_')}.parquet"
        if residuos_path.exists():
            res = pd.read_parquet(residuos_path)
            se_total = res["residuo"].std()
            coef_r  = np.polyfit(x, y, 1)
            y_fit   = np.polyval(coef_r, x_range)
            pred80  = 1.28 * se_total
            pred95  = 1.96 * se_total

            for alpha, width, name in [(pred95, 0.10, "Banda pred. 95%"),
                                        (pred80, 0.18, "Banda pred. 80%")]:
                fig.add_trace(go.Scatter(
                    x=np.concatenate([x_range, x_range[::-1]]),
                    y=np.concatenate([y_fit + alpha, (y_fit - alpha)[::-1]]),
                    fill="toself", fillcolor=f"rgba(200,100,50,{width})",
                    line=dict(color="rgba(0,0,0,0)"), name=name
                ))

    fig.update_layout(
        title=f"Votos Cámara vs Senado por mesa — {colectividad}",
        xaxis_title="Votos candidato Cámara (mesa)",
        yaxis_title="Votos colectividad Senado (misma mesa)",
        height=420, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Distribución de ratios ────────────────────────────────────────────────────
def render_distribucion(df, mesa_full, colectividad, candidato, modelo):
    ratios_all = mesa_full[mesa_full["colectividad"] == colectividad][
        "ratio_arrastre_mesa"].dropna()
    ratios_filt = df["ratio_arrastre_mesa"].dropna()

    fig = go.Figure()

    if "Probabilístico" in modelo and len(ratios_all) > 10:
        kde = gaussian_kde(ratios_all.clip(0, 5))
        x_kde = np.linspace(0, 5, 300)
        y_kde = kde(x_kde)
        fig.add_trace(go.Scatter(
            x=x_kde, y=y_kde, mode="lines", fill="toself",
            fillcolor=f"rgba(100,150,200,0.2)",
            line=dict(color=color(colectividad), width=2),
            name="KDE Boyacá completo"
        ))
        # Línea de ratio=1 (sin arrastre neto)
        fig.add_vline(x=1, line_dash="dash", line_color="gray",
                      annotation_text="ratio=1", annotation_position="top right")
        # Percentil del filtro actual
        if len(ratios_filt) > 0:
            med_filt = ratios_filt.median()
            pct = (ratios_all < med_filt).mean() * 100
            fig.add_vline(x=med_filt, line_dash="dot", line_color="#E30716",
                          annotation_text=f"Mediana filtro (p{pct:.0f})",
                          annotation_position="top left")

    if "Estadístico" in modelo and len(ratios_all) > 3:
        # Deciles
        for p in [10, 25, 50, 75, 90]:
            v = np.percentile(ratios_all, p)
            fig.add_vline(x=v, line_dash="dot", line_color="lightblue",
                          line_width=1,
                          annotation_text=f"p{p}", annotation_position="top")

    if "Determinístico" in modelo and len(ratios_filt) > 0:
        fig.add_trace(go.Histogram(
            x=ratios_filt.clip(0, 5), nbinsx=30,
            marker_color=color(colectividad), opacity=0.6,
            name="Mesas filtradas", yaxis="y2"
        ))
        fig.update_layout(
            yaxis2=dict(overlaying="y", side="right", showgrid=False,
                        title="Frecuencia (mesas filtradas)")
        )

    fig.update_layout(
        title=f"Distribución ratio arrastre — {colectividad}",
        xaxis_title="Ratio arrastre (votos SE / votos CA partido, por mesa)",
        yaxis_title="Densidad (KDE)",
        height=420, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Drill-down geográfico ─────────────────────────────────────────────────────
def render_drilldown(df):
    if len(df) == 0:
        st.warning("Sin datos para el filtro seleccionado.")
        return

    # Determinar nivel de agregación
    n_munis   = df["amb_municipio"].nunique()
    n_zonas   = df["amb_zona"].nunique()
    n_puestos = df["amb_puesto"].nunique()

    if n_munis > 1:
        grp_col, grp_lbl = "municipio", "Municipio"
    elif n_zonas > 1:
        grp_col, grp_lbl = "zona", "Zona / Localidad"
    elif n_puestos > 1:
        grp_col, grp_lbl = "puesto", "Puesto de votación"
    else:
        grp_col, grp_lbl = "num_mesa", "Mesa"

    agg = (df.groupby(grp_col).agg(
        votos_ca=("votos", "sum"),
        votos_se=("votos_se_colectividad_mesa", "sum"),
        mesas=("amb_mesa", "nunique")
    ).reset_index().sort_values("votos_ca", ascending=False).head(30))

    agg["ratio"] = (agg["votos_se"] / agg["votos_ca"]).round(3)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=agg[grp_col], y=agg["votos_ca"],
                         name="Votos Cámara", marker_color="#3366CC"))
    fig.add_trace(go.Bar(x=agg[grp_col], y=agg["votos_se"],
                         name="Votos Senado colectividad", marker_color="#FF7700",
                         text=agg["ratio"].apply(lambda r: f"ratio={r:.2f}"),
                         textposition="outside"))
    fig.update_layout(
        title=f"Votos CA y SE por {grp_lbl} (top 30)",
        barmode="group", height=380, template="plotly_white",
        xaxis_tickangle=-35,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Heatmap de mesas ──────────────────────────────────────────────────────────
def render_heatmap(df, outliers_df):
    if df["amb_puesto"].nunique() > 50:
        st.info("Demasiados puestos para el heatmap. Filtra por municipio o zona.")
        return
    if len(df) < 2:
        return

    pivot = df.pivot_table(
        index="puesto", columns="num_mesa",
        values="ratio_arrastre_mesa", aggfunc="mean"
    )
    if pivot.empty:
        return

    # Marcar outliers — usar clave compuesta amb_puesto + num_mesa
    outlier_keys = set()
    if not outliers_df.empty and "amb_puesto" in outliers_df.columns:
        outlier_keys = set(
            outliers_df["amb_puesto"].astype(str) + "_" + outliers_df["num_mesa"].astype(str)
        )

    fig = px.imshow(
        pivot, aspect="auto",
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=1.0,
        range_color=[0, 3],
        labels=dict(color="Ratio arrastre"),
        title="Heatmap ratio arrastre por puesto × mesa"
    )
    fig.update_layout(height=max(300, len(pivot) * 22 + 100),
                      template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)


# ── Tabla detalle ─────────────────────────────────────────────────────────────
def render_tabla(df, outliers_df):
    cols = ["municipio", "zona", "puesto", "num_mesa", "candidato",
            "votos", "votos_partido_ca_mesa", "votos_se_colectividad_mesa",
            "ratio_arrastre_mesa", "dominancia_mesa",
            "potencial_electoral", "votantes", "votos_validos"]
    cols_ok = [c for c in cols if c in df.columns]
    tabla = df[cols_ok].copy()
    tabla.columns = [c.replace("_"," ") for c in tabla.columns]
    tabla = tabla.sort_values("ratio arrastre mesa", ascending=False) if "ratio arrastre mesa" in tabla.columns else tabla

    # Marcar outliers usando clave compuesta
    if not outliers_df.empty and "amb_puesto" in df.columns:
        outlier_set = set(
            outliers_df["amb_puesto"].astype(str) + "_" + outliers_df["num_mesa"].astype(str)
        )
        tabla["es_outlier"] = (
            df["amb_puesto"].astype(str) + "_" + df["num_mesa"].astype(str)
        ).isin(outlier_set).values

    st.dataframe(tabla.head(500), use_container_width=True, height=320)
    csv = tabla.to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇ Descargar CSV", csv, "detalle_mesas.csv", "text/csv")


# ── Atribución CA candidato → SE senador ─────────────────────────────────────
def render_atribucion(df_ca_filt, colectividad, candidato_ca, senador_se, se_df):
    """
    Tabla mesa a mesa:
      V.CA candidato | SE atrib. | %SE total | V.Senador | Senador atrib. | %Senador

    df_ca_filt : mesa_consolidada ya filtrado por colectividad + candidato CA + geo
    senador_se : nombre del senador seleccionado (None = partido completo)
    """
    if candidato_ca == "Todos":
        st.info("Selecciona un candidato a Cámara en el sidebar para ver la atribución.")
        return
    if COLEC_TO_SE_CODPAR.get(colectividad) is None:
        st.warning("Colectividad sin mapeo a Senado definido.")
        return

    # BUG FIX 2: tuple ordenada para cache determinista
    mesas_tuple = tuple(sorted(df_ca_filt["amb_mesa"].unique()))
    if not mesas_tuple:
        return

    # Votos CA del candidato por mesa
    ca_mesa = (df_ca_filt[df_ca_filt["tipo_voto"] == "preferente"]
               .groupby(["num_mesa", "amb_mesa", "municipio", "puesto"])
               .agg(v_ca=("votos", "sum"),
                    v_partido_ca=("votos_partido_mesa", "first"))
               .reset_index())

    # SE partido total por mesa
    se_part = se_partido_por_mesa(colectividad, mesas_tuple)

    # SE candidato senador por mesa
    se_cands = senadores_por_colectividad(colectividad, mesas_tuple)

    if se_cands.empty or se_part.empty:
        st.warning("Sin datos de Senado para las mesas seleccionadas.")
        return

    # Seleccionar senador
    if senador_se:
        se_sen = (se_cands[se_cands["candidato"] == senador_se]
                  .rename(columns={"votos": "v_senador"}))
    else:
        # Suma de todos los candidatos SE (= partido)
        se_sen = (se_cands.groupby(["amb_mesa","num_mesa"])["votos"]
                  .sum().reset_index()
                  .rename(columns={"votos": "v_senador"}))
        se_sen["candidato"] = "PARTIDO COMPLETO"

    # Merge
    tbl = (ca_mesa
           .merge(se_part[["amb_mesa","v_se_partido"]], on="amb_mesa", how="left")
           .merge(se_sen[["amb_mesa","v_senador"]], on="amb_mesa", how="left"))
    tbl["v_se_partido"] = tbl["v_se_partido"].fillna(0)
    tbl["v_senador"]    = tbl["v_senador"].fillna(0)

    # Dominancia y atribuciones
    tbl["dominancia"]    = tbl["v_ca"] / tbl["v_partido_ca"].replace(0, np.nan)
    tbl["se_atrib"]      = tbl["dominancia"] * tbl["v_se_partido"]
    tbl["pct_se_total"]  = tbl["dominancia"] * 100
    tbl["sen_atrib"]     = tbl["dominancia"] * tbl["v_senador"]
    tbl["pct_senador"]   = tbl["dominancia"] * 100
    tbl = tbl.sort_values("num_mesa")

    # Totales
    T = {
        "v_ca"       : tbl["v_ca"].sum(),
        "v_partido_ca": tbl["v_partido_ca"].sum(),
        "v_se"       : tbl["v_se_partido"].sum(),
        "se_atrib"   : tbl["se_atrib"].sum(),
        "v_sen"      : tbl["v_senador"].sum(),
        "sen_atrib"  : tbl["sen_atrib"].sum(),
    }
    T["pct_se"]  = T["se_atrib"]  / T["v_se"]  * 100 if T["v_se"]  > 0 else 0
    T["pct_sen"] = T["sen_atrib"] / T["v_sen"] * 100 if T["v_sen"] > 0 else 0

    sen_label = senador_se or "PARTIDO SENADO"
    st.markdown(f"**Candidato CA:** `{candidato_ca}` &nbsp;→&nbsp; **Senado:** `{sen_label}`")

    # ── KPIs de atribución ────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Votos CA candidato",       f"{T['v_ca']:,}")
    k2.metric("SE partido (mismas mesas)",f"{T['v_se']:.0f}")
    k3.metric("SE partido atribuido",
              f"{T['se_atrib']:.1f}",
              delta=f"{T['pct_se']:.1f}% del SE partido")
    k4.metric(f"Atrib. a {sen_label[:20]}",
              f"{T['sen_atrib']:.1f}",
              delta=f"{T['pct_sen']:.1f}% de los votos del senador")

    # ── Tabla mesa a mesa ─────────────────────────────────────────────────────
    display = tbl[["num_mesa","municipio","puesto",
                    "v_ca","v_partido_ca","v_se_partido",
                    "se_atrib","pct_se_total",
                    "v_senador","sen_atrib","pct_senador"]].copy()
    display.columns = [
        "Mesa","Municipio","Puesto",
        "V.CA cand.","V.CA partido","V.SE partido",
        "SE atrib.","% SE total",
        f"V.{sen_label[:15]}","Atrib. senador","% senador"
    ]
    # Fila de totales
    total_row = pd.DataFrame([[
        "TOTAL","","",
        T["v_ca"], T["v_partido_ca"], round(T["v_se"],0),
        round(T["se_atrib"],1), round(T["pct_se"],1),
        round(T["v_sen"],0), round(T["sen_atrib"],1), round(T["pct_sen"],1)
    ]], columns=display.columns)
    display_full = pd.concat([display, total_row], ignore_index=True)
    st.dataframe(display_full, use_container_width=True, height=310,
                 column_config={
                     "% SE total"   : st.column_config.ProgressColumn("% SE total",   min_value=0, max_value=100, format="%.1f%%"),
                     "% senador"    : st.column_config.ProgressColumn("% senador",    min_value=0, max_value=100, format="%.1f%%"),
                     "SE atrib."    : st.column_config.NumberColumn(format="%.1f"),
                     "Atrib. senador": st.column_config.NumberColumn(format="%.1f"),
                 })

    # ── Gráfico doble barra por mesa ──────────────────────────────────────────
    c = color(colectividad)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Votos CA candidato", x=tbl["num_mesa"].astype(str),
        y=tbl["v_ca"], marker_color=c, opacity=0.85,
        text=tbl["v_ca"], textposition="outside"
    ))
    fig.add_trace(go.Bar(
        name=f"Atrib. {sen_label[:20]}", x=tbl["num_mesa"].astype(str),
        y=tbl["sen_atrib"].round(1), marker_color="#FF8C00", opacity=0.85,
        text=tbl["sen_atrib"].round(1), textposition="outside"
    ))
    fig.add_trace(go.Scatter(
        name="% atrib. senador", x=tbl["num_mesa"].astype(str),
        y=tbl["pct_senador"], mode="lines+markers+text",
        line=dict(color="white", dash="dot", width=2),
        marker=dict(size=8), yaxis="y2",
        text=tbl["pct_senador"].apply(lambda v: f"{v:.0f}%"),
        textposition="top center"
    ))
    fig.update_layout(
        title=f"Votos CA vs atribución a {sen_label} por mesa — {candidato_ca}",
        xaxis_title="Mesa", barmode="group",
        yaxis=dict(title="Votos"),
        yaxis2=dict(title="% atrib. senador", overlaying="y", side="right",
                    range=[0, 110], showgrid=False),
        height=400, template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Descarga ──────────────────────────────────────────────────────────────
    csv = display_full.to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇ Descargar tabla atribución", csv,
                       f"atribucion_{candidato_ca[:20]}_{sen_label[:20]}.csv", "text/csv")


# ── APP PRINCIPAL ─────────────────────────────────────────────────────────────
def main():
    st.title("🗳️ Arrastre Electoral Boyacá 2026")
    st.caption("Análisis de incidencia Cámara ↔ Senado · Elecciones Colombia 8-mar-2026 · **by WAM**")

    mesa, ca, se, idx, corr, reg, outliers = cargar_datos()
    modelos  = cargar_modelos_v2()
    cands_se = candidatos_senado(se)

    # Tabs principales
    tab_analisis, tab_modelo = st.tabs(["🗺️ Análisis de Arrastre", "📊 Modelo Estadístico Avanzado"])

    # Sidebar con filtros (fuera de los tabs — es global)
    df_filt, colectividad, candidato, modo, modelo, extra = render_sidebar(
        mesa, ca, se, idx, cands_se)
    senador_sel      = extra if modo == "CA → SE" else None
    candidato_ca_sel = extra if modo == "SE → CA" else None

    with tab_analisis:
        n_mesas = df_filt["amb_mesa"].nunique() if len(df_filt) > 0 else 0
        if n_mesas == 0:
            st.warning("Sin datos para el filtro seleccionado. Ajusta los criterios.")
        else:
            st.markdown(f"**Filtro activo:** `{colectividad}` · "
                        f"{'Candidato: ' + candidato if candidato != 'Todos' else 'Todos los candidatos'} · "
                        f"{n_mesas:,} mesas")

            modelos_activos = list(modelo)
            if n_mesas < 5 and ("Estadístico" in modelos_activos or "Probabilístico" in modelos_activos):
                st.warning("⚠ Menos de 5 mesas — modelos estadístico/probabilístico deshabilitados.")
                modelos_activos = ["Determinístico"]

            render_kpis(df_filt, mesa, colectividad, modo)
            st.divider()

            col_a, col_b = st.columns(2)
            with col_a:
                render_scatter(df_filt, colectividad, modelos_activos, reg, corr)
            with col_b:
                render_distribucion(df_filt, mesa, colectividad, candidato, modelos_activos)

            st.divider()
            st.subheader("📊 Votos por nivel geográfico")
            render_drilldown(df_filt)

            st.subheader("🟩 Heatmap: ratio arrastre por puesto × mesa")
            render_heatmap(df_filt, outliers)

            st.subheader("📋 Detalle de mesas")
            render_tabla(df_filt, outliers)

            st.divider()
            if modo == "CA → SE":
                st.subheader("🔗 Atribución CA → Senado")
                if candidato == "Todos":
                    st.info("👆 Selecciona un **candidato a Cámara** en el sidebar para ver "
                            "cuántos votos del Senado se atribuyen a sus votantes.")
                else:
                    st.caption("¿Qué % de los votos de un senador vinieron de los votantes de este candidato CA?")
                    render_atribucion(df_filt, colectividad, candidato, senador_sel, se)
            else:
                st.subheader("🔗 Atribución Senado → Cámara")
                st.caption("Para el senador seleccionado: ¿qué % de su apoyo vino de cada candidato a Cámara?")
                render_atribucion_inversa(df_filt, colectividad, candidato, candidato_ca_sel)

    with tab_modelo:
        col_sel = st.selectbox(
            "Filtrar resultados por colectividad",
            ["Todas"] + COLECTIVIDADES_VALIDAS,
            help="Filtra las tablas del modelo estadístico a una colectividad específica"
        )
        col_filter = col_sel if col_sel != "Todas" else None
        render_modelo_tab(modelos, col_filter)


# ── Sección: Modelo Estadístico Avanzado (Fases 1 y 2) ───────────────────────
def render_modelo_estadistico(modelos: dict, colectividad: str):
    """
    Sección expandible que muestra los resultados de las mejoras estadísticas:
      Fase 1: HLM con pendiente aleatoria + WAPE + Test De Finetti
      Fase 2: CLR + Bootstrap + Sesgo general
    """
    st.markdown("""
    > **¿Qué muestra esta sección?**
    > Compara tres generaciones del modelo de arrastre electoral:
    > el OLS original con efectos fijos, el HLM con pendiente aleatoria por municipio (Fase 1),
    > y el análisis bootstrap con corrección de intervalos de confianza (Fase 2).
    > Cada tabla y gráfico incluye una guía de lectura.
    """)

    hlm  = modelos["hlm"]
    dft  = modelos["definetti"]
    met  = modelos["metricas"]
    boot = modelos["bootstrap"]
    clr  = modelos["clr"]
    comp = modelos["comparativa"]

    # ── BLOQUE 1: Comparativa OLS → HLM ──────────────────────────────────────
    with st.expander("📐 Fase 1 — HLM vs OLS: ¿cuánto mejora el modelo multinivel?", expanded=True):
        st.markdown("""
        **¿Qué es el HLM?**
        El modelo lineal jerárquico (HLM) extiende el OLS permitiendo que el coeficiente de
        arrastre ($\\beta_p$) **varíe por municipio**. En el OLS todos los municipios comparten
        el mismo $\\beta$; en el HLM, cada municipio tiene su propia desviación ($u_m$) respecto
        a la media departamental. Esto produce estimaciones más honestas y predicciones más
        precisas en municipios pequeños (efecto *shrinkage*).

        **Cómo leer la tabla:**
        - **β OLS** vs **β HLM**: el HLM suele dar coeficientes mayores porque corrige el sesgo
          hacia la media que introduce el OLS con efectos fijos.
        - **ICC**: Intra-class Correlation — qué porcentaje de la varianza total se explica por
          diferencias *entre* municipios. ICC alto (>0.7) indica que el municipio importa mucho.
        - **τ² pendiente**: varianza del arrastre entre municipios. Si es alta, el arrastre no
          es uniforme en Boyacá.
        - **WAPE HLM < WAPE OLS**: el HLM comete menos error porcentual de atribución.
        """)

        if not hlm.empty:
            filt = hlm[hlm["colectividad"] == colectividad] if colectividad in hlm["colectividad"].values else hlm
            cols = ["colectividad","beta_ols_fe","beta_hlm","icc","tau2_slope",
                    "r2_ols_fe","r2_hlm_aprox","wape_ols_fe","wape_hlm","mejora_wape_pct"]
            cols_ok = [c for c in cols if c in filt.columns]
            st.dataframe(
                filt[cols_ok].rename(columns={
                    "beta_ols_fe":"β OLS","beta_hlm":"β HLM","icc":"ICC",
                    "tau2_slope":"τ² pendiente","r2_ols_fe":"R² OLS",
                    "r2_hlm_aprox":"R² HLM","wape_ols_fe":"WAPE OLS",
                    "wape_hlm":"WAPE HLM","mejora_wape_pct":"Mejora %"
                }),
                use_container_width=True,
                column_config={
                    "β OLS"      : st.column_config.NumberColumn(format="%.4f"),
                    "β HLM"      : st.column_config.NumberColumn(format="%.4f"),
                    "ICC"        : st.column_config.ProgressColumn("ICC", min_value=0, max_value=1, format="%.3f"),
                    "Mejora %"   : st.column_config.NumberColumn(format="+%.2f%%"),
                }
            )

        # Gráfico: β OLS vs β HLM
        if not hlm.empty:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name="β OLS (efectos fijos)", x=hlm["colectividad"],
                y=hlm["beta_ols_fe"], marker_color="#5588CC", opacity=0.8
            ))
            fig.add_trace(go.Bar(
                name="β HLM (pendiente aleatoria)", x=hlm["colectividad"],
                y=hlm["beta_hlm"], marker_color="#FF8C00", opacity=0.8
            ))
            fig.update_layout(
                title="Coeficiente de arrastre β: OLS vs HLM por colectividad",
                barmode="group", height=350, template="plotly_dark",
                xaxis_tickangle=-20,
                legend=dict(orientation="h", yanchor="bottom", y=1.02)
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "**Lectura:** barras naranjas (HLM) más altas que azules (OLS) indican que "
                "el OLS subestimaba el arrastre real al no modelar la varianza municipal. "
                "Centro Democrático muestra la mayor discrepancia: el OLS reportaba β=0.30 "
                "pero el HLM revela β=0.72 — el arrastre real era más del doble."
            )

    # ── BLOQUE 2: Métricas WAPE comparativas ─────────────────────────────────
    with st.expander("📏 Métricas de error: WAPE por modelo", expanded=False):
        st.markdown("""
        **¿Qué es el WAPE?**
        El *Weighted Absolute Percentage Error* pondera el error absoluto por el valor real,
        dando más peso a las mesas con más votos. A diferencia del R², captura el error
        de atribución en escala interpretable (porcentaje de votos mal asignados).

        $$\\text{WAPE} = \\frac{\\sum |\\hat{y}_i - y_i| \\cdot y_i}{\\sum y_i}$$

        **Cómo leer la tabla:**
        - **WAPE Determinístico**: el modelo proporcional puro (dominancia × SE partido). Alto error.
        - **WAPE OLS FE**: OLS con efectos fijos. Mejora sustancial.
        - **WAPE HLM**: el mejor de los tres. Menor error en todas las colectividades.
        - La mejora del HLM es modesta en %, pero sus intervalos de confianza son más honestos.
        """)

        if not met.empty:
            st.dataframe(
                met.rename(columns={
                    "WAPE_deterministico":"WAPE Det.", "WAPE_OLS_FE":"WAPE OLS",
                    "WAPE_HLM":"WAPE HLM", "R2_OLS_FE":"R² OLS",
                    "R2_HLM":"R² HLM", "mejora_HLM_vs_OLS_%":"Mejora HLM %",
                    "ICC":"ICC"
                }),
                use_container_width=True,
                column_config={
                    "WAPE Det." : st.column_config.NumberColumn(format="%.2f"),
                    "WAPE OLS"  : st.column_config.NumberColumn(format="%.2f"),
                    "WAPE HLM"  : st.column_config.NumberColumn(format="%.2f"),
                    "Mejora HLM %": st.column_config.NumberColumn(format="+%.2f%%"),
                    "ICC"       : st.column_config.ProgressColumn("ICC", min_value=0, max_value=1, format="%.3f"),
                }
            )

    # ── BLOQUE 3: Test De Finetti ─────────────────────────────────────────────
    with st.expander("🔬 Test de sobredispersión — Régimen de De Finetti", expanded=False):
        st.markdown("""
        **¿Por qué importa la sobredispersión?**
        El OLS estándar asume que los votantes de una mesa son **independientes** entre sí
        (modelo Binomial). Si hay cohesión social — votantes que se influyen — la varianza
        observada supera la esperada bajo independencia. A este fenómeno se le llama
        sobredispersión y significa que los **errores estándar del OLS están subestimados**.

        **Factor φ (phi):**
        - φ = Varianza observada / Varianza Binomial esperada
        - φ ≈ 1 → independencia (OLS válido)
        - φ >> 1 → cohesión social (IC del OLS demasiado estrechos)

        **Régimen de De Finetti:**
        - **Subcrítico** (α < 1): la correlación decae rápido con el tamaño de la mesa
        - **Crítico** (α ≈ 1): decaimiento neutro
        - **Supercrítico** (φ >> 1 sin decaimiento claro): fuerte cohesión, OLS no confiable

        **Todos los partidos muestran φ entre 14 y 29** → el OLS subestimaba los IC en
        factor √φ ≈ 3.8–5.4x. El HLM y el Bootstrap corrigen esta subestimación.
        """)

        if not dft.empty:
            cols_d = ["colectividad","phi_sobredispersion","p_valor_chi2",
                      "regimen_definetti","sobredispersion"]
            cols_ok = [c for c in cols_d if c in dft.columns]
            st.dataframe(
                dft[cols_ok].rename(columns={
                    "phi_sobredispersion":"φ sobredispersión",
                    "p_valor_chi2":"p-valor Chi²",
                    "regimen_definetti":"Régimen",
                    "sobredispersion":"¿Sobredispersión?"
                }),
                use_container_width=True,
                column_config={
                    "φ sobredispersión": st.column_config.NumberColumn(format="%.1f"),
                    "p-valor Chi²": st.column_config.NumberColumn(format="%.2e"),
                    "¿Sobredispersión?": st.column_config.CheckboxColumn(),
                }
            )

            fig_phi = px.bar(
                dft.sort_values("phi_sobredispersion", ascending=False),
                x="colectividad", y="phi_sobredispersion",
                color="phi_sobredispersion",
                color_continuous_scale=[[0,"#2ecc71"],[0.3,"#f39c12"],[1,"#e74c3c"]],
                labels={"phi_sobredispersion":"Factor φ","colectividad":"Partido"},
                title="Factor de sobredispersión φ por colectividad (φ=1 = independencia)"
            )
            fig_phi.add_hline(y=1, line_dash="dash", line_color="white",
                              annotation_text="φ=1 (independencia)", annotation_position="right")
            fig_phi.update_layout(height=320, template="plotly_dark",
                                  coloraxis_showscale=False, xaxis_tickangle=-15)
            st.plotly_chart(fig_phi, use_container_width=True)
            st.caption(
                "**Lectura:** todos los partidos están muy por encima de φ=1 (línea blanca). "
                "Alianza Verde (φ=29) tiene la mayor cohesión social: en mesas donde el partido "
                "es fuerte, los votantes tienden a votar en bloque tanto en CA como en SE. "
                "Esto valida el uso del HLM y del Bootstrap en lugar del OLS simple."
            )

    # ── BLOQUE 4: Bootstrap — IC corregidos ───────────────────────────────────
    with st.expander("🎲 Fase 2 — Bootstrap: intervalos de confianza corregidos", expanded=False):
        st.markdown("""
        **¿Por qué Bootstrap?**
        Dado que los votantes de una mesa no son independientes (φ >> 1), los IC del OLS
        están subestimados. El **cluster bootstrap** remuestrea mesas completas (no filas
        individuales), preservando la correlación intra-mesa y produciendo IC más amplios
        y más honestos.

        **Sesgo general de elección (b):**
        Representa la variabilidad adicional que ningún modelo puede capturar con una sola
        elección — el "ruido de año electoral":
        $$b = \\sqrt{\\max(0,\\ Var_{bootstrap} - Var_{modelo})}$$

        **Cómo leer la tabla:**
        - **Ratio SE**: cuántas veces más grande es el SE bootstrap vs el OLS (>1 = OLS subestimaba)
        - **IC OLS vs IC Bootstrap**: el bootstrap amplía el intervalo en los partidos con mayor
          cohesión social
        - **b sesgo**: mayor en Centro Democrático (ICC bajo = arrastre muy variable por municipio)
        """)

        if not boot.empty:
            boot_d = boot.copy()
            boot_d["ic_ols"] = boot_d.apply(
                lambda r: f"[{r['ic_ols_low']:.3f}, {r['ic_ols_high']:.3f}]", axis=1)
            boot_d["ic_boot"] = boot_d.apply(
                lambda r: f"[{r['ic_boot_low']:.3f}, {r['ic_boot_high']:.3f}]", axis=1)
            st.dataframe(
                boot_d[["colectividad","beta_ols","se_ols","se_bootstrap",
                         "ratio_se","b_sesgo_general","ic_ols","ic_boot"]].rename(columns={
                    "beta_ols":"β puntual","se_ols":"SE OLS","se_bootstrap":"SE Bootstrap",
                    "ratio_se":"Ratio SE","b_sesgo_general":"Sesgo b",
                    "ic_ols":"IC 95% OLS","ic_boot":"IC 95% Bootstrap"
                }),
                use_container_width=True,
                column_config={
                    "β puntual"   : st.column_config.NumberColumn(format="%.4f"),
                    "SE OLS"      : st.column_config.NumberColumn(format="%.4f"),
                    "SE Bootstrap": st.column_config.NumberColumn(format="%.4f"),
                    "Ratio SE"    : st.column_config.NumberColumn(format="×%.2f"),
                    "Sesgo b"     : st.column_config.NumberColumn(format="%.4f"),
                }
            )

            # Gráfico IC OLS vs Bootstrap
            fig_ic = go.Figure()
            for _, r in boot.iterrows():
                col = r["colectividad"]
                fig_ic.add_trace(go.Scatter(
                    x=[r["ic_ols_low"], r["ic_ols_high"]], y=[col, col],
                    mode="lines", line=dict(color="#5588CC", width=6),
                    name="IC OLS" if col == boot["colectividad"].iloc[0] else "",
                    showlegend=(col == boot["colectividad"].iloc[0]),
                    legendgroup="OLS"
                ))
                fig_ic.add_trace(go.Scatter(
                    x=[r["ic_boot_low"], r["ic_boot_high"]], y=[col, col],
                    mode="lines", line=dict(color="#FF8C00", width=3, dash="dot"),
                    name="IC Bootstrap" if col == boot["colectividad"].iloc[0] else "",
                    showlegend=(col == boot["colectividad"].iloc[0]),
                    legendgroup="Boot"
                ))
                fig_ic.add_trace(go.Scatter(
                    x=[r["beta_ols"]], y=[col],
                    mode="markers", marker=dict(color="white", size=8),
                    showlegend=False
                ))
            fig_ic.update_layout(
                title="IC 95% del coeficiente β: OLS (azul) vs Bootstrap (naranja)",
                xaxis_title="β (votos SE adicionales por voto CA adicional)",
                height=350, template="plotly_dark",
                legend=dict(orientation="h", yanchor="bottom", y=1.02)
            )
            st.plotly_chart(fig_ic, use_container_width=True)
            st.caption(
                "**Lectura:** el punto blanco es el estimado puntual. La barra azul es el IC "
                "del OLS (subestimado). La barra naranja punteada es el IC Bootstrap (correcto). "
                "Centro Democrático y CR-Nuevo Liberalismo tienen los IC más amplios — "
                "son los partidos con mayor variabilidad del arrastre entre municipios."
            )

    # ── BLOQUE 5: Diagnóstico CLR ─────────────────────────────────────────────
    with st.expander("🔄 Fase 2 — CLR: ¿mejora la transformación composicional?", expanded=False):
        st.markdown("""
        **¿Qué es la transformación CLR?**
        La dominancia $d_{ij}$ es una **composición** (suma a 1 dentro del partido en cada mesa),
        lo que viola el supuesto euclidiano del OLS. La transformación *Centered Log-Ratio* (CLR)
        mapea el simplex $\\Delta^{J-1}$ al espacio euclidiano $\\mathbb{R}^{J-1}$:

        $$\\text{clr}(d_{ij}) = \\log(d_{ij}) - \\frac{1}{J}\\sum_{j'} \\log(d_{ij'})$$

        **Resultado del diagnóstico:**
        En nuestro modelo, el predictor `log(dominancia)` directamente (**Modelo A**, WAPE bajo)
        supera al CLR en 5 de 6 colectividades. Esto ocurre porque en cada predicción usamos
        **un solo candidato** — el CLR pleno (con J predictores simultáneos) aplica cuando
        se modelan todos los candidatos en paralelo. El CLR queda como extensión futura
        para modelos composicionales completos.
        """)

        if not clr.empty and "beta_clr" in clr.columns:
            cols_c = ["colectividad","r2_hlm_dom","r2_hlm_clr","wape_hlm_dom","wape_hlm_clr","mejora_clr_pct"]
            cols_ok = [c for c in cols_c if c in clr.columns]
            st.dataframe(
                clr[cols_ok].rename(columns={
                    "r2_hlm_dom":"R² log(dom)","r2_hlm_clr":"R² CLR",
                    "wape_hlm_dom":"WAPE log(dom)","wape_hlm_clr":"WAPE CLR",
                    "mejora_clr_pct":"Mejora CLR %"
                }),
                use_container_width=True,
                column_config={
                    "Mejora CLR %": st.column_config.NumberColumn(format="+%.1f%%"),
                }
            )
            st.info(
                "💡 El CLR no mejora el modelo individual-candidato. Su valor está en "
                "modelar la composición completa del partido (todos los candidatos simultáneamente), "
                "que es el siguiente paso natural de la Fase 3."
            )


def render_modelo_tab(modelos, colectividad):
    """Wrapper que muestra el encabezado y llama a la función principal."""
    st.header("📊 Modelo Estadístico Avanzado")
    st.markdown("""
    Esta sección documenta la **evolución del modelo de arrastre electoral** a través de
    tres generaciones estadísticas, aplicando los hallazgos de la literatura especializada
    en modelos electorales composicionales y jerárquicos.

    | Fase | Modelo | Mejora principal |
    |---|---|---|
    | **Fase 0** | OLS con efectos fijos por municipio | Línea base |
    | **Fase 1** | HLM con pendiente aleatoria + WAPE + De Finetti | Corrige varianza municipal y sobredispersión |
    | **Fase 2** | Bootstrap cluster + diagnóstico CLR | IC honestos + estimación sesgo general |

    > ⚠️ **Advertencia metodológica:** todos los coeficientes describen relaciones entre
    > agregados por mesa. No implican causalidad ni comportamiento individual del votante
    > *(falacia ecológica de Robinson, 1950)*.
    """)
    render_modelo_estadistico(modelos, colectividad)


if __name__ == "__main__":
    main()
