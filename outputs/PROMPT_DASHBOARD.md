# Prompt — Dashboard Electoral Interactivo 2026
## Análisis de Arrastre Cámara ↔ Senado, Boyacá

> Pega este bloque en tu agente IA (Claude Code, Cursor, Copilot Agent).
> Trabaja fase por fase. Espera confirmación entre fases.

---

## CONTEXTO DEL PROYECTO

Tienes un proyecto Python en:
`/Users/willamaya/Desktop/Elecciones 2026/proyecto-electoral-2026/`

Los datos ya están procesados en:
- `data/processed/mesa_consolidada.parquet`  — 74 K filas, unidad = mesa × candidato CA
- `data/interim/votos_camara_mesa.parquet`  — 88 K filas tidy Cámara
- `data/interim/votos_senado_mesa.parquet`  — 130 K filas tidy Senado
- `data/processed/arrastre_candidato.parquet`  — 48 candidatos CA con índices
- `data/processed/residuos_*.parquet`  — residuos OLS por colectividad
- `outputs/correlaciones.csv`
- `outputs/regresion_resultados.csv`
- `config/settings.py`  — todas las rutas centralizadas

La DB cruda sigue en:
`/Users/willamaya/Desktop/Elecciones 2026/outputs/puestos_2026.db`

---

## OBJETIVO

Construir un **dashboard Streamlit** interactivo que permita analizar la incidencia
(arrastre/coattail) de candidatos a Cámara sobre los votos al Senado de su misma
colectividad, a cualquier nivel de granularidad (desde Boyacá completo hasta una
mesa individual), y también en sentido inverso (SE → CA).

---

## ARQUITECTURA DEL DASHBOARD

### Modos de operación (toggle en la barra lateral)
```
[CA → SE]   [SE → CA]
```
- **CA → SE**: Seleccionas partido/candidato de Cámara, ves el rendimiento de
  su colectividad en Senado en las mismas mesas.
- **SE → CA**: Seleccionas partido/candidato de Senado, ves el rendimiento de
  su colectividad en Cámara en las mismas mesas.

### Panel de filtros (sidebar, cascading)
```
Modo: [CA→SE] / [SE→CA]
─────────────────────────────────
Colectividad   [dropdown]
Candidato      [dropdown, filtrado por colectividad]
─────────────────────────────────
Municipio      [multiselect, "Todos" por defecto]
Localidad/Zona [multiselect, filtrado por municipio]
Puesto         [multiselect, filtrado por zona]
Mesa           [slider range o "Todas"]
─────────────────────────────────
Modelo visual  [Determinístico | Estadístico | Probabilístico | Todos]
```
Los filtros son en cascada: cambiar municipio actualiza zonas disponibles,
cambiar zona actualiza puestos, etc. La granularidad va de lo general a lo específico.

### Layout principal (3 filas)

**Fila 1 — KPIs (4 tarjetas)**
```
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ Votos CA     │ │ Votos SE     │ │ Ratio        │ │ Índice       │
│ candidato    │ │ colectividad │ │ arrastre     │ │ incidencia   │
│ seleccionado │ │ (mismo       │ │ = SE / CA    │ │ ponderado    │
│              │ │  mesas)      │ │ partido      │ │              │
└──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
```
Cada KPI muestra el valor filtrado vs el promedio departamental (delta arrow).

**Fila 2 — Gráficos de relación CA ↔ SE**
```
┌─────────────────────────┐  ┌─────────────────────────┐
│ Scatter: votos CA vs SE │  │ Distribución de ratios  │
│ • Determinístico:       │  │ de arrastre             │
│   puntos coloreados     │  │ • KDE (probabilístico)  │
│   por municipio         │  │ • Percentil candidato   │
│ • Estadístico:          │  │   seleccionado          │
│   línea OLS + R²        │  │ • Línea media depto     │
│ • Probabilístico:       │  │ • Deciles               │
│   banda de pred. 80/95% │  │                         │
└─────────────────────────┘  └─────────────────────────┘
```

**Fila 3 — Drill-down geográfico**
```
┌─────────────────────────────────────────────────────────────┐
│ Bar chart agrupado: votos CA + votos SE por municipio/zona/ │
│ puesto/mesa según nivel de filtro activo                    │
│ Ordenado por votos CA desc. Colores = colectividad          │
└─────────────────────────────────────────────────────────────┘
```

**Fila 4 — Heatmap de mesas**
```
┌─────────────────────────────────────────────────────────────┐
│ Heatmap: puestos (eje Y) × mesas (eje X)                   │
│ Color = ratio_arrastre_mesa (escala divergente 0–3)        │
│ Tooltip = municipio, puesto, mesa, votos CA, votos SE,     │
│           ratio, percentil en la distribución              │
│ Mesas anómalas (outliers) marcadas con borde rojo          │
└─────────────────────────────────────────────────────────────┘
```

**Fila 5 — Tabla detalle**
```
┌─────────────────────────────────────────────────────────────┐
│ Tabla interactiva (st.dataframe) con las mesas del filtro  │
│ Columnas: municipio, zona, puesto, mesa, votos_CA,         │
│   votos_SE, ratio_arrastre, percentil, residuo_OLS,        │
│   es_outlier, participación (votantes/potencial)           │
│ Descargable como CSV                                       │
└─────────────────────────────────────────────────────────────┘
```

---

## MODELOS A INTERCALAR

### 1. Determinístico (siempre visible)
- Sumas exactas de votos CA y SE en las mesas filtradas
- `ratio_arrastre = votos_SE_colectividad / votos_CA_partido`
- `indice_incidencia_pond` del candidato seleccionado
- Participación: `votantes / potencial_electoral`

### 2. Estadístico (activable)
- Línea de regresión OLS con `coef_votos_ca` e intervalo de confianza 95%
- R² ajustado del modelo con efectos fijos por municipio
- Pearson y Spearman para el subconjunto filtrado
- Residuos del OLS: positivo = la mesa "superó" lo esperado para ese candidato

### 3. Probabilístico (activable)
- **KDE** (kernel density estimation) de la distribución de ratios de arrastre
  en todas las mesas de esa colectividad/Boyacá
- **Bandas de predicción** al 80% y 95% (intervalos de predicción del OLS,
  más anchos que los de confianza: reflejan variabilidad individual de mesas)
- **Percentil del candidato seleccionado** dentro de la distribución:
  "Este candidato está en el percentil X de su colectividad en arrastre"
- **Probabilidad de arrastre positivo**: P(ratio > 1 | votos_CA = X) calculado
  de la distribución empírica condicional (binning de votos_CA)

### Diferencia visual clave
| | Estadístico | Probabilístico |
|---|---|---|
| Pregunta | ¿Cuál es la relación promedio? | ¿Qué tan probable es X resultado? |
| Visual | Línea + IC (certeza del estimado) | Banda de predicción + KDE (incertidumbre de la mesa) |
| Uso | Comparar colectividades | Evaluar si una mesa específica es anómala |

---

## ESPECIFICACIONES TÉCNICAS

### Stack
```python
streamlit>=1.32
plotly>=5.18
pandas>=2.0
pyarrow>=14.0
scipy>=1.11
statsmodels>=0.14
numpy>=1.26
```

### Archivos a crear
```
src/
└── dashboard.py        ← app Streamlit principal
config/
└── colores_partidos.py ← dict codpar → color hex (usa colores reales de partidos)
```

### Colores de colectividades (usar colores reales)
```python
COLORES = {
    "ALIANZA VERDE"            : "#007C34",
    "PACTO HISTORICO"          : "#8B1A4A",
    "CENTRO DEMOCRATICO"       : "#1E477D",
    "PARTIDO LIBERAL"          : "#E30716",
    "CONSERVADOR-SALV NACIONAL": "#0867B1",
    "CR-NUEVO LIBERALISMO"     : "#F95846",
    "PARTIDO DE LA U"          : "#48AB38",
}
```

### Rendimiento
- Cachear los parquets con `@st.cache_data`
- Pre-calcular los modelos estadísticos al cargar (no en cada interacción)
- Filtros en cascada: recomputar solo el subconjunto necesario

---

## FASE A — Estructura y carga de datos
1. Crear `src/dashboard.py` con layout básico
2. Cargar los 3 parquets cacheados
3. Implementar filtros en cascada en el sidebar
4. KPIs funcionales con delta vs departamento

## FASE B — Gráficos CA → SE
5. Scatter con modo determinístico (puntos coloreados)
6. Añadir overlay estadístico (OLS line + R²)
7. Añadir overlay probabilístico (bandas predicción + KDE)
8. Distribución de ratios con percentil del candidato

## FASE C — Drill-down geográfico
9. Bar chart por nivel de filtro activo (municipio/zona/puesto/mesa)
10. Heatmap de mesas con outliers marcados

## FASE D — Modo inverso SE → CA
11. Toggle SE→CA en sidebar
12. Reconfigurar filtros para candidatos SE
13. Adaptar todos los gráficos al modo inverso

## FASE E — Tabla y exportación
14. Tabla interactiva con todas las columnas
15. Botón descarga CSV del subconjunto filtrado

---

## REGLAS
- Trabajar fase por fase; mostrar preview antes de continuar
- No hardcodear rutas: usar `config/settings.py`
- Todos los gráficos deben tener título claro que indique qué modelo están usando
- La advertencia de inferencia ecológica debe aparecer en el sidebar siempre visible
- Si el filtro resulta en <5 mesas, mostrar aviso y deshabilitar modelos estadísticos

Empieza por la FASE A.
