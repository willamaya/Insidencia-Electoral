# Descripción Técnica del Modelo de Arrastre Electoral — v2
## Elecciones Legislativas Colombia 2026 — Boyacá
### Análisis de incidencia Cámara de Representantes → Senado de la República

> **Versión:** 2.0 · **Fecha:** 4 de junio de 2026
> **Cambios respecto a v1:** Incorpora HLM con pendiente aleatoria (Fase 1),
> Bootstrap cluster y diagnóstico CLR (Fase 2).

---

## 0. Notación y unidad de análisis

| Símbolo | Descripción |
|---|---|
| $i$ | Mesa electoral (unidad mínima de análisis) |
| $m$ | Municipio |
| $j$ | Candidato a Cámara |
| $k$ | Candidato / lista al Senado |
| $p$ | Partido o colectividad |
| $C_{ij}$ | Votos del candidato $j$ a Cámara en mesa $i$ |
| $P^{CA}_{ip}$ | Votos totales del partido $p$ a Cámara en mesa $i$ |
| $S^{total}_{ip}$ | Votos totales del partido $p$ al Senado en mesa $i$ |
| $N_i$ | Potencial electoral de la mesa $i$ |
| $V_i$ | Votantes efectivos en la mesa $i$ |
| $M$ | Total mesas Boyacá ($M = 3{,}282$) |
| $\phi$ | Factor de sobredispersión (test De Finetti) |
| $\tau^2$ | Varianza del arrastre entre municipios (HLM) |
| $b$ | Sesgo general de elección (Bootstrap) |

**Fuente:** API JSON Registraduría Nacional, `resultadospreccongreso2026.registraduria.gov.co`.
**Cobertura:** 3,282/3,282 mesas Cámara · 3,281/3,282 mesas Senado (Boyacá).

---

## 1. Componente Determinístico

### 1.1 Dominancia y atribución proporcional

$$d_{ij} = \frac{C_{ij}}{P^{CA}_{ip}}, \quad \sum_{j \in p} d_{ij} = 1, \quad d_{ij} \in \Delta^{J-1}$$

$$\hat{S}_{ij}^{det} = d_{ij} \cdot S^{total}_{ip}$$

### 1.2 Ratio de arrastre

$$r_{ip} = \frac{S^{total}_{ip}}{P^{CA}_{ip}}$$

| Estadístico | Valor |
|---|---|
| $\bar{r}$ | 1.084 |
| $\sigma_r$ | 1.683 |
| $p_{50}$ | 0.769 |
| Rango | [0.000, 146.000] |

### 1.3 Índices de incidencia

$$\mathcal{I}_j = \frac{\sum_{i: C_{ij}>0} S^{total}_{ip}}{\sum_{i: C_{ij}>0} C_{ij}}, \qquad
\mathcal{I}_j^{pond} = \frac{\sum_i S^{total}_{ip} \cdot d_{ij}}{\sum_i C_{ij}}$$

---

## 2. Componente Probabilístico — Evolución de Modelos

### 2.1 Fase 0 — OLS con efectos fijos (línea base)

$$\tilde{S}_{ip} = \beta_p \cdot C_{ij} + \sum_m \gamma_{pm} \cdot \mathbf{1}[i \in m] + \varepsilon_{ip}$$

**Limitación identificada:** los efectos fijos tratan a todos los municipios con la misma
varianza de arrastre, ignorando la heterogeneidad real. Los IC del OLS están subestimados
en factor $\sqrt{\phi} \approx 3.9$–$5.4$ debido a la sobredispersión.

### 2.2 Fase 1 — HLM con pendiente aleatoria (implementado)

$$\tilde{S}_{ip} = (\beta_p + u_{pm}) \cdot C_{ij} + \gamma_{pm} + \varepsilon_{ip}$$

$$u_{pm} \sim \mathcal{N}(0, \tau^2_p), \quad \varepsilon_{ip} \sim \mathcal{N}(0, \sigma^2_p)$$

El coeficiente de arrastre ahora **varía por municipio**: el estimado departamental $\beta_p$
se redistribuye con encogimiento (*shrinkage*) hacia la media en municipios con pocas mesas.

**Resultados Fase 1** (estimación REML, 123 municipios como grupos):

| Colectividad | $\hat\beta_{OLS}$ | $\hat\beta_{HLM}$ | $\hat\tau^2_{slope}$ | ICC | $R^2_{OLS}$ | $R^2_{HLM}$ | WAPE HLM | Mejora % |
|---|---|---|---|---|---|---|---|---|
| **Centro Democrático** | 0.297 | **0.716** | 0.495 | 0.398 | 0.294 | 0.349 | 11.39 | +4.16% |
| Alianza Verde | 0.230 | 0.351 | 54.99 | **0.886** | 0.531 | 0.548 | 8.84 | +1.48% |
| CR-Nuevo Lib. | 0.068 | 0.269 | 5.253 | 0.873 | 0.498 | 0.519 | 3.88 | +1.90% |
| Conservador | 0.119 | 0.169 | 0.027 | 0.635 | 0.512 | 0.538 | 4.19 | +2.67% |
| Partido Liberal | 0.081 | 0.098 | 18.970 | 0.780 | 0.521 | 0.531 | 4.62 | +0.75% |
| Partido de la U | 0.023 | 0.062 | 3.618 | 0.872 | 0.594 | 0.600 | 3.69 | +1.12% |

**Hallazgo crítico:** el OLS subestimaba el arrastre de Centro Democrático en 141% ($\beta_{OLS}=0.297$
vs $\beta_{HLM}=0.716$). Alianza Verde y CR-Nuevo Liberalismo muestran altísimo ICC (>0.87),
lo que indica que el municipio explica más del 87% de la varianza total del arrastre para
estos partidos — la heterogeneidad geográfica es determinante.

**ICC (Intra-class Correlation):**

$$ICC_p = \frac{\tau^2_{p,intercept}}{\tau^2_{p,intercept} + \sigma^2_p}$$

ICC alto → el municipio importa más que la mesa individual para predecir el arrastre.

### 2.3 Test de sobredispersión — Régimen de De Finetti (Fase 1)

**Hipótesis:**
- $H_0$: votantes independientes dentro de la mesa → Binomial$(N_i, \pi)$
- $H_1$: cohesión social (sobredispersión) → $\phi > 1$

$$\phi = \frac{Var\left(\frac{C_{ij}}{N_i}\right)}{\bar\pi(1-\bar\pi)/\bar N}$$

**Estadístico chi²:**

$$\chi^2 = \sum_i \frac{(C_{ij} - N_i\bar\pi)^2}{N_i\bar\pi}, \quad df = n - 1$$

**Resultados:**

| Colectividad | $\phi$ | $p$-valor $\chi^2$ | Régimen |
|---|---|---|---|
| Alianza Verde | **29.1** | $\approx 0$ | Crítico/Neutro ($\alpha=1.74$) |
| Partido de la U | **25.8** | $\approx 0$ | Subcrítico ($\alpha=0.89$) |
| Conservador | **21.9** | $\approx 0$ | Subcrítico ($\alpha=0.69$) |
| Partido Liberal | **21.1** | $\approx 0$ | Subcrítico ($\alpha=0.07$) |
| Centro Democrático | **20.1** | $\approx 0$ | Supercrítico |
| CR-Nuevo Lib. | **14.9** | $\approx 0$ | Crítico/Neutro ($\alpha=3.58$) |

**Todos los partidos rechazan $H_0$ con $p \approx 0$:** hay cohesión social real.
Los IC del OLS estaban subestimados en factor $\sqrt{\phi} \approx 3.9$–$5.4$.
El HLM y el Bootstrap (Fase 2) corrigen esta subestimación.

### 2.4 Fase 2 — Bootstrap cluster e IC corregidos

El **cluster bootstrap** remuestrea mesas completas (no filas individuales),
preservando la correlación intra-mesa:

```
Para b = 1, ..., B:
  1. Remuestrear M mesas con reemplazo: {i*_1, ..., i*_M}
  2. Estimar β* con regresión matricial:
     X = [1 | C_ij]  →  β* = (X^T X)^{-1} X^T S
  3. Registrar β*_b
```

**Sesgo general de elección:**

$$b_p = \sqrt{\max(0,\; Var_{bootstrap}(\hat\beta_p) - Var_{OLS}(\hat\beta_p))}$$

Representa la variabilidad del arrastre **no capturable con una sola elección**.
Con datos de múltiples años electorales, $b_p$ podría estimarse directamente.

**Resultados Bootstrap** ($B = 100$ réplicas, remuestreo por mesas):

| Colectividad | $\hat\beta$ | $SE_{OLS}$ | $SE_{boot}$ | Ratio $\times$ | $b$ sesgo | IC OLS | IC Bootstrap |
|---|---|---|---|---|---|---|---|
| Alianza Verde | 0.533 | 0.014 | 0.019 | ×1.41 | 0.013 | ±0.053 | ±0.082 |
| **Centro Democ.** | 0.392 | 0.020 | **0.044** | **×2.20** | **0.039** | ±0.078 | **±0.178** |
| **CR-Nuevo Lib.** | 0.183 | 0.010 | **0.021** | **×2.20** | **0.019** | ±0.038 | **±0.086** |
| Partido de la U | 0.189 | 0.006 | 0.011 | ×1.98 | 0.010 | ±0.022 | ±0.045 |
| Conservador | 0.266 | 0.008 | 0.015 | ×1.80 | 0.012 | ±0.032 | ±0.059 |
| Partido Liberal | 0.213 | 0.009 | 0.010 | ×1.13 | 0.005 | ±0.035 | ±0.037 |

Centro Democrático y CR-Nuevo Liberalismo tienen los IC más engañosos en el OLS
(subestimados en ×2.20). Partido Liberal es el más robusto (ratio ×1.13).

### 2.5 Diagnóstico CLR — Fase 2

La transformación Centered Log-Ratio:

$$\text{clr}(d_{ij}) = \log(d_{ij}) - \frac{1}{J_i}\sum_{j'} \log(d_{ij'})$$

**Resultado del diagnóstico:** el predictor $\log(d_{ij})$ directamente (**Modelo A**)
supera al CLR en 5 de 6 colectividades porque en el modelo individual usamos **un candidato**,
no la composición completa. El CLR pleno aplica cuando se modelan todos los candidatos
simultáneamente como vector composicional.

| Colectividad | WAPE log(dom) | WAPE CLR | Mejora CLR |
|---|---|---|---|
| Alianza Verde | 5.68 | 10.00 | −76.0% |
| Centro Democrático | 5.79 | 13.86 | −139.2% |
| Conservador | 2.58 | 4.73 | −83.3% |
| CR-Nuevo Lib. | 3.03 | 4.40 | −45.2% |
| **Partido de la U** | 4.84 | 4.35 | **+10.2%** ← único positivo |
| Partido Liberal | 3.33 | 5.20 | −56.0% |

**Conclusión:** el CLR completo (regresión composicional Dirichlet-multinomial con
todos los candidatos del partido como predictores simultáneos) es la Fase 3 natural.

---

## 3. Tratamiento de Datos

### 3.1 Cobertura y recuperación de mesas

| Elección | Esperadas | Descargadas | Recuperadas (reintento) | Pendiente |
|---|---|---|---|---|
| Cámara | 3,282 | 3,282 | 33 (HTTP 500) | 0 |
| Senado | 3,282 | 3,281 | 13 | 1 (0 votos válidos) |

### 3.2 Tratamiento de votos especiales

| Tipo | Tratamiento |
|---|---|
| Voto en blanco | Excluido de candidatos; incluido en denominador $V_i$ |
| Nulo / no marcado | Excluido de todos los cómputos |
| "Solo por la lista" ($codcan=0$) | Incluido en $P^{CA}_{ip}$ pero excluido del índice individual |
| Split-ticket ($S^{total}_{ip} > P^{CA}_{ip}$) | Masa no atribuible; cuantificada como $S - P$ |

---

## 4. Métricas — Versión 2

### 4.1 WAPE (métrica principal desde v2)

$$\text{WAPE} = \frac{\sum_i |\hat{y}_i - y_i| \cdot y_i}{\sum_i y_i}$$

Pondera el error por el volumen de votos — mesas grandes contribuyen más que mesas pequeñas.

### 4.2 IC OLS vs Bootstrap

$$IC^{OLS}_{95\%} = \hat\beta_p \pm 1.96 \cdot \widehat{SE}_{OLS}$$

$$IC^{Boot}_{95\%} = \left[q_{0.025}(\hat\beta^*_1, \ldots, \hat\beta^*_B),\;
                           q_{0.975}(\hat\beta^*_1, \ldots, \hat\beta^*_B)\right]$$

El IC Bootstrap es más conservador y correcto bajo sobredispersión.

### 4.3 ICC del HLM

$$ICC_p = \frac{\hat\tau^2_{p,int}}{\hat\tau^2_{p,int} + \hat\sigma^2_p}$$

Interpretación: qué fracción de la varianza total del arrastre se debe a diferencias
estructurales entre municipios (vs. ruido de mesa individual).

### 4.4 Detección de mesas anómalas

$$z_i = \frac{r_{ip} - \bar{r}_{pm}}{\sigma_{r_{pm}}} > 3 \;\Rightarrow\; \text{outlier}$$

918 mesas anómalas identificadas en Boyacá (28% del total), principalmente mesas
unipersonales con ratios extremos.

---

## 5. Comparativa de modelos — Resumen ejecutivo

```
FASE 0: OLS efectos fijos
  β_OLS ∈ [0.023, 0.297]   R² ∈ [0.29, 0.59]
  IC subestimados por √φ ≈ 3.9–5.4x   WAPE ∈ [3.7, 11.9]

FASE 1: HLM pendiente aleatoria
  β_HLM ∈ [0.062, 0.716]   R² ∈ [0.35, 0.60]
  ICC ∈ [0.40, 0.89] → el municipio explica hasta 89% de la varianza
  WAPE ∈ [3.7, 11.4]   Mejora WAPE: 0.75%–4.16% sobre OLS

FASE 2: Bootstrap + CLR diagnóstico
  SE_boot / SE_OLS ∈ [1.13, 2.20] → IC OLS hasta 2.2x más estrechos de lo real
  Sesgo b ∈ [0.005, 0.039]
  CLR no mejora el modelo individual → reservado para Fase 3 composicional

GANADOR ACTUAL: HLM Fase 1 + IC Bootstrap Fase 2
```

---

## 6. Limitaciones y hoja de ruta

| Limitación | Fase | Solución propuesta |
|---|---|---|
| Inferencia ecológica | Todas | Advertencia explícita; no inferir comportamiento individual |
| Solo 1 año electoral | b de sesgo | Añadir elecciones 2022, 2018 para estimar $b$ con datos reales |
| CLR individual subóptimo | Fase 2 | Fase 3: regresión composicional con vector $(d_{i1}, \ldots, d_{iJ})$ |
| MICE solo para cobertura completa | — | Fase 4: predicción en tiempo real con cobertura parcial |
| Un departamento | Boyacá | Escalar a Colombia con muestreo polietápico y factores $w_i$ |

---

*Código: `proyecto-electoral-2026-v2/src/` — Python 3.13, statsmodels 0.14.6, scikit-learn 1.8.0*
*Dashboard: `streamlit run src/dashboard.py` en http://localhost:8501*
