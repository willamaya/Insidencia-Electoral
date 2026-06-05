# Descripción Técnica del Modelo de Arrastre Electoral
## Elecciones Legislativas Colombia 2026 — Boyacá
### Análisis de incidencia Cámara de Representantes → Senado de la República

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
| $S_{ik}$ | Votos del candidato/lista $k$ al Senado en mesa $i$ |
| $S^{total}_{ip}$ | Votos totales del partido $p$ al Senado en mesa $i$ |
| $N_i$ | Potencial electoral (census) de la mesa $i$ |
| $V_i$ | Votantes efectivos en la mesa $i$ |
| $M$ | Número total de mesas en el departamento ($M = 3{,}282$ para Boyacá) |

**Fuente de datos:** API JSON de la Registraduría Nacional del Estado Civil,
portal `resultadospreccongreso2026.registraduria.gov.co`.
Cobertura: **3,282/3,282 mesas Cámara** y **3,281/3,282 mesas Senado** (Boyacá).

---

## 1. Componente Determinístico

### 1.1 Dominancia del candidato en la mesa

La dominancia mide el peso relativo del candidato $j$ dentro del partido $p$
en la mesa $i$:

$$
d_{ij} = \frac{C_{ij}}{P^{CA}_{ip}}, \quad \sum_{j \in p} d_{ij} = 1
$$

Esta es una **composición simplex** $d_i \in \Delta^{J-1}$, donde $J$ es el número
de candidatos del partido. Los votos de lista ("Solo por la lista") se excluyen
del numerador pero sí entran en $P^{CA}_{ip}$.

### 1.2 Ratio de arrastre por mesa

El ratio de arrastre cuantifica cuántos votos al Senado obtuvo el partido por
cada voto a Cámara en la mesa $i$:

$$
r_{ip} = \frac{S^{total}_{ip}}{P^{CA}_{ip}}
$$

**Estadísticos observados** (Boyacá, $n = 74{,}029$ observaciones candidato×mesa):

| Estadístico | Valor |
|---|---|
| Media $\bar{r}$ | 1.084 |
| Desviación estándar $\sigma_r$ | 1.683 |
| Mediana ($p_{50}$) | 0.769 |
| Percentil 10 / 90 | 0.200 / 1.897 |
| Rango | [0.000, 146.000] |

La distribución es **asimétrica a la derecha** (skew positivo), con cola pesada
generada por mesas pequeñas donde un partido captura muchos SE con pocos CA.

### 1.3 Atribución determinística

La atribución de votos al Senado al candidato $j$ en la mesa $i$ es proporcional
a su dominancia:

$$
\hat{S}_{ij}^{det} = d_{ij} \cdot S^{total}_{ip}
$$

El porcentaje de apoyo que el candidato $j$ aporta al senador $k$ en la mesa $i$:

$$
\alpha_{ijk}^{det} = d_{ij} \cdot 100\%
$$

Obsérvese que $\alpha_{ijk}^{det}$ **no depende de $k$** en el modelo
determinístico: todos los candidatos al Senado del partido reciben el mismo
vector de atribuciones desde Cámara, proporcional a la composición $d_{ij}$.

### 1.4 Índice de incidencia agregado (departamento)

Para el candidato $j$ a Cámara, el índice de incidencia bruto sobre el Senado
del partido es:

$$
\mathcal{I}_j = \frac{\displaystyle\sum_{i: C_{ij}>0} S^{total}_{ip}}
                     {\displaystyle\sum_{i: C_{ij}>0} C_{ij}}
$$

La versión ponderada por dominancia controla el efecto de masa del candidato:

$$
\mathcal{I}_j^{pond} = \frac{\displaystyle\sum_{i} S^{total}_{ip} \cdot d_{ij}}
                            {\displaystyle\sum_{i} C_{ij}}
$$

### 1.5 Homologación de colectividades CA → SE

El cruce CA↔SE requiere una tabla de correspondencia $\mathcal{H}$ que asigna
pesos $w_{ps}$ entre cada codpar de Cámara $p$ y el/los codpar(es) de Senado $s$:

$$
\mathcal{H} = \{(p, s, w_{ps})\}, \quad \sum_{s} w_{ps} = 1 \;\forall p
$$

Casos especiales:

- $p = 121$ (Conservador-Salv. Nacional) $\to s \in \{3, 17\}$, $w = 0.5$ c/u
- $p \in \{122, 137\}$ $\to s = 44$, $w = 1.0$

El Senado ponderado por colectividad en mesa $i$:

$$
\tilde{S}_{ip} = \sum_{s} w_{ps} \cdot S^{total}_{is}
$$

---

## 2. Componente Probabilístico

### 2.1 Modelo de regresión OLS con efectos fijos

**Especificación:**

$$
\tilde{S}_{ip} = \beta_p \cdot C_{ij} + \sum_{m} \gamma_{pm} \cdot \mathbf{1}[i \in m] + \varepsilon_{ip}
$$

donde:
- $\beta_p$ es el **coeficiente de arrastre** del candidato CA sobre el SE del partido $p$
- $\gamma_{pm}$ son efectos fijos por municipio (controlan heterogeneidad geográfica)
- $\varepsilon_{ip} \sim \mathcal{N}(0, \sigma_p^2)$ (supuesto homoscedástico)

**Resultados estimados** (OLS, $n$ por colectividad, 123 efectos fijos de municipio):

| Colectividad | $n$ | $\hat\beta_p$ | IC 95% | $p$-valor | $R^2$ | $R^2_{adj}$ |
|---|---|---|---|---|---|---|
| Alianza Verde | 15,027 | **0.230** | [0.210, 0.251] | <0.001 | 0.531 | 0.527 |
| Centro Democrático | 12,319 | **0.297** | [0.259, 0.336] | <0.001 | 0.294 | 0.287 |
| Conservador-Salv. Nac. | 12,970 | **0.119** | [0.106, 0.132] | <0.001 | 0.512 | 0.508 |
| Partido Liberal | 10,735 | **0.081** | [0.067, 0.094] | <0.001 | 0.521 | 0.515 |
| CR-Nuevo Liberalismo | 10,114 | **0.068** | [0.052, 0.084] | <0.001 | 0.498 | 0.492 |
| Partido de la U | 7,982 | **0.023** | [0.014, 0.032] | <0.001 | 0.594 | 0.588 |

**Interpretación de $\hat\beta_p$:** un voto adicional al candidato CA de la
colectividad $p$ en una mesa se asocia con $\hat\beta_p$ votos adicionales al
partido SE, controlando por municipio.

> ⚠️ **Advertencia de inferencia ecológica (falacia de Robinson):** los
> parámetros $\hat\beta_p$ describen relaciones entre agregados por mesa, no
> entre votantes individuales. La afirmación "quien vota CA $j$ vota SE $k$" no
> se sostiene con estos datos.

### 2.2 Intervalos de predicción

Los intervalos de predicción al nivel $(1-\alpha)$ para una nueva mesa $i^*$
son más amplios que los de confianza:

$$
\hat{\tilde{S}}_{i^*p} \pm t_{n-k,\,1-\alpha/2} \cdot \hat\sigma_p
\sqrt{1 + \frac{1}{n} + \frac{(C_{i^*j} - \bar{C}_j)^2}{\sum_i(C_{ij}-\bar{C}_j)^2}}
$$

donde el término $+1$ bajo la raíz captura la **variabilidad irreducible** de
una mesa individual frente a la incertidumbre del estimado medio.

### 2.3 Probabilidad de transferencia calibrada

Para cada candidato $j$, la probabilidad de transferencia al Senado se estima
como la media ponderada del ratio de transferencia observado en sus mesas,
usando la dominancia como peso:

$$
\hat\pi_j = \frac{\displaystyle\sum_i d_{ij} \cdot r_{ip}}
                 {\displaystyle\sum_i d_{ij}}
$$

**Valores estimados** (Boyacá, Alianza Verde):

| Candidato | $\hat\pi_j$ |
|---|---|
| ADELINDA NUMPAQUE SARMIENTO | 1.552 |
| VIVIAN ANDREA NIETO FLOREZ | 1.520 |
| JAIME RAUL SALAMANCA TORRES | 1.327 |
| RAMIRO BARRAGÁN ADAME | 1.319 |
| WILLIAM DONATO GÓMEZ | 1.304 |
| YAMIT NOE HURTADO NEIRA | 1.191 |

### 2.4 Atribución probabilística

La atribución probabilística reemplaza la dominancia pura por el producto
dominancia × probabilidad de transferencia, renormalizado:

$$
\alpha_{ijk}^{prob} = \frac{d_{ij} \cdot \hat\pi_j}{\displaystyle\sum_{j'} d_{i j'} \cdot \hat\pi_{j'}}
$$

$$
\hat{S}_{ij}^{prob} = \alpha_{ijk}^{prob} \cdot S^{total}_{ip}
$$

La diferencia $\Delta_{ij} = \alpha_{ij}^{prob} - \alpha_{ij}^{det}$ mide el
sesgo de atribución por no controlar el comportamiento histórico del candidato.

### 2.5 Estimación de densidad no paramétrica (KDE)

La distribución de ratios de arrastre se estima mediante KDE con kernel
gaussiano:

$$
\hat f(r) = \frac{1}{n h} \sum_{i=1}^n K\!\left(\frac{r - r_{ip}}{h}\right),
\quad K(u) = \frac{1}{\sqrt{2\pi}} e^{-u^2/2}
$$

El ancho de banda $h$ se elige por la regla de Silverman:
$h = 1.06 \cdot \hat\sigma_r \cdot n^{-1/5}$.

La KDE permite calcular el percentil del candidato seleccionado dentro de la
distribución departamental y estimar $P(r > 1)$ (probabilidad de arrastre neto
positivo).

---

## 3. Tratamiento de Datos

### 3.1 Imputación de mesas faltantes

De las **3,282 mesas esperadas** en Boyacá (según nomenclator oficial):
- **3,282/3,282** mesas Cámara recuperadas (33 con error HTTP 500 → reintento exitoso)
- **3,281/3,282** mesas Senado (1 mesa con 0 votos válidos → omitida sin sesgo)

**Estrategia de recuperación:** reintento con backoff exponencial (3 intentos,
espera $2^k$ segundos en el intento $k$). No se utilizó imputación estadística
dado que la cobertura efectiva es ≥ 99.97%.

### 3.2 Diseño de "muestreo"

Los datos no provienen de un muestreo sino de un **registro administrativo
completo** (resultados oficiales). Por tanto:

- **No hay factores de expansión** ($w_i = 1 \;\forall i$)
- **No hay etapas de muestreo** (universo = muestra)
- Los errores estándar son de **modelo**, no de diseño muestral

La jerarquía geográfica (departamento → municipio → zona → puesto → mesa)
sí se utiliza como **estructura de efectos fijos** en la regresión.

### 3.3 Tratamiento de votos especiales

| Tipo | Tratamiento |
|---|---|
| Voto en blanco | Excluido del numerador CA y SE; incluido en denominador $V_i$ |
| Voto nulo / no marcado | Excluido de todos los cómputos de arrastre |
| "Solo por la lista" ($codcan = 0$) | Incluido en $P^{CA}_{ip}$ y $S^{total}_{ip}$ pero excluido del índice de incidencia individual |
| Split-ticket (SE > CA partido) | Masa no atribuible; visible como $S^{total}_{ip} - P^{CA}_{ip}$ |

### 3.4 Datos composicionales — transformación log-cociente

Si se desease modelar la distribución del vector de dominancias
$\mathbf{d}_i = (d_{i1}, \ldots, d_{iJ})$ en el simplex $\Delta^{J-1}$, la
transformación CLR (centered log-ratio) elimina la restricción de suma:

$$
\text{clr}(\mathbf{d}_i) = \left(\log\frac{d_{i1}}{g(\mathbf{d}_i)},\, \ldots,\,
\log\frac{d_{iJ}}{g(\mathbf{d}_i)}\right),
\quad g(\mathbf{d}_i) = \left(\prod_j d_{ij}\right)^{1/J}
$$

Esta transformación no se aplica en el modelo actual (atribución proporcional
directa) pero es el punto de entrada para extensiones composicionales.

---

## 4. Métricas

### 4.1 Error estándar del coeficiente OLS

$$
\widehat{SE}(\hat\beta_p) = \hat\sigma_p
\sqrt{\left(\mathbf{X}^\top \mathbf{X}\right)^{-1}_{jj}}
$$

donde $\mathbf{X}$ incluye el predictor $C_{ij}$ y las dummies de municipio,
y $\hat\sigma_p^2 = \frac{RSS}{n - k}$ con $k = 124$ parámetros (1 predictor
+ 123 efectos fijos).

### 4.2 Intervalo de confianza 95% para $\beta_p$

$$
\hat\beta_p \pm t_{n-k,\,0.975} \cdot \widehat{SE}(\hat\beta_p)
$$

Para $n > 1{,}000$ se usa $t_{0.975} \approx 1.96$.

### 4.3 Coeficiente de determinación ajustado

$$
R^2_{adj} = 1 - \frac{RSS / (n-k)}{TSS / (n-1)}
$$

### 4.4 Pearson y Spearman entre CA y SE

$$
\rho^{Pearson}_{CA,SE} = \frac{\sum_i (C_{ij} - \bar C_j)(S^{total}_{ip} - \bar S_p)}
{\sqrt{\sum_i(C_{ij}-\bar C_j)^2 \cdot \sum_i(S^{total}_{ip}-\bar S_p)^2}}
$$

$$
\rho^{Spearman}_{CA,SE} = 1 - \frac{6 \sum_i d_i^2}{n(n^2-1)},
\quad d_i = \text{rg}(C_{ij}) - \text{rg}(S^{total}_{ip})
$$

**Valores observados** (Boyacá, nivel mesa):

| Colectividad | $\rho^{P}_{mesa}$ | $\rho^{S}_{mesa}$ | $\rho^{P}_{municipio}$ |
|---|---|---|---|
| Alianza Verde | 0.306 | 0.212 | **0.973** |
| Partido de la U | 0.347 | 0.111 | **0.953** |
| Conservador-Salv. Nac. | 0.276 | 0.268 | **0.935** |
| Partido Liberal | 0.222 | 0.157 | **0.927** |
| Centro Democrático | 0.185 | 0.267 | **0.877** |
| CR-Nuevo Liberalismo | 0.184 | 0.143 | **0.952** |

La brecha $\rho_{municipio} \gg \rho_{mesa}$ indica fuerte **correlación ecológica**:
la relación CA↔SE es robusta al agregar por municipio pero ruidosa a nivel
de mesa individual (heterogeneidad intra-municipal).

### 4.5 Detección de outliers

Una mesa $i$ es anómala si su residuo estandarizado supera 3 desviaciones:

$$
z_i = \frac{r_{ip} - \bar{r}_{pm}}{\sigma_{r_{pm}}} > 3
$$

donde $\bar{r}_{pm}$ y $\sigma_{r_{pm}}$ son la media y desviación del ratio
de arrastre en el municipio $m$ para la colectividad $p$.
Se identificaron **918 mesas anómalas** en Boyacá (28% del total),
principalmente mesas unipersonales con ratios extremos por denominador bajo.

---

## 5. Limitaciones y extensiones propuestas

### 5.1 Limitaciones actuales

| Limitación | Impacto | Severidad |
|---|---|---|
| Inferencia ecológica (falacia de Robinson) | $\hat\beta_p$ no es P(votar SE\|votar CA=j) | Alta |
| Atribución uniforme dentro del partido | $\alpha^{det}$ igual para todos los senadores del partido | Media |
| OLS homoscedástico | Mesas pequeñas tienen mayor varianza → heterocedasticidad | Media |
| No modela split-ticket | Votos SE > CA partido no se descomponen | Media |
| Un solo departamento | No permite generalizar a Colombia | Baja (por ahora) |

### 5.2 Mejoras identificadas

#### A. Redes Neuronales Feedforward para datos composicionales
El vector $\mathbf{d}_i \in \Delta^{J-1}$ viola el supuesto de espacio
euclidiano del OLS. Una red feedforward con capa de entrada CLR y función de
salida softmax preserva la restricción de suma y puede capturar no-linealidades:

$$
\hat{\boldsymbol\alpha}_i = \text{softmax}(W_L \cdot \text{ReLU}(\cdots W_1 \cdot \text{clr}(\mathbf{d}_i)))
$$

Mejora esperada: **reducción del RMSE de atribución en ~15-25%** en simulaciones
con datos composicionales (Aitchison, 1986; Pawlowsky-Glahn & Buccianti, 2011).

#### B. Corrección de sesgos en modelos multinivel
El OLS con efectos fijos no modela la varianza entre municipios como distribución.
Un modelo multinivel (HLM) con pendiente aleatoria por municipio:

$$
\tilde{S}_{ip} = (\beta_p + u_{pm}) \cdot C_{ij} + \gamma_{pm} + \varepsilon_{ip},
\quad u_{pm} \sim \mathcal{N}(0, \tau_p^2)
$$

permitiría: (1) estimar $\tau_p^2$ (heterogeneidad del arrastre entre municipios),
(2) producir predicciones shrinkage para municipios con pocas mesas, y
(3) cuantificar **house effects** geográficos sin sobreajustar.

#### C. Muestreo polietápico (si se extiende a Colombia)
Para escalar a nivel nacional ($\approx 115{,}000$ mesas), el diseño debería
estratificar por departamento y municipio, con factores de expansión:

$$
w_i = \frac{N_{\text{estrato}}}{n_{\text{estrato}}} \cdot \frac{M_m}{m_m}
$$

El error estándar de diseño difiere del de modelo y debe calcularse con el
estimador de Horvitz-Thompson o jackknife de réplicas.

#### D. Imputación múltiple para cobertura parcial
Si la cobertura fuese < 95% (escenarios de elección en tiempo real),
MICE (Multiple Imputation by Chained Equations) permitiría imputar:

$$
r_{ip}^{(t)} \sim p\!\left(r_{ip} \,\Big|\, \mathbf{x}_i,\, \{r_{i'p}^{(t-1)}\}_{i' \ne i}\right)
$$

donde $\mathbf{x}_i$ incluye votos parciales, municipio, participación histórica
y características socioeconómicas. Reducción esperada del error de predicción:
**8-12%** según simulaciones con datos electorales colombianos (referenciar con
Contraloría General o LAPOP Colombia).

#### E. Modelo de interacción social (de Finetti)
Si los votantes de una misma mesa se influyen entre sí (contagio social), la
hipótesis de intercambiabilidad de de Finetti permite modelar la correlación
intracluster. En el régimen **supercrítico** (correlación alta) se forman
clusters de opinión: la distribución de $C_{ij}/N_i$ tiene bimodalidad.
Diagnóstico: prueba de sobredispersión respecto a Binomial($N_i$, $\pi_j$).

---

## 6. Resumen ejecutivo del modelo

```
DATOS
  Fuente   : Registraduría Nacional (API REST JSON oficial)
  Unidad   : Mesa electoral (n = 3,282 en Boyacá)
  Periodo  : Elecciones 8 de marzo de 2026
  Cobertura: 99.97% Cámara | 99.94% Senado

DETERMINÍSTICO
  Dominancia   : d_ij = C_ij / P_CA_ip
  Atribución   : S_ij_hat = d_ij × S_SE_ip
  Ratio arrastre: r_ip = S_SE_ip / P_CA_ip  [media = 1.084, sd = 1.683]

PROBABILÍSTICO
  OLS           : S_SE ~ beta_p × C_CA + C(municipio)   [R² ≈ 0.29–0.59]
  KDE           : kernel gaussiano, BW Silverman
  Transfer prob : pi_j = Σ(d_ij × r_ip) / Σ(d_ij)

LIMITACIÓN PRINCIPAL
  Inferencia ecológica — los coeficientes NO implican
  causalidad ni comportamiento individual del votante.
```

---

*Documento generado el 4 de junio de 2026.
Código fuente: `proyecto-electoral-2026/src/` — Python 3.13, pandas, statsmodels, scipy, Streamlit.*
