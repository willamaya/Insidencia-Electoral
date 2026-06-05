# Análisis de Arrastre Cámara → Senado — Boyacá 2026

## Metodología
- **Unidad de análisis**: mesa electoral (3.282 mesas, Boyacá)
- **Modelo determinístico**: ratio_arrastre = votos_SE_colectividad / votos_CA_partido por mesa
- **Índice de incidencia ponderado**: Σ(votos_SE × dominancia_candidato) / Σ(votos_CA_candidato)
- **Homologación**: cada codpar de Cámara mapeado a su equivalente de Senado
- **⚠ Advertencia**: datos agregados por mesa, no inferencia individual (falacia ecológica)

## Cobertura
- Cámara: 3.282/3.282 mesas | 398 puestos | 123 municipios | 89.893 filas
- Senado: 3.281/3.282 mesas | 398 puestos | 123 municipios | 130.732 filas

## Top 5 candidatos por índice de incidencia ponderado
| Candidato | Colectividad | Votos CA | Votos SE asoc. | Índice pond. | Mesas |
|---|---|---|---|---|---|
| DELCI DEL CARMEN URBANO | CENTRO DEMOCRATICO | 6,296 | 71,809 | 1.545 | 2,372 |
| KENDA LUCIA CALDERA GARAVITO | CENTRO DEMOCRATICO | 1,890 | 37,926 | 1.464 | 1,139 |
| FREDY ALEXANDER GARZON ROJAS | CENTRO DEMOCRATICO | 3,088 | 50,122 | 1.445 | 1,622 |
| JHONATAN DAVID HOLGUIN CRUZ | CENTRO DEMOCRATICO | 3,102 | 50,310 | 1.439 | 1,643 |
| JICLY ESGARDO MUTIS ISAZA | CENTRO DEMOCRATICO | 15,046 | 79,451 | 1.234 | 2,682 |

## Correlaciones Pearson (votos CA candidato vs votos SE colectividad, por mesa)
| Colectividad | Pearson mesa | Pearson municipio |
|---|---|---|
| PARTIDO DE LA U | 0.3466 | 0.9529 |
| ALIANZA VERDE | 0.3058 | 0.9726 |
| CONSERVADOR-SALV NACIONAL | 0.2758 | 0.9345 |
| PARTIDO LIBERAL | 0.2217 | 0.9269 |
| CENTRO DEMOCRATICO | 0.1849 | 0.8772 |
| CR-NUEVO LIBERALISMO | 0.1842 | 0.952 |

## Regresión OLS (votos_SE ~ votos_CA + efectos_fijos_municipio)
| Colectividad | Coef. votos_CA | p-valor | R² | Significativo |
|---|---|---|---|---|
| ALIANZA VERDE | 0.2304 | 0.0000 | 0.5314 | ✓ |
| CENTRO DEMOCRATICO | 0.2974 | 0.0000 | 0.2939 | ✓ |
| CONSERVADOR-SALV NACIONAL | 0.1189 | 0.0000 | 0.5124 | ✓ |
| CR-NUEVO LIBERALISMO | 0.0678 | 0.0000 | 0.4977 | ✓ |
| PARTIDO DE LA U | 0.0231 | 0.0000 | 0.5940 | ✓ |
| PARTIDO LIBERAL | 0.0808 | 0.0000 | 0.5210 | ✓ |

> **Interpretación**: el coeficiente indica cuántos votos adicionales al Senado se asocian con 1 voto más a ese candidato a Cámara, controlando por municipio. **No implica causalidad individual.**