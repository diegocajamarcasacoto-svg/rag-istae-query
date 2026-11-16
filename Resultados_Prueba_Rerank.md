# Resultados de la prueba de reranking: con o sin `bge-reranker-base` (B y C)

**Fecha de ejecución:** 2026-11-13 · **Decisión aplicada:** 2026-11-15
**Resultado:** el servicio `/retrieve` usa **`rerank: false` por defecto**. Sin rerank, la recuperación es igual o mejor en todas las métricas de orden y de contexto, en B y en C, y unas **6 veces más rápida**. Ninguna diferencia en recall es estadísticamente significativa; la conclusión es «el rerank no aporta», no «el rerank empeora».
**Alcance:** las 50 preguntas del Gold Standard v4 (Q31–Q50 en borrador), sin LLM.

---

## 1. Motivo de la prueba

LB-1 incluía el reranking con un cross-encoder (`bge-reranker-base`) sobre los 20 candidatos de la búsqueda híbrida, como es habitual en la literatura (Nogueira & Cho, 2019). Había dos razones para dudar de su aporte en este corpus:

- **Indicio previo:** en los datos de la Prueba 5, medidos por fragmentos, el MRR era de 0,58 sin rerank y de 0,60 con rerank, una diferencia mínima.
- **Antecedente en el dominio legal:** Pipitone & Alami (2024) encontraron que un reranker de propósito general rendía peor que no reranquear.

Además, el rerank era la etapa más costosa del servicio en CPU: unos 3,7 s por consulta. La decisión quedó registrada como pendiente en el README («Decisión: rerank sí o no»), con este diseño y este criterio fijados **antes** de ejecutar la prueba.

## 2. Diseño

| | |
|---|---|
| Factor | Reranking con `bge-reranker-base`: sí (LB-1) / no (orden de la fusión RRF). |
| Fijo | E5 `multilingual-e5-base` + BM25 nativo de Qdrant, RRF k = 60, `prefetch_k` = `candidate_k` = 20, `top_k` = 5, contexto por artículos (4.000 tokens, artículos de hasta 1.500) y capa jerárquica de LB-1 en C. |
| Variantes | B con rerank (LB-1), B sin rerank, C con rerank (LB-1), C sin rerank. En C, el factor se aplica a las dos etapas: el Top-5 base y la selección de la capa jerárquica. |
| Preguntas | Las 50 del Gold Standard v4: 25 directas y 25 de jerarquía (tipos 1–5, 5 por tipo). |
| Ejecución | Motor residente `query.engine.RetrievalEngine` (el mismo pipeline que el servicio `/retrieve` y que `query_rag.py --expansion units`), en CPU, sin LLM. |
| Estadística | Comparación pareada por pregunta: victorias, derrotas y empates del rerank, y prueba de Wilcoxon de rangos con signo. |
| Criterio (acordado antes) | Quitar el rerank si no mejora; mantenerlo si mejora de forma consistente; si solo mejora una etapa (por ejemplo, la jerárquica), mantenerlo solo en esa etapa como variante explícita. |

### 2.1 Criterio de relevancia

Las fuentes esperadas de cada pregunta (Fuente 1 y Fuente 2 del Gold Standard) se traducen a unidades de `units.jsonl` (`DOC-artN`). Las secciones, indicadores y numerales se mapean a mano (9 preguntas, pendientes de validar). Las fuentes complementarias no cuentan.

### 2.2 Métricas

Se miden sobre los **artículos deduplicados** que recibe el LLM, en su orden de entrada: primero los artículos base, en el orden del rerank o de RRF, y al final los de la capa jerárquica.

- **Recall@k (k = 1, 3, 5):** fuentes esperadas con algún artículo entre los k primeros, dividido por el número de fuentes. La hipótesis de la tesis se formula sobre Recall@3 ≥ 0,80.
- **MRR:** promedio de 1 / posición del primer artículo esperado.
- **Contexto final:** recall de fuentes, recall de artículos y precisión de artículos (esperados presentes / artículos en el contexto).
- **Norma y artículo de mayor rango en el contexto:** solo en las preguntas de jerarquía.
- **Techo:** Recall@20 de los candidatos, que es igual con y sin rerank porque el rerank solo reordena los mismos 20.
- **Costo:** tokens de contexto y tiempo por consulta.
- Las mismas métricas sobre fragmentos (definición literal de la Guía del Gold Standard) se conservan como referencia.

Los totales son macropromedios: cada pregunta pesa lo mismo.

## 3. Resultados agregados (50 preguntas)

| Métrica | B con rerank | B sin rerank | C con rerank | C sin rerank |
|---|---|---|---|---|
| Recall@1 | 0,28 | **0,37** | 0,28 | **0,37** |
| Recall@3 (hipótesis ≥ 0,80) | 0,49 | **0,59** | 0,49 | **0,59** |
| Recall@5 | 0,68 | 0,69 | 0,69 | 0,69 |
| MRR | 0,548 | **0,645** | 0,554 | **0,651** |
| Recall fuentes (contexto) | 0,68 | 0,69 | 0,74 | 0,75 |
| Recall artículos (contexto) | 0,539 | 0,587 | 0,582 | 0,628 |
| Precisión artículos (contexto) | 0,239 | **0,272** | 0,175 | 0,197 |
| Norma de mayor rango en contexto (jerarquía) | 0,52 | 0,44 | 0,80 | **0,96** |
| Art. de mayor rango en contexto (jerarquía) | 0,40 | 0,32 | **0,56** | 0,52 |
| Artículos en contexto | 4,7 | 4,6 | 7,1 | 7,0 |
| Tokens de contexto | 1.731 | 1.658 | 3.378 | 3.264 |
| Tiempo por consulta (promedio / máximo) | 4,6 s / 6,4 s | **0,7 s / 1,0 s** | 9,2 s / 12,9 s | **1,5 s / 1,7 s** |
| Recall@20 candidatos (techo) | 0,78 | 0,78 | 0,78 | 0,78 |
| Ref.: Recall@3 por fragmentos | 0,49 | 0,56 | 0,49 | 0,56 |
| Ref.: MRR por fragmentos | 0,552 | 0,638 | 0,552 | 0,638 |

El rerank tarda en promedio 3,8 s en B y 3,6 s en la etapa base de C.

### 3.1 Comparación pareada (Wilcoxon; victorias / derrotas / empates del rerank)

| Métrica | B: sin → con | B: V / D / E | B: p | C: sin → con | C: V / D / E | C: p |
|---|---|---|---|---|---|---|
| Recall@1 | 0,37 → 0,28 | 6 / 11 / 33 | 0,116 | 0,37 → 0,28 | 6 / 11 / 33 | 0,116 |
| Recall@3 | 0,59 → 0,49 | 4 / 11 / 35 | 0,072 | 0,59 → 0,49 | 4 / 11 / 35 | 0,072 |
| Recall@5 | 0,69 → 0,68 | 3 / 4 / 43 | 0,792 | 0,69 → 0,69 | 4 / 4 / 42 | 1,000 |
| RR | 0,645 → 0,548 | 8 / 18 / 24 | 0,097 | 0,651 → 0,554 | 9 / 18 / 23 | 0,091 |
| Recall fuentes (contexto) | 0,69 → 0,68 | 3 / 4 / 43 | 0,792 | 0,75 → 0,74 | 6 / 6 / 38 | 0,796 |
| Recall artículos (contexto) | 0,587 → 0,539 | 3 / 10 / 37 | 0,169 | 0,628 → 0,582 | 6 / 12 / 32 | 0,292 |
| Precisión artículos (contexto) | 0,272 → 0,239 | 6 / 16 / 28 | **0,037** | 0,197 → 0,175 | 12 / 23 / 15 | 0,070 |

## 4. Resultados por estrato y por tipo

| Recall@3 / Recall fuentes (contexto) | B con | B sin | C con | C sin |
|---|---|---|---|---|
| Directas (25) | 0,62 / 0,82 | **0,76 / 0,88** | 0,62 / 0,82 | **0,76 / 0,88** |
| Jerarquía (25) | 0,36 / 0,54 | 0,42 / 0,50 | 0,36 / **0,66** | 0,42 / 0,62 |

- **Directas:** quitar el rerank es lo que más ayuda. Recall@3 sube 14 puntos y llegan al contexto más fuentes y más artículos esperados (recall de artículos: 0,70 → 0,80).
- **Jerarquía:** el rerank gana en Recall@5 (0,54–0,56 frente a 0,50) y en el recall de fuentes de C (0,66 frente a 0,62), pero pierde en Recall@3 y MRR (en C, 0,52 frente a 0,63).

Por tipo de jerarquía (5 preguntas cada uno; orientativo):

| Tipo | Lo más relevante |
|---|---|
| 1. Fundamento | El más débil en todas las variantes (Recall@3 0,3; techo 0,5). En C, el rerank trae una fuente más al contexto (0,5 frente a 0,4). |
| 2. Remisión explícita | Recall@3 0,3 en todas. C sin rerank lleva la norma superior en el 100 % de los casos (con rerank, en el 60 %), pero el artículo casi nunca. |
| 3. Complementariedad | Iguales en el orden. En C, la norma superior llega en el 100 % de los casos en ambas variantes. |
| 4. Prevalencia o conflicto | Sin rerank, MRR 0,90 frente a 0,71. El rerank gana en Recall@5 (0,7 frente a 0,5). |
| 5. Jerarquía interna | Sin rerank, mejor orden (MRR de C: 0,66 frente a 0,47). Con rerank, algo más de recall en el contexto. |

## 5. Resultados por pregunta

En las dos arquitecturas, el **Recall@3** cambia en las mismas preguntas:
- **el rerank gana** en Q07, Q10, Q35 y Q38;
- **el rerank pierde** en Q04, Q06, Q11, Q14, Q21, Q25, Q28, Q34, Q45, Q47 y Q50.

**Recall de fuentes en el contexto de C:**
- **el rerank gana** en Q27, Q33, Q35, Q38, Q40 y Q43;
- **el rerank pierde** en Q04, Q06, Q25, Q32, Q44 y Q45.

Un caso ilustrativo es **Q27**, las prácticas de hace 7 años, con el conflicto entre los 10 años del RRA y los 5 de la norma interna. Con rerank, C trae `EXT02-art43` (el artículo de mayor rango del conflicto); sin rerank, no. La decisión es estadística, no pregunta por pregunta: en el promedio de las 50 preguntas pesan más las pérdidas que las ganancias.

El detalle por pregunta, con la lista ordenada de artículos del contexto, está en `logs/recall_retrieve/Pruebas_Recall_Retrieve_LB1.xlsx` (una hoja por variante).

## 6. Análisis

1. **El rerank empeora el orden más de lo que mejora.** En Recall@3 pierde en 11 preguntas y gana en 4, y en RR pierde en 18 y gana en 8–9. El patrón es igual en B y en C, porque la etapa base es la misma.
2. **El contexto final casi no cambia.** Las dos variantes eligen sus 5 artículos base entre los mismos 20 candidatos, y el recall de fuentes en el contexto difiere en 0,01. El rerank cambia sobre todo **en qué posición** entra lo pertinente, no **si** entra.
3. **El límite está antes del rerank.** El techo de candidatos (0,78; 0,68 en jerarquía) es igual en las cuatro variantes. Ningún reordenamiento recupera lo que la búsqueda híbrida no trajo. El diagnóstico posterior (`Diagnostico_Fuentes_Perdidas.md`, rama `mejoras-recuperacion`) atribuye esas pérdidas a la representación: lenguaje coloquial frente a normativo, fragmentos sin el nombre de la norma y tablas de EXT09.
4. **El rerank aporta algo solo en la parte jerárquica:** el artículo de mayor rango en C (0,56 frente a 0,52) y Recall@5 en el estrato de jerarquía. Es coherente con un reranker de propósito general que reconoce mejor un artículo cuando ya está restringido a la norma superior.
5. **El costo es grande y seguro; el beneficio, nulo o negativo.** Con rerank, B tarda 6,6 veces más y C 6,3 veces más. En CPU, C pasa de 1,5 s a 9,2 s por consulta, con picos de 12,9 s.
6. **Por qué un cross-encoder general puede no ayudar aquí:** `bge-reranker-base` no está ajustado al dominio jurídico ni al español normativo. Además, la fusión RRF ya combina dos señales complementarias (léxica y semántica). Es una explicación posible, no verificada en esta prueba.

## 7. Decisión

- **Etapa base (B y C): sin rerank.** No mejora en ninguna métrica, empeora el orden y cuesta unas 6 veces más tiempo. Se aplicó en el servicio `/retrieve` (`rerank: false` por defecto, commit `4ecff2b` de `rag-istae-query`) y es lo que está publicado en Hugging Face Spaces. El request todavía puede pedir `rerank: true`.
- **Etapa jerárquica de C: decisión abierta.** Hay un indicio a favor (artículo de mayor rango 0,56 frente a 0,52). Siguiendo el criterio acordado, falta probar la variante **rerank solo en la capa jerárquica** antes de cerrar esa parte.
- **Reproducibilidad de LB-1:** `query_rag.py` conserva `--rerank` por defecto, para reproducir la línea base tal como se definió.

## 8. Limitaciones y condiciones para reabrir la decisión

- **Tamaño de la muestra:** con 50 preguntas, las diferencias de Recall@3 (p = 0,072) y RR (p ≈ 0,09) no alcanzan significación. Los resultados por tipo (5 preguntas) son orientativos.
- **Gold Standard:** Q31–Q50 están en borrador y hay 9 mapeos manuales sin validar (secciones e indicadores de EXT09, numeración repetida en INT20 y artículos fusionados en EXT02).
- **Se mide recuperación, no respuesta:** un artículo en el contexto no garantiza una buena respuesta del LLM, y su ausencia no implica una mala. La prueba de punta a punta queda pendiente.
- **Un solo reranker:** se evaluó `bge-reranker-base`, el liviano de LB-1. `bge-reranker-v2-m3` se conserva en `storage/models` para una comparación futura.

**La decisión se reabre si** cambia el reranker (otro modelo, o uno ajustado al dominio), si la variante con rerank solo en la etapa jerárquica muestra una mejora consistente, si una mejora de la búsqueda de candidatos (colección v2) cambia el techo, o si la evaluación de punta a punta con el LLM muestra respuestas peores sin rerank.

## Reproducción

```powershell
cd <ruta>\rag-istae-query
.\.venv\Scripts\Activate.ps1
python .\scripts\eval_recall_retrieve.py run          # 4 variantes × 50 preguntas, sin LLM (solo lee Qdrant)
python .\scripts\eval_recall_retrieve.py recompute    # recalcula las métricas desde resultados.jsonl
```

No hay que ejecutarlo con el servicio `/retrieve` local activo, porque Qdrant local es de un solo proceso. Salidas (fuera de git), en `logs/recall_retrieve/`:
- `Pruebas_Recall_Retrieve_LB1.xlsx`: hojas Material, una por variante, Resumen y Metodología;
- `resultados.jsonl`;
- `gold_v4_unidades.json`, el mapeo del Gold Standard a unidades;
- `Interpretacion_Recall_Retrieve_C.docx`, la interpretación de columnas y resultados de C.

## Referencias

- Nogueira, R. y Cho, K. (2019). *Passage Re-ranking with BERT*. arXiv:1901.04085.
- Pipitone, N. y Alami, G. H. (2024). *LegalBench-RAG: A Benchmark for Retrieval-Augmented Generation in the Legal Domain*. arXiv:2408.10343.
- Cormack, G. V., Clarke, C. L. A. y Büttcher, S. (2009). *Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods*. SIGIR '09.
