# Resultados de la Prueba 5: profundidad de candidatos antes del reranking (20 frente a 50)

**Fecha de ejecución:** 2026-11-07
**Resultado:** se ratifica `candidate_k` = `prefetch_k` = **20** en la línea base LB-1. La variante 50 se descarta.
**Alcance:** muestra de 10 preguntas del Gold Standard (ver «Limitaciones»).

Documento de respaldo para el Trabajo de Fin de Máster. Complementa la sección 2.5 de [`Justificacion_Linea_Base.md`](Justificacion_Linea_Base.md) y la Prueba 5 de [`Guia_Pruebas_Recuperación.md`](Guia_Pruebas_Recuperación.md).

---

## 1. Motivo de la prueba

La cantidad de candidatos que se reranquean era el parámetro de LB-1 con **menos respaldo bibliográfico**. La literatura reranquea grupos mucho mayores: Soriano et al. (2026) recuperan 1.000 candidatos por consulta, y Nogueira & Cho (2019) reranquean los 1.000 primeros de BM25. El valor 20 se había fijado por el costo del cross-encoder en CPU, y la variante 50 quedó registrada como hipótesis.

La prueba responde dos preguntas:
- **5a:** ¿20 candidatos dejan fuera fragmentos relevantes que 50 sí alcanzan?
- **5b:** ¿reranquear 50 en lugar de 20 mejora el top 5 final, y a qué costo de latencia?

## 2. Diseño

| Elemento | Valor |
|---|---|
| Factor variado | `candidate_k` = `prefetch_k`: 20 (LB-1) frente a 50 |
| Constantes | `top_k` 5; E5 `multilingual-e5-base` + BM25 nativo de Qdrant; RRF k = 60; `bge-reranker-base` (lotes de 4); colección `istae_chunks_e5_base` (9.623 puntos) |
| Etapa evaluada | recuperación híbrida → reranking → top 5 de fragmentos. Es la etapa común a B y C: **no incluye** la expansión a artículos ni la capa jerárquica de C |
| Muestra | las 5 últimas preguntas (Q26–Q30) y 5 sorteadas de Q01–Q25 con `random.Random(20261107)`: Q02, Q13, Q15, Q19, Q22 |
| Generación (LLM) | no se usa |
| Equipo | inferencia en CPU (Windows 11, Python 3.11), sin otros procesos usando Qdrant |
| Script | [`scripts/eval_candidate_depth.py`](scripts/eval_candidate_depth.py) |

Comando ejecutado (desde la raíz de `rag-istae-query`, con `$env:PYTHONPATH = "src;scripts"`):

```powershell
python .\scripts\eval_candidate_depth.py --questions Q02,Q13,Q15,Q19,Q22,Q26,Q27,Q28,Q29,Q30
```

Salidas: `logs/prueba5/prueba5_20261107-215458.{csv,txt,json}`. `logs/` está excluido de git; los datos completos se transcriben en las secciones 3 y 4.

### 2.1 Criterio de relevancia

El Gold Standard anota la **norma y el artículo** esperados («INT08 – … | Arts. 12 y 13»), no `chunk_id`. Por eso:

- Las columnas «Fuente esperada» y «Artículo/sección» se convierten en unidades de `units.jsonl`. Por ejemplo, «INT08, Arts. 12 y 13» da `INT08-art12` y `INT08-art13`; los literales («Art. 5, literal b») se reducen al artículo.
- Las fuentes que no son artículos se asignaron a mano. Son los indicadores del modelo CACES (EXT09):
  - Q29, indicador 1.1.3 → `EXT09-sec37`;
  - Q30, indicador 3.2.3 → `EXT09-sec116`.

  Se descartó `EXT09-sec02`, que es el índice del documento y también menciona el 3.2.3.
- Un fragmento recuperado es **relevante** si pertenece a una unidad esperada.

### 2.2 Métricas

Todas se calculan por pregunta sobre el top 5 después del reranking:

| Métrica | Definición |
|---|---|
| P@5 | fragmentos relevantes del top 5 / 5 |
| R@5 por unidad | unidades esperadas presentes en el top 5 / unidades esperadas |
| MRR | 1 / posición del primer fragmento relevante (0 si no hay ninguno) |
| nDCG@5 | ganancia binaria por fragmento; el ideal es min(5, n.° de fragmentos de las unidades esperadas) |

Además, sin reranking:
- la posición del primer fragmento relevante entre los 50 candidatos (5a);
- cuántas unidades esperadas aparecen entre los candidatos.

Prueba estadística: Wilcoxon de rangos con signo pareado por pregunta (`scipy.stats.wilcoxon`, `zero_method="wilcox"`), como Soriano et al. (2026).

---

## 3. Resultados agregados

| Métrica (media de 10 preguntas) | 20 candidatos (LB-1) | 50 candidatos | Wilcoxon |
|---|---|---|---|
| P@5 | **0,260** | 0,200 | W = 0, p = 0,500 (2 pares distintos) |
| R@5 por unidad | **0,625** | 0,500 | W = 0, p = 0,500 (2) |
| MRR | **0,600** | 0,508 | W = 0, p = 0,250 (3) |
| nDCG@5 | **0,338** | 0,266 | W = 0, p = 0,062 (5) |
| Unidades esperadas entre los candidatos | 1,3 | 1,5 | — |
| Fragmentos relevantes entre los candidatos | 1,9 | 2,3 | — |
| Tiempo de la búsqueda híbrida | 0,72 s | 0,65 s | — |
| Tiempo de reranking | **4,4 s** | 13,8 s | — |

Carga de modelos, índice de unidades y Qdrant: 16,6 s, una vez por proceso.

**Comparación por pregunta (nDCG@5):**
- 50 mejora en **0** preguntas;
- empeora en **5**: Q13, Q22, Q26, Q27 y Q28;
- no cambia en 5.

W = 0 en todas las métricas significa que **ninguna diferencia favorece a 50**.

**5a. Primer fragmento relevante entre los 50 candidatos, sin reranking:**
- posición 1–20, ya alcanzado por LB-1: **8** preguntas;
- posición 21–50, solo lo alcanza la variante: **2** (Q29 en la 41 y Q30 en la 36);
- ausente: 0.

---

## 4. Resultados por pregunta

Los pares se leen «20 / 50».

| ID | Requiere jerarquía | Unidades esperadas | 1.er relevante entre los candidatos | P@5 | R@5 por unidad | MRR | nDCG@5 | Reranking (s) |
|---|---|---|---|---|---|---|---|---|
| Q02 | No | `INT08-art19`, `INT08-art20` | 1 / 1 | 0,20 / 0,20 | 0,50 / 0,50 | 1,00 / 1,00 | 0,61 / 0,61 | 3,8 / 9,3 |
| Q13 | No | `EXT02-art64` | 3 / 3 | 0,40 / **0,00** | 1,00 / **0,00** | 0,33 / **0,00** | 0,42 / **0,00** | 5,7 / 10,4 |
| Q15 | No | `EXT02-art74`, `EXT01-art84` | 2 / 2 | 0,20 / 0,20 | 0,50 / 0,50 | 1,00 / 1,00 | 0,34 / 0,34 | 3,2 / 10,8 |
| Q19 | No | `EXT01-art87` | 2 / 2 | 0,20 / 0,20 | 1,00 / 1,00 | 0,33 / 0,33 | 0,23 / 0,23 | 3,0 / 9,0 |
| Q22 | No | `EXT01-art86` | 2 / 2 | 0,20 / 0,20 | 1,00 / 1,00 | 1,00 / **0,50** | 0,34 / **0,21** | 4,8 / 31,4 |
| Q26 | No | `EXT02-art67`, `INT07-art48`, `-art49`, `-art50` | 1 / 1 | 0,60 / **0,40** | 0,75 / **0,50** | 1,00 / 1,00 | 0,64 / **0,51** | 4,7 / 12,8 |
| Q27 | Sí | `EXT02-art43`, `INT15-art22` | 1 / 1 | 0,40 / 0,40 | 1,00 / 1,00 | 1,00 / 1,00 | 0,49 / **0,47** | 3,8 / 16,7 |
| Q28 | Sí | `EXT01-art207`, `INT23-art4`, `-art22`, `-art25` | 1 / 1 | 0,40 / 0,40 | 0,50 / 0,50 | 0,33 / **0,25** | 0,32 / **0,28** | 4,3 / 10,9 |
| Q29 | No | `EXT09-sec37`, `INT06-art19`, `INT06-art20` | — / 41 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 5,7 / 12,5 |
| Q30 | No | `EXT09-sec116`, `INT18-art13`, `INT18-art14` | — / 36 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 4,8 / 14,1 |

Top 5 de los casos que explican el resultado (unidad del fragmento y puntaje del reranker):

| ID | 20 candidatos | 50 candidatos |
|---|---|---|
| Q13 | EXT13-art269 (0,918) · INT27-art20 (0,911) · **EXT02-art64** (0,826) · INT07-art36 (0,584) · **EXT02-art64** (0,530) | INT27-art15 (0,998) · INT07-art36 (0,984) · EXT13-art269 (0,918) · INT27-art20 (0,911) · EXT13-art156 (0,852) |
| Q22 | **EXT01-art86** (0,995) · EXT13-art111 (0,750) · INT01-art57 · INT01-art55 · INT02-sec16 | INT13-art16 (0,999) · **EXT01-art86** (0,995) · EXT01-art207 (0,993) · INT12-preambulo · INT01-art56 |
| Q29 | INT02-sec13 (0,939) · INT02-sec17 · EXT09-sec05 · EXT01-art96 · EXT09-sec40 | INT02-sec13 (0,939) · INT02-sec12 (0,935) · INT12-art1 · INT02-sec17 · EXT09-sec05 |
| Q30 | INT02-sec17 (0,947) · INT02-sec13 · INT02-sec17 · INT02-sec12 · INT02-sec17 | igual en los 4 primeros · EXT13-disp-general-tercera |

---

## 5. Análisis

1. **Más candidatos no mejoran el resultado final; lo empeoran.** Los 30 candidatos adicionales casi nunca son relevantes: en promedio aportan 0,4 fragmentos relevantes. En cambio, sí aportan artículos parecidos a la pregunta que el cross-encoder puntúa por encima de los correctos, a menudo con 0,99 o más.
   - En Q13, el artículo correcto (`EXT02-art64`) sale del top 5.
   - En Q22 baja del primer al segundo puesto.

   Es el riesgo descrito en la sección 8 de la guía y coincide con Pipitone & Alami (2024): un reranker general sin ajuste al dominio legal puede rendir peor que no reranquear.
2. **Los dos rescates de 5a no llegan a la respuesta.** En Q29 y Q30 el fragmento relevante aparece recién en las posiciones 41 y 36, y el reranker no lo sube al top 5. Un fragmento que la fusión RRF coloca tan abajo tampoco se beneficia del reranker. Ampliar la profundidad no resuelve esas dos preguntas.
3. **El costo es alto:** el reranking tarda unas 3 veces más (4,4 s frente a 13,8 s en promedio, con un máximo de 31 s en Q22). Para un servicio en CPU, esa latencia sería determinante aun si hubiera una mejora pequeña.
4. **Q29 y Q30 fallan con ambas configuraciones.** Las secciones de `INT02` acaparan el top 5, y no aparecen ni el indicador del modelo CACES ni los artículos de INT06 o INT18. Es un error de la recuperación base, no de la profundidad. Queda registrado como error observado de LB-1 para el protocolo de variantes. No se prueba ninguna variante por ahora.

## 6. Decisión

- **`candidate_k` = `prefetch_k` = 20 se ratifica en LB-1.** Ya no es solo una restricción de costo: en la muestra evaluada, 20 obtiene métricas iguales o mejores que 50 en todas las preguntas, con un tercio del tiempo de reranking.
- La hipótesis «50 candidatos antes del rerank» se **retira** del registro de variantes. No se crea una LB-2.
- `tests/test_baseline.py` sigue fijando el valor 20, sin cambios en el código.

## 7. Limitaciones y condiciones para reabrir la decisión

- **Muestra de 10 preguntas.** Con solo 2 a 5 pares distintos, el Wilcoxon no alcanza significancia: el menor p fue 0,062, en nDCG@5. La decisión se apoya en la **dirección consistente** de los resultados (ninguna pregunta favorece a 50) y en el costo, no en un resultado significativo.
- **Relevancia por artículo, no por fragmento.** El Gold Standard no anota `chunk_id_esperado`. Cualquier fragmento del artículo esperado cuenta como relevante, aunque no contenga el pasaje exacto que responde. Con ese criterio, P@5 y nDCG@5 pueden ser algo optimistas para ambas configuraciones por igual, lo que no afecta la comparación.
- **Asignación manual de los indicadores CACES** en Q29 y Q30 (sección 2.1).
- **Una sola ejecución.** Los tiempos dependen de este equipo y de la carga del momento; el valor de 31 s en Q22 parece un valor atípico. La recuperación y el reranking son deterministas, así que las métricas de calidad no varían entre ejecuciones.
- **Solo la etapa base.** No se evaluó la búsqueda de la capa jerárquica de C, que tiene su propio número de candidatos (`--hierarchy-candidate-k` 20).

**La decisión se reabre si** la ejecución sobre las 30 preguntas (mismo comando, sin `--questions`) muestra preguntas en las que 50 mejore de forma consistente, o si se cambia el reranker (por ejemplo, a `bge-reranker-v2-m3` o a un modelo ajustado al dominio).

## Referencias

- Nogueira, R. & Cho, K. (2019). *Passage Re-ranking with BERT*. arXiv:1901.04085.
- Pipitone, N. & Alami, G. H. (2024). *LegalBench-RAG: A Benchmark for Retrieval-Augmented Generation in the Legal Domain*. arXiv:2408.10343.
- Soriano et al. (2026). Evaluación de recuperación sobre el benchmark JUÁ (actos normativos brasileños), STIL. Detalle en `Justificacion_Linea_Base.md`, sección 5.
