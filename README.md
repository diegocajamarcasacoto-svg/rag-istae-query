# RAG ISTAE Query

Pipeline de consulta **de solo lectura** del sistema RAG sobre el corpus normativo del ISTAE: E5 (`query:`) + BM25 nativo de Qdrant sobre `payload.text` -> RRF (`k=60`) -> reranker BGE (CrossEncoder, opcional) -> Context Builder -> generación con Gemini u OpenRouter. No crea colecciones, índices ni puntos.

La colección `istae_chunks_e5_base` la genera `rag_istae_ingestion`; los modelos y la persistencia Qdrant se administran en `rag-istae-resources`.

## Relación con `rag-istae-api`

`rag-istae-api` es el backend FastAPI del chat. Es un repositorio externo y, para este proyecto, de solo lectura. Lo consume la interfaz `rag-istae-web` (Next.js 16, también de solo lectura). La interfaz muestra un selector de arquitectura que lista las de `GET /architectures`, con las no disponibles como "próximamente". También muestra citas numeradas que abren el artículo completo (`GET /units/{id}`) y un panel de fuentes con su estado de verificación. Sus tipos (`src/lib/types.ts`) reflejan el contrato `Answer`, así que una arquitectura B que lo cumpla aparece en el chat sin cambios en el frontend. Implementa el diseño 2×2 de la tesis (`app/architectures/base.py`):

| ID | Arquitectura | Estado en la API |
|---|---|---|
| `A` | Contexto largo: las 46 normas completas (unos 830.000 tokens) en cada consulta | Implementada |
| `AJ` | Contexto largo + jerarquía normativa (rango y art. 425 de la Constitución) | Implementada |
| `B` | RAG híbrido (semántica + palabras clave + reordenamiento) | Pendiente: corresponde al pipeline de este repositorio |
| `C` | RAG jerárquico | Pendiente |

Puntos a tener en cuenta para integrar o comparar:
- **Contrato:** una arquitectura nueva implementa `ChatArchitecture.stream(question, history)`. Entrega fragmentos de texto y, al final, un `Answer` con `body`, `sources` verificadas, tokens (`input`, `cached`, `output`, `reasoning`), `finish_reason`, `latency_ms`, `model` y `corpus_version`. Se registra en `main.build_architectures`; el endpoint `/chat` y el frontend no cambian.
- **Mismo modelo:** la API usa `nvidia/nemotron-3-ultra-550b-a55b:free` vía OpenRouter, con `reasoning_effort=medium` y `max_output_tokens=8192`. Para comparar A/AJ con B en igualdad de condiciones, use `config/generation.openrouter.json` con el mismo modelo y un esfuerzo de razonamiento equivalente (`"extra_body": {"reasoning": {"effort": "medium"}}`). En la evaluación de la tesis, la API debe correr con `LLM_FALLBACK_MODELS=[]`.
- **Datos distintos:** la API no usa Qdrant ni chunks. Lee su copia de `document_mapping.jsonl` y `extracted_text/` (idénticos a `storage/outputs`) y `catalogo_normativo.json`. Los `document_id` (`EXT##`/`INT##`) coinciden con el `document_id` del payload Qdrant, de modo que las fuentes de ambas arquitecturas son comparables por documento.
- **Registro para la evaluación:** `logs/runs_index.*` de este repositorio registra `input_tokens`, `output_tokens`, `finish_reason` y tiempos con nombres alineados a los campos de `Answer`.

## Arquitectura

```text
pregunta
   │  retrieval.encode_query (E5, prefijo "query:", normalizado)
   ▼
retrieval.hybrid_search ── Prefetch vectorial + Prefetch TextQuery(text) ── RRF k=60 ──► candidate_k candidatos
   │
   ▼
reranking.rerank (CrossEncoder BGE, CPU)            ← se omite con --no-rerank
   │
   ▼
context.build_context (semillas + vecinos, --expansion neighbors)   ó   units.build_unit_context (artículo completo, --expansion units)
   │
   ▼  solo --retrieval-mode hierarchical (arquitectura C)
hierarchy.expand_with_hierarchy (catálogo → normas superiores → búsqueda filtrada por document_id → +artículos)
   │
   ▼  se omite con --no-generation
generation.generate_answer (System Prompt externo + Gemini vía google-genai)
```

| Módulo | Responsabilidad |
|---|---|
| `src/query/config.py` | Carga `config/local.json`, valida claves obligatorias y resuelve rutas absolutas. |
| `src/query/retrieval.py` | Carga E5, abre el cliente Qdrant y ejecuta búsqueda vectorial, BM25 e híbrida con RRF. |
| `src/query/reranking.py` | Carga el CrossEncoder local y reordena los candidatos por score. |
| `src/query/units.py` | Expansión por artículo (*small-to-big*): índice `chunk_id → unidad` desde `units.jsonl` y construcción del contexto con artículos completos. |
| `src/query/baseline.py` | Parámetros de la línea base (`LB1`), compartidos por el CLI y el servicio. |
| `src/query/pipeline.py` | Pipeline B/C por artículos (híbrida → rerank opcional → artículos → jerarquía opcional → presentación), compartido por `query_rag.py` y el servicio. |
| `src/query/engine.py` | Motor residente: carga los recursos una vez y ejecuta el pipeline con la línea base. |
| `src/query/service/` | Servicio HTTP `POST /retrieve` (FastAPI): configuración por variables de entorno, contrato y caché. |
| `src/query/hierarchy/` | Recuperación jerárquica (arquitectura C): `catalog.py` carga el catálogo normativo (rangos y relaciones); `expansion.py` calcula las normas superiores relacionadas y agrega sus artículos. |
| `src/query/context/` | Context Builder: detector y validador de estructura (artículo, capítulo, sección, listas, notas), expansión de vecinos, fusión, deduplicación y presupuesto de tokens. |
| `src/query/generation/` | Configuración, construcción del prompt, adaptador de proveedor (`providers/gemini.py`), normalización de respuestas y errores explícitos (`GenerationConfigurationError`, `GenerationProviderError`). |
| `scripts/` | Ejecutables de línea de comandos de cada etapa y del pipeline completo. |

Modelo de reranking actual: `bge-reranker-base` (más liviano que `bge-reranker-v2-m3`, que se utilizó en las primeras pruebas). El modelo se toma de `reranker_model_path` o de `--reranker-model`.

## Instalacion

1. Clone este repositorio y cree un entorno virtual Python 3.11.
2. Instale `pip install -r requirements.txt` (`sentence-transformers`, `qdrant-client`, `torch`, `google-genai`, `httpx`, y `fastapi`/`uvicorn` para el servicio).
3. Prepare y valide `rag-istae-resources`.
4. Copie `config/local.example.json` a `config/local.json` y asigne rutas absolutas al E5, BGE y Qdrant compartidos.
5. Para generar respuestas, copie `config/generation.example.json` a `config/generation.json` y defina la variable de entorno `GEMINI_API_KEY`.

Todos los scripts usan `config/local.json` por defecto y aceptan `--config` para indicar otra ruta. Con los recursos ya disponibles, la recuperación funciona offline; los scripts nunca descargan modelos. Solo la generación requiere conexión. Ejemplos:

```powershell
python .\scripts\search_vectors.py "Art. 357 Constitucion" 
python .\scripts\search_qdrant_bm25.py "Art. 357 Constitucion"
python .\scripts\search_hybrid.py "Art. 357 Constitucion" --rrf-k 60
python .\scripts\search_reranked.py "Art. 357 Constitucion" --candidate-k 20 --prefetch-k 20 --rrf-k 60
python .\scripts\inspect_qdrant.py --chunk-id INT08:00055 --neighbors 1
```

No ejecute Query mientras Ingestion traslada o indexa la persistencia Qdrant local.

## Context Builder

`query.context.build_context()` recibe la consulta, la lista `(point, score)` producida por `rerank()`, el cliente Qdrant ya abierto y un diccionario de configuración. No carga modelos, no ejecuta retrieval y no abre ni modifica colecciones. Devuelve un `ContextResult` con el texto consolidado, chunks tipados y metadata de trazabilidad y rendimiento.

```python
from pathlib import Path

from query.config import load_config
from query.context import build_context
from query.reranking import load_reranker, rerank
from query.retrieval import (
    encode_query,
    hybrid_search,
    load_e5,
    open_client,
)

query = "¿Qué requisitos establece el artículo 18?"
config = load_config(Path("config/local.json"))
e5 = load_e5(config)
reranker = load_reranker(config["reranker_model_path"])
client = open_client(config)
try:
    vector = encode_query(e5, query, config)
    candidates = hybrid_search(client, config, vector, query, 20, 20, 60)
    ranked_results = rerank(reranker, query, candidates, batch_size=4)
    result = build_context(query, ranked_results, client, config)
finally:
    client.close()

print(result.context)
print(result.metadata["provenance"])
```

La expansión consulta solamente el mismo `document_id` y el intervalo de `chunk_index` alrededor de cada semilla; usa `scroll()` con filtro y no carga el corpus. Las tres primeras posiciones del reranker son las semillas predeterminadas. Un vecino debe pertenecer al mismo documento y ser consecutivo. Si el detector identifica un artículo, capítulo o sección diferente, el vecino se descarta; las estructuras desconocidas consecutivas se permiten y quedan etiquetadas como `unknown`. Listas y literales consecutivos pueden incorporarse. Notas de reforma y concordancias son condicionales: la consulta debe justificar su inclusión.

El detector combina `text`, `headings` e `item_types`, sin tratar las páginas o los encabezados como identificadores definitivos. En los payloads inspeccionados, `page_numbers` y `headings` suelen estar vacíos en DOCX; ambos campos se conservan vacíos cuando faltan. Los identificadores estructurales son heurísticos y pueden permanecer desconocidos.

Las opciones del builder se pasan en `config["context_builder"]`:

| Opción | Predeterminado | Descripción |
|---|---:|---|
| `seed_count` | `3` | Semillas máximas, en orden de reranking. |
| `neighbors_before` / `neighbors_after` | `1` / `1` | Vecinos a consultar por semilla. |
| `require_same_document` | `True` | Rechaza vecinos de otro documento o sin documento. |
| `require_consecutive` | `True` | Exige diferencia de un índice para cada vecino. |
| `max_list_continuation_chunks` | `20` | Límite por semilla para seguir listas o numerales fragmentados, consultando vecinos acotados del mismo documento. |
| `complementary_query_terms` | términos de reforma, concordancia, vigencia y nota | Señales configurables para justificar contenido complementario. |
| `max_tokens` | `None` | Sin presupuesto predeterminado hasta seleccionar el LLM. |
| `tokenizer` / `token_counter` | ausente | Un tokenizador puede exponer `encode(text)` o un contador callable; sin él se informa una estimación UTF-8 aproximada, no un conteo exacto. |

Las semillas se preservan aunque superen el presupuesto. Los chunks adicionales se agregan en orden hasta agotarlo; nunca se trunca un chunk. Los descartes por presupuesto, por validación o por condición complementaria quedan en `metadata["discarded_chunks"]`. La metadata también registra timings internos, documentos/páginas, semillas de origen, duplicados y conteos funcionales.

## Expansión por artículo (`--expansion units`)

Alternativa al Context Builder que no usa heurísticas en tiempo de consulta. La recuperación híbrida y el reranking siguen sobre los chunks finos de Docling. Después, cada chunk del top-k se **sube al artículo** al que pertenece según `units.jsonl`, generado offline por `rag_istae_ingestion` (patrón *small-to-big* o *parent-document retrieval*).

- **Índice:** `units.jsonl` lista por unidad sus `chunk_ids` (rango contiguo). `query.units.load_units` lo carga en memoria como diccionario `chunk_id → unidad`. Ruta en `units_path` de `config/local.json` o con `--units-path`.
- **Deduplicación:** varias coincidencias del mismo artículo producen una sola unidad, en el puesto de su mejor chunk.
- **Unidades largas** (más de `--max-unit-tokens`, por defecto 1.500 tokens estimados; p. ej. una disposición transitoria con anexos): se envían solo los chunks acertados con ±1 vecino **dentro de la misma unidad**, separando tramos con `[…]`. Los textos se leen de Qdrant por `chunk_id` (solo lectura).
- **Presupuesto:** `--max-context-tokens`, por defecto 4.000 en este modo. La primera unidad siempre entra; las siguientes solo si caben, y las demás se registran en `metadata["excluded_units"]`.
- **Formato del contexto:** cada unidad va precedida por `[INT08-art12] INT08 · Art. 12 · Declaración de egresado`. Los `unit_id` son los mismos que usa `rag-istae-api` para citar.
- **En este modo cada elemento de `ContextResult.chunks` es una unidad**: `chunk_id` contiene el `unit_id`, e `included_by_seed_ids` los chunks recuperados que la seleccionaron. `--seed-k` y `--neighbors-*` no se usan; la cantidad de unidades la determina `--top-k`.
- **Coste:** comparación inicial con `--top-k 5` sobre tres preguntas, sin LLM:
  - con vecinos: 319–682 tokens de contexto estimados;
  - con artículos: 750–1.463 tokens.

  El registro guarda `expansion` y `context_tokens` para medirlo en la evaluación.

```powershell
python .\scripts\query_rag.py "¿Qué requisitos necesito cumplir para que me declaren egresado?" --expansion units --generation-config .\config\generation.openrouter.json --tag units
```

`--expansion neighbors` (predeterminado) conserva el Context Builder como línea base de comparación.

## Recuperación jerárquica (`--retrieval-mode hierarchical`, arquitectura C)

`--retrieval-mode hybrid` (predeterminado) corresponde a la arquitectura B. `hierarchical` corresponde a C: parte del mismo contexto por artículos y le **agrega artículos de las normas de rango superior** relacionadas con las normas recuperadas. Implica `--expansion units`.

1. **Normas destino:** desde los `document_id` del contexto base se siguen las relaciones del catálogo normativo solo hacia normas de **rango estrictamente superior** (número menor), hasta `--hierarchy-depth` saltos (2 por defecto). Las relaciones se eligen con `--hierarchy-relations` (`desarrolla`, `compatible`; por defecto ambas). No se vuelven a buscar normas ya presentes en el contexto base, y el recorrido no entra en ciclos.
2. **Búsqueda acotada:** la misma pregunta (mismo vector, BM25, RRF y reranker) se busca con filtro por esas normas (`hybrid_search(..., document_ids=...)`, solo lectura). Hay dos estrategias en `--hierarchy-search`:
   - `joint` (predeterminada): una sola búsqueda con 20 candidatos en total;
   - `per-document`: una búsqueda por norma, con 5 candidatos por norma y rerank conjunto.

   `--hierarchy-candidate-k` cambia esas cantidades.
3. **Selección:**
   - `--hierarchy-max-units` (3) y `--hierarchy-max-per-document` (1) limitan las unidades nuevas, y `--hierarchy-min-score` (sin mínimo por defecto) descarta las de baja puntuación del reranker;
   - las unidades se suben a artículos con `build_unit_context`, con su propio presupuesto `--hierarchy-max-tokens` (1.500), aparte del contexto base;
   - los descartes y sus motivos quedan en `metadata["hierarchy"]["skipped"]`.

**Tiempos:** la expansión jerárquica es una etapa propia, separada de `context_builder`, que mide solo la expansión base, igual que en B. Hay dos valores:
- `hierarchy`: búsqueda filtrada, rerank, selección y subida a artículos;
- `hierarchy_search`: la búsqueda con su rerank, contenida en `hierarchy`.

En CPU domina el rerank: unos 4,6 s de 4,8 s en la pregunta del egresado con `joint`. En el índice figura como `hierarchy_ms`.

**No se impulsa por rango:** el rango decide *dónde* se busca, no el orden de los resultados. El contexto incluye el rango en el encabezado de cada unidad (`… · Jerarquía 6: Reglamento o normativa interna`). Los artículos agregados van al final, bajo `=== NORMAS DE RANGO SUPERIOR RELACIONADAS ===`, con `role="hierarchy"`. El orden por rango y la regla del art. 425 corresponden a quien arma el prompt final (`rag-istae-api` en B/C).

**Catálogo:** es una copia de `catalogo_normativo.json` de `rag-istae-api`, generado desde `Matriz_Reglamentos_TESIS_ISTAE_v5.xlsx`. Tiene 46 normas, 214 relaciones y rangos 2–6: el corpus no incluye la Constitución. Los rangos figuran como «pendiente de validación jurídica». Se importa a `storage/catalog/` verificando que su `source_sha256` coincida con la matriz de `storage/corpus` (el script solo lee la API):

```powershell
python .\scripts\import_catalog.py --source <rag-istae-api>\data\catalogo_normativo.json --matrix <storage>\corpus\Matriz_Reglamentos_TESIS_ISTAE_v5.xlsx --dest <storage>\catalog\catalogo_normativo.json
```

La ruta se configura en `catalog_path` de `config/local.json`, o con `--catalog-path`. El registro guarda el `sha256` del catálogo usado.

**Exploración sin LLM:** `--no-generation` ejecuta recuperación y contexto sin llamar al modelo ni requerir clave. Imprime las normas destino, las unidades agregadas y los descartes, y registra el contexto en `logs/` (`context_result`). `--show-context` imprime además el texto del contexto.

```powershell
python .\scripts\query_rag.py "¿Qué requisitos necesito cumplir para que me declaren egresado?" --retrieval-mode hierarchical --no-generation --show-context --tag explor-c
python .\scripts\query_rag.py "¿Qué requisitos necesito cumplir para que me declaren egresado?" --retrieval-mode hierarchical --hierarchy-search per-document --hierarchy-min-score 0.3 --no-generation --tag explor-c-doc
```

**Observación inicial (2026-11-06), con la pregunta del egresado y sin LLM:**
- Las normas destino fueron 8 (EXT01, EXT02, EXT03, EXT04, EXT12, EXT13, EXT09 e INT01).
- Con `joint`, 15 de los 20 candidatos fueron de EXT13. Los artículos agregados coincidían por palabras («requisitos» de ingreso del personal académico), no por tema.
- `per-document` da más variedad, pero tarda unas 3 veces más en CPU y, con pocos candidatos por norma, puede perder el mejor resultado de una norma.
- Hay que calibrar la estrategia, el mínimo de score y las relaciones con el Gold Standard antes de fijarlos.

## Línea base B/C (LB-1, 2026-11-06)

Configuración de referencia para comparar B (`hybrid`) y C (`hierarchical`). **C = B + una sola capa**: hasta 3 artículos de las normas de rango superior con las que las normas recuperadas tienen relación `Desarrolla` o `Debe ser compatible con`. Los valores están en `src/query/baseline.py` (`LB1`), son los predeterminados del CLI y del servicio `/retrieve`, y `tests/test_baseline.py` los fija: cambiarlos debe ser una decisión explícita.

| Etapa | Parámetro | B | C |
|---|---|---|---|
| Recuperación | híbrida E5 + BM25, RRF k=60, `candidate_k`=`prefetch_k`=20 | ✓ | ✓ |
| Reranking | BGE `bge-reranker-base`, `top_k`=5 | ✓ | ✓ |
| Expansión | `--expansion units`; contexto máx. 4.000 tokens; artículo parcial si supera 1.500 | ✓ | ✓ |
| Jerarquía: relaciones | `Desarrolla` + `Debe ser compatible con`, solo hacia rango estrictamente superior | — | ✓ |
| Jerarquía: profundidad | 2 saltos | — | ✓ |
| Jerarquía: consulta y búsqueda | la pregunta del usuario; búsqueda conjunta filtrada por las normas superiores, 20 candidatos, con rerank | — | ✓ |
| Jerarquía: selección | máx. 3 artículos, 1 por norma, 1.500 tokens aparte del contexto base; sin puntuación mínima | — | ✓ |
| Presentación | por documentos, formato `rag-istae-api` | en orden de relevancia | como AJ: escala, orden por rango, «incluida por jerarquía», relaciones entre normas presentes |
| Prompt (`prompt_strategy=modular`) | `config/prompts/modular/` | base + rag | base + rag + jerarquía |
| Generación | `config/generation.openrouter.json` (Nemotron `:free`, temperatura 0,2, `max_output_tokens` 2.048, `reasoning.effort` none) | ✓ | ✓ |

Por qué incluir las dos relaciones: `Desarrolla` forma un árbol fijo (reglamento interno → Estatuto → LOES) que nunca alcanza el Reglamento de Régimen Académico desde una norma interna, porque esa relación solo existe como «Debe ser compatible con». Para un reglamento interno, las normas superiores con las que debe ser coherente son las que pueden prevalecer en caso de conflicto. El riesgo, la dilución entre muchas normas destino, se controla con el tope de un artículo por norma y queda medido en `hierarchy_unit_ids`.

La justificación bibliográfica de cada parámetro está en [`Justificacion_Linea_Base.md`](Justificacion_Linea_Base.md). Indica la fuerza de la evidencia, las referencias y qué valores son decisiones de diseño, como los topes de la jerarquía. `candidate_k` = 20 está además ratificado empíricamente en [`Resultados_Prueba5_Candidatos.md`](Resultados_Prueba5_Candidatos.md).

Pendiente de decisión: la API usa `LLM_REASONING_EFFORT=medium` y `LLM_MAX_OUTPUT_TOKENS=8192`, mientras que la generación local usa `effort: none` y 2.048. Para comparar B/C con A/AJ, ambos deben coincidir.

Comandos (con `--question-id` para el Gold Standard):

```powershell
python .\scripts\query_rag.py "<pregunta>" --expansion units --generation-config .\config\generation.openrouter.json --tag LB1-B --question-id Q27
python .\scripts\query_rag.py "<pregunta>" --retrieval-mode hierarchical --generation-config .\config\generation.openrouter.json --tag LB1-C --question-id Q27
```

Protocolo:
1. Ejecutar B y C de la línea base sobre el Gold Standard. Las respuestas no son deterministas (con el mismo contexto, Q27 dio conclusiones distintas), así que conviene repetir las preguntas de validación cruzada.
2. Clasificar los errores de C frente a B: norma superior irrelevante, norma correcta con artículo equivocado, conflicto mal resuelto o falsa abstención.
3. Solo probar una variante si responde a un error observado, registrándola antes con un único factor cambiado respecto de LB-1.

Hipótesis registradas, sin aplicar:
- solo `Desarrolla`;
- puntuación mínima de rerank (ojo: en Q27 todos los scores base fueron < 0,03);
- búsqueda por norma;
- otra consulta para la jerarquía.

Hipótesis retiradas:
- 50 candidatos antes del rerank en lugar de 20. Se evaluó en la Prueba 5 (2026-11-07, 10 preguntas). Ninguna pregunta mejora con 50, cinco empeoran en nDCG@5 y el reranking tarda unas 3 veces más. Se ratifica 20: ver [`Resultados_Prueba5_Candidatos.md`](Resultados_Prueba5_Candidatos.md).

Errores observados de LB-1, todavía sin variante:
- Q29 y Q30: las secciones de `INT02` acaparan el top 5, y no se recuperan ni el indicador CACES (EXT09) ni los artículos de INT06 o INT18.

### Decisión pendiente: rerank sí o no

LB-1 mantiene el rerank mientras no haya evidencia en contra; el servicio lo deja elegir por consulta (`rerank`, por defecto `true`). La decisión se tomará con una prueba de un solo factor (con o sin rerank, mismos 20 candidatos), medida a nivel de **artículos** sobre la salida real de B y de C, sin LLM y con las 30 preguntas:
- por pregunta: artículos esperados presentes en el contexto (recall), artículos pertinentes / artículos enviados (precisión), posición del primer artículo pertinente, tokens del contexto y tiempo de rerank;
- en C, además, cuántos artículos agregados por la capa jerárquica son esperados y cuántos son ruido (la capa jerárquica tiene su propio rerank, una segunda etapa);
- criterio: quitar el rerank si no mejora; mantenerlo si mejora de forma consistente; si solo mejora una etapa (p. ej. la jerárquica), mantenerlo solo en esa etapa como variante explícita.

Indicio previo (datos de la Prueba 5, por fragmento): MRR ≈ 0,58 sin rerank frente a 0,60 con rerank. Pipitone & Alami (2024) encontraron que un reranker general rendía peor que no reranquear en el dominio legal.

### Conversación: una conversación = un tema

La primera pregunta del usuario define el tema y el contexto. Las preguntas de seguimiento se responden **con los artículos recuperados para la primera pregunta, sin recuperar de nuevo**; para otro tema se abre otra conversación.

- **Cómo se aplica:** `rag-istae-web` envía en cada turno todo el historial (`useChat.ts`), así que la primera pregunta siempre está en `history[0]`. Las clases B/C de la API envían a `/retrieve` esa primera pregunta (o `question` si la conversación empieza); el historial completo sigue yendo al LLM, como en A/AJ. Como la recuperación es determinista, cada turno obtiene el mismo contexto, y el servicio lo cachea. `/retrieve` no recibe `history`.
- **Prompt:** `rag_es.txt` aclara que el contexto corresponde a la primera pregunta y que, si un seguimiento trata otro tema o necesita artículos ausentes, el modelo use la frase de abstención y sugiera abrir una conversación nueva.
- **Ya existente en el frontend:** la arquitectura queda fija por conversación (`isArchitectureLocked`). Límite de la API: como máximo 20 mensajes de historial (unos 10 turnos).
- **Limitaciones:**
  - el «tema» para el usuario es más amplio que los ~5 artículos del contexto: a medida que avanza la conversación, aumentan las abstenciones;
  - si la primera recuperación falla (como Q29/Q30) o la primera pregunta es vaga, toda la conversación hereda ese contexto; reformular no ayuda, hay que abrir otra conversación;
  - A/AJ no tienen esta limitación porque siempre ven todo el corpus: es una diferencia de diseño de B/C que se declara en la tesis.
- **Evaluación prevista:** el Gold Standard tiene solo preguntas de un turno. Para medir esto hace falta un conjunto pequeño de seguimientos (1 o 2 por pregunta), registrando cuántos se responden con el contexto fijo y cuántos terminan en abstención (`turn` en el registro del servicio distingue primer turno y seguimientos).
- **Hipótesis registrada, no aplicada:** «complemento por turno»: además del contexto fijo, buscar con la pregunta actual y agregar como máximo 1 o 2 artículos nuevos. Se prueba solo si las abstenciones resultan frecuentes.

Casos pendientes de validación jurídica:
- **Q27 (convalidación de prácticas de hace 7 años): remisión con techo.** El RRA (`EXT02-art43`) dice que las prácticas «son objeto de» convalidación «según lo establecido en la normativa interna de cada IES, hasta un máximo de diez (10) años». El ISTAE (`INT15-art22`, `INT13-art48`) fija 5 años. El texto admite dos lecturas:
  - la IES solo regula el procedimiento y los 10 años prevalecen por jerarquía, que es la respuesta esperada del Gold Standard;
  - la IES fija su plazo dentro del techo, y entonces valen los 5 años.

  Lo debe validar un abogado. Mientras tanto, `jerarquia_es.txt` no se ajusta a ninguna de las dos lecturas: codificar el criterio en el prompt sería ajustarlo a una interpretación no validada. Para calificar Q27 se propone una rúbrica en lugar de exigir solo el plazo: identifica los dos plazos y cita ambas normas; reconoce la mayor jerarquía del RRA; explica la consecuencia según la lectura validada, o al menos la tensión entre la remisión y el máximo; e incluye el trámite de `INT15-art22`.

## Servicio de recuperación (`POST /retrieve`)

Servicio HTTP (FastAPI) que carga E5, el reranker, `units.jsonl`, el catálogo y Qdrant **una sola vez** y atiende consultas de B y C. No genera respuesta: `rag-istae-api` arma el prompt y llama al LLM con su propio modelo. Contrato completo, ejemplos y variables de entorno: [`docs/contrato_retrieve.md`](docs/contrato_retrieve.md).

- **Request:** `question` (en una conversación, la primera pregunta del usuario), `rerank` (por defecto `true`), `hierarchy` (`false` = B, `true` = C), `document_ids` opcional; `turn`, `question_id` y `tag` solo para el registro; `debug`. Los parámetros de la línea base no se aceptan en el request (422): los fija `query.baseline.CURRENT_BASELINE` (LB-1).
- **Response:** artículos (`unit_id`, norma, rango, texto, `score`, `origin` base o jerarquía), `context` ya renderizado en el formato de la API, tokens estimados, `cached`, `index_version` y tiempos.
- **Mismo pipeline que los scripts:** `query.pipeline.run_units_pipeline` lo usan `query_rag.py` (rama `--expansion units`) y el motor residente `query.engine.RetrievalEngine`.
- **Caché:** LRU por (pregunta, `rerank`, `hierarchy`, `document_ids`); los turnos de seguimiento de una conversación no repiten la recuperación.
- **Registro:** cada llamada (también las cacheadas y las fallidas) va a `logs/service/` con el formato de `query.run_log`, sin la clave.

```powershell
$env:PYTHONPATH = "src;scripts"
$env:RAG_API_KEY = "<clave>"              # o $env:RAG_AUTH_DISABLED = "true" solo en local
python .\scripts\serve_retrieval.py       # http://127.0.0.1:8100/docs
```

Qdrant local admite un solo proceso: mientras el servicio está activo no se pueden ejecutar los scripts de consulta sobre el mismo `vector_store/qdrant`.

## Validar el pipeline completo

El ejecutable independiente `scripts/validate_context_pipeline.py` conecta la configuración, E5, Qdrant, retrieval híbrido, RRF, reranker y Context Builder. Cada ejecución inicializa los modelos; sus tiempos de carga no representan consultas posteriores con los modelos ya cargados.

```powershell
python .\scripts\validate_context_pipeline.py "¿Qué requisitos establece el artículo 18?" --top-k 5 --candidate-k 20 --prefetch-k 20 --rrf-k 60
python .\scripts\validate_context_pipeline.py "¿Qué requisitos establece el artículo 18?" --no-rerank
python .\scripts\validate_context_pipeline.py "¿Qué reformas se relacionan con el artículo 18?" --max-context-tokens 1800 --json-output .\validation.json
```

El reranking se ejecuta por defecto. `--no-rerank` evita tanto cargar BGE como ejecutar el CrossEncoder; el Context Builder recibe los candidatos en orden RRF, y el score de reranker de las semillas queda como `null`. El JSON indica `reranker_enabled`, deja `reranker_results` vacío y registra `context_selection_results` con el origen del orden y su score. `--rerank` lo activa explícitamente. Parámetros del builder: `--seed-k`, `--neighbors-before`, `--neighbors-after` y `--max-context-tokens`. El JSON opcional incluye configuración relevante, candidatos/scores, resultados de reranking si se ejecutó, semillas, chunks incluidos y descartados, contexto, procedencia y rendimiento; no incluye modelos ni documentos originales aparte del texto ya incluido en el contexto final.

La salida distingue candidatos de recuperación y chunks finales. Los tiempos reportan inicialización, cada etapa de consulta y el total; el desglose `context_*` forma parte del tiempo total del builder y no se debe sumar otra vez. El contador de tokens se identifica como estimado salvo que la API reciba un tokenizador real. El pipeline mide comportamiento y coste, no demuestra una mejora en la calidad semántica de respuestas.

## Generación de respuestas (Gemini, OpenRouter)

El paquete `query.generation` recibe un `ContextResult` ya construido; no ejecuta retrieval, reranking ni consultas a Qdrant. Su configuración JSON y el System Prompt son independientes de `config/local.json` y de `load_config()`. El adaptador usa `google-genai` 1.x de forma diferida; la credencial se obtiene de la variable indicada en `api_key_env` (por defecto `GEMINI_API_KEY`) y nunca se guarda en la configuración.

Copie `config/generation.example.json` a una ruta local, configure el modelo y exporte la clave API en el entorno. `system_prompt_path` se resuelve respecto al archivo JSON:

```python
from pathlib import Path

from query.generation import generate_answer, load_generation_config

generation_config = load_generation_config(Path("config/generation.json"))
answer = generate_answer(context_result, generation_config)
print(answer.answer)
print(answer.provider_metrics)
print(answer.context_result.metadata)
```

El modelo recibe solo la pregunta y `ContextResult.context`, empaquetados como datos JSON; la metadata y los chunks se retienen en `GenerationResult.context_result` para trazabilidad. El System Prompt indica que el contexto es material documental no confiable y que las instrucciones contenidas en él no deben obedecerse. Las métricas de tokens se reportan solo si Gemini las incluye; `application_latency_ms` es una medición local del tiempo de la solicitud. Una llamada real a Gemini requiere credenciales y puede generar costos.

Para ejecutar una pregunta a través del pipeline RAG completo (E5 + recuperación híbrida en Qdrant + reranking BGE + Context Builder + generación Gemini), utilice `scripts/query_rag.py`. Asegúrese de haber instalado `requirements.txt`, de tener la configuración local de E5/Qdrant y de definir `GEMINI_API_KEY` en el entorno. El script usa `config/local.json` y `config/generation.json` por defecto; `--generation-config` permite indicar otro archivo. Acepta además los mismos parámetros de recuperación y contexto que `validate_context_pipeline.py` (`--top-k`, `--candidate-k`, `--prefetch-k`, `--rrf-k`, `--batch-size`, `--reranker-model`, `--rerank/--no-rerank`, `--seed-k`, `--neighbors-before`, `--neighbors-after`, `--max-context-tokens`, `--json-output`):

```powershell
$env:PYTHONPATH = "src;scripts"
$env:GEMINI_API_KEY = "TU_API_KEY"
python .\scripts\query_rag.py "¿Qué requisitos establece el artículo 18?"
python .\scripts\query_rag.py "¿Qué requisitos establece el artículo 18?" --no-rerank --json-output .\rag-result.json
python .\scripts\query_rag.py "¿Qué reformas aplican?" --generation-config .\config\generation.json
```

La respuesta se muestra junto con las fuentes del contexto y los tiempos de ejecución. El JSON opcional incluye la respuesta, el contexto y su trazabilidad. Cada ejecución real invoca Gemini y puede generar costos; `--no-rerank` evita cargar BGE, pero no omite los demás pasos.

### Registro automático de ejecuciones

Cada ejecución de `query_rag.py` se registra en `logs/` (excluida de git), sin necesidad de `--json-output`. También se registran las ejecuciones que fallan una vez iniciado el pipeline (p. ej. un HTTP 429 del proveedor), con `status=error` y la etapa en que ocurrió el error. Los errores de argumentos y la falta de clave API no se registran.

```
logs/
  runs/<run_id>.json   # detalle: respuesta, contexto, rankings, tiempos, parámetros y configuración (sin secretos)
  runs_index.jsonl     # una línea resumen por ejecución
  runs_index.csv       # mismo resumen; separador ";" y UTF-8 con BOM (se abre directo en Excel)
```

`run_id` tiene la forma `YYYYMMDD-HHMMSS_<hex>`. Columnas del índice:
- **Identificación y estado:** `run_id`, `timestamp`, `status`, `error_stage`, `error`, `tag`, `question_id`, `query`.
- **Generación:** `provider`, `model`, `extra_body`, `temperature`, `max_output_tokens`.
- **Recuperación y contexto:** `rerank`, `top_k`, `candidate_k`, `prefetch_k`, `rrf_k`, `expansion`, `retrieval_mode` (`hybrid`/`hierarchical`), `seed_k`, `neighbors_before`, `neighbors_after`, `max_context_tokens`, `n_candidates`, `n_context_chunks`, `context_chunk_ids` (separados por `|`; en `--expansion units` son `unit_id`), `context_tokens` (estimados), `hierarchy_unit_ids` (unidades agregadas por jerarquía).
- **Resultado y costo:** `input_tokens`, `output_tokens`, `finish_reason`, `hierarchy_ms` (expansión jerárquica; vacío en `hybrid`), `generation_ms`, `load_ms`, `execution_ms`, `total_ms`, `answer`, `detail_file`.

Los tiempos se agrupan en dos bloques, definidos en `query.run_log.LOAD_STAGES` / `EXECUTION_STAGES` y `group_timings()`:
- **carga** (`load_ms`): configuración, modelos E5 y reranker, catálogo, `units.jsonl` y apertura de Qdrant. Se paga una vez por proceso y desaparece por consulta en un servicio persistente que mantenga todo en memoria;
- **ejecución** (`execution_ms`): embedding, búsqueda híbrida, rerank, contexto, jerarquía y generación. Es el costo de cada consulta.

Las subetapas contenidas en otra, como `hierarchy_search` dentro de `hierarchy`, se muestran pero no se suman. La consola imprime ambos bloques con su subtotal. Con `--no-generation` las columnas de generación quedan vacías y el detalle guarda el contexto en `context_result`. Los tiempos del índice están en milisegundos enteros. Si cambian las columnas, `runs_index.csv` se regenera automáticamente desde `runs_index.jsonl`. El detalle guarda el nombre de la variable de la clave API, nunca su valor.

Opciones:
- `--tag` (opcional): etiqueta del experimento.
- `--question-id` (opcional): ID de la pregunta del Gold Standard.
- `--log-dir`: otra carpeta de registro.
- `--no-log`: no registrar esa ejecución.

```powershell
python .\scripts\query_rag.py "¿Qué requisitos establece el artículo 18?" --generation-config .\config\generation.openrouter.json --tag nemotron-effort-none --question-id P01
```

```python
import pandas as pd
runs = pd.read_json("logs/runs_index.jsonl", lines=True)
```

### Estrategia de prompt (`prompt_strategy`)

El JSON de generación elige cómo se arma el mensaje al modelo:

| `prompt_strategy` | `system_prompt_path` | Armado |
|---|---|---|
| `single` (predeterminada) | archivo (`prompts/system_prompt_es.txt`) | System Prompt único; la pregunta y el contexto van como JSON en el mensaje del usuario. |
| `modular` | carpeta (`prompts/modular/`) | Mismo armado que `rag-istae-api`: las instrucciones y `<documentos>…</documentos>` en el system; `<pregunta>…</pregunta>` y el recordatorio en el mensaje del usuario. Requiere `--expansion units`. |

`config/generation.openrouter.example.json` usa `modular`. Es la estrategia recomendada para B/C, la que se propondrá a `rag-istae-api` (Fase 5). Los prompts se componen igual que en el diseño 2×2 de la API:

| | Sin jerarquía | Con jerarquía |
|---|---|---|
| Contexto largo (API) | A = base | AJ = base + jerarquía |
| RAG (este repo) | B = `base_es` + `rag_es` | C = `base_es` + `rag_es` + `jerarquia_es` |

- `base_es.txt` y `recordatorio_es.txt` son **copias sin cambios** de `rag-istae-api/prompts`, con las mismas reglas de cita (`[unit_id]` y la sección «Fuentes:»), la misma frase de abstención y el mismo formato que verifica la API.
- `rag_es.txt` (nuevo) aclara que el contexto es **una selección de artículos**, no la normativa completa.
  - La frase de abstención «No encontré en la normativa disponible», la que detecta la API, se reserva para cuando **ningún** artículo es pertinente. Así se evita la falsa abstención observada en Q27.
  - Si la información es parcial, ambigua o hay disposiciones que difieren, se responde lo que se puede afirmar y se señala lo que no.
  - Pide no suponer el contenido de artículos que no aparecen.
- `jerarquia_es.txt` (versión corregida respecto de la API) aplica la escala de rangos y la prevalencia **solo con los documentos proporcionados**. No invoca el art. 425 de la Constitución, que no está en el corpus y no se puede verificar como cita. También explica qué significan las normas «incluida por jerarquía».

Con `--expansion units`, el contexto se presenta **por documentos** (`query.presentation.render_documents`), en el mismo formato que la API:
- `=== [INT08] Reglamento de Titulación y Graduación ===` y, debajo, `[INT08-art12] Artículo 12. …`;
- en B, los documentos van en orden de relevancia;
- en C, como en AJ: la escala al inicio, los documentos por rango con rango y emisor en el encabezado, la marca `· incluida por jerarquía` en las normas que solo aportan artículos superiores, y al final `=== RELACIONES ENTRE NORMAS ===`, solo con las relaciones entre normas presentes.

Los nombres, rangos y emisores salen del catálogo (`catalog_path`). En B, sin catálogo, se usa el `document_id`. La etapa se mide como `presentation`.

### Cambiar de proveedor o modelo

El proveedor se elige con la clave `provider` del JSON de generación; el resto del pipeline no cambia. Proveedores registrados en `query.generation.providers.PROVIDERS`:

| `provider` | Adaptador | Credencial por defecto | Claves adicionales |
|---|---|---|---|
| `gemini` | `GeminiProvider` (`google-genai`) | `GEMINI_API_KEY` | — |
| `openrouter` | `OpenAICompatibleProvider` (httpx, `POST {base_url}/chat/completions`) | `OPENROUTER_API_KEY` | `base_url` (por defecto `https://openrouter.ai/api/v1`), `extra_headers`, `extra_body` |

`config/generation.openrouter.example.json` configura `nvidia/nemotron-3-ultra-550b-a55b:free`. `extra_body` se mezcla en el cuerpo de la solicitud (no puede sobrescribir `model` ni `messages`); el ejemplo usa `{"reasoning": {"exclude": true}}` para que el modelo de razonamiento no devuelva su cadena de pensamiento. Para cambiar de modelo basta con otro JSON:

```powershell
$env:OPENROUTER_API_KEY = "TU_API_KEY"
python .\scripts\query_rag.py "¿Qué requisitos establece el artículo 18?" --generation-config .\config\generation.openrouter.json
```

Los modelos `:free` de OpenRouter tienen límites de tasa (un HTTP 429 se informa como `GenerationProviderError`) y sus endpoints gratuitos pueden registrar los prompts; considérelo antes de enviar contexto del corpus institucional. Como el adaptador sigue el formato Chat Completions, otro servicio compatible (OpenAI, vLLM, Ollama) solo requiere registrar un nombre en `PROVIDERS` y ajustar `base_url`.

## Pruebas y estado de validación

Las pruebas unitarias se ejecutan sin conectar a Qdrant ni a Gemini:

```powershell
$env:PYTHONPATH = "src;scripts"
python -m unittest discover -s tests -p "test_*.py" -v              # suite completa (127 pruebas)
python -m unittest discover -s tests -p "test_context_builder.py" -v
python -m unittest discover -s tests -p "test_generation.py" -v
python -m unittest discover -s tests -p "test_structure_detector.py" -v
```

`tests/test_configuration.py` está escrita con el fixture `tmp_path` de pytest y no se ejecuta con `unittest`; requiere instalar `pytest`, que no forma parte de `requirements.txt`.

Los tests usan mocks/fakes y no requieren conectar con Gemini o Qdrant, reindexar ni cambiar la colección. Se verifican selección de semillas, filtros de vecinos, continuidad y límites estructurales, complementarios condicionales, deduplicación, trazabilidad, presupuesto, metadata ausente, tiempos, exportación y normalización de respuestas del proveedor. La ejecución de pruebas no implica por sí sola una evaluación semántica del contexto ni una solicitud real a Gemini.

## Estado actual

| Componente | Estado |
|---|---|
| Búsqueda vectorial E5 | Implementada |
| BM25 nativo de Qdrant | Implementado |
| Recuperación híbrida con RRF | Implementada |
| Reranking BGE (`bge-reranker-base`) | Implementado, opcional (`--no-rerank`) |
| Inspección de chunks y vecinos | Implementada (`inspect_qdrant.py`) |
| Context Builder (expansión por vecinos) | Implementado, con pruebas unitarias |
| Expansión por artículo (`--expansion units`) | Implementada, con pruebas unitarias; pendiente de evaluar contra el Gold Standard |
| Recuperación jerárquica (`--retrieval-mode hierarchical`, arquitectura C) | Implementada, con pruebas unitarias; en exploración (estrategia de búsqueda y mínimo de score por calibrar) |
| Generación con Gemini | Implementada (`query_rag.py`), con pruebas unitarias |
| Generación con OpenRouter (Nemotron) | Implementada (`--generation-config`), con pruebas unitarias; sin validación real |
| Evaluación sistemática con Gold Standard | En curso: ver `Guia_Pruebas_Recuperación.md` |
| Servicio persistente `POST /retrieve` (modelos y Qdrant cargados una sola vez) | Implementado, con pruebas unitarias (`test_service.py`, `test_pipeline.py`); ver «Servicio de recuperación» |
| Hosting del servicio de recuperación (arquitectura B del chat) | **Pendiente.** Prototipo académico: buscar un entorno cloud gratuito (candidatos por verificar: Hugging Face Spaces CPU, Oracle Cloud Always Free) o, si se consigue, una máquina del instituto. La web (Vercel) y `rag-istae-api` (Render) ya están publicadas en hostings gratuitos independientes. |

Cada ejecución de script carga E5, BGE y abre Qdrant de nuevo; en las mediciones iniciales esa inicialización representó la mayor parte de la latencia (del orden de 4–5 s por cada recurso), mientras que la búsqueda híbrida tomó alrededor de 0,3 s. La evaluación de la recuperación (vectorial, BM25, híbrida e híbrida con reranking frente al Gold Standard) se describe en [Guia_Pruebas_Recuperación.md](Guia_Pruebas_Recuperación.md).
