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
   ▼
generation.generate_answer (System Prompt externo + Gemini vía google-genai)
```

| Módulo | Responsabilidad |
|---|---|
| `src/query/config.py` | Carga `config/local.json`, valida claves obligatorias y resuelve rutas absolutas. |
| `src/query/retrieval.py` | Carga E5, abre el cliente Qdrant y ejecuta búsqueda vectorial, BM25 e híbrida con RRF. |
| `src/query/reranking.py` | Carga el CrossEncoder local y reordena los candidatos por score. |
| `src/query/units.py` | Expansión por artículo (*small-to-big*): índice `chunk_id → unidad` desde `units.jsonl` y construcción del contexto con artículos completos. |
| `src/query/context/` | Context Builder: detector y validador de estructura (artículo, capítulo, sección, listas, notas), expansión de vecinos, fusión, deduplicación y presupuesto de tokens. |
| `src/query/generation/` | Configuración, construcción del prompt, adaptador de proveedor (`providers/gemini.py`), normalización de respuestas y errores explícitos (`GenerationConfigurationError`, `GenerationProviderError`). |
| `scripts/` | Ejecutables de línea de comandos de cada etapa y del pipeline completo. |

Modelo de reranking actual: `bge-reranker-base` (más liviano que `bge-reranker-v2-m3`, que se utilizó en las primeras pruebas). El modelo se toma de `reranker_model_path` o de `--reranker-model`.

## Instalacion

1. Clone este repositorio y cree un entorno virtual Python 3.11.
2. Instale `pip install -r requirements.txt` (`sentence-transformers`, `qdrant-client`, `torch`, `google-genai`).
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
- **Recuperación y contexto:** `rerank`, `top_k`, `candidate_k`, `prefetch_k`, `rrf_k`, `seed_k`, `neighbors_before`, `neighbors_after`, `max_context_tokens`, `expansion`, `n_candidates`, `n_context_chunks`, `context_chunk_ids` (separados por `|`; en `--expansion units` son `unit_id`), `context_tokens` (estimados).
- **Resultado y costo:** `input_tokens`, `output_tokens`, `finish_reason`, `generation_ms`, `total_ms`, `answer`, `detail_file`.

Los tiempos del índice están en milisegundos enteros. Si cambian las columnas, `runs_index.csv` se regenera automáticamente desde `runs_index.jsonl`. El detalle guarda el nombre de la variable de la clave API, nunca su valor.

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
python -m unittest discover -s tests -p "test_*.py" -v              # suite completa (72 pruebas)
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
| Generación con Gemini | Implementada (`query_rag.py`), con pruebas unitarias |
| Generación con OpenRouter (Nemotron) | Implementada (`--generation-config`), con pruebas unitarias; sin validación real |
| Evaluación sistemática con Gold Standard | En curso: ver `Guia_Pruebas_Recuperación.md` |
| Servicio persistente (modelos y Qdrant cargados una sola vez) | Pendiente |
| Hosting del servicio de recuperación (arquitectura B del chat) | **Pendiente.** Prototipo académico: buscar un entorno cloud gratuito (candidatos por verificar: Hugging Face Spaces CPU, Oracle Cloud Always Free) o, si se consigue, una máquina del instituto. La web (Vercel) y `rag-istae-api` (Render) ya están publicadas en hostings gratuitos independientes. |

Cada ejecución de script carga E5, BGE y abre Qdrant de nuevo; en las mediciones iniciales esa inicialización representó la mayor parte de la latencia (del orden de 4–5 s por cada recurso), mientras que la búsqueda híbrida tomó alrededor de 0,3 s. La evaluación de la recuperación (vectorial, BM25, híbrida e híbrida con reranking frente al Gold Standard) se describe en [Guia_Pruebas_Recuperación.md](Guia_Pruebas_Recuperación.md).
