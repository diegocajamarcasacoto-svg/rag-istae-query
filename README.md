# RAG ISTAE Query

Consulta de solo lectura: E5 (`query:`) + BM25 nativo de Qdrant sobre `payload.text` -> RRF (`k=60`) -> BGE CrossEncoder. No crea colecciones, indices ni puntos.

## Instalacion

1. Clone este repositorio y cree un entorno virtual Python 3.11.
2. Instale `pip install -r requirements.txt`.
3. Prepare y valide `rag-istae-resources`.
4. Copie `config/local.example.json` a `config/local.json` y asigne rutas absolutas al E5, BGE y Qdrant compartidos.

Con los recursos ya disponibles, funciona offline; los scripts nunca descargan modelos. Ejemplos:

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

## Validar el pipeline completo

El ejecutable independiente `scripts/validate_context_pipeline.py` conecta la configuración, E5, Qdrant, retrieval híbrido, RRF, reranker y Context Builder. Cada ejecución inicializa los modelos; sus tiempos de carga no representan consultas posteriores con los modelos ya cargados.

```powershell
python .\scripts\validate_context_pipeline.py "¿Qué requisitos establece el artículo 18?" --top-k 5 --candidate-k 20 --prefetch-k 20 --rrf-k 60
python .\scripts\validate_context_pipeline.py "¿Qué requisitos establece el artículo 18?" --no-rerank
python .\scripts\validate_context_pipeline.py "¿Qué reformas se relacionan con el artículo 18?" --max-context-tokens 1800 --json-output .\validation.json
```

El reranking se ejecuta por defecto. `--no-rerank` evita tanto cargar BGE como ejecutar el CrossEncoder; el Context Builder recibe los candidatos en orden RRF, y el score de reranker de las semillas queda como `null`. El JSON indica `reranker_enabled`, deja `reranker_results` vacío y registra `context_selection_results` con el origen del orden y su score. `--rerank` lo activa explícitamente. Parámetros del builder: `--seed-k`, `--neighbors-before`, `--neighbors-after` y `--max-context-tokens`. El JSON opcional incluye configuración relevante, candidatos/scores, resultados de reranking si se ejecutó, semillas, chunks incluidos y descartados, contexto, procedencia y rendimiento; no incluye modelos ni documentos originales aparte del texto ya incluido en el contexto final.

La salida distingue candidatos de recuperación y chunks finales. Los tiempos reportan inicialización, cada etapa de consulta y el total; el desglose `context_*` forma parte del tiempo total del builder y no se debe sumar otra vez. El contador de tokens se identifica como estimado salvo que la API reciba un tokenizador real. El pipeline mide comportamiento y coste, no demuestra una mejora en la calidad semántica de respuestas.

## Generación de respuestas con Gemini

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

Para ejecutar una pregunta a través del pipeline RAG completo (E5 + recuperación híbrida en Qdrant + reranking BGE + Context Builder + generación Gemini), utilice `scripts/query_rag.py`. Asegúrese de haber instalado `requirements.txt`, de tener la configuración local de E5/Qdrant y de definir `GEMINI_API_KEY` en el entorno. El script usa `config/generation.example.json` por defecto; puede copiarlo a `config/generation.json` y pasar esa ruta explícitamente:

```powershell
$env:PYTHONPATH = "src;scripts"
$env:GEMINI_API_KEY = "TU_API_KEY"
python .\scripts\query_rag.py "¿Qué requisitos establece el artículo 18?"
python .\scripts\query_rag.py "¿Qué requisitos establece el artículo 18?" --no-rerank --json-output .\rag-result.json
python .\scripts\query_rag.py "¿Qué reformas aplican?" --generation-config .\config\generation.json
```

La respuesta se muestra junto con las fuentes del contexto y los tiempos de ejecución. El JSON opcional incluye la respuesta, el contexto y su trazabilidad. Cada ejecución real invoca Gemini y puede generar costos; `--no-rerank` evita cargar BGE, pero no omite los demás pasos.

## Pruebas y estado de validación

Las pruebas unitarias del Context Builder se pueden ejecutar sin conectar a Qdrant:

```powershell
$env:PYTHONPATH = "src;scripts"
python -m unittest discover -s tests -p "test_context_builder.py" -v
python -m unittest discover -s tests -p "test_generation.py" -v
```

Los tests usan mocks/fakes y no requieren conectar con Gemini o Qdrant, reindexar ni cambiar la colección. Se verifican selección de semillas, filtros de vecinos, continuidad y límites estructurales, complementarios condicionales, deduplicación, trazabilidad, presupuesto, metadata ausente, tiempos, exportación y normalización de respuestas del proveedor. La ejecución de pruebas no implica por sí sola una evaluación semántica del contexto ni una solicitud real a Gemini.
