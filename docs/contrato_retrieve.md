# Contrato del servicio `POST /retrieve` (versión 1.0)

Servicio de recuperación de `rag-istae-query` para las arquitecturas **B** (RAG híbrido) y **C** (RAG jerárquico) de `rag-istae-api`. Devuelve los artículos pertinentes y el contexto ya armado en el formato de la API. **No genera respuesta**: la API arma el prompt (`base_es` + `rag_es` [+ `jerarquia_es`] + `<documentos>`), llama al LLM con su propio modelo y verifica las citas.

- Código: `src/query/service/` (FastAPI), `src/query/engine.py` (motor residente), `src/query/pipeline.py` (mismo pipeline que `scripts/query_rag.py --expansion units`).
- Parámetros de recuperación: los de la línea base vigente (`src/query/baseline.py`, **LB-1**). No se pueden cambiar desde el request.

## Arranque

```powershell
cd rag-istae-query
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "src;scripts"
$env:RAG_API_KEY = "<clave compartida con rag-istae-api>"
python .\scripts\serve_retrieval.py --host 127.0.0.1 --port 8100
```

Documentación interactiva: `http://127.0.0.1:8100/docs`. La carga inicial (E5, reranker, `units.jsonl`, catálogo y Qdrant) tarda unos 17 s y se hace una sola vez.

**Qdrant local admite un solo proceso.** Mientras el servicio está activo, no se pueden ejecutar los scripts de consulta (`query_rag.py`, `search_*.py`) sobre el mismo `vector_store/qdrant`. El servicio corre con un solo worker y atiende las consultas de a una.

### Variables de entorno

| Variable | Uso | Por defecto |
|---|---|---|
| `RAG_CONFIG` | JSON con el formato de `config/local.json` | `config/local.json` del repositorio |
| `RAG_QDRANT_PATH`, `RAG_E5_MODEL_PATH`, `RAG_RERANKER_MODEL_PATH`, `RAG_UNITS_PATH`, `RAG_CATALOG_PATH`, `RAG_COLLECTION` | Sobrescriben claves del JSON (contenedores u otro host) | — |
| `RAG_API_KEY` | Clave que los clientes envían en `X-API-Key` | **obligatoria** |
| `RAG_AUTH_DISABLED` | `true` desactiva la autenticación (solo pruebas locales) | `false` |
| `RAG_LOG_DIR` | Carpeta del registro de llamadas (`-` lo desactiva) | `logs/service` |
| `RAG_CACHE_SIZE` | Consultas en el caché LRU (`0` lo desactiva) | `128` |

## `GET /health`

Sin autenticación. Responde 200 con el estado, la línea base, el número de artículos y normas, la versión del índice y los tiempos de carga, o 503 si el motor no cargó. No incluye secretos ni rutas locales.

```json
{
  "status": "ok",
  "contract_version": "1.0",
  "baseline": "LB-1",
  "units": 2748,
  "documents": 46,
  "index_version": {
    "collection": "istae_chunks_e5_base",
    "units_sha256": "2079998ac1e9aab5…",
    "catalog_sha256": "02c3a9b0c19c7ee7…"
  },
  "load_seconds": {"e5_load": 5.6, "reranker_load": 6.6, "units_load": 0.05, "catalog_load": 0.003, "qdrant_open": 4.7},
  "service": {"auth": "api_key", "cache_size": 128}
}
```

## `POST /retrieve`

Cabecera obligatoria: `X-API-Key: <RAG_API_KEY>`.

### Request

| Campo | Tipo | Por defecto | Descripción |
|---|---|---|---|
| `question` | string (1–2.000) | — | Consulta. **En una conversación, la API envía la primera pregunta del usuario** (ver «Conversación»). Los espacios repetidos se normalizan. |
| `rerank` | bool | `true` | Reordenar los candidatos con `bge-reranker-base`. Decisión pendiente en la tesis: se deja elegible. |
| `hierarchy` | bool | `false` | `false` = arquitectura **B**; `true` = **C** (agrega artículos de normas de rango superior relacionadas en el catálogo). |
| `document_ids` | string[] | todo el corpus | Limita la búsqueda base a esas normas (p. ej. `["INT15"]`). La capa jerárquica puede agregar normas superiores fuera del filtro. |
| `turn` | int ≥ 1 | — | Solo para el registro: 1 = primera pregunta, > 1 = seguimiento. No afecta la búsqueda ni el caché. |
| `question_id`, `tag` | string | — | Solo para el registro (Gold Standard, experimento). |
| `debug` | bool | `false` | Incluye tiempos por etapa, candidatos, ranking y detalle de la jerarquía. |

Cualquier otro campo, como `top_k`, `candidate_k` o `history`, se rechaza con 422.

### Response (200)

| Campo | Descripción |
|---|---|
| `contract_version` | `"1.0"` |
| `run_id` | Identificador de la llamada en el registro (`logs/service/runs/<run_id>.json`). |
| `baseline` | Línea base aplicada (`"LB-1"`). |
| `mode` | `{rerank, hierarchy, architecture: "B" \| "C"}` |
| `units[]` | Artículos en el orden en que se envían: primero los de la búsqueda base por relevancia, después los de la jerarquía. Campos: `unit_id` (los mismos IDs de `GET /units/{id}` de la API), `document_id`, `document_name`, `rank` (1 = mayor jerarquía), `label`, `title`, `text` (texto del artículo; si es parcial, con `[…]` en los tramos omitidos), `partial`, `score` (del reranker; sin rerank, el de la fusión RRF), `origin` (`base` \| `hierarchy`) y `matched_chunk_ids`. |
| `context` | Bloque de documentos listo para `<documentos>…</documentos>`, en el formato de la API: `=== [DOC] Nombre ===` y `[unit_id] texto`. En C, como AJ: escala jerárquica, normas por rango con emisor, «incluida por jerarquía» y relaciones entre las normas presentes. |
| `tokens_estimated` | Tokens estimados de `context` (bytes UTF-8 / 4). |
| `warnings` | Avisos del pipeline, por ejemplo «sin normas superiores relacionadas». |
| `cached` | `true` si el resultado salió del caché. |
| `index_version` | Colección y sha256 de `units.jsonl` y del catálogo: identifican qué índice produjo la respuesta. |
| `timings_ms` | `{total}` de la llamada. |
| `debug` | Solo con `debug: true`: `candidates`, `ranked`, `excluded_units`, `hierarchy`, `document_order`, `timings_ms` y `baseline_params`. |

### Errores

| Código | Cuándo |
|---|---|
| 401 | Falta `X-API-Key` o no es válida. |
| 422 | Pregunta vacía, `document_ids` vacío o desconocido, `turn` < 1, o un campo no admitido (`top_k`, `history`, …). |
| 503 | El motor no cargó (por ejemplo, falta `units.jsonl`). `/health` informa la causa. |
| 500 | Error interno. El detalle solo incluye el `run_id`; la causa queda en el registro. |

### Ejemplo real (Q27, arquitectura C, 2026-11-07)

```powershell
$body = @{ question = "Hice prácticas hace 7 años, ¿todavía me las pueden convalidar?"; hierarchy = $true; question_id = "Q27"; turn = 1 } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8100/retrieve -Headers @{ "X-API-Key" = $env:RAG_API_KEY } -ContentType "application/json; charset=utf-8" -Body ([Text.Encoding]::UTF8.GetBytes($body))
```

Respuesta resumida:

```json
{
  "contract_version": "1.0",
  "run_id": "20261107-225714_2423e4",
  "baseline": "LB-1",
  "mode": {"rerank": true, "hierarchy": true, "architecture": "C"},
  "units": [
    {"unit_id": "INT15-art22", "document_id": "INT15", "document_name": "Reglamento de Prácticas Preprofesionales",
     "rank": 6, "label": "Art. 22", "title": "Experiencia laboral", "text": "**Artículo 22. Experiencia laboral** . Las horas …",
     "partial": false, "score": 0.0292, "origin": "base", "matched_chunk_ids": ["INT15:…"]},
    {"unit_id": "INT15-art25", "origin": "base", "…": "…"},
    {"unit_id": "INT13-art50", "origin": "base", "…": "…"},
    {"unit_id": "EXT02-art43", "rank": 3, "origin": "base", "…": "…"},
    {"unit_id": "INT13-art48", "origin": "base", "…": "…"},
    {"unit_id": "EXT01-disp-general-cuarta-2", "rank": 2, "origin": "hierarchy", "…": "…"},
    {"unit_id": "EXT09-sec189", "rank": 4, "origin": "hierarchy", "…": "…"}
  ],
  "context": "=== ESCALA DE JERARQUÍA NORMATIVA (1 = mayor rango) ===\n1. Constitución\n…\n=== [EXT01] Ley Orgánica de Educación Superior (LOES) · Jerarquía 2: Ley orgánica · Emisor: Asamblea Nacional · incluida por jerarquía ===\n[EXT01-disp-general-cuarta-2] CUARTA.- …\n…\n=== RELACIONES ENTRE NORMAS ===\n- [INT15] Reglamento de Prácticas Preprofesionales — Debe ser compatible con → [EXT02] Reglamento de Régimen Académico\n…",
  "tokens_estimated": 3066,
  "warnings": [],
  "cached": false,
  "index_version": {"collection": "istae_chunks_e5_base", "units_sha256": "2079998a…", "catalog_sha256": "02c3a9b0…"},
  "timings_ms": {"total": 6795}
}
```

Tiempos medidos en CPU, en este equipo, para esa consulta:
- **B:** unos 3,2 s, de los cuales unos 2,6 s son del rerank.
- **C:** unos 6,8 s, porque se suma la capa jerárquica (unos 3,8 s).
- **Seguimiento con la misma pregunta:** 0 ms (`cached: true`).
- **B sin rerank:** unos 0,7 s.

Los artículos coinciden con los de `query_rag.py --no-generation` para la misma pregunta.

## Conversación (línea base: una conversación = un tema)

La primera pregunta del usuario define el contexto. Las preguntas de seguimiento se responden con **los mismos artículos**, sin recuperar de nuevo. Para otro tema, se abre otra conversación.

- `/retrieve` no recibe `history`: es una función pura pregunta → contexto.
- La arquitectura B/C de la API elige la consulta: el **primer mensaje del usuario** en `history`, o `question` si la conversación empieza. Como `rag-istae-web` envía todo el historial en cada turno, la primera pregunta siempre está disponible.
- Como la recuperación es determinista, cada turno obtiene el mismo contexto, y el caché del servicio evita recalcularlo.
- `turn` registra si la llamada corresponde al primer turno o a un seguimiento.
- El prompt `rag_es` indica que, si un seguimiento no puede responderse con esos artículos, el modelo use la frase de abstención y sugiera abrir una conversación nueva.

Limitaciones y evaluación prevista: ver la sección «Conversación» del README de `rag-istae-query`.

## Integración prevista en `rag-istae-api` (propuesta; ese repositorio no se modifica desde aquí)

```python
# Esquema de una clase B/C en rag-istae-api (ChatArchitecture.stream)
anchor = next((m.content for m in history if m.role == "user"), question)
turn = sum(1 for m in history if m.role == "user") + 1
data = httpx.post(f"{RETRIEVAL_URL}/retrieve",
                  json={"question": anchor, "hierarchy": variant == "C", "turn": turn},
                  headers={"X-API-Key": RETRIEVAL_API_KEY}, timeout=60).json()
system = base + rag (+ jerarquia en C) + f"\n\n<documentos>\n{data['context']}\n</documentos>"
# El LLM recibe system, el historial reciente (MAX_HISTORY_MESSAGES) y la pregunta actual con el recordatorio.
# parse_answer y GET /units/{id} funcionan sin cambios: los unit_id son los mismos.
```

- Variables propuestas para la API: `RETRIEVAL_URL` y `RETRIEVAL_API_KEY`.
- Para mantener la comparación pareja con A/AJ, se propone usar en ellas la misma regla de conversación (un tema por conversación) y mostrar un aviso en `rag-istae-web`.
