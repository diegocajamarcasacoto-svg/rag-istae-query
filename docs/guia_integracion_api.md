# Guía de integración: arquitecturas B y C en `rag-istae-api` y `rag-istae-web`

**Destinatario:** el responsable de `rag-istae-api` y `rag-istae-web`. Desde `rag-istae-query` esos repositorios no se modifican: esta guía describe los cambios para que los implemente su responsable.

**Estado:** propuesta, versión 1 (2026-11-08). El código de la clase B/C ([`integracion_api/rag.py`](integracion_api/rag.py)) se probó contra el código real de la API con un LLM simulado y una respuesta real de `/retrieve`. Los cambios en `main.py`, `config.py` y en la web son fragmentos de referencia.

---

## 1. Decisiones que guían esta integración

1. **B y C son independientes de A y AJ.** El contexto largo (A/AJ) queda como antecedente histórico: no se publica, ni se carga por defecto. B y C no usan el corpus completo de la API (`Library`) para nada.
2. **El contexto lo arma `rag-istae-query`.** La API llama a `POST /retrieve`, que devuelve los artículos pertinentes y el contexto ya presentado en el formato de la API (`=== [DOC] Nombre ===` y `[unit_id] texto`). La API no recupera ni corta artículos.
3. **Los artículos salen de la respuesta de `/retrieve`.** Es la fuente de los artículos para todo lo que hace la API en B/C:
   - el **contexto** del LLM: `context`;
   - la **verificación de citas**: `units[].text`. Se verifica contra lo que realmente se envió, así que una cita a un artículo que el modelo no recibió queda como desconocida;
   - el **panel de fuentes y la ventana del artículo** en la web: la API agrega `label`, `title` y `text` de `units[]` a cada fuente del evento `answer`, y la web los muestra **sin llamar a `GET /units/{id}`**.
4. **Sin pantalla de Documentos por ahora.** `GET /documents` no se usa en B/C y el menú se oculta en la web.
5. **Conversación de un solo tema.** La recuperación usa la **primera pregunta del usuario** de la conversación. Los seguimientos se responden con los mismos artículos (el servicio los entrega desde su caché), y el historial reciente va al LLM como en A/AJ. Para otro tema, el usuario abre otra conversación.
6. **Parámetros de recuperación fijos.** La línea base **LB-1** la fija el servicio; la API solo elige B o C (`hierarchy`).

## 2. Flujo

```text
rag-istae-web ── POST /chat {question, history, architecture: "B"|"C"} ──► rag-istae-api
                                                                              │ RagAssistant.stream()
                                                                              │ 1. consulta = 1.ª pregunta del usuario en history (o question)
                                                                              ▼
                                      rag-istae-query  POST /retrieve {question, hierarchy, turn}
                                                       ◄── {units[], context, index_version, baseline, …}
                                                                              │ 2. LLM (OpenAICompatibleClient, el mismo de hoy):
                                                                              │    system = base_es + rag_es (+ jerarquia_rag_es en C)
                                                                              │             + <documentos>{context}</documentos>
                                                                              │    historial reciente + <pregunta>{question}</pregunta> + recordatorio
                                                                              │ 3. parse_answer(texto, artículos de /retrieve)
rag-istae-web ◄── SSE: delta… / answer {body, sources[+label,title,text], …} ─┘
   └─ clic en una cita → muestra source.text (sin GET /units)
```

## 3. Cambios en `rag-istae-api`

### 3.1 Prompts (`prompts/`)

| Archivo | Acción | Origen |
|---|---|---|
| `base_es.txt`, `recordatorio_es.txt` | **sin cambios** (`rag-istae-query` usa copias idénticas) | — |
| `rag_es.txt` | **agregar** | `rag-istae-query/config/prompts/modular/rag_es.txt` |
| `jerarquia_rag_es.txt` | **agregar** | `rag-istae-query/config/prompts/modular/jerarquia_es.txt` (copiar con este nombre) |
| `jerarquia_es.txt` | **sin cambios**: queda para AJ (histórico) | — |

Qué hace cada prompt nuevo:
- **`rag_es.txt`:**
  - el contexto es una selección de artículos para la primera pregunta de la conversación, no la normativa completa;
  - la frase de abstención («No encontré en la normativa disponible») se reserva para cuando ningún artículo es pertinente;
  - si un seguimiento cambia de tema, el modelo se abstiene y sugiere abrir otra conversación.
- **`jerarquia_rag_es.txt`** (solo C): aplica la escala y la prevalencia solo con los documentos proporcionados. **No invoca el art. 425 de la Constitución**, que no está en el corpus y no se puede citar ni verificar. Se usa otro nombre para no alterar el `jerarquia_es.txt` de AJ.

Composición: **B = base + rag**; **C = base + rag + jerarquía RAG**. El recordatorio va al final del mensaje del usuario, como en A/AJ.

### 3.2 Nuevo módulo `app/architectures/rag.py`

Copiar [`integracion_api/rag.py`](integracion_api/rag.py). Contiene:
- **`RetrievalClient`**: cliente `httpx` de `POST /retrieve`. Envía la cabecera `X-API-Key` y convierte los tiempos agotados, los errores de conexión y las respuestas con error en `RetrievalError` (con `status_code`).
- **`load_rag_prompts(prompts_dir, variant)`**: B → `base_es` + `rag_es`; C → además `jerarquia_rag_es`.
- **`conversation_anchor(question, history)`**: devuelve la primera pregunta del usuario (o `question`) y el número de turno.
- **`RagAssistant(variant, retrieval, client, prompts_dir, max_history_messages, max_output_tokens)`**: cumple `ChatArchitecture.stream()`:
  1. llama a `/retrieve` con la consulta ancla, `hierarchy = (variant == "C")` y `turn`;
  2. arma `LLMRequest(system, context=data["context"], question, history reciente, reminder)` y reenvía los fragmentos del LLM;
  3. arma el `Answer`:
     - verifica las citas con `parse_answer(texto, {unit_id: (document_id, text)})` usando **los artículos de `/retrieve`**;
     - arma las fuentes con `sources_with_text()`, que es `sources_by_document` más `label`, `title`, `text`, `partial` y `origin`;
     - calcula `corpus_version` a partir de la variante, los prompts, la línea base, el `index_version` del servicio y el modelo.

`Answer` no cambia de forma: las fuentes llevan campos extra, que el frontend actual ignora hasta que se apliquen los cambios de la sección 4.

### 3.3 Configuración (`app/config.py`)

```python
class Settings(BaseSettings):
    ...
    # Servicio de recuperación (rag-istae-query) para B y C.
    retrieval_url: str = "http://127.0.0.1:8100"
    retrieval_api_key: SecretStr
    retrieval_timeout_seconds: float = 90   # C tarda hasta ~12 s en CPU; más si el servicio estaba dormido
    # A/AJ (contexto largo) quedan como históricos: no se cargan salvo que se pida.
    enable_long_context: bool = False
```

| Variable de entorno | Ejemplo | Nota |
|---|---|---|
| `RETRIEVAL_URL` | `http://127.0.0.1:8100` (local) | URL pública del servicio cuando se publique (hosting pendiente) |
| `RETRIEVAL_API_KEY` | (secreto) | la misma que `RAG_API_KEY` del servicio |
| `RETRIEVAL_TIMEOUT_SECONDS` | `90` | |
| `ENABLE_LONG_CONTEXT` | `false` | `true` solo para reproducir A/AJ |
| `LLM_*`, `MAX_HISTORY_MESSAGES`, `CORS_ORIGINS` | sin cambios | para la evaluación de la tesis, `LLM_FALLBACK_MODELS=[]` |

### 3.4 Arranque y endpoints (`app/main.py`)

```python
from app.architectures.rag import RagAssistant, RetrievalClient, RetrievalError

def build_architectures(settings: Settings) -> tuple[Library | None, dict[str, ChatArchitecture]]:
    client = OpenAICompatibleClient(...)          # sin cambios
    options = {"max_history_messages": settings.max_history_messages, "max_output_tokens": settings.llm_max_output_tokens}
    retrieval = RetrievalClient(
        base_url=settings.retrieval_url,
        api_key=settings.retrieval_api_key.get_secret_value(),
        timeout_seconds=settings.retrieval_timeout_seconds,
    )
    architectures: dict[str, ChatArchitecture] = {
        "B": RagAssistant("B", retrieval, client, settings.prompts_dir, **options),
        "C": RagAssistant("C", retrieval, client, settings.prompts_dir, **options),
    }
    library = None
    if settings.enable_long_context:              # histórico: carga las 46 normas solo si se pide
        catalog = load_catalog(settings.data_dir / "catalogo_normativo.json")
        library = Library(load_corpus(settings.data_dir, catalog))
        architectures["A"] = LongContextAssistant(library, catalog, "A", client, settings.prompts_dir, **options)
        architectures["AJ"] = LongContextAssistant(library, catalog, "AJ", client, settings.prompts_dir, **options)
    return library, architectures
```

- **Cierre:** el `lifespan` cierra `architecture.client`. Hay que cerrar también `retrieval` (es compartido; basta cerrarlo una vez).
- **`GET /health`:** sin `Library`, `documents` y `units` pasan a `null`. Conviene agregar `"retrieval_url"`.
- **`GET /units/{id}` y `GET /documents`:**
  - responden **404** si `library is None`: B/C no los necesitan;
  - con `ENABLE_LONG_CONTEXT=true` siguen sirviendo los chats históricos de A/AJ;
  - opcionalmente, `/units` puede mantenerse como respaldo para chats guardados antes del cambio.
- **`GET /architectures`:** sin cambios de código. `available` ya refleja lo registrado, así que A/AJ aparecen como no disponibles. Conviene actualizar sus descripciones en `ARCHITECTURES` («histórico»).
- **`POST /chat`:** agregar un manejo propio de `RetrievalError`, antes del `except Exception`:

```python
            except RetrievalError as exc:
                logger.error("Error del servicio de recuperación (%s): %s", exc.status_code, exc)
                yield _sse("error", {
                    "message": "El servicio de búsqueda de normativa no respondió. Intente nuevamente en unos minutos.",
                    "status": exc.status_code,
                })
```

- **`app/schemas.py`:** `ChatRequest.architecture: str = "B"` (hoy `"A"`).

### 3.5 Pruebas sugeridas (pytest, sin red)

- **`RetrievalClient` con `httpx.MockTransport`:**
  - el cuerpo enviado (`question`, `hierarchy`, `turn`) y la cabecera `X-API-Key`;
  - los errores 401, 503 y el tiempo agotado se convierten en `RetrievalError`.
- **`RagAssistant` con un LLM simulado:**
  - la consulta es la primera pregunta del historial y `turn` es el número de pregunta;
  - C pide `hierarchy=true` y su prompt incluye `jerarquia_rag_es`;
  - el contexto va dentro de `<documentos>`;
  - una cita a un artículo enviado queda verificada, y una cita a un artículo no enviado va a `unknown_citations`;
  - las fuentes incluyen `text`, `label` y `title`.
- **`/chat` (B y C):** emite `delta` y `answer`, y emite `error` si falla `/retrieve`.
- **Sin `ENABLE_LONG_CONTEXT`:** no se lee `data/`; `/units` y `/documents` responden 404, y `/architectures` marca A/AJ como no disponibles.

El script de validación usado para esta guía cubre los dos primeros grupos y sirve de punto de partida.

## 4. Cambios en `rag-istae-web`

### 4.1 Tipos (`src/lib/types.ts`)

```ts
export type SourceUnit = {
  unit_id: string;
  quote: string | null;
  quote_verified: boolean | null;
  // B/C: el artículo tal como se envió al modelo (viene en el evento "answer").
  label?: string;
  title?: string;
  text?: string;
  partial?: boolean;                       // true: se envió un fragmento del artículo ("[…]")
  origin?: "base" | "hierarchy";           // C: "hierarchy" = agregado por jerarquía normativa
};

export const DEFAULT_ARCHITECTURE = "B";   // hoy "A", que deja de estar disponible
```

Las fuentes ya se guardan con el chat en `localStorage` (`chatStorage.ts`), así que el texto queda disponible al reabrir un chat sin pedir nada al backend.

### 4.2 Ventana del artículo (`ArticleModal.tsx` y `AssistantMessage.tsx`)

`AssistantMessage` busca la fuente citada en `message.sources` y se la pasa al modal. El modal usa ese texto y solo llama a `GET /units/{id}` como respaldo, para chats viejos sin `text`:

```tsx
// AssistantMessage.tsx
const sourceFor = (unitId: string) =>
  message.sources?.flatMap((document) =>
    document.units.map((unit) => ({ ...unit, document_id: document.document_id, document_name: document.document_name })),
  ).find((unit) => unit.unit_id === unitId);
...
<ArticleModal key={openUnitId} unitId={openUnitId} source={sourceFor(openUnitId)} number={...} onClose={...} />

// ArticleModal.tsx
type ArticleModalProps = { unitId: string; source?: SourceUnit & { document_id: string; document_name: string }; number?: number; onClose: () => void };

export function ArticleModal({ unitId, source, number, onClose }: ArticleModalProps) {
  const embedded = source?.text ? source : undefined;
  // Solo se pide al backend si la fuente no trae el texto (chats anteriores al cambio).
  const { data: fetched, error, isLoading } = useSWRImmutable(embedded ? null : unitId, fetchUnit);
  const unit = embedded ?? fetched;
  ...
  {embedded?.partial && <p className="…">Fragmento del artículo enviado al modelo.</p>}
  {embedded?.origin === "hierarchy" && <p className="…">Incluido por jerarquía normativa.</p>}
}
```

### 4.3 Navegación y avisos
- **Ocultar «Documentos»** en `Sidebar.tsx` y `AppShell.tsx` (enlaces a `/documentos`), porque B/C no ofrecen `GET /documents`.
- **Aviso de un tema por conversación**, por ejemplo en `WelcomeMessage` o junto a `ChatInput`: «Cada conversación trata un solo tema. Para consultar otro tema, inicie una conversación nueva.»
- **Selector de arquitectura:** sin cambios, porque ya muestra como «próximamente» las no disponibles. Con A/AJ desactivadas quedan B y C.

## 5. Puesta en marcha local (las tres piezas)

1. **Servicio de recuperación:** según [`guia_ambiente_local.md`](guia_ambiente_local.md), pasos 1–9, con clave: `$env:RAG_API_KEY = "clave-local"` y `python .\scripts\serve_retrieval.py` (puerto 8100).
2. **API:** en `.env`:

   ```env
   LLM_API_KEY=<clave de OpenRouter>
   RETRIEVAL_URL=http://127.0.0.1:8100
   RETRIEVAL_API_KEY=clave-local
   ENABLE_LONG_CONTEXT=false
   ```

   Luego, `uvicorn app.main:create_app --factory --reload` (puerto 8000).
3. **Web:** `NEXT_PUBLIC_API_URL=http://localhost:8000`, y luego `npm run dev` (puerto 3000).

## 6. Verificación de punta a punta

| Prueba | Resultado esperado |
|---|---|
| `GET http://localhost:8000/architectures` | B y C `available: true`; A y AJ `false` |
| Chat nuevo con C: «Hice prácticas hace 7 años, ¿todavía me las pueden convalidar?» | Respuesta con citas a `INT15-art22`, `EXT02-art43` y/o `INT13-art48`. El panel de fuentes las muestra verificadas. Al hacer clic, se abre el texto **sin** una llamada a `/units` (verificar en la pestaña Red del navegador) |
| Seguimiento en el mismo chat: «¿Y qué documentos necesito?» | Responde con el trámite de `INT15-art22`. En `logs/service/runs_index.csv` del servicio, la llamada aparece con el mismo contexto, desde el caché |
| Seguimiento de otro tema: «¿Cuál es la nota mínima para aprobar una materia?» | Abstención y sugerencia de abrir otra conversación |
| Detener el servicio de recuperación y preguntar | Evento `error`: «El servicio de búsqueda de normativa no respondió…» |
| Reabrir el chat desde Recientes y hacer clic en una cita | Abre el artículo desde `localStorage`, sin backend |

## 7. Limitaciones y riesgos conocidos
- **Latencia en CPU:** B tarda unos 3–5 s y C unos 7–12 s de recuperación, más el tiempo del LLM. Si el servicio se publica en un hosting que se duerme, la primera consulta tarda más: el tiempo de espera de 90 s lo cubre.
- **Artículos parciales:** un artículo de más de 1.500 tokens se envía recortado (`partial: true`, con `[…]`). La ventana muestra ese recorte, que es lo que vio el modelo.
- **59 artículos que en el texto original se separan, pero que nuestra división deja dentro del anterior** (ver `rag_istae_ingestion`, `units_api_comparison.json`). Como verificación y ventana usan el mismo texto que vio el modelo, son consistentes. La única imprecisión es la etiqueta: el art. 140 del RRA aparece dentro de `[EXT02-art139]`.
- **Contexto acotado por conversación:** un seguimiento que necesite artículos que no estaban en la primera recuperación termina en abstención. Está documentado como limitación de la línea base.
- **Modelos `:free` de OpenRouter:** tienen límites de uso y pueden registrar los prompts, que incluyen texto de la normativa interna.

## 8. Referencias
- Contrato de `/retrieve`: [`contrato_retrieve.md`](contrato_retrieve.md).
- Instalación del servicio: [`guia_ambiente_local.md`](guia_ambiente_local.md).
- Línea base y su justificación: README de `rag-istae-query` («Línea base B/C») y `Justificacion_Linea_Base.md`.
