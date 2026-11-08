# Guía: levantar rag-istae-query en un equipo local (Windows)

Esta guía instala, desde cero, el pipeline de recuperación de las arquitecturas **B** (RAG híbrido) y **C** (RAG jerárquico) y el servicio `POST /retrieve`. Al terminar podrá:

- consultar la normativa del ISTAE por línea de comandos, con o sin LLM;
- levantar el servicio `/retrieve` y probarlo desde PowerShell o el navegador.

**No hay que reindexar nada.** Los modelos y el índice validado (colección `istae_chunks_e5_base`, 9.623 fragmentos) vienen en un paquete de datos compartido por OneDrive. Volver a generar el índice cambiaría los identificadores y dejaría de coincidir con el Gold Standard.

Tiempo estimado: unos 45 minutos, la mayor parte en descargas.

---

## 0. Requisitos

| Requisito | Detalle |
|---|---|
| Sistema | Windows 10 u 11 de 64 bits |
| Memoria | 8 GB de RAM o más (el servicio usa unos 3 GB, con picos de casi 4 GB) |
| Disco | unos 8 GB libres: zip (1,7 GB) + datos (2,4 GB) + entorno de Python (unos 2 GB) |
| Python | **3.11** de 64 bits (no 3.12 ni 3.13) |
| Git | cualquier versión reciente |
| Acceso | al repositorio `rag-istae-query` en GitHub y al enlace de OneDrive del paquete de datos |
| Internet | solo para instalar dependencias y, si se usa, para generar respuestas con el LLM |

Todos los comandos son de **PowerShell**. Las rutas de ejemplo usan la carpeta `C:\RAG-ISTAE`. Si usa otra, cámbiela en todos los pasos y en el paso 6.

---

## 1. Instalar y verificar Python 3.11 y Git

1. Descargue Python 3.11 (64 bits) desde <https://www.python.org/downloads/windows/>. En el instalador, marque **«Add python.exe to PATH»** y deje marcado **«py launcher»**.
2. Instale Git desde <https://git-scm.com/download/win>.
3. Abra **una PowerShell nueva** y verifique:

```powershell
py -3.11 --version
git --version
```

Debe ver `Python 3.11.x` y `git version …`. Si `py -3.11` falla, Python 3.11 no quedó instalado o falta el launcher `py`.

---

## 2. Clonar el repositorio

```powershell
New-Item -ItemType Directory -Force C:\RAG-ISTAE | Out-Null
cd C:\RAG-ISTAE
git clone https://github.com/diegocajamarcasacoto-svg/rag-istae-query.git
```

Resultado: `C:\RAG-ISTAE\rag-istae-query\`.

---

## 3. Descargar y descomprimir el paquete de datos

1. Descargue de OneDrive el archivo **`rag-istae-storage-LB1.zip`** (1,7 GB), con el enlace que le compartió el responsable del proyecto. Guárdelo, por ejemplo, en `C:\RAG-ISTAE\`.
2. Descomprímalo en `C:\RAG-ISTAE`. El zip ya contiene la carpeta `storage`:

```powershell
Expand-Archive -Path C:\RAG-ISTAE\rag-istae-storage-LB1.zip -DestinationPath C:\RAG-ISTAE
```

`Expand-Archive` tarda varios minutos y no muestra avance. Debe quedar esta estructura:

```text
C:\RAG-ISTAE\storage\
├── catalog\catalogo_normativo.json
├── models\bge-reranker-base\...
├── models\multilingual-e5-base\...
├── outputs\units.jsonl
├── vector_store\qdrant\...
├── LEEME.txt
└── MANIFEST.json
```

---

## 4. Verificar la integridad del paquete

Este comando compara el sha256 de cada archivo con `MANIFEST.json`. Tarda uno o dos minutos:

```powershell
$storage = "C:\RAG-ISTAE\storage"
$manifest = Get-Content "$storage\MANIFEST.json" -Raw | ConvertFrom-Json
$errores = $manifest.files | Where-Object { (Get-FileHash (Join-Path $storage $_.path) -Algorithm SHA256).Hash.ToLower() -ne $_.sha256 }
if ($errores) { "ARCHIVOS CON ERROR:"; $errores.path } else { "Paquete íntegro: $($manifest.files.Count) archivos, versión $($manifest.version)" }
```

Debe ver `Paquete íntegro: 20 archivos, versión LB1`. Si aparece algún archivo con error, vuelva a descargar el zip.

---

## 5. Crear el entorno de Python e instalar las dependencias

```powershell
cd C:\RAG-ISTAE\rag-istae-query
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

- La instalación tarda entre 10 y 20 minutos (torch, sentence-transformers, qdrant-client, FastAPI…). En Windows, el torch que se instala es la versión para CPU; no hace falta GPU.
- Si `Activate.ps1` da el error *«la ejecución de scripts está deshabilitada en este sistema»*, ejecute una vez lo siguiente y vuelva a activar:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Con el entorno activo, el prompt empieza con `(.venv)`. **Cada vez que abra una PowerShell nueva** para trabajar con el proyecto, active el entorno y defina `PYTHONPATH`:

```powershell
cd C:\RAG-ISTAE\rag-istae-query
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "src;scripts"
```

---

## 6. Configurar las rutas a los datos

```powershell
Copy-Item config\local.example.json config\local.json
Get-Content config\local.json
```

El ejemplo ya apunta a `C:/RAG-ISTAE/storage/...`, así que **si usó `C:\RAG-ISTAE` no hay que editar nada**. Si descomprimió los datos en otra carpeta, abra `config\local.json` (por ejemplo con `notepad config\local.json`) y reemplace `C:/RAG-ISTAE/storage` por su ruta, **con barras `/`** y absoluta, en las cinco rutas: `e5_model_path`, `reranker_model_path`, `qdrant_path`, `units_path` y `catalog_path`.

`config\local.json` no se sube a git: cada equipo tiene el suyo.

---

## 7. Comprobar la instalación con las pruebas unitarias

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

Las pruebas no usan los modelos ni Qdrant. Al final debe aparecer `OK`, con 124 pruebas a la fecha. La primera vez puede tardar un minuto en importar torch.

---

## 8. Primera consulta por línea de comandos (sin LLM)

`--no-generation` ejecuta toda la recuperación sin llamar a ningún modelo de lenguaje ni necesitar claves. `--show-context` imprime el contexto que recibiría el LLM.

```powershell
# Arquitectura B (RAG híbrido)
python .\scripts\query_rag.py "¿Qué requisitos necesito cumplir para que me declaren egresado?" --no-generation --show-context

# Arquitectura C (RAG jerárquico: agrega artículos de normas de mayor rango)
python .\scripts\query_rag.py "Hice prácticas hace 7 años, ¿todavía me las pueden convalidar?" --retrieval-mode hierarchical --no-generation --show-context
```

- **Qué debe ver:** la lista de artículos recuperados (`chunk=INT08-art12; rol=seed`, …) y los tiempos de cada etapa.
- **Primera ejecución:** la carga de modelos (bloque «Carga») tarda entre 15 y 30 s, y la consulta (bloque «Ejecución») entre 3 y 12 s en CPU.
- **Resultado de referencia para la segunda pregunta** (en C):
  - artículos base: `INT15-art22`, `INT15-art25`, `INT13-art50`, `EXT02-art43` e `INT13-art48`;
  - agregados por jerarquía: `EXT01-disp-general-cuarta-2` y `EXT09-sec189`.

Cada ejecución queda registrada en `logs\runs_index.csv`, que se abre en Excel.

---

## 9. Levantar el servicio `/retrieve` y probarlo

Se necesitan **dos PowerShell**: una mantiene el servicio encendido y en la otra se hacen las consultas. En ambas, active el entorno y defina `PYTHONPATH` como en el paso 5.

**Mientras el servicio está activo, no se pueden ejecutar los scripts del paso 8**, porque Qdrant local admite un solo proceso. Detenga el servicio con `Ctrl+C` antes de volver a usarlos.

### 9.1 Terminal 1: arrancar el servicio

```powershell
$env:RAG_AUTH_DISABLED = "true"      # solo en el equipo local: no pide clave
python .\scripts\serve_retrieval.py
```

Está listo cuando aparece `Uvicorn running on http://127.0.0.1:8100`, después de 15 a 30 s.

Para probarlo con clave, como lo usará la API, defina en su lugar `$env:RAG_API_KEY = "una-clave-local"` y agregue a cada consulta la cabecera `-Headers @{ "X-API-Key" = "una-clave-local" }`.

### 9.2 Terminal 2: consultar

```powershell
# Estado: debe responder status = ok, baseline = LB-1, units = 2748, documents = 46
Invoke-RestMethod http://127.0.0.1:8100/health

# Función auxiliar: envía el JSON en UTF-8 para que no se rompan las tildes en PowerShell 5.1
function Retrieve($body) { Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8100/retrieve -ContentType "application/json; charset=utf-8" -Body ([Text.Encoding]::UTF8.GetBytes(($body | ConvertTo-Json -Depth 5))) }

# Arquitectura B
$b = Retrieve @{ question = "Hice prácticas hace 7 años, ¿todavía me las pueden convalidar?"; question_id = "Q27"; turn = 1 }
$b.units | Format-Table unit_id, origin, rank, score

# Arquitectura C y el contexto que recibiría el LLM
$c = Retrieve @{ question = "Hice prácticas hace 7 años, ¿todavía me las pueden convalidar?"; hierarchy = $true; turn = 1 }
$c.units | Format-Table unit_id, origin, rank, score
$c.context

# Turno de seguimiento: misma pregunta, sale del caché (cached = True, total cercano a 0 ms)
$c2 = Retrieve @{ question = "Hice prácticas hace 7 años, ¿todavía me las pueden convalidar?"; hierarchy = $true; turn = 2 }
$c2.cached; $c2.timings_ms

# Sin rerank, limitado a una norma y con detalle de tiempos
$d = Retrieve @{ question = "Hice prácticas hace 7 años, ¿todavía me las pueden convalidar?"; rerank = $false; document_ids = @("INT15"); debug = $true }
$d.units | Format-Table unit_id, score; $d.debug.timings_ms.execution

# Los parámetros de la línea base no se aceptan: debe responder un error 422
Retrieve @{ question = "prueba"; top_k = 10 }
```

También puede probarlo desde el navegador en <http://127.0.0.1:8100/docs>: despliegue `POST /retrieve`, pulse **Try it out**, edite el JSON y pulse **Execute**.

Las llamadas quedan registradas en `logs\service\runs_index.csv`. El contrato completo (campos, errores y variables de entorno) está en [`contrato_retrieve.md`](contrato_retrieve.md).

---

## 10. Opcional: generar respuestas con el LLM (OpenRouter)

Usa el mismo modelo que `rag-istae-api` (`nvidia/nemotron-3-ultra-550b-a55b:free`). Requiere una cuenta y una clave propias de <https://openrouter.ai>.

Antes de seguir, dos advertencias:
- Los endpoints `:free` tienen límites diarios y pueden registrar los prompts, que incluyen texto de la normativa interna del ISTAE.
- **Detenga el servicio del paso 9** antes de seguir, porque este comando también usa Qdrant.

```powershell
Copy-Item config\generation.openrouter.example.json config\generation.openrouter.json
$env:OPENROUTER_API_KEY = "<su clave de OpenRouter>"

python .\scripts\query_rag.py "¿Qué requisitos necesito cumplir para que me declaren egresado?" --generation-config .\config\generation.openrouter.json
python .\scripts\query_rag.py "Hice prácticas hace 7 años, ¿todavía me las pueden convalidar?" --retrieval-mode hierarchical --generation-config .\config\generation.openrouter.json
```

La clave no se guarda en ningún archivo: los registros solo anotan el nombre de la variable.

---

## 11. Opcional: el chat completo (`rag-istae-api` y `rag-istae-web`)

Son repositorios aparte, cada uno con su propio README:
- `rag-istae-api`: Python 3.11, `pip install -r requirements-dev.txt`, `uvicorn app.main:create_app --factory`, y `LLM_API_KEY` en un `.env`;
- `rag-istae-web`: Node.js 20 o superior, `npm install` y `npm run dev`, con `NEXT_PUBLIC_API_URL=http://localhost:8000`.

**A la fecha, la API solo ofrece las arquitecturas A y AJ.** El chat todavía no usa este servicio; B y C se prueban directamente contra `/retrieve` (paso 9) o con `query_rag.py` (pasos 8 y 10).

---

## 12. Actualizar

- **Código:** con el servicio detenido, `git pull` en `C:\RAG-ISTAE\rag-istae-query`. Si cambió `requirements.txt`, ejecute de nuevo `pip install -r requirements.txt` con el entorno activo.
- **Datos:** si se publica un paquete nuevo (`rag-istae-storage-<versión>.zip`), detenga el servicio, borre `C:\RAG-ISTAE\storage`, descomprima el nuevo y repita el paso 4.
  - Para verificar que corre la versión correcta, compare los valores `units_sha256` y `catalog_sha256` de `LEEME.txt` con el campo `index_version` de `GET /health`.

---

## 13. Problemas frecuentes

| Síntoma | Causa | Solución |
|---|---|---|
| `py -3.11` no se reconoce | Python 3.11 no instalado o sin launcher `py` | Reinstale Python 3.11 con «py launcher» y «Add to PATH» |
| `Activate.ps1` «no se puede cargar… ejecución de scripts deshabilitada» | Política de ejecución de PowerShell | `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` |
| `ModuleNotFoundError: No module named 'query'` | Falta `PYTHONPATH` en esa terminal | `$env:PYTHONPATH = "src;scripts"` |
| `FileNotFoundError: No existe el modelo E5 local` o `No existe la ruta Qdrant` | Rutas de `config\local.json` incorrectas | Revise el paso 6: rutas absolutas, con `/`, hacia `storage` |
| `Storage folder … is already accessed by another instance of Qdrant client` | Hay otro proceso usando Qdrant (servicio o script) | Cierre el servicio (`Ctrl+C`) o el otro script, y reintente |
| `/health` responde 503 | El servicio no pudo cargar los datos | Lea el error en la terminal 1; casi siempre son las rutas o el bloqueo de Qdrant |
| `[WinError 10048]` al arrancar el servicio | El puerto 8100 está ocupado | `python .\scripts\serve_retrieval.py --port 8101` y use ese puerto en las consultas |
| Las tildes llegan rotas o da 422 con texto raro | PowerShell 5.1 envía el cuerpo sin UTF-8 | Use la función `Retrieve` del paso 9.2 |
| La primera importación tarda varios minutos (`import torch`) | El antivirus analiza las bibliotecas de torch | Espere a que termine. Si se repite siempre, excluya la carpeta `.venv` del análisis en tiempo real |
| El equipo se queda sin memoria | Menos de 8 GB de RAM libres | Cierre otras aplicaciones; el servicio necesita unos 3–4 GB |
| `GenerationProviderError: HTTP 429` | Límite del modelo gratuito de OpenRouter | Espere y reintente, o use otra clave o modelo |

## 14. Lo que no se debe hacer

- No ejecutar la ingestión (`rag_istae_ingestion`) ni reindexar: el índice del paquete es el validado y coincide con el Gold Standard.
- No modificar archivos dentro de `C:\RAG-ISTAE\storage`.
- No ejecutar el servicio y los scripts de consulta al mismo tiempo sobre el mismo `storage`.
- No subir `config\local.json`, `config\generation*.json` con datos propios, ni claves, al repositorio.
