# Guía: publicar el servicio `/retrieve` en Hugging Face Spaces

Publica el servicio de recuperación de las arquitecturas B y C como un **Space de Docker**, para que `rag-istae-api` (en Render) lo consuma por HTTPS.

## Por qué Hugging Face y con qué plan

| Necesidad del servicio | Hugging Face |
|---|---|
| Unos 3–4 GB de RAM en CPU | Hardware **CPU Basic**: 2 vCPU y 16 GB, sin costo por hora |
| Imagen Docker propia | Los Spaces de Docker requieren el plan **PRO (9 USD/mes)** en cuentas personales |
| Normativa interna privada | Visibilidad **«protegida»** (incluida en PRO): código y datos privados, aplicación accesible por URL y protegida con `X-API-Key` |
| HTTPS | Incluido: `https://<usuario>-<space>.hf.space` |

**Limitación:** en hardware gratuito, el Space **se duerme tras 48 h sin uso**. La primera consulta lo despierta: reconstruye el contenedor y carga los modelos, lo que tarda de uno a varios minutos. Durante ese tiempo la API puede devolver un error de tiempo agotado; reintentar basta. Mantenerlo siempre encendido requiere hardware pago (CPU Upgrade, 0,03 USD/h).

## Qué se publica

El repositorio del Space **no se edita a mano**: lo arma `scripts/prepare_space.py` con:

```text
<carpeta del Space>/
├── Dockerfile              ← deploy/huggingface/Dockerfile (Python 3.11, torch CPU, uid 1000, puerto 7860)
├── README.md               ← configuración del Space (sdk: docker, app_port: 7860)
├── requirements-space.txt  ← dependencias fijadas a las versiones validadas
├── src/query/              ← código del servicio
└── data/                   ← paquete de datos LB1: modelos, Qdrant, units.jsonl, catálogo, MANIFEST.json (2,36 GB)
```

Dentro de la imagen, los datos quedan en `/app/data`, no en `/data`, porque en los Spaces esa ruta está reservada para el almacenamiento persistente opcional. La configuración va por variables `RAG_*` definidas en el `Dockerfile`. La única que se define en el Space es la clave.

---

## Pasos

### 1. Cuenta y plan
1. Cree una cuenta en <https://huggingface.co> y active **PRO** en <https://huggingface.co/pricing>.
2. Cree un **token de acceso** con permiso de escritura: *Settings → Access Tokens → Create new token → Write*. No lo comparta ni lo suba a ningún repositorio.

### 2. Instalar la CLI de Hugging Face e iniciar sesión (PowerShell)

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://hf.co/cli/install.ps1 | iex"
hf auth login
```

`hf auth login` pide el token del paso 1. Verifique con `hf auth whoami`.

### 3. Armar la carpeta del Space
Requiere el paquete de datos descomprimido, la carpeta `storage` del zip de OneDrive o la salida de `build_data_bundle.py`:

```powershell
cd <ruta>\rag-istae-query
.\.venv\Scripts\Activate.ps1
python .\scripts\prepare_space.py --data E:\rag-istae-paquete\storage --out E:\rag-istae-space --force
```

El script verifica el sha256 de cada archivo contra `MANIFEST.json` antes de copiar.

### 4. (Opcional, recomendado) Probar la imagen localmente
Con Docker Desktop en ejecución:

```powershell
cd E:\rag-istae-space
docker build -t rag-istae-retrieval:lb1 .
docker run --rm -p 7860:7860 -e RAG_API_KEY=clave-local rag-istae-retrieval:lb1
```

En otra PowerShell:

```powershell
Invoke-RestMethod http://127.0.0.1:7860/health
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:7860/retrieve -Headers @{ "X-API-Key" = "clave-local" } -ContentType "application/json; charset=utf-8" -Body ([Text.Encoding]::UTF8.GetBytes('{"question":"Hice prácticas hace 7 años, ¿todavía me las pueden convalidar?","hierarchy":true}'))
```

Debe responder `status: ok`, `units: 2748`, y en C los artículos `INT15-art22`, `EXT02-art43`, `INT13-art48`, etc.

**Resultado de la prueba local (2026-11-09, Docker Desktop, 8 CPU):**

| Medición | Valor |
|---|---|
| Imagen | 6,18 GB (2,36 GB de datos + Python, torch CPU y dependencias) |
| Arranque hasta `/health` OK | 22 s |
| Memoria en uso | unos 2,0 GB (en Linux usa menos que en Windows) |
| B / C (Q27) | 5,2 s / 10,1 s; el seguimiento sale del caché en 0 s |
| Artículos de C | idénticos a la instalación local (`query_rag.py`) |
| Sin `X-API-Key` o con clave errónea | 401 · `top_k` en el request: 422 |
| Usuario y permisos | uid 1000; Qdrant crea su `.lock` en `/app/data` sin errores |
| Sin `RAG_API_KEY` | el contenedor no arranca: «Defina RAG_API_KEY…» |

En el Space (2 vCPU) los tiempos pueden ser algo mayores.

### 5. Crear el Space

```powershell
hf repos create <usuario>/rag-istae-retrieval --repo-type space --space-sdk docker --private
```

Si la opción de SDK no es aceptada por su versión de la CLI, cree el Space desde la web (<https://huggingface.co/new-space>): elija **SDK: Docker**, plantilla **Blank**, hardware **CPU Basic** y visibilidad **Private**. Después, en *Settings → Visibility*, cámbiela a **Protected**: el código y los datos quedan privados y la aplicación queda accesible por su URL.

### 6. Definir la clave del servicio
En el Space: *Settings → Variables and secrets → New secret*:
- **Name:** `RAG_API_KEY`
- **Value:** una clave larga y aleatoria. Puede generarla con:

  ```powershell
  -join ((48..57)+(65..90)+(97..122) | Get-Random -Count 40 | % {[char]$_})
  ```

Guárdela: es la misma que irá en `RAG_API_KEY` de la API (mismo nombre y valor en ambos servicios). Sin este secret el servicio no arranca, porque se niega a funcionar sin autenticación.

### 7. Subir la carpeta

```powershell
cd E:\rag-istae-space
hf upload <usuario>/rag-istae-retrieval . . --repo-type=space --commit-message "Servicio /retrieve LB-1"
```

La subida es de unos 2,4 GB; los archivos grandes se suben con el almacenamiento de archivos grandes de Hugging Face, sin configurar nada. Al terminar, el Space se construye solo: en la pestaña **Logs** se ve primero *Building* (unos 5–10 min la primera vez) y luego *Running*. La carga de modelos tarda entre 20 y 60 s más.

### 8. Verificar el Space publicado

```powershell
$url = "https://<usuario>-rag-istae-retrieval.hf.space"
$key = "<RAG_API_KEY>"
Invoke-RestMethod "$url/health"
Invoke-RestMethod -Method Post -Uri "$url/retrieve" -Headers @{ "X-API-Key" = $key } -ContentType "application/json; charset=utf-8" -Body ([Text.Encoding]::UTF8.GetBytes('{"question":"Hice prácticas hace 7 años, ¿todavía me las pueden convalidar?","hierarchy":true}'))
```

- `/health`:
  - `index_version.units_sha256` debe ser `2079998ac1e9aab5ff490160bc6989e761b5e6f5ed37df9a7beb659b50807579`;
  - `catalog_sha256` debe ser `02c3a9b0c19c7ee7472419306df65b70f89578fb4d943cb91c35d92134705530`.
- **Sin la cabecera `X-API-Key`:** `/retrieve` debe responder 401.
- **Visibilidad protegida:** `/health` puede responder una página de autenticación de Hugging Face. Lo importante es que `/retrieve` funcione con la clave desde fuera.

### 9. Conectar la API (Render)
En el servicio de `rag-istae-api` en Render, agregue estas variables de entorno, según [`guia_integracion_api.md`](guia_integracion_api.md):

```env
RETRIEVAL_URL=https://<usuario>-rag-istae-retrieval.hf.space
RAG_API_KEY=<la misma clave del secret RAG_API_KEY del Space>
RETRIEVAL_TIMEOUT_SECONDS=90
ENABLE_LONG_CONTEXT=false
```

---

## Actualizar el Space
- **Cambió el código** de `rag-istae-query`: repita los pasos 3 y 7. Los datos no cambian, pero el script los vuelve a copiar; la subida solo envía los archivos modificados.
- **Nuevo paquete de datos:** regenérelo con `build_data_bundle.py` y repita los pasos 3, 7 y 8, y compare el nuevo `units_sha256`.
- **Logs del servicio:** el registro de llamadas va a `/tmp/rag-logs` dentro del contenedor y se pierde al reiniciarse el Space. Los mensajes de arranque y de errores se ven en la pestaña **Logs** del Space.

## Problemas frecuentes

| Síntoma | Causa | Solución |
|---|---|---|
| El Space queda en *Runtime error* con `Defina RAG_API_KEY` | Falta el secret | Paso 6; luego *Settings → Restart Space* |
| *Runtime error* con `FileNotFoundError` sobre `/app/data/...` | La carpeta subida no tenía `data/` completa | Repita los pasos 3 y 7 sin modificar la carpeta |
| La API recibe tiempo agotado en la primera consulta del día | El Space estaba dormido | Reintentar; para evitarlo, hardware pago |
| 401 desde la API | `RAG_API_KEY` de Render distinta del secret `RAG_API_KEY` del Space | Igualar ambas |
| La construcción falla al instalar `torch` | Versión no disponible en el índice CPU de PyTorch | Ajustar `torch==` en `deploy/huggingface/Dockerfile` a una versión disponible y regenerar |
