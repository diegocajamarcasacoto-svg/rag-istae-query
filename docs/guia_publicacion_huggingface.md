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

El **repositorio de un Space admite como máximo 1 GB** (la primera subida, del 2026-11-15, falló con «Storage limit reached for this space (Max: 1 GB)»). El paquete de datos LB1 pesa 2,36 GB, así que se publica en dos repositorios:

| Repositorio | Tipo y visibilidad | Contenido | Tamaño |
|---|---|---|---|
| `<usuario>/rag-istae-data-lb1` | **Dataset, privado** | Paquete de datos LB1 tal cual: modelos, Qdrant, `units.jsonl`, catálogo, `MANIFEST.json` | 2,36 GB |
| `<usuario>/rag-istae-retrieval` | Space Docker, **protegido** | Código y receta de la imagen | < 1 MB |

No se pueden descargar los modelos de sus repositorios públicos: `bge-reranker-base` coincide con `BAAI/bge-reranker-base`, pero el `model.safetensors` y el `tokenizer.json` de E5 locales **no** coinciden con los de `intfloat/multilingual-e5-base` (comprobado por sha256), y el índice se construyó con la copia local.

El repositorio del Space **no se edita a mano**: lo arma `scripts/prepare_space.py` con:

```text
<carpeta del Space>/
├── Dockerfile              ← deploy/huggingface/Dockerfile, con el repositorio de datos, el commit y el sha256 de MANIFEST.json fijados
├── README.md               ← configuración del Space (sdk: docker, app_port: 7860, license: other)
├── requirements-space.txt  ← dependencias fijadas a las versiones validadas
├── fetch_data.py           ← descarga y verificación del paquete de datos
└── src/query/              ← código del servicio
```

Al construir la imagen, `fetch_data.py` descarga el repositorio de datos **fijado a un commit** y verifica el sha256 de `MANIFEST.json` y de cada archivo. Si algo no coincide, la construcción falla. Para descargar, usa el secret `HF_TOKEN` del Space, de solo lectura y montado solo en ese paso. Los datos quedan en `/app/data`, no en `/data`, porque esa ruta está reservada para el almacenamiento persistente. La configuración va por variables `RAG_*` definidas en el `Dockerfile`.

---

## Pasos

### 1. Cuenta y plan
1. Cree una cuenta en <https://huggingface.co> y active **PRO** en <https://huggingface.co/pricing>.
2. La CLI `hf` viene con `huggingface_hub`, que ya está en el `.venv` de `rag-istae-query`. Si no, instálela con `powershell -ExecutionPolicy ByPass -c "irm https://hf.co/cli/install.ps1 | iex"`.

### 2. Crear los dos repositorios (web)
1. **Datos:** en <https://huggingface.co/new-dataset> cree `rag-istae-data-lb1`, con visibilidad **Private**.
2. **Space:** en <https://huggingface.co/new-space> cree `rag-istae-retrieval`:
   - **SDK** Docker, plantilla Blank, **CPU Basic**, license *other* y visibilidad **Private**;
   - sin Storage Bucket ni Dev Mode;
   - después, en *Settings → Visibility*, cámbiela a **Protected**: el código queda privado y la aplicación accesible por su URL.

No siga la página de bienvenida del Space (`git clone`, `app.py` de ejemplo): la carpeta la arma `prepare_space.py`.

### 3. Tokens (*Settings → Access Tokens → Fine-grained*)
- **Token de subida**, para la CLI: lectura y **escritura** solo sobre `datasets/<usuario>/rag-istae-data-lb1` y `spaces/<usuario>/rag-istae-retrieval`. Inicie sesión con él:

  ```powershell
  cd <ruta>\rag-istae-query
  .\.venv\Scripts\Activate.ps1
  hf auth login      # pegue el token; a «Add token as git credential?» responda n
  hf auth whoami
  ```
- **Token de construcción**, para el secret `HF_TOKEN` del Space: **solo lectura** sobre `datasets/<usuario>/rag-istae-data-lb1`.

Ninguno se comparte ni se sube a un repositorio. Se pueden revocar cuando ya no se usen; el de construcción se necesita cada vez que el Space se reconstruye.

### 4. Secrets del Space
En el Space: *Settings → Variables and secrets → New secret*:
- `HF_TOKEN`: el token de construcción del paso 3.
- `RAG_API_KEY`: una clave larga y aleatoria. Puede generarla con:

  ```powershell
  -join ((48..57)+(65..90)+(97..122) | Get-Random -Count 40 | % {[char]$_})
  ```

  Guárdela: es la misma que irá en `RAG_API_KEY` de la API (mismo nombre y valor en ambos servicios). Sin este secret el servicio no arranca, porque se niega a funcionar sin autenticación.

### 5. Subir el paquete de datos (una vez por versión del paquete)
Requiere el paquete descomprimido: la carpeta `storage` del zip de OneDrive o la salida de `build_data_bundle.py`.

```powershell
hf upload <usuario>/rag-istae-data-lb1 E:\rag-istae-paquete\storage . --repo-type=dataset --commit-message "Paquete de datos LB1"
```

Son unos 2,4 GB. Si la subida se corta, repita el mismo comando: solo envía lo que falte.

### 6. Armar la carpeta del Space

```powershell
python .\scripts\prepare_space.py --data E:\rag-istae-paquete\storage --data-repo <usuario>/rag-istae-data-lb1 --out E:\rag-istae-space --force
```

El script:
- verifica el paquete local contra su manifiesto;
- comprueba que el repositorio de datos sea privado y tenga los mismos archivos y el mismo `MANIFEST.json`;
- fija en el `Dockerfile` el commit actual del repositorio de datos (o el de `--data-revision`).

### 7. (Opcional, recomendado) Probar la imagen localmente
Con Docker Desktop en ejecución. La construcción descarga los datos como lo hará el Space, usando la sesión de `hf auth login` como secret; el token no queda en la imagen:

```powershell
cd E:\rag-istae-space
docker build --secret id=HF_TOKEN,src=$env:USERPROFILE\.cache\huggingface\token -t rag-istae-retrieval:lb1 .
docker run --rm -p 7860:7860 -e RAG_API_KEY=clave-local rag-istae-retrieval:lb1
```

En otra PowerShell:

```powershell
Invoke-RestMethod http://127.0.0.1:7860/health
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:7860/retrieve -Headers @{ "X-API-Key" = "clave-local" } -ContentType "application/json; charset=utf-8" -Body ([Text.Encoding]::UTF8.GetBytes('{"question":"Hice prácticas hace 7 años, ¿todavía me las pueden convalidar?","hierarchy":true}'))
```

Debe responder `status: ok`, `units: 2748` y `mode.rerank: false`. En C, sin rerank (valor por defecto), los artículos son `INT15-art22`, `EXT01-disp-general-tercera-2`, `EXT09-sec100`, `INT15-art25`, `EXT02-art122` y, por jerarquía, `EXT13-art36`. Con `"rerank": true` en el request aparecen además `EXT02-art43` e `INT13-art48`.

**Resultado de la prueba local (2026-11-15, Docker Desktop, 8 CPU, `rerank: false` por defecto):**

| Medición | Valor |
|---|---|
| Arranque hasta `/health` OK | 12 s |
| Memoria en uso | unos 1,7 GB |
| B / C (Q27) | 0,8 s / 1,8 s (con `rerank: true`, C tarda 9,9 s); el seguimiento sale del caché en 0,03 s |
| Sin `X-API-Key` o con clave errónea | 401 · `top_k` en el request: 422 |

**Resultado de la prueba local anterior (2026-11-09, LB-1 con rerank por defecto):**

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

### 8. Subir la carpeta del Space

```powershell
cd E:\rag-istae-space
hf upload <usuario>/rag-istae-retrieval . . --repo-type=space --commit-message "Servicio /retrieve LB-1"
```

La subida pesa menos de 1 MB. Al terminar, el Space se construye solo: en la pestaña **Logs** se ve primero *Building* y luego *Running*. La primera vez tarda unos 10–15 min, porque instala torch y descarga y verifica 2,4 GB. La carga de modelos tarda entre 20 y 60 s más.

### 9. Verificar el Space publicado

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

### 10. Conectar la API (Render)
En el servicio de `rag-istae-api` en Render, agregue estas variables de entorno, según [`guia_integracion_api.md`](guia_integracion_api.md):

```env
RETRIEVAL_URL=https://<usuario>-rag-istae-retrieval.hf.space
RAG_API_KEY=<la misma clave del secret RAG_API_KEY del Space>
RETRIEVAL_TIMEOUT_SECONDS=90
ENABLE_LONG_CONTEXT=false
```

---

## Actualizar el Space
- **Cambió el código** de `rag-istae-query`: repita los pasos 6 y 8. La subida solo envía los archivos modificados, y la imagen reutiliza la capa de datos si no cambió el commit.
- **Nuevo paquete de datos:** regenérelo con `build_data_bundle.py`, súbalo al repositorio de datos (paso 5), repita los pasos 6, 8 y 9, y compare el nuevo `units_sha256`.
- **Logs del servicio:** el registro de llamadas va a `/tmp/rag-logs` dentro del contenedor y se pierde al reiniciarse el Space. Los mensajes de arranque y de errores se ven en la pestaña **Logs** del Space.

## Problemas frecuentes

| Síntoma | Causa | Solución |
|---|---|---|
| `403 Forbidden: Storage limit reached for this space (Max: 1 GB)` al subir | Se intentó subir el paquete de datos al Space | Los datos van al repositorio de dataset (paso 5); el Space solo lleva lo que arma `prepare_space.py` |
| La construcción falla con «Defina el secret HF_TOKEN» o un error 401/404 al descargar | Falta el secret `HF_TOKEN`, o no tiene lectura sobre el repositorio de datos | Paso 4; luego *Settings → Factory rebuild* |
| La construcción falla con «no coincide con el manifiesto» | El repositorio de datos cambió o la descarga quedó incompleta | Verifique el commit con `prepare_space.py` (paso 6) y vuelva a subir la carpeta del Space |
| El Space queda en *Runtime error* con `Defina RAG_API_KEY` | Falta el secret | Paso 4; luego *Settings → Restart Space* |
| La API recibe tiempo agotado en la primera consulta del día | El Space estaba dormido | Reintentar; para evitarlo, hardware pago |
| 401 desde la API | `RAG_API_KEY` de Render distinta del secret `RAG_API_KEY` del Space | Igualar ambas |
| La construcción falla al instalar `torch` | Versión no disponible en el índice CPU de PyTorch | Ajustar `torch==` en `deploy/huggingface/Dockerfile` a una versión disponible y regenerar |
