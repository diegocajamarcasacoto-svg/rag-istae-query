# Servicio de recuperación de rag-istae-query (POST /retrieve) para un Space Docker de Hugging Face.
#
# La carpeta del Space la arma scripts/prepare_space.py:
#   Dockerfile, README.md, requirements-space.txt, fetch_data.py y src/ (paquete query).
# El repositorio de un Space admite como máximo 1 GB, así que el paquete de datos LB1 (2,36 GB: modelos,
# Qdrant, units.jsonl, catálogo) vive en un repositorio de dataset PRIVADO y se descarga al construir la
# imagen, fijado a un commit y verificado con sha256 (fetch_data.py). prepare_space.py completa los valores
# {data_repo}, {data_revision} y {manifest_sha256}; las variables del Space con el mismo nombre los reemplazan.
# Los datos quedan en /app/data: en Hugging Face, /data está reservado para el almacenamiento persistente.
#
# Secrets del Space (Settings → Variables and secrets):
#   HF_TOKEN     token de LECTURA sobre el repositorio de datos (solo se usa al construir la imagen);
#   RAG_API_KEY  clave del servicio (se usa al ejecutar; sin ella el servicio no arranca).
#
# Prueba local (desde la carpeta armada; el token se lee de un archivo y no queda en la imagen):
#   docker build --secret id=HF_TOKEN,src=%USERPROFILE%\.cache\huggingface\token -t rag-istae-retrieval .
#   docker run --rm -p 7860:7860 -e RAG_API_KEY=clave-local rag-istae-retrieval

FROM python:3.11-slim

ARG RAG_DATA_REPO={data_repo}
ARG RAG_DATA_REVISION={data_revision}
ARG RAG_DATA_MANIFEST_SHA256={manifest_sha256}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Hugging Face ejecuta el contenedor con el usuario uid 1000.
RUN useradd -m -u 1000 user
WORKDIR /app

# Dependencias: torch para CPU desde el índice de PyTorch (el de PyPI en Linux trae CUDA, varios GB).
COPY requirements-space.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.15.0 \
 && pip install -r requirements-space.txt

# /app/data pertenece a uid 1000 (Qdrant local crea su .lock ahí) y se crea vacío antes de descargar:
# descargar ya como uid 1000 evita un chown recursivo que duplicaría los 2,4 GB en otra capa.
RUN mkdir -p /app/data /tmp/rag-logs && chown 1000:1000 /app/data /tmp/rag-logs
COPY --chown=1000:1000 fetch_data.py /app/fetch_data.py

USER user
ENV HOME=/home/user

# Datos: descarga fijada a un commit y verificación sha256. El token se monta como secret solo en este paso.
RUN --mount=type=secret,id=HF_TOKEN,mode=0444,required=true \
    HF_TOKEN="$(cat /run/secrets/HF_TOKEN)" python /app/fetch_data.py \
      --repo "$RAG_DATA_REPO" --revision "$RAG_DATA_REVISION" \
      --manifest-sha256 "$RAG_DATA_MANIFEST_SHA256" --out /app/data \
 && rm -rf /app/data/.cache /home/user/.cache/huggingface

# Código (al final: un cambio de código no vuelve a descargar los datos).
COPY --chown=1000:1000 src/ /app/src/

ENV PYTHONPATH=/app/src \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    RAG_E5_MODEL_PATH=/app/data/models/multilingual-e5-base \
    RAG_RERANKER_MODEL_PATH=/app/data/models/bge-reranker-base \
    RAG_QDRANT_PATH=/app/data/vector_store/qdrant \
    RAG_UNITS_PATH=/app/data/outputs/units.jsonl \
    RAG_CATALOG_PATH=/app/data/catalog/catalogo_normativo.json \
    RAG_COLLECTION=istae_chunks_e5_base \
    RAG_LOG_DIR=/tmp/rag-logs \
    PORT=7860

EXPOSE 7860
# RAG_API_KEY se define como secret del Space; sin ella el servicio no arranca.
CMD ["sh", "-c", "uvicorn query.service.app:create_app --factory --host 0.0.0.0 --port ${PORT} --workers 1"]
