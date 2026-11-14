# Servicio de recuperación de rag-istae-query (POST /retrieve) para un Space Docker de Hugging Face.
#
# La carpeta del Space la arma scripts/prepare_space.py:
#   Dockerfile, README.md, requirements-space.txt, src/ (paquete query) y data/ (paquete de datos LB1).
# Los datos van DENTRO de la imagen (/app/data). En Hugging Face, /data está reservado para el almacenamiento persistente opcional.
#
# Prueba local (desde la carpeta armada):
#   docker build -t rag-istae-retrieval .
#   docker run --rm -p 7860:7860 -e RAG_API_KEY=clave-local rag-istae-retrieval

FROM python:3.11-slim

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

# Código y datos. Qdrant local crea su archivo .lock: /app/data debe ser escribible por uid 1000.
COPY --chown=1000:1000 src/ /app/src/
COPY --chown=1000:1000 data/ /app/data/
RUN mkdir -p /tmp/rag-logs && chown 1000:1000 /tmp/rag-logs

USER user
ENV HOME=/home/user \
    PYTHONPATH=/app/src \
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
# RAG_API_KEY se define como «secret» del Space (Settings → Variables and secrets); sin ella el servicio no arranca.
CMD ["sh", "-c", "uvicorn query.service.app:create_app --factory --host 0.0.0.0 --port ${PORT} --workers 1"]
