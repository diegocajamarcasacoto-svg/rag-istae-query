---
title: RAG ISTAE Retrieval
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: other
short_description: Recuperación de normativa del ISTAE (arquitecturas B y C)
---

# Servicio de recuperación RAG ISTAE

Servicio `POST /retrieve` de `rag-istae-query`: recupera los artículos de la normativa del ISTAE pertinentes a una pregunta (arquitectura B, RAG híbrido; C, RAG jerárquico) y devuelve el contexto listo para el LLM. No genera respuestas.

- `GET /health`: estado, línea base (`LB-1`) y versión del índice.
- `POST /retrieve`: requiere la cabecera `X-API-Key`. El contrato está en `docs/contrato_retrieve.md` del repositorio `rag-istae-query`.

Este Space se arma con `scripts/prepare_space.py` del repositorio `rag-istae-query`; no se edita a mano. Paquete de datos: `{bundle_version}`, `units_sha256` = `{units_sha256}`.

## Licencia y uso

Prototipo académico, sin licencia abierta (`license: other`):

- **Código del servicio** (`src/`, `Dockerfile`): forma parte de un trabajo de titulación de maestría (UIDE). Su uso es académico; no se concede permiso de redistribución.
- **Modelos** (`data/models/`): `intfloat/multilingual-e5-base` y `BAAI/bge-reranker-base` conservan sus licencias originales, publicadas en sus tarjetas de modelo en Hugging Face.
- **Textos normativos** (`data/`: índice Qdrant, `units.jsonl` y catálogo): pertenecen a sus emisores. La normativa interna del Instituto Superior Tecnológico Alberto Enríquez (ISTAE) se incluye solo para este prototipo y **no se redistribuye**.

El Space es protegido: el código y los datos no son públicos, y el servicio requiere `X-API-Key`.
