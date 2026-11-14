---
title: RAG ISTAE Retrieval
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: Recuperación de normativa del ISTAE (arquitecturas B y C)
---

# Servicio de recuperación RAG ISTAE

Servicio `POST /retrieve` de `rag-istae-query`: recupera los artículos de la normativa del ISTAE pertinentes a una pregunta (arquitectura B, RAG híbrido; C, RAG jerárquico) y devuelve el contexto listo para el LLM. No genera respuestas.

- `GET /health`: estado, línea base (`LB-1`) y versión del índice.
- `POST /retrieve`: requiere la cabecera `X-API-Key`. El contrato está en `docs/contrato_retrieve.md` del repositorio `rag-istae-query`.

Este Space se arma con `scripts/prepare_space.py` del repositorio `rag-istae-query`; no se edita a mano. Paquete de datos: `{bundle_version}`, `units_sha256` = `{units_sha256}`.
