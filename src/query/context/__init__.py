"""Construcción reutilizable y de solo lectura de contexto desde Qdrant.

La API pública empieza después de la recuperación y el ordenamiento: quien
invoca proporciona la consulta original, los puntos Qdrant ordenados, un
cliente ya abierto y la configuración de colección/builder. El paquete
devuelve chunks con procedencia, decisiones estructurales y metadata de
tokens/tiempos.
"""

from query.context.builder import build_context
from query.context.models import ContextChunk, ContextResult

__all__ = ["ContextChunk", "ContextResult", "build_context"]
