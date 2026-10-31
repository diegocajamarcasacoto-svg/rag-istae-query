"""Errores explícitos de configuración y ejecución de generación."""


class GenerationError(RuntimeError):
    """Error base del módulo de generación."""


class GenerationConfigurationError(GenerationError):
    """La configuración o credencial necesaria no está disponible."""


class GenerationProviderError(GenerationError):
    """El proveedor no devolvió una respuesta de texto utilizable."""
