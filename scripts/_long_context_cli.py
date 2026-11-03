"""Utilidades compartidas por los scripts de contexto largo."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from query.long_context.catalog import load_catalog  # noqa: E402
from query.long_context.corpus import load_corpus  # noqa: E402
from query.long_context.settings import LongContextSettings, build_client, load_settings  # noqa: E402
from query.long_context.variant import LongContextVariant  # noqa: E402

DEFAULT_SETTINGS = ROOT / "config" / "long_context.json"


def run(main) -> int:
    """Ejecuta un script mostrando errores de configuración como mensajes legibles."""
    from query.long_context.providers.gemini import GeminiConfigurationError

    try:
        return main()
    except (GeminiConfigurationError, FileNotFoundError, KeyError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        if isinstance(exc, GeminiConfigurationError) and "variable de entorno" in str(exc):
            print('Configure la clave en PowerShell con: setx GEMINI_API_KEY "su_clave" y abra una terminal nueva.', file=sys.stderr)
        return 2


def save_results(results_dir: Path, name: str, data: dict, markdown: str, override: Path | None = None) -> tuple[Path, Path]:
    """Guarda el resultado en JSON (para análisis) y en Markdown (para leer).

    Sin ``override``, los archivos se nombran con fecha y hora en ``results_dir``,
    así ninguna ejecución sobrescribe a otra.
    """
    import json
    from datetime import datetime

    if override:
        json_path = Path(override).expanduser().resolve()
    else:
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        json_path = (Path(results_dir) / f"{name}_{stamp}.json").resolve()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path = json_path.with_suffix(".md")
    md_path.write_text(markdown, encoding="utf-8")
    return json_path, md_path


def parse_documents(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {item.strip().upper() for item in value.split(",") if item.strip()}


def build_variants(
    settings_path: Path,
    variants: list[str],
    documents: set[str] | None,
) -> tuple[LongContextSettings, object, dict[str, LongContextVariant]]:
    settings = load_settings(settings_path)
    catalog, relations, meta = load_catalog(settings.catalog) if settings.catalog.exists() else ({}, [], {})
    corpus = load_corpus(settings.outputs_dir, catalog, documents)
    client = build_client(settings)
    built = {
        name: LongContextVariant(
            corpus,
            name,
            client,
            settings.prompts_dir,
            relations,
            meta.get("rank_labels", {}),
            temperature=settings.temperature,
            max_output_tokens=settings.max_output_tokens,
            max_history_messages=settings.max_history_messages,
        )
        for name in variants
    }
    return settings, client, built
