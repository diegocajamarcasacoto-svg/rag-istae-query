"""Cuenta los tokens reales de los corpus A y A+J con el tokenizador del modelo.

Uso:
    python scripts/count_long_context_tokens.py --settings config/long_context.json
    python scripts/count_long_context_tokens.py --documents INT08,EXT01   (corpus reducido)

Indica si cada corpus cabe en la ventana de contexto del modelo, dejando un
margen para la pregunta y la respuesta. Contar tokens no genera respuestas.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from _long_context_cli import DEFAULT_SETTINGS, build_variants, parse_documents, run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS)
    parser.add_argument("--variants", nargs="+", default=["A", "AJ"], choices=["A", "AJ"])
    parser.add_argument("--documents", help="Lista de document_id separada por comas (corpus reducido).")
    parser.add_argument("--margin", type=float, default=0.05, help="Fracción de la ventana reservada (por defecto 5 %%).")
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args(argv)

    settings, client, variants = build_variants(args.settings, args.variants, parse_documents(args.documents))
    window = client.context_window()
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provider": settings.provider,
        "model": settings.model,
        "context_window": window,
        "margin": args.margin,
        "documents": sorted(parse_documents(args.documents) or []) or "todos",
        "variants": {},
    }
    print(f"Modelo: {settings.model} · ventana de entrada informada por la API: {window if window else 'no disponible'} tokens")
    for name, variant in variants.items():
        tokens = variant.count_tokens()
        chars = len(variant.context)
        limit = int(window * (1 - args.margin)) if window else None
        fits = None if limit is None else tokens <= limit
        report["variants"][name] = {
            "tokens": tokens,
            "context_chars": chars,
            "chars_per_token": round(chars / tokens, 3) if tokens else None,
            "usable_limit": limit,
            "fits": fits,
            "corpus_version": variant.corpus_version,
        }
        verdict = "sin dato de ventana" if fits is None else ("CABE" if fits else "NO CABE")
        print(f"  {name}: {tokens:,} tokens ({chars:,} caracteres, {chars / tokens:.2f} car./token) → {verdict}")
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Informe: {args.json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run(main))
