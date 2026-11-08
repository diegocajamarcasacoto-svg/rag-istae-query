"""Prueba 5 de la guía de recuperación: profundidad de candidatos antes del reranking.

Compara la línea base (``candidate_k`` = ``prefetch_k`` = 20) con la variante de 50
sobre preguntas del Gold Standard. Solo lee Qdrant y los modelos locales: no usa LLM.

Relevancia: un chunk es relevante si pertenece a una unidad esperada (``units.jsonl``).
Las unidades esperadas se derivan de las columnas «Fuente esperada» y «Artículo/sección»
del Gold Standard («INT08 … | Arts. 12 y 13» → INT08-art12, INT08-art13). Las fuentes
que no son artículos (secciones o indicadores) se resuelven con ``EXPECTED_OVERRIDES``.

Uso:
    python .\\scripts\\eval_candidate_depth.py --questions Q02,Q13,Q26 [--out-dir .\\logs\\prueba5]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
import time
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from query.config import load_config  # noqa: E402
from query.reranking import load_reranker, rerank  # noqa: E402
from query.retrieval import encode_query, hybrid_search, load_e5, open_client  # noqa: E402
from query.units import load_units  # noqa: E402

GOLD_SHEET = "Gold Standard"
# Fuentes que no se expresan como artículos: se mapean a mano a unidades de units.jsonl.
EXPECTED_OVERRIDES: dict[tuple[str, str], list[str]] = {
    ("Q29", "EXT09"): ["EXT09-sec37"],   # 1.1.3. Indicador Aseguramiento interno de la calidad
    ("Q30", "EXT09"): ["EXT09-sec116"],  # 3.2.3. Indicador Evaluación de los profesores
}
VARIANTS = {"base20": 20, "var50": 50}
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
REL_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"


# --- Lectura mínima del xlsx (sin openpyxl) -------------------------------------------

def _col_index(ref: str) -> int:
    letters = re.match(r"[A-Z]+", ref).group(0)
    value = 0
    for ch in letters:
        value = value * 26 + (ord(ch) - 64)
    return value - 1


def read_sheet(path: Path, sheet_name: str) -> list[list[str]]:
    with zipfile.ZipFile(path) as zf:
        shared = []
        if "xl/sharedStrings.xml" in zf.namelist():
            for si in ET.fromstring(zf.read("xl/sharedStrings.xml")).findall("m:si", NS):
                shared.append("".join(t.text or "" for t in si.iter(f"{{{NS['m']}}}t")))
        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        targets = {r.get("Id"): r.get("Target") for r in rels}
        sheet = next((s for s in workbook.find("m:sheets", NS) if s.get("name") == sheet_name), None)
        if sheet is None:
            raise ValueError(f"No existe la hoja «{sheet_name}» en {path}")
        target = targets[sheet.get(REL_NS)].lstrip("/")
        target = target if target.startswith("xl/") else f"xl/{target}"
        rows = []
        for row in ET.fromstring(zf.read(target)).iter(f"{{{NS['m']}}}row"):
            values: list[str] = []
            for cell in row.findall("m:c", NS):
                idx = _col_index(cell.get("r"))
                values.extend([""] * (idx + 1 - len(values)))
                v = cell.find("m:v", NS)
                if cell.get("t") == "s" and v is not None:
                    values[idx] = shared[int(v.text)]
                elif cell.get("t") == "inlineStr":
                    values[idx] = "".join(t.text or "" for t in cell.iter(f"{{{NS['m']}}}t"))
                elif v is not None:
                    values[idx] = v.text or ""
            rows.append(values)
        return rows


def _articles(spec: str) -> list[int]:
    """«Arts. 16, 17 y 60» → [16, 17, 60]; «Art. 5, literal b» → [5]; sin «Art» → []."""
    spec = re.sub(r"\([^)]*\)", " ", spec)
    if not re.search(r"\bArts?\b", spec, re.IGNORECASE):
        return []
    spec = re.sub(r"literal\s+\w+", " ", spec, flags=re.IGNORECASE)
    return [int(n) for n in re.findall(r"\b(\d+)\b", spec)]


def load_gold(path: Path) -> dict[str, dict]:
    rows = read_sheet(path, GOLD_SHEET)
    header = [h.strip() for h in rows[0]]
    col = {name: header.index(name) for name in header if name}
    gold = {}
    for row in rows[1:]:
        row = row + [""] * (len(header) - len(row))
        qid = row[col["ID"]].strip()
        if not qid:
            continue
        expected, unresolved = [], []
        for n in (1, 2):
            source = row[col[f"Fuente esperada {n}"]].strip()
            spec = row[col[f"Artículo/sección {n}"]].strip()
            if not source:
                continue
            doc = source.split("–")[0].split("-")[0].strip()
            override = EXPECTED_OVERRIDES.get((qid, doc))
            arts = _articles(spec)
            if override:
                expected.extend(override)
            elif arts:
                expected.extend(f"{doc}-art{a}" for a in arts)
            else:
                unresolved.append(f"{doc}: {spec}")
        gold[qid] = {
            "question": row[col["Pregunta"]].strip(),
            "expected_units": list(dict.fromkeys(expected)),
            "unresolved": unresolved,
            "requires_hierarchy": row[col["Requiere jerarquía normativa"]].strip(),
        }
    return gold


# --- Métricas -------------------------------------------------------------------------

def _unit_of(point, index) -> str | None:
    unit = index.unit_for_chunk((point.payload or {}).get("chunk_id"))
    return unit.unit_id if unit else None


def first_relevant_position(unit_ids: list[str | None], expected: set[str]) -> int | None:
    return next((i for i, u in enumerate(unit_ids, 1) if u in expected), None)


def ndcg_at_k(unit_ids: list[str | None], expected: set[str], ideal_relevant: int, k: int) -> float:
    dcg = sum(1 / math.log2(i + 1) for i, u in enumerate(unit_ids[:k], 1) if u in expected)
    idcg = sum(1 / math.log2(i + 1) for i in range(1, min(k, ideal_relevant) + 1))
    return dcg / idcg if idcg else 0.0


def wilcoxon(a: list[float], b: list[float]) -> str:
    diffs = [y - x for x, y in zip(a, b)]
    nonzero = [d for d in diffs if d != 0]
    if not nonzero:
        return "sin diferencias (todas las parejas iguales)"
    try:
        from scipy.stats import wilcoxon as _w
        res = _w(a, b, zero_method="wilcox")
        return f"W={res.statistic:.1f}, p={res.pvalue:.3f} (n efectivo={len(nonzero)})"
    except Exception as exc:  # noqa: BLE001
        return f"no calculable ({exc})"


# --- Programa -------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=ROOT / "config/local.json")
    parser.add_argument("--gold", type=Path, help="xlsx del Gold Standard (por defecto, en storage/corpus/GOLD STANDARD).")
    parser.add_argument("--questions", help="IDs separados por coma (por defecto, todas).")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "logs/prueba5")
    args = parser.parse_args()

    config = load_config(args.config)
    gold_path = args.gold or (
        Path(config["qdrant_path"]).parents[1] / "corpus/GOLD STANDARD/Gold_Standard_Bateria_Pruebas_TFM_v2.xlsx"
    )
    gold = load_gold(gold_path)
    qids = [q.strip() for q in args.questions.split(",")] if args.questions else list(gold)
    missing = [q for q in qids if q not in gold]
    if missing:
        parser.error(f"Preguntas inexistentes en el Gold Standard: {missing}")

    t0 = time.perf_counter()
    index = load_units(Path(config["units_path"]))
    e5 = load_e5(config)
    reranker = load_reranker(config["reranker_model_path"])
    client = open_client(config)
    load_s = time.perf_counter() - t0

    rows = []
    try:
        for qid in qids:
            item = gold[qid]
            expected = set(item["expected_units"])
            unknown = [u for u in expected if index.unit(u) is None]
            if unknown:
                print(f"[aviso] {qid}: unidades esperadas inexistentes en units.jsonl: {unknown}")
            expected -= set(unknown)
            if not expected:
                print(f"[omitida] {qid}: sin unidades esperadas resolubles {item['unresolved']}")
                continue
            ideal = sum(len(index.unit(u).chunk_ids) for u in expected)
            vector = encode_query(e5, item["question"], config)
            row = {"question_id": qid, "requires_hierarchy": item["requires_hierarchy"],
                   "expected_units": ";".join(sorted(expected)),
                   "unresolved_sources": ";".join(item["unresolved"])}
            for name, k in VARIANTS.items():
                t = time.perf_counter()
                candidates = hybrid_search(client, config, vector, item["question"], k, k, args.rrf_k)
                search_ms = (time.perf_counter() - t) * 1000
                cand_units = [_unit_of(p, index) for p in candidates]
                t = time.perf_counter()
                ranked = rerank(reranker, item["question"], candidates, args.batch_size)
                rerank_ms = (time.perf_counter() - t) * 1000
                top_units = [_unit_of(p, index) for p, _ in ranked[: args.top_k]]
                pos = first_relevant_position(top_units, expected)
                row.update({
                    f"{name}_first_relevant_candidate": first_relevant_position(cand_units, expected) or "",
                    f"{name}_units_in_candidates": len(expected & set(cand_units)),
                    f"{name}_relevant_chunks_in_candidates": sum(u in expected for u in cand_units),
                    f"{name}_P@5": sum(u in expected for u in top_units) / args.top_k,
                    f"{name}_unitR@5": len(expected & set(top_units)) / len(expected),
                    f"{name}_MRR": 1 / pos if pos else 0.0,
                    f"{name}_nDCG@5": ndcg_at_k(top_units, expected, ideal, args.top_k),
                    f"{name}_top5_units": ";".join(u or "?" for u in top_units),
                    f"{name}_top5_scores": ";".join(f"{float(s):.4f}" for _, s in ranked[: args.top_k]),
                    f"{name}_search_ms": round(search_ms, 1),
                    f"{name}_rerank_ms": round(rerank_ms, 1),
                })
            row["n_expected_units"] = len(expected)
            rows.append(row)
            print(f"{qid}: base20 MRR={row['base20_MRR']:.2f} nDCG={row['base20_nDCG@5']:.2f} | "
                  f"var50 MRR={row['var50_MRR']:.2f} nDCG={row['var50_nDCG@5']:.2f} | "
                  f"1.ª pos. en 50 cand.={row['var50_first_relevant_candidate'] or '—'}")
    finally:
        client.close()

    if not rows:
        print("Sin resultados.")
        return
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_csv = args.out_dir / f"prueba5_{stamp}.csv"
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with out_csv.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)

    lines = [f"Prueba 5 — {stamp} — {len(rows)} preguntas: {', '.join(r['question_id'] for r in rows)}",
             f"Carga (una vez): {load_s:.1f} s", ""]
    lines.append(f"{'Métrica':<34}{'base20':>10}{'var50':>10}  Wilcoxon (base20 vs var50)")
    for metric in ("P@5", "unitR@5", "MRR", "nDCG@5"):
        a = [r[f"base20_{metric}"] for r in rows]
        b = [r[f"var50_{metric}"] for r in rows]
        lines.append(f"{metric:<34}{statistics.mean(a):>10.3f}{statistics.mean(b):>10.3f}  {wilcoxon(a, b)}")
    for metric in ("units_in_candidates", "relevant_chunks_in_candidates", "search_ms", "rerank_ms"):
        a = [r[f"base20_{metric}"] for r in rows]
        b = [r[f"var50_{metric}"] for r in rows]
        lines.append(f"{metric + ' (media)':<34}{statistics.mean(a):>10.1f}{statistics.mean(b):>10.1f}")
    buckets = {"1–20": 0, "21–50": 0, "ausente": 0}
    for r in rows:
        p = r["var50_first_relevant_candidate"]
        buckets["ausente" if p == "" else "1–20" if p <= 20 else "21–50"] += 1
    lines.append("")
    lines.append("Primer chunk relevante entre 50 candidatos (sin rerank): "
                 + ", ".join(f"{k}: {v}" for k, v in buckets.items()))
    better = [r["question_id"] for r in rows if r["var50_nDCG@5"] > r["base20_nDCG@5"]]
    worse = [r["question_id"] for r in rows if r["var50_nDCG@5"] < r["base20_nDCG@5"]]
    lines.append(f"nDCG@5 mejora con 50: {better or '—'} | empeora: {worse or '—'}")
    summary = "\n".join(lines)
    (args.out_dir / f"prueba5_{stamp}.txt").write_text(summary + "\n", encoding="utf-8")
    (args.out_dir / f"prueba5_{stamp}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + summary)
    print(f"\nCSV: {out_csv}")


if __name__ == "__main__":
    main()
