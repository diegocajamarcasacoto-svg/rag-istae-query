"""Prueba de recall de la etapa /retrieve (B y C, con y sin rerank) contra el Gold Standard v4.

Dos pasos:

    python scripts/eval_recall_retrieve.py material   # Excel con el material (sin resultados)
    python scripts/eval_recall_retrieve.py run        # ejecuta las 4 variantes y llena el Excel

Variantes (parámetros LB-1 fijos, solo cambia un factor):
    B-rerank (LB-1), B-sin-rerank, C-rerank (LB-1), C-sin-rerank.

Métricas (definición de la hoja «Guía» del Gold Standard):
    - Recall@k (k = 1, 3, 5): fuentes esperadas (Fuente 1 y Fuente 2) con al menos un
      fragmento de sus artículos entre los k primeros fragmentos que pasan al contexto
      (orden del rerank, o de RRF sin rerank). Se mide a nivel de artículo vía units.jsonl.
    - Diseño acordado para la decisión del rerank (README, «Decisión pendiente»):
      recall y precisión de los artículos esperados en el contexto final, posición del
      primer pertinente (RR), tokens y, en C, aportes y ruido de la capa jerárquica.
    - Norma y artículo de mayor rango: si llegan al contexto final y si los aportó la capa jerárquica.

Requiere openpyxl. Solo lee Qdrant (no correr con el servicio /retrieve activo).
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from query.baseline import CURRENT_BASELINE  # noqa: E402
from query.config import load_config  # noqa: E402
from query.hierarchy import load_catalog  # noqa: E402
from query.hierarchy.expansion import hierarchy_targets  # noqa: E402
from query.units import load_units  # noqa: E402

GOLD_NAME = "Gold_Standard_Bateria_Pruebas_TFM_v4.xlsx"
OUT_DIR = ROOT / "logs/recall_retrieve"
WORKBOOK = "Pruebas_Recall_Retrieve_LB1.xlsx"
DOC_RE = re.compile(r"\b((?:EXT|INT)\d{2})\b")

# Fuentes que no se citan por artículo (secciones, indicadores, numerales) o cuya
# numeración se repite: mapeo manual a unidades de units.jsonl. A validar.
OVERRIDES: dict[tuple[str, str], tuple[list[str], str]] = {
    ("Q23", "EXT09"): (["EXT09-sec06", "EXT09-sec07"], "Sección 3 «Estructura del modelo» y Tabla 1 (componentes)"),
    ("Q24", "EXT09"): (["EXT09-sec11"], "4.4 Indicador cualitativo: contiene las categorías y sus valores"),
    ("Q29", "EXT09"): ([f"EXT09-sec{n}" for n in range(37, 42)], "Indicador 1.1.3 completo (encabezado, estándar, descripción, elementos, evidencias)"),
    ("Q30", "EXT09"): ([f"EXT09-sec{n}" for n in range(116, 119)], "Indicador 3.2.3 completo (encabezado, elementos, evidencias)"),
    ("Q48", "EXT09"): ([f"EXT09-sec{n}" for n in range(179, 184)], "Indicador 4.1.8 completo (encabezado, estándar, descripción, elementos, evidencias)"),
    ("Q33", "INT09"): (["INT09-sec01", "INT09-sec07"], "Secciones «Objetivo» y «Defensa pública»"),
    ("Q42", "INT16"): ([f"INT16-sec0{n}" for n in range(4, 9)], "Secciones «Evaluación» a «Aprobación» (numerales 8 a 17)"),
    ("Q41", "EXT02"): (["EXT02-art44"], "units.jsonl no separó los arts. 45 a 47 del RRA: quedaron dentro de EXT02-art44"),
    ("Q46", "INT20"): (["INT20-art40", "INT20-art40-2", "INT20-art41", "INT20-art42", "INT20-art43", "INT20-art48"],
                       "Cap. VII: el art. 39 repetido quedó dentro de INT20-art40 y el art. 40 repetido es INT20-art40-2"),
}

VARIANTS = [
    ("B-rerank", "B · con rerank (LB-1)", False, True),
    ("B-sin-rerank", "B · sin rerank", False, False),
    ("C-rerank", "C · con rerank (LB-1)", True, True),
    ("C-sin-rerank", "C · sin rerank", True, False),
]


# --- Gold Standard ------------------------------------------------------------------------

def _articles(spec: str) -> list[int]:
    """«Arts. 38, numeral 10, y 42» → [38, 42]; «Art. 6 (tabla) y Art. 21» → [6, 21]; sin «Art» → []."""
    spec = re.sub(r"\([^)]*\)", " ", spec or "")
    if not re.search(r"\bArts?\b", spec, re.IGNORECASE):
        return []
    spec = re.sub(r"(literal|literales|numeral|numerales)\s+\w+(\s+y\s+\w+)?", " ", spec, flags=re.IGNORECASE)
    return [int(n) for n in re.findall(r"\b(\d+)\b", spec)]


def _expected_units(qid: str, doc: str, spec: str) -> tuple[list[str], str, str]:
    """(unidades, mapeo, nota)."""
    if (qid, doc) in OVERRIDES:
        units, note = OVERRIDES[(qid, doc)]
        return list(units), "manual", note
    arts = _articles(spec)
    return [f"{doc}-art{a}" for a in arts], "automático", ""


def read_gold(path: Path, index, catalog) -> list[dict]:
    import openpyxl

    ws = openpyxl.load_workbook(path, data_only=True)["Gold Standard"]
    rows = list(ws.iter_rows(values_only=True))
    header = [str(h).strip() if h else "" for h in rows[0]]
    col = {name: i for i, name in enumerate(header)}
    gold = []
    for row in rows[1:]:
        qid = (row[col["ID"]] or "").strip()
        if not qid:
            continue
        val = lambda name: str(row[col[name]]).strip() if row[col[name]] is not None else ""  # noqa: E731
        item = {
            "id": qid, "origen": val("Origen"), "dominio": val("Dominio"), "estrato": val("Estrato"),
            "tipo_jerarquia": val("Tipo de jerarquía"), "pregunta": val("Pregunta"),
            "sources": [], "notes": [],
        }
        for n in (1, 2):
            source = val(f"Fuente esperada {n}")
            if not source:
                continue
            doc = DOC_RE.search(source).group(1)
            spec = val(f"Artículo/sección {n}")
            units, mapping, note = _expected_units(qid, doc, spec)
            missing = [u for u in units if index.unit(u) is None]
            if missing:
                item["notes"].append(f"F{n}: unidades inexistentes {missing}")
            if not units:
                item["notes"].append(f"F{n}: sin unidades mapeadas ({spec})")
            if note:
                item["notes"].append(f"F{n} (manual): {note}")
            item["sources"].append({"n": n, "document_id": doc, "spec": spec,
                                    "units": [u for u in units if u not in missing], "mapping": mapping})

        # Norma y artículo de mayor rango
        norma = val("Norma de mayor rango esperada")
        compl = val("Fuentes complementarias (no cuentan para Recall)")
        item["norma_mayor_texto"] = norma
        item["mayor"] = None
        match = DOC_RE.search(norma)
        if match:
            doc = match.group(1)
            units, origin = [], ""
            for source in item["sources"]:
                if source["document_id"] == doc:
                    units += source["units"]
                    origin = "fuente esperada"
            if not units:
                for segment in compl.split(";"):
                    if doc in segment:
                        units = [f"{doc}-art{a}" for a in _articles(segment) if index.unit(f"{doc}-art{a}")]
                        origin = "fuente complementaria (no cuenta para Recall)"
                        break
            others = [d for d in DOC_RE.findall(norma) if d != doc]
            if others:
                item["notes"].append(f"Mayor rango: el Gold Standard también nombra {', '.join(others)}; se toma {doc}")
            info = catalog.document(doc)
            lower = [s["document_id"] for s in item["sources"] if s["document_id"] != doc]
            reachable = None
            if lower:
                targets, _ = hierarchy_targets(catalog, lower, CURRENT_BASELINE.hierarchy_relations,
                                               CURRENT_BASELINE.hierarchy_depth)
                reachable = any(t.document_id == doc for t in targets)
            lower_ranks = [catalog.document(d).rank for d in lower if catalog.document(d)]
            if info and lower_ranks and not all(info.rank < r for r in lower_ranks):
                item["notes"].append(f"Catálogo: {doc} (rango {info.rank}) no es de rango superior a {lower}")
            item["mayor"] = {"document_id": doc, "units": units, "origin": origin,
                             "rank": info.rank if info else None, "rank_label": info.rank_label if info else "",
                             "reachable_from": lower, "reachable": reachable}
        gold.append(item)
    return gold


# --- Métricas -----------------------------------------------------------------------------

def _found(units: list[str | None], source_units: set[str]) -> bool:
    return any(u in source_units for u in units if u)


def evaluate(item: dict, out, index) -> dict:
    ranked_units = [index.unit_for_chunk(r["chunk_id"]) for r in out.debug["ranked"]]
    ranked_units = [u.unit_id if u else None for u in ranked_units]
    cand_units = [index.unit_for_chunk(c["chunk_id"]) for c in out.debug["candidates"]]
    cand_units = [u.unit_id if u else None for u in cand_units]
    context_units = [u["unit_id"] for u in out.units]
    hierarchy_units = [u["unit_id"] for u in out.units if u["origin"] == "hierarchy"]

    sources = [set(s["units"]) for s in item["sources"] if s["units"]]
    n_src = len(sources)
    row = {"top5": ranked_units, "context_units": context_units, "hierarchy_units": hierarchy_units}
    # Referencia (definición de la Guía): posiciones de fragmentos.
    for k in (1, 3, 5):
        row[f"recall@{k}_frag"] = (sum(_found(ranked_units[:k], s) for s in sources) / n_src) if n_src else None
    first = next((i + 1 for i, u in enumerate(ranked_units) if any(u in s for s in sources)), None)
    row["rr_frag"] = 1 / first if first else 0.0
    row["recall@20_candidatos"] = (sum(_found(cand_units, s) for s in sources) / n_src) if n_src else None
    row["tokens"] = out.tokens_estimated
    row["tiempo_total_ms"] = round((out.timings.get("total") or 0) * 1000)
    row["tiempo_rerank_ms"] = round(out.timings["reranking"] * 1000) if out.timings.get("reranking") else None
    row["tiempo_jerarquia_ms"] = round(out.timings["hierarchy"] * 1000) if out.timings.get("hierarchy") else None
    row["warnings"] = out.warnings
    return article_metrics(item, row)


def article_metrics(item: dict, row: dict) -> dict:
    """Métricas sobre los artículos deduplicados del contexto (lo que recibe el LLM).

    Orden: artículos base en el orden en que entran (rerank o RRF) y luego los de la capa
    jerárquica (C). Es ``context_units`` tal como lo devuelve /retrieve.
    """
    context_units = row["context_units"]
    hierarchy_units = row["hierarchy_units"]
    sources = [set(s["units"]) for s in item["sources"] if s["units"]]
    expected = set().union(*sources) if sources else set()
    n_src = len(sources)
    row["n_fuentes"] = n_src
    row["n_articulos"] = len(expected)
    row["n_articulos_contexto"] = len(context_units)
    for k in (1, 3, 5):
        hits = sum(_found(context_units[:k], s) for s in sources)
        row[f"fuentes_top{k}"] = hits
        row[f"recall@{k}"] = hits / n_src if n_src else None
    first = next((i + 1 for i, u in enumerate(context_units) if u in expected), None)
    row["primer_pertinente"] = first
    row["rr"] = 1 / first if first else 0.0
    found_ctx = expected & set(context_units)
    row["fuentes_contexto"] = sum(_found(context_units, s) for s in sources)
    row["recall_fuentes_contexto"] = row["fuentes_contexto"] / n_src if n_src else None
    row["recall_articulos_contexto"] = len(found_ctx) / len(expected) if expected else None
    row["precision_articulos_contexto"] = (len([u for u in context_units if u in expected]) / len(context_units)
                                          if context_units else None)
    row["jerarquia_aportes"] = [u for u in hierarchy_units if u in expected]
    row["jerarquia_ruido"] = [u for u in hierarchy_units if u not in expected]
    mayor = item.get("mayor")
    if mayor:
        doc = mayor["document_id"]
        row["norma_mayor_en_contexto"] = any(_doc(u) == doc for u in context_units)
        row["art_mayor_en_contexto"] = bool(set(mayor["units"]) & set(context_units)) if mayor["units"] else None
        row["mayor_por_jerarquia"] = any(_doc(u) == doc for u in hierarchy_units)
    return row


def _doc(unit_id: str) -> str:
    return unit_id.split("-", 1)[0]


# --- Excel --------------------------------------------------------------------------------

def _yn(value):
    return "N/A" if value is None else ("Sí" if value else "No")


def _mean(values):
    values = [v for v in values if v is not None]
    return round(statistics.fmean(values), 3) if values else None


def write_workbook(path: Path, gold: list[dict], results: dict[str, dict[str, dict]] | None, meta: dict,
                   variants: list[tuple] | None = None) -> None:
    variants = variants or VARIANTS
    import openpyxl
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    head_fill = PatternFill("solid", fgColor="4A6585")
    res_fill = PatternFill("solid", fgColor="FFF2CC")
    manual_fill = PatternFill("solid", fgColor="FCE4D6")
    wb = openpyxl.Workbook()

    def sheet(title, header, widths, rows, result_from=None):
        ws = wb.create_sheet(title)
        ws.append(header)
        for i, cell in enumerate(ws[1], 1):
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = head_fill if not (result_from and i >= result_from) else PatternFill("solid", fgColor="BF8F00")
            cell.alignment = Alignment(wrap_text=True, vertical="center")
        for row in rows:
            ws.append(row)
        for i, w in enumerate(widths, 1):
            ws.column_dimensions[get_column_letter(i)].width = w
        for r in ws.iter_rows(min_row=2):
            for c in r:
                c.alignment = Alignment(wrap_text=True, vertical="top")
                if result_from and c.column >= result_from:
                    c.fill = res_fill
        ws.freeze_panes = "B2"
        ws.auto_filter.ref = ws.dimensions
        ws.row_dimensions[1].height = 45
        return ws

    # Material
    header = ["ID", "Estrato", "Tipo de jerarquía", "Pregunta",
              "Fuente 1 (doc)", "Artículo/sección 1", "Fuente 2 (doc)", "Artículo/sección 2",
              "Norma de mayor rango (Gold Standard)", "Norma de mayor rango (doc)", "Rango (catálogo)",
              "Artículo de mayor rango", "Origen del art. de mayor rango",
              "¿C alcanza la norma superior por el catálogo?",
              "Unidades esperadas F1", "Unidades esperadas F2", "Mapeo", "Observaciones del mapeo"]
    n_material = len(header)
    for key, label, *_ in variants:
        header += [f"Recall@3 · {label}", f"Art. mayor rango en contexto · {label}"]
    rows = []
    for item in gold:
        s = {src["n"]: src for src in item["sources"]}
        m = item["mayor"]
        row = [item["id"], item["estrato"], item["tipo_jerarquia"], item["pregunta"],
               s.get(1, {}).get("document_id", ""), s.get(1, {}).get("spec", ""),
               s.get(2, {}).get("document_id", ""), s.get(2, {}).get("spec", ""),
               item["norma_mayor_texto"], m["document_id"] if m else "N/A",
               f'{m["rank"]} · {m["rank_label"]}' if m and m["rank"] else "N/A",
               ", ".join(m["units"]) if m else "N/A", m["origin"] if m else "N/A",
               (_yn(m["reachable"]) + (f' (desde {", ".join(m["reachable_from"])})' if m["reachable_from"] else "")) if m else "N/A",
               ", ".join(s.get(1, {}).get("units", [])), ", ".join(s.get(2, {}).get("units", [])),
               "manual" if any(x["mapping"] == "manual" for x in item["sources"]) else "automático",
               " | ".join(item["notes"])]
        for key, *_ in variants:
            r = (results or {}).get(key, {}).get(item["id"])
            row += [r["recall@3"] if r else None, _yn(r.get("art_mayor_en_contexto")) if r and m else (None if not r else "N/A")]
        rows.append(row)
    ws = sheet("Material", header, [6, 10, 16, 50, 8, 22, 8, 22, 26, 10, 18, 26, 22, 22, 26, 26, 11, 45] + [14] * (2 * len(variants)),
               rows, result_from=n_material + 1)
    for r in ws.iter_rows(min_row=2):
        if r[16].value == "manual":
            r[16].fill = manual_fill

    # Una solapa por variante
    vheader = ["ID", "Estrato", "Tipo de jerarquía", "N.º fuentes", "N.º artículos esperados",
               "Artículos del contexto en orden (✓ = esperado; J = capa jerárquica)", "N.º artículos en contexto",
               "Fuentes en Top-1", "Fuentes en Top-3", "Fuentes en Top-5",
               "Recall@1", "Recall@3", "Recall@5", "Recall@20 candidatos (techo)", "Posición 1.er pertinente", "RR",
               "Recall fuentes (contexto)", "Recall artículos (contexto)",
               "Precisión artículos (contexto)", "Aportes jerarquía (esperados)",
               "Ruido jerarquía", "Norma de mayor rango en contexto", "Art. de mayor rango en contexto",
               "Mayor rango aportado por jerarquía", "Tokens contexto", "Tiempo total (ms)", "Rerank (ms)",
               "Jerarquía (ms)", "Ref. Guía: Top-5 fragmentos → artículo", "Ref. Recall@1 fragm.",
               "Ref. Recall@3 fragm.", "Ref. Recall@5 fragm.", "Ref. RR fragm.", "Avisos"]
    for key, label, hier, _ in variants:
        rows = []
        for item in gold:
            r = (results or {}).get(key, {}).get(item["id"])
            if not r:
                rows.append([item["id"], item["estrato"], item["tipo_jerarquia"]])
                continue
            m = item["mayor"]
            expected = set().union(*[set(s["units"]) for s in item["sources"]]) if item["sources"] else set()
            hier_set = set(r["hierarchy_units"])
            listing = "\n".join(
                f"{i}. {u}{' ✓' if u in expected else ''}{' (J)' if u in hier_set else ''}"
                for i, u in enumerate(r["context_units"], 1))
            rows.append([item["id"], item["estrato"], item["tipo_jerarquia"], r["n_fuentes"], r["n_articulos"],
                         listing, r["n_articulos_contexto"],
                         r["fuentes_top1"], r["fuentes_top3"], r["fuentes_top5"],
                         r["recall@1"], r["recall@3"], r["recall@5"], r["recall@20_candidatos"],
                         r["primer_pertinente"] or "—", round(r["rr"], 3),
                         r["recall_fuentes_contexto"],
                         r["recall_articulos_contexto"] and round(r["recall_articulos_contexto"], 3),
                         r["precision_articulos_contexto"] and round(r["precision_articulos_contexto"], 3),
                         ", ".join(r["jerarquia_aportes"]) if hier else "N/A",
                         ", ".join(r["jerarquia_ruido"]) if hier else "N/A",
                         _yn(r.get("norma_mayor_en_contexto")) if m else "N/A",
                         _yn(r.get("art_mayor_en_contexto")) if m else "N/A",
                         (_yn(r.get("mayor_por_jerarquia")) if hier else "N/A") if m else "N/A",
                         r["tokens"], r["tiempo_total_ms"], r["tiempo_rerank_ms"], r["tiempo_jerarquia_ms"],
                         "\n".join(f"{i}. {u}" for i, u in enumerate(r["top5"], 1)),
                         r["recall@1_frag"], r["recall@3_frag"], r["recall@5_frag"], round(r["rr_frag"], 3),
                         " | ".join(r["warnings"])])
        sheet(label, vheader, [6, 10, 16, 8, 9, 34, 9, 8, 8, 8, 8, 8, 8, 10, 10, 7, 9, 9, 9, 22, 28,
                               10, 10, 10, 9, 9, 9, 9, 26, 9, 9, 9, 9, 30], rows, result_from=4)

    # Resumen
    ws = wb.create_sheet("Resumen")
    ws.append(["Resumen por variante (promedios sobre las preguntas evaluadas)"])
    ws["A1"].font = Font(bold=True, size=13)
    ws.append([f"Línea base {meta.get('baseline')} · Gold Standard v4 · {meta.get('fecha', '')}"])
    ws.append([])
    metrics = [("recall@1", "Recall@1 (artículos)"), ("recall@3", "Recall@3 (artículos)"),
               ("recall@5", "Recall@5 (artículos)"),
               ("recall@20_candidatos", "Recall@20 candidatos (techo)"), ("rr", "MRR (artículos)"),
               ("n_articulos_contexto", "N.º de artículos en contexto"),
               ("recall_fuentes_contexto", "Recall fuentes (contexto)"),
               ("recall_articulos_contexto", "Recall artículos (contexto)"),
               ("precision_articulos_contexto", "Precisión artículos (contexto)"),
               ("art_mayor_en_contexto", "Art. de mayor rango en contexto (% jerarquía)"),
               ("norma_mayor_en_contexto", "Norma de mayor rango en contexto (% jerarquía)"),
               ("tokens", "Tokens de contexto"), ("tiempo_total_ms", "Tiempo total (ms)"),
               ("recall@3_frag", "Ref. Guía: Recall@3 por fragmentos"), ("rr_frag", "Ref. MRR por fragmentos")]
    groups = [("Total", lambda it: True), ("Directa", lambda it: it["estrato"] == "Directa"),
              ("Jerarquía", lambda it: it["estrato"] == "Jerarquía")]
    groups += [(f"Tipo {t}", (lambda t: lambda it: it["tipo_jerarquia"] == t)(t))
               for t in sorted({it["tipo_jerarquia"] for it in gold if it["tipo_jerarquia"] not in ("", "N/A")})]
    for gname, pred in groups:
        items = [it for it in gold if pred(it)]
        ws.append([f"{gname} (n = {len(items)})"] + [label for _, label, *_ in variants])
        for c in ws[ws.max_row]:
            c.font = Font(bold=True, color="FFFFFF")
            c.fill = head_fill
        for mkey, mlabel in metrics:
            line = [mlabel]
            for vkey, *_ in variants:
                vals = []
                for it in items:
                    r = (results or {}).get(vkey, {}).get(it["id"])
                    if r is None or (mkey.endswith("mayor_en_contexto") and not it["mayor"]):
                        continue
                    v = r.get(mkey)
                    vals.append(float(v) if isinstance(v, bool) else v)
                line.append(_mean(vals))
            ws.append(line)
        ws.append([])
    if meta.get("comparisons"):
        ws.append(["Comparación pareada con y sin rerank (Wilcoxon; victorias / derrotas / empates del rerank)"])
        ws[f"A{ws.max_row}"].font = Font(bold=True)
        for line in meta["comparisons"]:
            ws.append(line)
    ws.column_dimensions["A"].width = 44
    for col in "BCDE":
        ws.column_dimensions[col].width = 22

    # Metodología
    ws = wb.create_sheet("Metodología")
    for line in meta.get("methodology", []):
        ws.append(line)
    ws.column_dimensions["A"].width = 28
    ws.column_dimensions["B"].width = 120
    for r in ws.iter_rows():
        for c in r:
            c.alignment = Alignment(wrap_text=True, vertical="top")
        r[0].font = Font(bold=True)

    del wb["Sheet"]
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


METHODOLOGY = [
    ["Objetivo", "Decidir si se mantiene el rerank (bge-reranker-base) en la etapa /retrieve: prueba de un solo factor "
                 "(con / sin rerank) sobre B y C, con los demás parámetros de LB-1 fijos. Sin LLM."],
    ["Preguntas", "Las 50 del Gold Standard v4 (Q31–Q50 en borrador hasta su validación)."],
    ["Recall@k", "Sobre artículos deduplicados (lo que recibe el LLM): una fuente esperada (Fuente 1 o 2) cuenta si "
                 "uno de sus artículos está entre los k primeros artículos del contexto. Orden: artículos base en el "
                 "orden en que entran (rerank o, sin rerank, fusión RRF) y luego los de la capa jerárquica (C). "
                 "k = 1, 3, 5. Recall = fuentes halladas / fuentes. RR = 1 / posición del primer artículo esperado."],
    ["Ref. fragmentos", "Como referencia se conserva la definición literal de la Guía (k primeros fragmentos del "
                        "Top-5, donde dos fragmentos del mismo artículo ocupan dos posiciones)."],
    ["Recall@20 (techo)", "Fuentes con algún fragmento entre los 20 candidatos de la búsqueda híbrida: es igual con y "
                          "sin rerank y es el máximo que el rerank puede llevar al contexto base."],
    ["Contexto final", "Artículos que llegan al prompt (base + capa jerárquica en C). Recall de fuentes y de artículos, "
                       "precisión de artículos (esperados / artículos del contexto), tokens."],
    ["Mayor rango", "Norma y artículo de mayor rango según el Gold Standard: si llegan al contexto y si los aportó la "
                    "capa jerárquica. Solo preguntas de jerarquía con norma de mayor rango definida."],
    ["Mapeo", "Artículos «Art. N» → unidad DOC-artN de units.jsonl. Secciones, indicadores y numerales: mapeo manual "
              "(color salmón en «Material»), pendiente de validar."],
    ["Estadística", "Comparación pareada por pregunta (Wilcoxon de rangos con signo) y victorias / derrotas / empates. "
                    "Con 50 preguntas, diferencias pequeñas no serán significativas."],
]


# --- Programa -----------------------------------------------------------------------------

def _common(args):
    config = load_config(args.config)
    gold_path = args.gold or Path(config["qdrant_path"]).parents[1] / "corpus/GOLD STANDARD" / GOLD_NAME
    index = load_units(Path(config["units_path"]))
    catalog = load_catalog(Path(config["catalog_path"]))
    gold = read_gold(gold_path, index, catalog)
    return config, gold, index


def _wilcoxon(a, b):
    diffs = [y - x for x, y in zip(a, b)]
    wins, losses = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
    ties = len(diffs) - wins - losses
    text = f"{wins} / {losses} / {ties}"
    if wins + losses == 0:
        return text, "sin diferencias"
    try:
        from scipy.stats import wilcoxon
        res = wilcoxon(a, b, zero_method="wilcox")
        return text, f"W = {res.statistic:.1f}; p = {res.pvalue:.3f}"
    except Exception as exc:  # noqa: BLE001
        return text, f"no calculable ({exc})"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=["material", "run", "recompute"])
    parser.add_argument("--config", type=Path, default=ROOT / "config/local.json")
    parser.add_argument("--gold", type=Path)
    parser.add_argument("--questions", help="IDs separados por coma (por defecto, las 50).")
    parser.add_argument("--variants", help="Variantes separadas por coma (por defecto, las 4).")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--workbook", default=WORKBOOK,
                        help="Nombre del Excel. Con «recompute» y --variants, arma un libro solo con esas variantes.")
    args = parser.parse_args()

    config, gold, index = _common(args)
    methodology = list(METHODOLOGY)
    if args.variants and all(v.startswith("C-") for v in args.variants.split(",")):
        methodology.insert(0, ["Alcance", "Libro solo con la arquitectura C (RAG jerárquico, la integrada en la app), "
                                          "con y sin rerank. Mismos resultados que el libro completo de B y C."])
    meta = {"baseline": CURRENT_BASELINE.id, "methodology": methodology,
            "fecha": datetime.now().strftime("%Y-%m-%d %H:%M")}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "gold_v4_unidades.json").write_text(json.dumps(gold, ensure_ascii=False, indent=2), encoding="utf-8")
    for item in gold:
        for note in item["notes"]:
            print(f"[mapeo] {item['id']}: {note}")

    if args.step == "material":
        write_workbook(args.out_dir / args.workbook, gold, None, meta)
        print(f"Material: {args.out_dir / args.workbook}")
        return

    if args.step == "recompute":
        # Recalcula las métricas desde resultados.jsonl, sin volver a recuperar.
        by_id = {it["id"]: it for it in gold}
        results = {}
        for line in open(args.out_dir / "resultados.jsonl", encoding="utf-8"):
            row = json.loads(line)
            key, qid = row.pop("variant"), row.pop("id")
            if "recall@1_frag" not in row:  # ejecución anterior: Recall@k y RR eran por fragmentos
                for k in (1, 3, 5):
                    row[f"recall@{k}_frag"] = row[f"recall@{k}"]
                row["rr_frag"] = row["rr"]
            results.setdefault(key, {})[qid] = article_metrics(by_id[qid], row)
        with open(args.out_dir / "resultados.jsonl", "w", encoding="utf-8") as fh:
            for key, rows in results.items():
                for qid, row in rows.items():
                    fh.write(json.dumps({"variant": key, "id": qid, **row}, ensure_ascii=False, default=str) + "\n")
        finish(args, gold, results, meta)
        return

    from query.engine import RetrievalEngine

    qids = set(q.strip() for q in args.questions.split(",")) if args.questions else None
    items = [it for it in gold if qids is None or it["id"] in qids]
    wanted = set(args.variants.split(",")) if args.variants else None
    variants = [v for v in VARIANTS if wanted is None or v[0] in wanted]

    t0 = time.perf_counter()
    engine = RetrievalEngine.from_config(config)
    print(f"Recursos cargados en {time.perf_counter() - t0:.1f} s")
    results: dict[str, dict[str, dict]] = {}
    raw = open(args.out_dir / "resultados.jsonl", "w", encoding="utf-8")
    try:
        for key, label, hier, rr in variants:
            results[key] = {}
            for item in items:
                out = engine.retrieve(item["pregunta"], rerank=rr, hierarchy=hier)
                row = evaluate(item, out, index)
                results[key][item["id"]] = row
                raw.write(json.dumps({"variant": key, "id": item["id"], **row}, ensure_ascii=False, default=str) + "\n")
                raw.flush()
                print(f"{key:13} {item['id']}  R@3={row['recall@3']}  ctx={row['recall_fuentes_contexto']}  "
                      f"{row['tiempo_total_ms']} ms", flush=True)
    finally:
        raw.close()
        engine.close()
    finish(args, gold, results, meta)


def finish(args, gold, results, meta) -> None:
    comparisons = []
    archs = sorted({v.split("-")[0] for v in args.variants.split(",")}) if args.variants else ("B", "C")
    for arch in archs:
        on, off = results.get(f"{arch}-rerank"), results.get(f"{arch}-sin-rerank")
        if not on or not off:
            continue
        ids = [i for i in on if i in off]
        for mkey, mlabel in [("recall@1", "Recall@1"), ("recall@3", "Recall@3"), ("recall@5", "Recall@5"),
                             ("rr", "RR"), ("recall_fuentes_contexto", "Recall fuentes (contexto)"),
                             ("recall_articulos_contexto", "Recall artículos (contexto)"),
                             ("recall@3_frag", "Ref. Recall@3 por fragmentos")]:
            a = [off[i][mkey] or 0 for i in ids]
            b = [on[i][mkey] or 0 for i in ids]
            wlt, test = _wilcoxon(a, b)
            comparisons.append([f"{arch} · {mlabel}", f"sin {statistics.fmean(a):.3f} → con {statistics.fmean(b):.3f}",
                                wlt, test])
    meta["comparisons"] = comparisons
    wanted = set(args.variants.split(",")) if args.variants else None
    variants = [v for v in VARIANTS if (wanted is None or v[0] in wanted) and v[0] in results]
    if wanted:  # libro parcial (p. ej. solo C): también su propio JSONL
        stem = Path(args.workbook).stem.replace("Pruebas_Recall_Retrieve_", "")
        with open(args.out_dir / f"resultados_{stem}.jsonl", "w", encoding="utf-8") as fh:
            for key, *_ in variants:
                for qid, row in results[key].items():
                    fh.write(json.dumps({"variant": key, "id": qid, **row}, ensure_ascii=False, default=str) + "\n")
    write_workbook(args.out_dir / args.workbook, gold, results, meta, variants)
    print(f"Resultados: {args.out_dir / args.workbook}")
    for line in comparisons:
        print(" | ".join(line))


if __name__ == "__main__":
    main()
