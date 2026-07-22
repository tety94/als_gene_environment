"""
Utility di export (CSV + Word) e di verifica automatica (assert-based)
per test_vqtl_pipeline.py. Vedi in fondo l'uso previsto.
"""
from __future__ import annotations

import os
import json
from dataclasses import dataclass, field

import pandas as pd
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

CAUSAL_TYPES = {"gxe_meanshift", "pure_variance"}

COLOR_HEADER_BG = "2F5496"
COLOR_HEADER_TXT = RGBColor(0xFF, 0xFF, 0xFF)
COLOR_CAUSAL_BG = "FFF2CC"


# ============================================================
# CSV export
# ============================================================

def export_csv(df: pd.DataFrame, tables_dir: str, name: str) -> str:
    os.makedirs(tables_dir, exist_ok=True)
    path = os.path.join(tables_dir, f"{name}.csv")
    df.to_csv(path, index=False)
    print(f"[export] {path}")
    return path


# ============================================================
# Controlli automatici (PASS / WARN / FAIL) -- non solo stampa per
# revisione umana: qui si decide se il test e' passato o no.
# ============================================================

@dataclass
class CheckResult:
    name: str
    level: str  # "PASS" | "WARN" | "FAIL"
    detail: str


@dataclass
class CheckSuite:
    results: list = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str, warn_only: bool = False) -> None:
        if ok:
            level = "PASS"
        elif warn_only:
            level = "WARN"
        else:
            level = "FAIL"
        self.results.append(CheckResult(name, level, detail))

    def skip(self, name: str, detail: str) -> None:
        self.results.append(CheckResult(name, "SKIP", detail))

    @property
    def has_failures(self) -> bool:
        return any(r.level == "FAIL" for r in self.results)

    def print_report(self) -> None:
        print("\n" + "-" * 78)
        print(f"{'CHECK':<45}{'ESITO':<8}DETTAGLIO")
        print("-" * 78)
        for r in self.results:
            print(f"{r.name:<45}{r.level:<8}{r.detail}")
        print("-" * 78)
        n_fail = sum(1 for r in self.results if r.level == "FAIL")
        n_warn = sum(1 for r in self.results if r.level == "WARN")
        n_pass = sum(1 for r in self.results if r.level == "PASS")
        print(f"PASS={n_pass}  WARN={n_warn}  FAIL={n_fail}")

    def to_list(self) -> list[dict]:
        return [{"name": r.name, "level": r.level, "detail": r.detail} for r in self.results]


def run_checks(
    lambda_gc: float,
    all_causal: set,
    found_causal: set,
    candidates: pd.DataFrame,
    interaction_df_display: pd.DataFrame,
    perm_df_display: pd.DataFrame,
    alpha: float = 0.05,
) -> CheckSuite:
    suite = CheckSuite()

    # 1. lambda_GC ragionevolmente vicino a 1. La P asintotica e' nota per
    #    essere anti-conservativa (vedi WARNING di filter_candidates), quindi
    #    qui trattiamo uno scostamento forte come WARN, non FAIL: e' un
    #    segnale da controllare in filter_candidates/scan, non un crash del
    #    test in se'.
    suite.add(
        "lambda_GC vicino a 1",
        0.8 <= lambda_gc <= 1.5,
        f"lambda_GC={lambda_gc:.3f} (atteso in [0.8, 1.5])",
        warn_only=True,
    )

    # 2. Almeno una variante causale recuperata fra i candidati: se ZERO,
    #    e' un fallimento vero (lo scan/filtro non funziona affatto).
    suite.add(
        "Almeno 1 variante causale fra i candidati",
        len(found_causal) >= 1,
        f"{len(found_causal)}/{len(all_causal)} causali recuperate: {sorted(found_causal) or '[]'}",
    )

    # 3. Maggioranza delle causali recuperata (soglia indicativa, non 100%:
    #    e' un test statistico su dati sintetici, non deterministico).
    frac = len(found_causal) / len(all_causal) if all_causal else 0.0
    suite.add(
        "Recupero causali >= 50%",
        frac >= 0.5,
        f"{frac:.0%} delle {len(all_causal)} causali recuperate fra i candidati",
        warn_only=True,
    )

    if candidates.empty or interaction_df_display.empty:
        suite.skip("Step 5+: interazione/permutazione", "nessun candidato disponibile, step 5-7 non eseguiti")
        return suite

    gxe_rows = interaction_df_display[interaction_df_display["effect_type"] == "gxe_meanshift"]
    pv_rows = interaction_df_display[interaction_df_display["effect_type"] == "pure_variance"]

    # 4. Le G×E causali fra i candidati devono avere interazione
    #    significativa con segno coerente.
    if gxe_rows.empty:
        suite.skip("G×E: interazione significativa e segno coerente", "nessuna variante G×E fra i candidati in questo run")
    else:
        import numpy as np
        sign_ok = (
            (gxe_rows["pval"] < alpha)
            & (pd_sign(gxe_rows["beta_I"]) == pd_sign(gxe_rows["true_beta_interaction"]))
        )
        n_ok = int(sign_ok.sum())
        suite.add(
            "G×E: interazione significativa e segno coerente",
            n_ok == len(gxe_rows),
            f"{n_ok}/{len(gxe_rows)} G×E con pval<{alpha} e segno coerente",
        )

    # 5. Le vQTL pure NON devono mostrare interazione significativa (falso
    #    positivo del test di interazione).
    if pv_rows.empty:
        suite.skip("vQTL pure: nessun falso positivo di interazione", "nessuna vQTL pura fra i candidati in questo run")
    else:
        n_falsepos = int((pv_rows["pval"] < alpha).sum())
        suite.add(
            "vQTL pure: nessun falso positivo di interazione",
            n_falsepos == 0,
            f"{n_falsepos}/{len(pv_rows)} vQTL pure con interazione falsamente significativa",
        )

    # 6. Empirical pval (Step 7) basso per le G×E causali fra i top loci
    #    permutati -- conferma indipendente dello Step 5.
    perm_gxe = perm_df_display[perm_df_display["effect_type"] == "gxe_meanshift"] if not perm_df_display.empty else perm_df_display
    if perm_gxe is None or perm_gxe.empty:
        suite.skip("G×E: empirical_pval basso (Step 7)", "nessuna G×E fra i top loci permutati in questo run")
    else:
        n_ok = int((perm_gxe["empirical_pval"] < alpha).sum())
        suite.add(
            "G×E: empirical_pval basso (Step 7)",
            n_ok == len(perm_gxe),
            f"{n_ok}/{len(perm_gxe)} G×E con empirical_pval<{alpha}",
            warn_only=True,  # 500 permutazioni: potenza limitata, non deterministico
        )

    # 7. Levene basso per le vQTL pure fra i top loci permutati (segnale di
    #    eteroschedasticita' anche senza interazione).
    perm_pv = perm_df_display[perm_df_display["effect_type"] == "pure_variance"] if not perm_df_display.empty else perm_df_display
    if perm_pv is None or perm_pv.empty:
        suite.skip("vQTL pure: levene_pval basso (Step 7)", "nessuna vQTL pura fra i top loci permutati in questo run")
    else:
        n_ok = int((perm_pv["levene_pval"] < 0.1).sum())
        suite.add(
            "vQTL pure: levene_pval basso (Step 7)",
            n_ok == len(perm_pv),
            f"{n_ok}/{len(perm_pv)} vQTL pure con levene_pval<0.10",
            warn_only=True,
        )

    return suite


def pd_sign(series: pd.Series) -> pd.Series:
    import numpy as np
    return series.apply(np.sign)


# ============================================================
# Word export (python-docx)
# ============================================================

def _set_cell_shading(cell, hex_color: str) -> None:
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tcPr.append(shd)


def _add_table(doc: Document, df: pd.DataFrame, highlight_col: str | None = "effect_type") -> None:
    if df.empty:
        doc.add_paragraph("(nessun dato)").italic = True
        return
    table = doc.add_table(rows=1, cols=len(df.columns))
    table.style = "Table Grid"
    hdr_cells = table.rows[0].cells
    for i, col in enumerate(df.columns):
        hdr_cells[i].text = str(col)
        for p in hdr_cells[i].paragraphs:
            for r in p.runs:
                r.font.bold = True
                r.font.color.rgb = COLOR_HEADER_TXT
                r.font.size = Pt(9)
        _set_cell_shading(hdr_cells[i], COLOR_HEADER_BG)

    for _, row in df.iterrows():
        cells = table.add_row().cells
        is_causal = highlight_col in df.columns and row.get(highlight_col) in CAUSAL_TYPES
        for i, col in enumerate(df.columns):
            val = row[col]
            if isinstance(val, float):
                text = f"{val:.3g}"
            else:
                text = str(val)
            cells[i].text = text
            for p in cells[i].paragraphs:
                for r in p.runs:
                    r.font.size = Pt(9)
            if is_causal:
                _set_cell_shading(cells[i], COLOR_CAUSAL_BG)


def export_docx(
    out_path: str,
    generation: int,
    summary: dict,
    check_suite: CheckSuite,
    tables: list[tuple[str, str, pd.DataFrame]],
) -> str:
    """tables: lista di (titolo, nota, dataframe) nell'ordine desiderato."""
    doc = Document()

    title = doc.add_heading(f"Report pipeline vQTL — generazione {generation}", level=0)

    # ---- Riepilogo ----
    doc.add_heading("Riepilogo", level=1)
    for line in [
        f"lambda_GC: {summary['lambda_gc']}",
        f"Causali recuperate come candidati: {summary['n_found_causal']}/{summary['n_causal_total']} "
        f"({', '.join(summary['found_causal']) or 'nessuna'})",
        f"Falsi positivi fra i candidati: {summary['n_false_positives']}/{summary['n_null_truth']}",
        f"G×E con interazione significativa (Step 5): {summary['n_gxe_sig']}/{summary['n_gxe_total']}",
        f"vQTL pure con interazione falsamente significativa (Step 5, atteso 0): "
        f"{summary['n_pv_falsepos']}/{summary['n_pv_total']}",
    ]:
        doc.add_paragraph(line)

    # ---- Esito controlli automatici ----
    doc.add_heading("Esito dei controlli automatici", level=1)
    overall = "FALLITO" if check_suite.has_failures else "SUPERATO"
    p = doc.add_paragraph()
    run = p.add_run(f"Esito complessivo: {overall}")
    run.bold = True
    run.font.color.rgb = RGBColor(0xC0, 0x00, 0x00) if check_suite.has_failures else RGBColor(0x00, 0x80, 0x00)

    check_df = pd.DataFrame(check_suite.to_list())
    if not check_df.empty:
        check_df = check_df.rename(columns={"name": "controllo", "level": "esito", "detail": "dettaglio"})
        _add_table(doc, check_df, highlight_col=None)

    # ---- Tabelle per step ----
    for title_text, note_text, df in tables:
        doc.add_heading(title_text, level=1)
        if note_text:
            note_p = doc.add_paragraph(note_text)
            note_p.runs[0].italic = True
            note_p.runs[0].font.size = Pt(9)
        _add_table(doc, df)
        doc.add_paragraph("")

    doc.save(out_path)
    print(f"[export] {out_path}")
    return out_path