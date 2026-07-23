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
from __future__ import annotations

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

from docx import Document
from docx.shared import Pt, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH

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



"""
Recap automatico dei risultati del test G×E: incrocia ground_truth.csv
(verita' nota, da gen_fake_data.py) con l'output REALE della pipeline
(pipeline_results.csv, da modeling.process_single_variant) e produce
sempre 3 file nella cartella indicata:

  - recap_summary.json  -> numeri aggregati (potenza per segno/magnitudine,
                            falsi positivi sulle nulle, ecc.)
  - recap_detail.csv    -> una riga per variante, con esito (TP/FN/FP/TN)
  - recap_report.docx   -> stesso contenuto in tabelle Word, leggibile
                            senza aprire CSV/JSON

Pensato per essere chiamato IN AUTOMATICO alla fine di run_pipeline_test.py
e di ogni scenario in run_scenarios.py -- non serve lanciarlo a mano.

Dipendenza aggiuntiva: python-docx (pip install python-docx).
"""


MAGNITUDE_BINS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, np.inf]
MAGNITUDE_LABELS = ["0-1", "1-2", "2-3", "3-4", "4-5", "5-6", "6-7", "7-8", "8-9", "9+"]


# ============================================================
# 1) Incrocio ground_truth / pipeline_results -> detail + summary
# ============================================================

def _classify_row(row: pd.Series, pvalue_threshold: float) -> dict:
    effect_type = row["effect_type"]
    true_inter = row.get("true_beta_interaction", 0.0)
    true_main = row.get("true_beta_main", 0.0)
    p_emp = row.get("p_emp", np.nan)
    obs_coef = row.get("obs_coef", np.nan)

    detected = bool(pd.notna(p_emp) and p_emp < pvalue_threshold)

    if effect_type == "gxe_meanshift" and true_inter != 0:
        category, sign = "gxe_interaction", ("pos" if true_inter > 0 else "neg")
        outcome = "TP (trovata)" if detected else "FN (mancata)"
    elif effect_type == "gxe_meanshift" and true_inter == 0:
        # solo main effect, interazione vera = 0 -> controllo falsi positivi
        category, sign = "main_only_control", "zero"
        outcome = "FP (falsa interazione)" if detected else "TN (corretto)"
    elif effect_type == "pure_variance":
        category, sign = "pure_variance", "n/a"
        outcome = "n/a (vedi step vQTL)"
    else:
        category, sign = "null", "zero"
        outcome = "FP (falso positivo)" if detected else "TN (corretto)"

    return dict(
        category=category, sign=sign, magnitude=abs(true_inter), detected=detected,
        outcome=outcome, true_beta_interaction=true_inter, true_beta_main=true_main,
        obs_coef=obs_coef, p_emp=p_emp,
    )


def build_recap(ground_truth_df: pd.DataFrame, pipeline_results_df: pd.DataFrame,
                 pvalue_threshold: float = 0.05) -> tuple[pd.DataFrame, dict]:
    m = ground_truth_df.merge(pipeline_results_df, on="variant", how="left", suffixes=("", "_pr"))
    classified = m.apply(lambda r: _classify_row(r, pvalue_threshold), axis=1, result_type="expand")
    detail = pd.concat([m[["variant"]], classified], axis=1)
    detail["magnitude_bin"] = pd.cut(detail["magnitude"], bins=MAGNITUDE_BINS,
                                      labels=MAGNITUDE_LABELS, right=False)

    gxe = detail[detail.category == "gxe_interaction"]
    main_only = detail[detail.category == "main_only_control"]
    null_ = detail[detail.category == "null"]
    pv = detail[detail.category == "pure_variance"]

    def rate(df, mask=None):
        d = df if mask is None else df[mask]
        return None if len(d) == 0 else round(float(d["detected"].mean()), 4)

    by_sign = {
        s: dict(n=int((gxe.sign == s).sum()), n_detected=int(gxe[gxe.sign == s]["detected"].sum()),
                power=rate(gxe, gxe.sign == s))
        for s in ["pos", "neg"]
    }

    by_magnitude = {}
    for lab in MAGNITUDE_LABELS:
        sub = gxe[gxe.magnitude_bin == lab]
        if len(sub) == 0:
            continue
        by_magnitude[lab] = dict(
            n=int(len(sub)), n_detected=int(sub["detected"].sum()), power=rate(sub),
            n_pos=int((sub.sign == "pos").sum()), power_pos=rate(sub, sub.sign == "pos"),
            n_neg=int((sub.sign == "neg").sum()), power_neg=rate(sub, sub.sign == "neg"),
        )

    summary = dict(
        generated_at=datetime.now().isoformat(timespec="seconds"),
        pvalue_threshold=pvalue_threshold,
        n_variants_total=int(len(detail)),
        gxe_interaction=dict(
            n=int(len(gxe)), n_detected=int(gxe["detected"].sum()), power_overall=rate(gxe),
            by_sign=by_sign, by_magnitude=by_magnitude,
        ),
        main_only_control=dict(
            n=int(len(main_only)), n_false_interaction=int(main_only["detected"].sum()),
            false_positive_rate=rate(main_only),
        ),
        null_genomewide=dict(
            n=int(len(null_)), n_false_positive=int(null_["detected"].sum()),
            false_positive_rate=rate(null_),
        ),
        pure_variance=dict(
            n=int(len(pv)),
            note="Non testate dal modello G×E (obs_coef/p_emp non applicabili); "
                 "vedi lo step vQTL (step3-7) per l'esito su queste varianti.",
        ),
    )
    return detail, summary


# ============================================================
# 2) Scrittura JSON + CSV
# ============================================================

def write_json(summary: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def write_csv(detail: pd.DataFrame, path: str) -> None:
    detail.to_csv(path, index=False)


# ============================================================
# 3) Scrittura DOCX (python-docx, nessuna dipendenza da Node/docx-js)
# ============================================================

def _add_table(doc: Document, headers: list[str], rows: list[list[str]], col_widths_cm: list[float] | None = None):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Light Grid Accent 1"
    hdr_cells = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr_cells[i].text = h
        for p in hdr_cells[i].paragraphs:
            for r in p.runs:
                r.font.bold = True
    for row in rows:
        cells = table.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = "" if val is None else str(val)
    if col_widths_cm:
        for row in table.rows:
            for i, w in enumerate(col_widths_cm):
                row.cells[i].width = Cm(w)
    return table


def _pct(x) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def write_docx(summary: dict, detail: pd.DataFrame, path: str, title: str = "Report di validazione — pipeline G×E") -> None:
    doc = Document()

    h = doc.add_heading(title, level=0)
    h.alignment = WD_ALIGN_PARAGRAPH.LEFT

    meta = doc.add_paragraph()
    meta.add_run(f"Generato: {summary['generated_at']}    ").italic = True
    meta.add_run(f"Soglia p-value: {summary['pvalue_threshold']}").italic = True

    # ---- Riepilogo generale ----
    doc.add_heading("Riepilogo generale", level=1)
    gxe, mo, nu, pv = (summary["gxe_interaction"], summary["main_only_control"],
                       summary["null_genomewide"], summary["pure_variance"])
    _add_table(
        doc,
        ["Categoria", "N", "N rilevate/FP", "Tasso"],
        [
            ["Varianti causali G×E (potenza)", gxe["n"], gxe["n_detected"], _pct(gxe["power_overall"])],
            ["Controllo solo-main (falsi positivi attesi 0)", mo["n"], mo["n_false_interaction"], _pct(mo["false_positive_rate"])],
            ["Varianti nulle genome-wide (falsi positivi attesi ≈ soglia)", nu["n"], nu["n_false_positive"], _pct(nu["false_positive_rate"])],
            ["Pure variance (vQTL, non nel modello G×E)", pv["n"], "-", "vedi step vQTL"],
        ],
        col_widths_cm=[8, 2, 3, 3],
    )

    # ---- Potenza per segno ----
    doc.add_heading("Potenza per segno dell'interazione", level=1)
    rows = []
    for s, lab in [("pos", "Positivo"), ("neg", "Negativo")]:
        d = gxe["by_sign"][s]
        rows.append([lab, d["n"], d["n_detected"], _pct(d["power"])])
    _add_table(doc, ["Segno", "N", "N rilevate", "Potenza"], rows, col_widths_cm=[5, 3, 3, 3])

    # ---- Potenza per magnitudine ----
    doc.add_heading("Potenza per magnitudine di |beta_interazione|", level=1)
    rows = []
    for lab, d in gxe["by_magnitude"].items():
        rows.append([lab, d["n"], _pct(d["power"]), d["n_pos"], _pct(d["power_pos"]), d["n_neg"], _pct(d["power_neg"])])
    _add_table(
        doc,
        ["Bin |beta|", "N tot", "Potenza tot", "N pos", "Potenza pos", "N neg", "Potenza neg"],
        rows, col_widths_cm=[3, 2, 3, 2, 3, 2, 3],
    )

    # ---- Dettaglio varianti causali G×E ----
    doc.add_heading("Dettaglio varianti causali G×E", level=1)
    gxe_detail = detail[detail.category == "gxe_interaction"].sort_values("magnitude", ascending=False)
    rows = []
    for _, r in gxe_detail.iterrows():
        rows.append([
            r["variant"], f"{r['true_beta_interaction']:.2f}",
            "" if pd.isna(r["obs_coef"]) else f"{r['obs_coef']:.2f}",
            "" if pd.isna(r["p_emp"]) else f"{r['p_emp']:.3f}",
            r["outcome"],
        ])
    _add_table(doc, ["Variante", "beta vero", "beta osservato", "p_emp", "esito"], rows,
               col_widths_cm=[4, 2.5, 2.5, 2, 4])

    # ---- Controllo falsi positivi solo-main ----
    if mo["n"] > 0:
        doc.add_heading("Controllo falsi positivi (solo main effect, interazione vera = 0)", level=1)
        mo_detail = detail[detail.category == "main_only_control"]
        rows = [[r["variant"], f"{r['true_beta_main']:.2f}",
                 "" if pd.isna(r["p_emp"]) else f"{r['p_emp']:.3f}", r["outcome"]]
                for _, r in mo_detail.iterrows()]
        _add_table(doc, ["Variante", "beta_main vero", "p_emp", "esito"], rows, col_widths_cm=[4, 3, 3, 5])

    # ---- Falsi positivi sulle nulle ----
    doc.add_heading("Falsi positivi su varianti nulle genome-wide", level=1)
    p = doc.add_paragraph()
    p.add_run(
        f"{nu['n_false_positive']} / {nu['n']} varianti nulle risultate significative "
        f"(p_emp < {summary['pvalue_threshold']}) — tasso {_pct(nu['false_positive_rate'])}."
    )
    fp_null = detail[(detail.category == "null") & (detail.detected)]
    if len(fp_null) > 0:
        rows = [[r["variant"], "" if pd.isna(r["obs_coef"]) else f"{r['obs_coef']:.2f}",
                 "" if pd.isna(r["p_emp"]) else f"{r['p_emp']:.3f}"]
                for _, r in fp_null.iterrows()]
        _add_table(doc, ["Variante", "beta osservato", "p_emp"], rows, col_widths_cm=[5, 4, 4])

    for run in doc.paragraphs[0].runs:
        run.font.size = Pt(20)

    doc.save(path)


# ============================================================
# 4) Funzione "tutto in uno", da chiamare a fine pipeline
# ============================================================

def generate_recap(ground_truth_path: str, pipeline_results_path: str, out_dir: str,
                    pvalue_threshold: float = 0.05) -> dict:
    """Legge i due CSV, scrive recap_summary.json / recap_detail.csv / recap_report.docx
    in out_dir, e ritorna il dict di summary (utile per loggarlo o per il
    report riassuntivo multi-scenario in run_scenarios.py)."""
    os.makedirs(out_dir, exist_ok=True)
    gt = pd.read_csv(ground_truth_path)
    pr = pd.read_csv(pipeline_results_path)
    detail, summary = build_recap(gt, pr, pvalue_threshold=pvalue_threshold)

    write_json(summary, os.path.join(out_dir, "recap_summary.json"))
    write_csv(detail, os.path.join(out_dir, "recap_detail.csv"))
    write_docx(summary, detail, os.path.join(out_dir, "recap_report.docx"))

    return summary


def generate_recap(ground_truth_path: str, pipeline_results_path: str, out_dir: str,
                    pvalue_threshold: float = 0.05) -> dict:
    """Legge i due CSV, scrive recap_summary.json / recap_detail.csv / recap_report.docx
    in out_dir, e ritorna il dict di summary (utile per loggarlo o per il
    report riassuntivo multi-scenario, vedi generate_multi_scenario_recap sotto).
    Ritorna anche 'detail' dentro il dict (chiave privata "_detail") cosi'
    run_scenarios.py non deve rileggere il CSV da disco per l'aggregazione."""
    os.makedirs(out_dir, exist_ok=True)
    gt = pd.read_csv(ground_truth_path)
    pr = pd.read_csv(pipeline_results_path)
    detail, summary = build_recap(gt, pr, pvalue_threshold=pvalue_threshold)

    write_json(summary, os.path.join(out_dir, "recap_summary.json"))
    write_csv(detail, os.path.join(out_dir, "recap_detail.csv"))
    write_docx(summary, detail, os.path.join(out_dir, "recap_report.docx"))

    summary["_detail"] = detail  # comodo per l'aggregazione, non finisce nel JSON (vedi write_json)
    return summary


# ============================================================
# 5) Report finale UNICO su tutti gli scenari
# ============================================================

def load_scenario_recap(recap_dir: str) -> tuple[pd.DataFrame, dict]:
    """Rilegge da disco l'output di generate_recap() per uno scenario
    (utile se non hai piu' in memoria il dict/detail, es. run separati)."""
    detail = pd.read_csv(os.path.join(recap_dir, "recap_detail.csv"))
    with open(os.path.join(recap_dir, "recap_summary.json")) as f:
        summary = json.load(f)
    return detail, summary


def _rate(df: pd.DataFrame, mask=None):
    d = df if mask is None else df[mask]
    return None if len(d) == 0 else round(float(d["detected"].mean()), 4)


def generate_multi_scenario_recap(scenario_summaries: dict[str, dict], out_dir: str,
                                   pvalue_threshold: float = 0.05,
                                   title: str = "Report di validazione — riepilogo tutti gli scenari") -> dict:
    """
    scenario_summaries: {nome_scenario: summary_dict}, dove summary_dict e'
    quello ritornato da generate_recap() per QUELLO scenario (deve avere
    la chiave "_detail" con il DataFrame -- se invece lo stai rileggendo da
    disco in un secondo momento, usa load_scenario_recap() e costruisci tu
    il dict {nome: {**summary, "_detail": detail}} prima di chiamare questa
    funzione).

    Scrive in out_dir:
      - all_scenarios_summary.json  (per-scenario + aggregato su tutti)
      - all_scenarios_detail.csv    (concatenato, con colonna 'scenario')
      - all_scenarios_report.docx   (tabella di confronto + potenza aggregata)
    """
    os.makedirs(out_dir, exist_ok=True)

    combined_parts = []
    for name, summary in scenario_summaries.items():
        d = summary["_detail"].copy()
        d["scenario"] = name
        combined_parts.append(d)
    combined = pd.concat(combined_parts, ignore_index=True)

    gxe = combined[combined.category == "gxe_interaction"]
    null_ = combined[combined.category == "null"]
    mo = combined[combined.category == "main_only_control"]

    # ---- confronto per scenario ----
    per_scenario_rows = []
    for name, summary in scenario_summaries.items():
        g = summary["gxe_interaction"]
        n = summary["null_genomewide"]
        per_scenario_rows.append(dict(
            scenario=name,
            n_causal=g["n"], power_overall=g["power_overall"],
            power_pos=g["by_sign"]["pos"]["power"], power_neg=g["by_sign"]["neg"]["power"],
            fp_rate_null=n["false_positive_rate"],
        ))

    # ---- aggregato su tutti gli scenari insieme ----
    by_sign_agg = {
        s: dict(n=int((gxe.sign == s).sum()), n_detected=int(gxe[gxe.sign == s]["detected"].sum()),
                power=_rate(gxe, gxe.sign == s))
        for s in ["pos", "neg"]
    }
    by_magnitude_agg = {}
    for lab in MAGNITUDE_LABELS:
        sub = gxe[gxe.magnitude_bin == lab]
        if len(sub) == 0:
            continue
        by_magnitude_agg[lab] = dict(
            n=int(len(sub)), power=_rate(sub),
            n_pos=int((sub.sign == "pos").sum()), power_pos=_rate(sub, sub.sign == "pos"),
            n_neg=int((sub.sign == "neg").sum()), power_neg=_rate(sub, sub.sign == "neg"),
        )

    agg_summary = dict(
        generated_at=datetime.now().isoformat(timespec="seconds"),
        pvalue_threshold=pvalue_threshold,
        n_scenarios=len(scenario_summaries),
        per_scenario=per_scenario_rows,
        aggregate=dict(
            n_gxe_observations=int(len(gxe)),  # nota: stesse varianti replicate su piu' scenari, non indipendenti
            power_overall=_rate(gxe),
            by_sign=by_sign_agg,
            by_magnitude=by_magnitude_agg,
            fp_rate_null_pooled=_rate(null_),
            fp_rate_main_only_pooled=_rate(mo) if len(mo) else None,
        ),
    )

    with open(os.path.join(out_dir, "all_scenarios_summary.json"), "w") as f:
        json.dump(agg_summary, f, indent=2, ensure_ascii=False)
    combined.to_csv(os.path.join(out_dir, "all_scenarios_detail.csv"), index=False)

    _write_multi_docx(agg_summary, os.path.join(out_dir, "all_scenarios_report.docx"), title=title)

    return agg_summary


def _write_multi_docx(agg_summary: dict, path: str, title: str) -> None:
    doc = Document()
    h = doc.add_heading(title, level=0)
    h.alignment = WD_ALIGN_PARAGRAPH.LEFT

    meta = doc.add_paragraph()
    meta.add_run(f"Generato: {agg_summary['generated_at']}    ").italic = True
    meta.add_run(f"Scenari: {agg_summary['n_scenarios']}    ").italic = True
    meta.add_run(f"Soglia p-value: {agg_summary['pvalue_threshold']}").italic = True

    doc.add_heading("Confronto tra scenari", level=1)
    rows = []
    for r in agg_summary["per_scenario"]:
        rows.append([
            r["scenario"], r["n_causal"], _pct(r["power_overall"]),
            _pct(r["power_pos"]), _pct(r["power_neg"]), _pct(r["fp_rate_null"]),
        ])
    _add_table(
        doc,
        ["Scenario", "N causali", "Potenza tot", "Potenza pos", "Potenza neg", "FP rate (nulle)"],
        rows, col_widths_cm=[5, 2, 2.5, 2.5, 2.5, 3],
    )

    doc.add_heading("Potenza aggregata su tutti gli scenari — per segno", level=1)
    a = agg_summary["aggregate"]
    rows = []
    for s, lab in [("pos", "Positivo"), ("neg", "Negativo")]:
        d = a["by_sign"][s]
        rows.append([lab, d["n"], d["n_detected"], _pct(d["power"])])
    _add_table(doc, ["Segno", "N", "N rilevate", "Potenza"], rows, col_widths_cm=[5, 3, 3, 3])

    doc.add_heading("Potenza aggregata su tutti gli scenari — per magnitudine", level=1)
    rows = []
    for lab, d in a["by_magnitude"].items():
        rows.append([lab, d["n"], _pct(d["power"]), d["n_pos"], _pct(d["power_pos"]), d["n_neg"], _pct(d["power_neg"])])
    _add_table(
        doc,
        ["Bin |beta|", "N tot", "Potenza tot", "N pos", "Potenza pos", "N neg", "Potenza neg"],
        rows, col_widths_cm=[3, 2, 3, 2, 3, 2, 3],
    )

    p = doc.add_paragraph()
    p.add_run(
        f"Falsi positivi pooled su varianti nulle: {_pct(a['fp_rate_null_pooled'])}. "
        + (f"Falsi positivi pooled su controllo solo-main: {_pct(a['fp_rate_main_only_pooled'])}."
           if a["fp_rate_main_only_pooled"] is not None else "")
    )
    note = doc.add_paragraph()
    note.add_run(
        "Nota: 'N' nella potenza aggregata conta le osservazioni variante×scenario, "
        "non varianti indipendenti (se i DEFAULT_CAUSAL_VARIANTS sono gli stessi in ogni "
        "scenario, con lo stesso seed non sono repliche indipendenti — vedi analisi in "
        "conversazione)."
    ).italic = True

    for run in doc.paragraphs[0].runs:
        run.font.size = Pt(20)

    doc.save(path)