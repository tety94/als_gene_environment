"""
Test isolato di potenza: UNA variante causale alla volta, contro il pool di
varianti nulle del pannello, per misurare la curva di potenza per
magnitudine/segno SENZA la contaminazione che si ha quando piu' causali
sono attive insieme nello stesso dataset (vedi spiegazione in cima a
run_scenarios.py: con 46 G×E + 16 pure_variance insieme il rumore
"nascosto" per ogni singolo test saliva a ~4.5x il noise_sd dichiarato).

Qui, per ciascuna variante nei pool di default di gen_fake_data.py
(DEFAULT_CAUSAL_VARIANTS per la parte gene-ambiente, DEFAULT_PURE_VARIANCE_
VARIANTS per la parte vQTL), si genera un dataset con QUELLA SOLA variante
marcata come causale (nessun'altra causale/pure_variance attiva) e si
verifica se la pipeline la recupera. Tutte le altre varianti del pannello
restano nulle per costruzione di gen_fake_data.py, quindi questo isola
esattamente il contributo della singola variante.

*** ASSUNZIONI SU gen_fake_data.py (verificale, non ho il file in questa
chat) ***
  - DEFAULT_CAUSAL_VARIANTS: dict[str, tuple[float, float]]  (beta_inter, beta_main)
  - DEFAULT_PURE_VARIANCE_VARIANTS: dict[str, dict[int, float]]  (sd per dosage)
  - generate_dataset(out_dir=..., causal_variants=..., pure_variance_variants=..., **kw)
    con "tutto il resto nullo di default" se non elencato nei due dict sopra.
Se uno di questi nomi/comportamenti non corrisponde, aggiusta gli import e
la chiamata a generate_dataset() qui sotto -- il resto della logica non
cambia.

COSA TESTA:
  - parte G×E: per ogni variante in DEFAULT_CAUSAL_VARIANTS, gira
    run_ge_interaction() (importata da run_scenarios.py, stessa logica di
    run_pipeline_test.py) e controlla se quella variante risulta
    significativa (p_emp < PVALUE_THRESHOLD) nel risultato.
  - parte vQTL: per ogni variante in DEFAULT_PURE_VARIANCE_VARIANTS, gira
    run_vqtl_debug() (importata da run_scenarios.py) -- che confronta
    asymptotic vs bootstrap sulle causali + un campione di nulle lette dal
    ground_truth di QUESTO dataset (che qui contiene solo 1 causale) -- e
    controlla se quella variante risulta significativa.

OUTPUT: isolated/<tipo>/<variant_label>/... (dati + risultati grezzi) e un
riepilogo unico isolated/isolated_power_curve.csv + .json con, per ogni
variante: beta/sd dichiarati, se recuperata, p-value, SE.

COME LANCIARLO (gira da TE, non da questa chat, stessa cartella di
run_scenarios.py):
    python run_isolated_causal_test.py                 # tutte le varianti default, sequenziale
    python run_isolated_causal_test.py --workers 4      # parallelo, processi separati
    python run_isolated_causal_test.py --only-ge         # solo parte G×E
    python run_isolated_causal_test.py --only-vqtl       # solo parte vQTL/pure_variance
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless: nessun display richiesto
import matplotlib.pyplot as plt

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# Riuso diretto della logica gia' scritta in run_scenarios.py -- niente
# duplicazione: stessa run_ge_interaction/run_vqtl_debug/_set_common_env/
# _force_cfg_overrides usate li'.
import run_scenarios as rs
from gen_fake_data import (
    generate_dataset,
    DEFAULT_CAUSAL_VARIANTS,
    DEFAULT_PURE_VARIANCE_VARIANTS,
)

ISOLATED_ROOT = os.path.join(SCRIPT_DIR, "isolated")
PVALUE_THRESHOLD = 0.05


def section(title: str) -> None:
    print("\n" + "#" * 88)
    print(title)
    print("#" * 88)


# ============================================================
# Cache: se una variante ha gia' un isolated_summary.json con status "ok",
# non la riesegue -- utile per riprendere dopo un errore/interruzione senza
# rifare da capo le 46+16 varianti. Bypassabile con --force.
# ============================================================

def _load_cached_result(var_dir: str, force: bool = False) -> dict | None:
    if force:
        return None
    summary_path = os.path.join(var_dir, "isolated_summary.json")
    if not os.path.isfile(summary_path):
        return None
    try:
        with open(summary_path) as f:
            cached = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None  # file corrotto/incompleto: ricalcola
    if cached.get("status") != "ok":
        return None  # un run fallito in precedenza va sempre riprovato
    return cached


# ============================================================
# Una variante G×E alla volta
# ============================================================

def run_isolated_ge_variant(label: str, beta_inter: float, beta_main: float,
                             n_workers: int = 1, force: bool = False) -> dict:
    var_dir = os.path.join(ISOLATED_ROOT, "ge", label)
    fake_dir = os.path.join(var_dir, "fake_data")

    cached = _load_cached_result(var_dir, force=force)
    if cached is not None:
        print(f"[GxE isolata] {label}: risultato gia' presente (status ok), salto il ricalcolo.")
        return cached

    section(f"[GxE isolata] {label}  (beta_inter={beta_inter}, beta_main={beta_main})")
    os.makedirs(fake_dir, exist_ok=True)

    # Nel pool DEFAULT_CAUSAL_VARIANTS alcune voci hanno beta_inter=0.0
    # (solo main effect, "controllo falsi positivi" -- vedi commento in
    # gen_fake_data.py): per QUESTE il test corretto e' l'assenza di
    # significativita' sull'interazione, non la sua presenza. Le
    # trattiamo come test di FPR, non di potenza.
    is_fpr_control = (beta_inter == 0.0)
    result: dict = {
        "variant": label, "kind": "gxe", "beta_inter": beta_inter, "beta_main": beta_main,
        "role": "fpr_control" if is_fpr_control else "power",
        "status": "ok", "error": None,
    }
    try:
        generate_dataset(
            out_dir=fake_dir, verbose=True,
            causal_variants={label: (beta_inter, beta_main)},
            pure_variance_variants={},
        )
        vqtl_n_jobs = max(1, (os.cpu_count() or 2) // max(1, n_workers))
        rs._set_common_env(fake_dir, var_dir, vqtl_n_jobs=vqtl_n_jobs)

        ge_res = rs.run_ge_interaction(fake_dir, var_dir)
        result["ge_result"] = ge_res

        res_df = pd.read_csv(ge_res["results_csv"])
        row = res_df.loc[res_df["variant"] == label]
        if row.empty:
            result["found_in_results"] = False
            result["significant"] = False
        else:
            p_emp = float(row["p_emp"].iloc[0]) if "p_emp" in row.columns else None
            result["found_in_results"] = True
            result["p_emp"] = p_emp
            result["significant"] = (p_emp is not None) and (p_emp < PVALUE_THRESHOLD)

        # "recovered" ora significa "esito coerente con l'aspettativa":
        # per un controllo FPR e' coerente se NON significativo, per una
        # variante di potenza e' coerente se significativo.
        if not result["found_in_results"]:
            # la pipeline non ha nemmeno testato la variante: sempre un
            # problema, indipendentemente dal ruolo.
            result["recovered"] = False
        elif is_fpr_control:
            result["recovered"] = not result["significant"]
        else:
            result["recovered"] = result["significant"]

    except Exception as exc:
        result["status"] = "FAILED"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        print(f"\n*** VARIANTE '{label}' (GxE) FALLITA: {result['error']} ***")
        traceback.print_exc()

    with open(os.path.join(var_dir, "isolated_summary.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)
    return result


# ============================================================
# Una variante pure_variance (vQTL) alla volta
# ============================================================

def run_isolated_vqtl_variant(label: str, sd_by_dosage: dict, n_workers: int = 1, force: bool = False) -> dict:
    var_dir = os.path.join(ISOLATED_ROOT, "vqtl", label)

    cached = _load_cached_result(var_dir, force=force)
    if cached is not None:
        print(f"[vQTL isolata] {label}: risultato gia' presente (status ok), salto il ricalcolo.")
        return cached

    section(f"[vQTL isolata] {label}  (sd_by_dosage={sd_by_dosage})")
    fake_dir = os.path.join(var_dir, "fake_data")
    os.makedirs(fake_dir, exist_ok=True)

    result: dict = {
        "variant": label, "kind": "pure_variance", "sd_by_dosage": sd_by_dosage,
        "status": "ok", "error": None,
    }
    try:
        generate_dataset(
            out_dir=fake_dir, verbose=True,
            causal_variants={},
            pure_variance_variants={label: sd_by_dosage},
        )
        vqtl_n_jobs = max(1, (os.cpu_count() or 2) // max(1, n_workers))
        rs._set_common_env(fake_dir, var_dir, vqtl_n_jobs=vqtl_n_jobs)

        # run_vqtl_debug legge le causali/nulle dal ground_truth di QUESTO
        # dataset: qui contiene solo `label` come pure_variance, quindi il
        # confronto e' automaticamente isolato (1 causale + 20 nulle campionate).
        debug_res = rs.run_vqtl_debug(fake_dir, var_dir)
        result["vqtl_debug"] = debug_res

        comparison = pd.read_csv(debug_res["comparison_csv"])
        row = comparison.loc[comparison["SNP"] == label]
        if row.empty:
            result["found_in_results"] = False
            result["recovered"] = False
        else:
            p_asym = float(row["P_asym"].iloc[0]) if "P_asym" in row.columns else None
            result["found_in_results"] = True
            result["p_asymptotic"] = p_asym
            result["recovered"] = (p_asym is not None) and (p_asym < PVALUE_THRESHOLD)

    except Exception as exc:
        result["status"] = "FAILED"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        print(f"\n*** VARIANTE '{label}' (pure_variance) FALLITA: {result['error']} ***")
        traceback.print_exc()

    with open(os.path.join(var_dir, "isolated_summary.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)
    return result


# ============================================================
# Wrapper top-level per ProcessPoolExecutor (deve essere picklabile)
# ============================================================

def _worker_ge(label: str, beta_inter: float, beta_main: float, n_workers: int, force: bool) -> dict:
    return run_isolated_ge_variant(label, beta_inter, beta_main, n_workers=n_workers, force=force)


def _worker_vqtl(label: str, sd_by_dosage: dict, n_workers: int, force: bool) -> dict:
    return run_isolated_vqtl_variant(label, sd_by_dosage, n_workers=n_workers, force=force)


# ============================================================
# Reportistica: grafici (matplotlib) + report Word (python-docx),
# scritti sempre sotto ISOLATED_ROOT/plots/ e ISOLATED_ROOT/report.docx.
# Richiede: pip install matplotlib python-docx
# ============================================================

PLOTS_DIR = os.path.join(ISOLATED_ROOT, "plots")


def _fmt_p(p) -> str:
    if p is None or pd.isna(p):
        return "n/a"
    return f"{p:.4f}" if p >= 0.0005 else f"{p:.2e}"


def generate_plots(summary_df: pd.DataFrame) -> dict:
    """Genera i grafici in PNG e ritorna {nome_logico: path} per poterli
    poi inserire nell'ordine giusto nel report Word."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    paths = {}

    ge = summary_df[summary_df["kind"] == "ge"].copy()
    ge_power = ge[ge["role"] == "power"].copy()
    ge_fpr = ge[ge["role"] == "fpr_control"].copy()
    vqtl = summary_df[summary_df["kind"] == "vqtl"].copy()

    # ---- 1) scatter di potenza: |beta_inter| vs esito, colore per esito ----
    if not ge_power.empty:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        abs_beta = ge_power["beta_inter"].abs()
        colors = ge_power["recovered"].map({True: "#1D9E75", False: "#D85A30"})
        markers_pos = ge_power["beta_inter"] > 0
        ax.scatter(abs_beta[markers_pos], [1] * markers_pos.sum(), c=colors[markers_pos],
                   marker="o", s=70, label="positive sign", edgecolor="white", linewidth=0.5)
        ax.scatter(abs_beta[~markers_pos], [0] * (~markers_pos).sum(), c=colors[~markers_pos],
                   marker="s", s=70, label="negative sign", edgecolor="white", linewidth=0.5)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["negative beta_inter", "positive beta_inter"])
        ax.set_xlabel("|beta_inter|")
        ax.set_title("GxE isolated power test — outcome by effect magnitude and sign")
        from matplotlib.lines import Line2D
        legend_elems = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#1D9E75", markersize=9, label="recovered (significant)"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#D85A30", markersize=9, label="missed (not significant)"),
        ]
        ax.legend(handles=legend_elems, loc="lower right", frameon=False)
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        p = os.path.join(PLOTS_DIR, "ge_power_scatter.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths["ge_power_scatter"] = p

        # ---- 2) potenza per fascia di magnitudine (bin da 1 unita') ----
        bins = list(range(0, 10))
        ge_power["bin"] = pd.cut(abs_beta, bins=bins, right=True)
        by_bin = ge_power.groupby("bin", observed=True)["recovered"].agg(["sum", "count"])
        by_bin["power_pct"] = 100 * by_bin["sum"] / by_bin["count"]
        by_bin = by_bin[by_bin["count"] > 0]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar([str(b) for b in by_bin.index], by_bin["power_pct"], color="#2a78d6")
        for i, (n_rec, n_tot) in enumerate(zip(by_bin["sum"], by_bin["count"])):
            ax.text(i, by_bin["power_pct"].iloc[i] + 2, f"{int(n_rec)}/{int(n_tot)}", ha="center", fontsize=8)
        ax.set_ylim(0, 110)
        ax.set_ylabel("power (%)")
        ax.set_xlabel("|beta_inter| bin")
        ax.set_title("GxE detection rate by effect-size bin")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        p = os.path.join(PLOTS_DIR, "ge_power_by_bin.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths["ge_power_by_bin"] = p

    # ---- 3) controlli FPR (beta_inter = 0): p-value per variante ----
    if not ge_fpr.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        colors = ge_fpr["recovered"].map({True: "#1D9E75", False: "#E24B4A"})
        ax.bar(ge_fpr["variant"], ge_fpr["p_value"].fillna(1.0), color=colors)
        ax.axhline(0.05, color="#898781", linestyle="--", linewidth=1, label="p = 0.05 threshold")
        ax.set_ylabel("p_emp (interaction test)")
        ax.set_title("Main-effect-only controls (beta_inter = 0) — false-positive check")
        ax.tick_params(axis="x", rotation=60, labelsize=7)
        ax.legend(frameon=False, loc="upper right")
        fig.tight_layout()
        p = os.path.join(PLOTS_DIR, "ge_fpr_controls.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths["ge_fpr_controls"] = p

    # ---- 4) vQTL / pure_variance (se presenti) ----
    if not vqtl.empty:
        def _sd_ratio(d):
            if not isinstance(d, dict) or not d:
                return None
            vals = list(d.values())
            return max(vals) / min(vals) if min(vals) > 0 else None

        vqtl["sd_ratio"] = vqtl["sd_by_dosage"].apply(_sd_ratio)
        fig, ax = plt.subplots(figsize=(7, 4.2))
        colors = vqtl["recovered"].map({True: "#1D9E75", False: "#D85A30"})
        ax.scatter(vqtl["sd_ratio"], vqtl["p_value"].fillna(1.0), c=colors, s=70, edgecolor="white", linewidth=0.5)
        ax.axhline(0.05, color="#898781", linestyle="--", linewidth=1, label="p = 0.05 threshold")
        ax.set_xlabel("sd ratio (max/min across dosage)")
        ax.set_ylabel("p (asymptotic)")
        ax.set_title("vQTL isolated power test — pure_variance variants")
        ax.legend(frameon=False, loc="upper right")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        p = os.path.join(PLOTS_DIR, "vqtl_power_scatter.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths["vqtl_power_scatter"] = p

    return paths


def generate_word_report(summary_df: pd.DataFrame, plot_paths: dict) -> str:
    """Genera il report Word riassuntivo (recap + tabelle + immagini),
    sempre in inglese, sotto ISOLATED_ROOT/isolated_causal_test_report.docx."""
    ge = summary_df[summary_df["kind"] == "ge"].copy()
    ge_power = ge[ge["role"] == "power"].copy()
    ge_fpr = ge[ge["role"] == "fpr_control"].copy()
    vqtl = summary_df[summary_df["kind"] == "vqtl"].copy()

    doc = Document()

    title = doc.add_heading("Isolated causal variant test — results report", level=0)
    meta = doc.add_paragraph()
    meta.add_run(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}").italic = True

    doc.add_heading("Summary", level=1)

    if not ge_power.empty:
        n_rec = int(ge_power["recovered"].sum())
        n_tot = len(ge_power)
        doc.add_paragraph(
            f"GxE power test: {n_rec}/{n_tot} causal variants recovered "
            f"(overall power {100 * n_rec / n_tot:.1f}%), tested one at a time "
            f"against the null background (no other causal variant active in the same dataset)."
        )
    if not ge_fpr.empty:
        n_fp = int((~ge_fpr["recovered"]).sum())
        n_tot = len(ge_fpr)
        doc.add_paragraph(
            f"GxE false-positive controls (main effect only, beta_inter = 0): "
            f"{n_fp}/{n_tot} incorrectly flagged as significant on the interaction test "
            f"(false-positive rate {100 * n_fp / n_tot:.1f}%)."
        )
    if not vqtl.empty:
        n_rec = int(vqtl["recovered"].sum())
        n_tot = len(vqtl)
        doc.add_paragraph(
            f"vQTL power test (pure_variance): {n_rec}/{n_tot} causal variants recovered "
            f"(overall power {100 * n_rec / n_tot:.1f}%)."
        )
    failed = summary_df[summary_df["status"] == "FAILED"]
    if not failed.empty:
        p = doc.add_paragraph()
        run = p.add_run(f"{len(failed)} variant run(s) raised an exception and could not be evaluated — see the detail table.")
        run.font.color.rgb = RGBColor(0xE2, 0x4B, 0x4A)

    # ---- immagini ----
    if plot_paths:
        doc.add_heading("Plots", level=1)
        captions = {
            "ge_power_scatter": "GxE isolated power test — outcome by effect magnitude and sign.",
            "ge_power_by_bin": "GxE detection rate by effect-size bin.",
            "ge_fpr_controls": "Main-effect-only controls — false-positive check.",
            "vqtl_power_scatter": "vQTL isolated power test — pure_variance variants.",
        }
        for key in ["ge_power_scatter", "ge_power_by_bin", "ge_fpr_controls", "vqtl_power_scatter"]:
            if key in plot_paths:
                doc.add_picture(plot_paths[key], width=Inches(6))
                cap = doc.add_paragraph(captions[key])
                cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in cap.runs:
                    run.italic = True
                    run.font.size = Pt(9)

    # ---- tabella dettaglio (stessa info del CSV) ----
    doc.add_heading("Detailed results", level=1)
    cols = ["kind", "variant", "role", "beta_inter", "beta_main", "sd_by_dosage",
            "p_value", "significant", "recovered", "status"]
    cols = [c for c in cols if c in summary_df.columns]
    table = doc.add_table(rows=1, cols=len(cols))
    table.style = "Light Grid Accent 1"
    for i, c in enumerate(cols):
        table.rows[0].cells[i].text = c
        table.rows[0].cells[i].paragraphs[0].runs[0].bold = True

    sort_cols = [c for c in ["kind", "role"] if c in summary_df.columns]
    ordered = summary_df.sort_values(sort_cols + ["beta_inter"]) if sort_cols else summary_df
    for _, row in ordered.iterrows():
        cells = table.add_row().cells
        for i, c in enumerate(cols):
            val = row[c]
            if c == "p_value":
                val = _fmt_p(val)
            cells[i].text = "" if pd.isna(val) else str(val)
        for c in table.rows[-1].cells:
            for r in c.paragraphs[0].runs:
                r.font.size = Pt(8)

    out_path = os.path.join(ISOLATED_ROOT, "isolated_causal_test_report.docx")
    doc.save(out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Test isolato: una causale alla volta")
    parser.add_argument("--workers", type=int, default=1,
                         help="Numero di varianti da testare in parallelo (processi separati). Default 1.")
    parser.add_argument("--only-ge", action="store_true", help="Testa solo le varianti G×E")
    parser.add_argument("--only-vqtl", action="store_true", help="Testa solo le varianti pure_variance")
    parser.add_argument("--force", action="store_true",
                         help="Ricalcola anche le varianti che hanno gia' un isolated_summary.json con status ok "
                              "(default: vengono saltate e si riusa il risultato salvato)")
    args = parser.parse_args()

    do_ge = not args.only_vqtl
    do_vqtl = not args.only_ge
    n_workers = max(1, args.workers)
    force = args.force
    os.makedirs(ISOLATED_ROOT, exist_ok=True)

    jobs = []
    if do_ge:
        for label, (beta_inter, beta_main) in DEFAULT_CAUSAL_VARIANTS.items():
            jobs.append(("ge", label, beta_inter, beta_main))
    if do_vqtl:
        for label, sd_by_dosage in DEFAULT_PURE_VARIANCE_VARIANTS.items():
            jobs.append(("vqtl", label, sd_by_dosage, None))

    t0 = time.time()
    all_results = []

    if n_workers == 1:
        for job in jobs:
            if job[0] == "ge":
                _, label, beta_inter, beta_main = job
                all_results.append(run_isolated_ge_variant(label, beta_inter, beta_main, n_workers=1, force=force))
            else:
                _, label, sd_by_dosage, _ = job
                all_results.append(run_isolated_vqtl_variant(label, sd_by_dosage, n_workers=1, force=force))
    else:
        print(f"Eseguo {len(jobs)} varianti isolate con {n_workers} processi paralleli...")
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {}
            for job in jobs:
                if job[0] == "ge":
                    _, label, beta_inter, beta_main = job
                    fut = pool.submit(_worker_ge, label, beta_inter, beta_main, n_workers, force)
                else:
                    _, label, sd_by_dosage, _ = job
                    fut = pool.submit(_worker_vqtl, label, sd_by_dosage, n_workers, force)
                futures[fut] = (job[0], job[1])
            for fut in as_completed(futures):
                kind, label = futures[fut]
                try:
                    all_results.append(fut.result())
                except Exception as exc:
                    all_results.append({
                        "variant": label, "kind": kind, "status": "FAILED",
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                    print(f"\n*** VARIANTE '{label}' FALLITA nel processo worker: {exc} ***")
                print(f"[{kind}/{label}] completato ({all_results[-1]['status']}).")

    section("CURVA DI POTENZA — riepilogo")
    rows = []
    for r in all_results:
        rows.append({
            "kind": r["kind"],
            "variant": r["variant"],
            "role": r.get("role", "power"),  # "power" o "fpr_control" (solo per kind=ge)
            "status": r["status"],
            "error": r.get("error"),
            "beta_inter": r.get("beta_inter"),
            "beta_main": r.get("beta_main"),
            "sd_by_dosage": r.get("sd_by_dosage"),
            "p_value": r.get("p_emp") if r["kind"] == "ge" else r.get("p_asymptotic"),
            "significant": r.get("significant"),
            "recovered": r.get("recovered"),  # = esito coerente con l'aspettativa (potenza O fpr_control)
        })
    summary_df = pd.DataFrame(rows)
    print(summary_df.to_string(index=False))

    summary_csv = os.path.join(ISOLATED_ROOT, "isolated_power_curve.csv")
    summary_df.to_csv(summary_csv, index=False)
    with open(os.path.join(ISOLATED_ROOT, "isolated_power_curve.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[export] {summary_csv}")

    section("REPORT — grafici e Word")
    plot_paths = generate_plots(summary_df)
    for name, p in plot_paths.items():
        print(f"[plot] {name}: {p}")
    report_path = generate_word_report(summary_df, plot_paths)
    print(f"[export] {report_path}")

    print(f"\nCompletato in {time.time() - t0:.0f}s.")

    missed_power = [r["variant"] for r in all_results
                    if r["status"] == "ok" and r.get("role", "power") == "power" and not r.get("recovered")]
    false_positives = [r["variant"] for r in all_results
                        if r["status"] == "ok" and r.get("role") == "fpr_control" and not r.get("recovered")]
    not_recovered = missed_power + false_positives
    failed = [r["variant"] for r in all_results if r["status"] == "FAILED"]
    if missed_power:
        print(f"\n*** VARIANTI CAUSALI NON RECUPERATE (potenza, isolate): {missed_power} ***")
    if false_positives:
        print(f"*** CONTROLLI FPR RISULTATI SIGNIFICATIVI (falso positivo, isolati): {false_positives} ***")
    if failed:
        print(f"*** VARIANTI FALLITE (eccezione): {failed} ***")
    if not_recovered or failed:
        sys.exit(1)
    print("\n*** Tutte le varianti causali isolate recuperate correttamente. ***")


if __name__ == "__main__":
    main()