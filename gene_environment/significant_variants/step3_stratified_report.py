"""
SCRIPT 3/3 — mergia il file genetico ridotto (output dello script 2, solo
varianti significative) con il file ambientale, filtra per generazione, e
calcola per ciascuna variante:
  - statistiche onset_age POOLED (mutati vs non mutati, come oggi)
  - statistiche onset_age STRATIFICATE per esposizione:
      mutati-esposti vs non_mutati-esposti
      mutati-non_esposti vs non_mutati-non_esposti
  - boxplot a 4 gruppi

"esposto" = valore GREZZO dell'esposizione > 0, "non esposto" = 0 (come
confermato). Nessuna scrittura a DB: tutto ricalcolabile da file, riusa
`compute_onset_age_result` (stessa funzione usata da modeling.py, stessa
definizione di p-value/CI/low_power — coerenza garantita) e
`NON_GEN_COLS`/convenzioni di build_dataset.py per restare allineato al
resto della pipeline.
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from gene_environment.analysis.onset_age_stats import compute_onset_age_result
from gene_environment.config import get_config
from gene_environment.logging_utils import configure_logging, get_logger
from gene_environment.utils.id_utils import clean_sample_id
from gene_environment.vcf_pipeline.build_dataset import NON_GEN_COLS

log = get_logger(__name__)


def _load_merged(genetic_path: str, target_generation: int, cfg) -> tuple[pd.DataFrame, list[str]]:
    df_gen = pd.read_parquet(genetic_path)
    df_gen["id"] = df_gen["id"].astype(str).map(clean_sample_id)
    variant_cols = [c for c in df_gen.columns if c != "id"]

    env_cols = [cfg.sample_id_col, cfg.target_col] + cfg.covariates
    if cfg.exposure and cfg.exposure not in env_cols:
        env_cols.append(cfg.exposure)
    df_env = pd.read_csv(cfg.env_file, sep=cfg.sep, decimal=cfg.decimal, usecols=lambda c: c in env_cols or c == cfg.sample_id_col)
    df_env = df_env.rename(columns={cfg.sample_id_col: "id"})
    df_env["id"] = df_env["id"].astype(str)

    df = pd.merge(df_env, df_gen, on="id", how="inner")
    log.info("Merge ambiente(%d) x genetica(%d) -> %d righe", len(df_env), len(df_gen), len(df))

    map_path = cfg.sample_generation_map or os.path.join(cfg.output_folder, "sample_generation_map.csv")
    if os.path.exists(map_path):
        gen_map = pd.read_csv(map_path, dtype={"id": str})
        n_before = len(df)
        df = df.merge(gen_map, on="id", how="left")
        df = df[df["generation"] == target_generation]
        log.info("Filtro generation=%d: %d -> %d righe", target_generation, n_before, len(df))
    else:
        log.warning("Mappa id->generazione non trovata in %s: nessun filtro per generazione applicato.", map_path)

    if not cfg.exposure or cfg.exposure not in df.columns:
        raise RuntimeError(
            f"cfg.exposure={cfg.exposure!r} non presente nel file ambientale dopo il merge: "
            f"non posso stratificare per esposizione."
        )

    return df, variant_cols


def _stratified_stats_for_variant(df: pd.DataFrame, variant_col: str, cfg) -> dict:
    sub = df[df[variant_col] != "."].copy()
    sub[variant_col] = sub[variant_col].astype(int)
    sub["_mutato"] = (sub[variant_col] > 0).astype(int)
    sub["_esposto"] = (sub[cfg.exposure] > 0).astype(int)

    def _onset(mask_mutato, mask_esposto=None):
        m = sub["_mutato"] == mask_mutato
        if mask_esposto is not None:
            m = m & (sub["_esposto"] == mask_esposto)
        return sub.loc[m, cfg.target_col]

    onset_kwargs = dict(
        use_mann_whitney=cfg.use_mann_whitney, alpha=cfg.onset_alpha,
        min_group_size=cfg.onset_min_group_size, low_power_threshold=cfg.onset_low_power_threshold,
        n_boot=cfg.n_boot, seed=cfg.random_state,
    )

    pooled = compute_onset_age_result(_onset(1), _onset(0), **onset_kwargs)
    exposed = compute_onset_age_result(_onset(1, 1), _onset(0, 1), **onset_kwargs)
    non_exposed = compute_onset_age_result(_onset(1, 0), _onset(0, 0), **onset_kwargs)

    groups = {
        "mutati_esposti": _onset(1, 1), "non_mutati_esposti": _onset(0, 1),
        "mutati_non_esposti": _onset(1, 0), "non_mutati_non_esposti": _onset(0, 0),
    }
    return {"pooled": pooled, "exposed": exposed, "non_exposed": non_exposed, "groups": groups}


def _boxplot_4gruppi(groups: dict, variant: str, out_dir: str) -> None:
    order = ["non_mutati_non_esposti", "mutati_non_esposti", "non_mutati_esposti", "mutati_esposti"]
    labels_base = ["WT\nnon esposti", "Mutato\nnon esposti", "WT\nesposti", "Mutato\nesposti"]
    plot_groups, plot_labels = [], []
    for key, label in zip(order, labels_base):
        g = groups[key]
        if len(g) > 0:
            plot_groups.append(g)
            plot_labels.append(f"{label}\n(n={len(g)})")
    if not plot_groups:
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.boxplot(plot_groups, tick_labels=plot_labels, showmeans=True)
    ax.set_ylabel("Età d'esordio")
    ax.set_title(variant)
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"{variant}.png"), dpi=150)
    plt.close(fig)


def _flatten(prefix: str, onset) -> dict:
    if onset is None:
        return {f"{prefix}{k}": None for k in
                ["n_mutati", "n_non_mutati", "median_mutati", "median_non_mutati",
                 "delta_median", "ci_low", "ci_high", "p_value", "effect_size", "low_power", "method"]}
    d = onset.__dict__
    return {f"{prefix}{k}": v for k, v in d.items()}


def run(genetic_path: str, target_generation: int, out_csv: str, plots_dir: str) -> str:
    cfg = get_config()
    configure_logging(cfg.log_dir)

    df, variant_cols = _load_merged(genetic_path, target_generation, cfg)
    log.info("%d varianti da processare su generation=%d", len(variant_cols), target_generation)

    rows = []
    for variant_col in variant_cols:
        stats = _stratified_stats_for_variant(df, variant_col, cfg)
        row = {"variant": variant_col, "generation": target_generation}
        row.update(_flatten("pooled_", stats["pooled"]))
        row.update(_flatten("exposed_", stats["exposed"]))
        row.update(_flatten("non_exposed_", stats["non_exposed"]))
        rows.append(row)
        _boxplot_4gruppi(stats["groups"], variant_col, plots_dir)

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    log.info("Report completato: %d varianti -> %s (boxplot in %s)", len(rows), out_csv, plots_dir)
    return out_csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--genetic-file", type=str, default="./output/significant_genetic_matrix.parquet")
    parser.add_argument("--target-generation", type=int, required=True)
    parser.add_argument("--out-csv", type=str, default="./output/significant_stratified_stats.csv")
    parser.add_argument("--plots-dir", type=str, default="./output/significant_boxplots")
    args = parser.parse_args()
    run(args.genetic_file, args.target_generation, args.out_csv, args.plots_dir)