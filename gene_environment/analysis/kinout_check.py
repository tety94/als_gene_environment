"""Leave-k-out ("k_in_out") robustness check on the significant G x E variants.

WHAT IT DOES
    For every (variant, exposure) pair returned by the `get_significant_results`
    stored procedure, on the generation set in the .env (GENERATION):

      1. rebuilds the dataset exactly as the pipeline does (genetic file =
         RAW_FILE, environmental file = ENV_FILE, id->generation map, PCs, sex,
         exposure standardisation: `_load_genetic_data` + `_build_narrow_covariates`);
      2. reproduces the observed matched sample and ranks the carriers by how much
         each one SUPPORTS the observed interaction (DFBETAS of the G x E coefficient,
         signed so that "large positive" = removing that carrier shrinks |beta|);
      3. removes the top `frac` (default 20%, at least 1) of the carriers;
      4. re-runs the pipeline's OWN test (`modeling.process_single_variant`, untouched:
         same matching, same SMD gate, same permutation scheme and seed) on the reduced
         data, with `n_perm` permutations (default 1000);
      5. decides, and for the variants that do not hold up sets
         `to_use = 0, reason = 'k_in_out'` in `variant_results_significant`.

    The carriers are removed by setting their genotype to "." in the dataframe that is
    handed to `process_single_variant` (it already drops "." rows), so none of the
    files that implement the test (modeling.py, matching.py, fast_ols.py,
    orchestrator.py) is modified: only the input of the test changes.

DECISION RULE (per variant x exposure x generation)
    FAIL            the variant does not hold after the removal, if ANY of:
                      - empirical p > PVALUE_THRESHOLD
                      - the sign of the interaction beta is inverted vs the observed one
                      - |beta| < MIN_OBS_COEF (the pipeline's own minimum effect size)
    PASS            evaluable and none of the above
    NOT_EVALUABLE   no evidence either way, the variant is NOT touched:
                      - fewer than MIN_TREATED carriers would be left
                      - the pipeline's gates reject the reduced sample (SMD > MAX_SMD,
                        too few units, matching failure) or |beta| > 100 (unstable fit)
    SKIP / ERROR    variant not in the genetic file / unexpected exception

DB UPDATE
    Every FAIL sets `to_use = 0, reason = 'k_in_out'` on ALL the rows of that variant
    for the same land use (both generations, both radii, e.g. seminativi_1000 and
    seminativi_1500), only where `to_use = 1` (a reason already written by an earlier
    step is never overwritten). `--dry-run` prints what would change and writes nothing.

GENERATIONS
    One generation per run, taken from the .env like every other command. For gen 2
    change RAW_FILE (and PCA path / GENERATION) in the .env and run again: a FAIL in
    either run sets to_use = 0 everywhere, which is the "both generations" rule.

OUTPUT (cfg.kinout_check_dir, default ./output/kinout_check/gen<N>/)
    kinout_results_gen<N>_<timestamp>.csv   one row per variant x exposure
    kinout_recap_gen<N>_<timestamp>.txt     human-readable recap
    kinout_run_info_gen<N>_<timestamp>.json parameters, counts, DB updates
  plus everything in the pipeline log (cfg.log_dir/pipeline.log).

Usage:
    python -m gene_environment.cli kinout-check
    python -m gene_environment.cli kinout-check --dry-run
    python -m gene_environment.cli kinout-check --frac 0.2 --n-perm 1000
"""
from __future__ import annotations

import contextlib
import dataclasses
import gc
import io
import json
import math
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import OLSInfluence

from gene_environment.analysis import modeling
from gene_environment.analysis.matching import match_control_units
from gene_environment.config import get_config
from gene_environment.db.connection import cursor_scope, get_connection
from gene_environment.db.repository import get_significant_results
from gene_environment.logging_utils import configure_logging, get_logger
from gene_environment.vcf_pipeline.build_dataset import _build_narrow_covariates, _load_genetic_data

log = get_logger(__name__)

REASON = "k_in_out"
DEFAULT_FRAC = 0.20
DEFAULT_N_PERM = 1000

PASS, FAIL, NOT_EVALUABLE, ERROR, SKIP = "PASS", "FAIL", "NOT_EVALUABLE", "ERROR", "SKIP"

# Per-process state of the worker (same pattern as orchestrator.init_worker).
_BASE_DF: pd.DataFrame | None = None


# ============================================================================
# Small helpers
# ============================================================================
def _land_use(exposure: str) -> str:
    """'seminativi_1000' -> 'seminativi' (the radius is the trailing number)."""
    head, _, tail = str(exposure).rpartition("_")
    return head if head and tail.isdigit() else str(exposure)


def _quiet(fn, *a, **k):
    """Run fn swallowing its stdout (build_dataset helpers print debug lines)."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def _f(x, nd: int = 4):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), nd)


# ============================================================================
# Core: carrier ranking, decision rule
# ============================================================================
def _rank_carriers(base_df: pd.DataFrame, col: str, Ecols: list[str], Ccols: list[str], cfg) -> dict:
    """Reproduce the pipeline's observed matched sample for `col` and rank its carriers.

    Same steps as modeling.process_single_variant up to the observed fit (drop "." and
    NaN, carrier = dosage > 0, match_control_units, smf.ols with the pipeline formula),
    with the sample id kept as a plain column so the carriers can be traced back.
    """
    needed = list(dict.fromkeys(["id", cfg.target_col, col] + list(Ecols) + list(Ccols)))
    df = base_df.loc[base_df[col] != ".", needed].copy()
    df[col] = df[col].astype(int)
    df["_match_variant"] = (df[col] > 0).astype(int)
    df_model = df.dropna()

    out = {
        "n_carriers": int(df_model["_match_variant"].sum()),
        "n_non_carriers": int((df_model["_match_variant"] == 0).sum()),
        "ranked_ids": [], "base_beta": None, "problem": None,
    }

    matched = match_control_units(df_model, "_match_variant", k=cfg.match_k,
                                  covariates_for_matching=list(Ecols) + list(Ccols))
    if matched is None or matched.shape[0] < cfg.min_sample_size:
        out["problem"] = "observed matched sample not available (matching failed or too small)"
        return out

    formula = modeling.build_formula(cfg.target_col, col, list(Ecols), list(Ccols), matched)
    mod = smf.ols(formula=formula, data=matched).fit()
    term = modeling._find_interaction_term(mod.params.index, col)
    if term is None:
        out["problem"] = "interaction term not found in the observed model"
        return out

    beta = float(mod.params[term])
    out["base_beta"] = beta
    dfb = np.asarray(OLSInfluence(mod).dfbetas)[:, list(mod.params.index).index(term)]
    # dfbetas_i > 0 means "including unit i raises beta". The carriers that SUPPORT the
    # effect are the ones that push |beta| up: sign-align with beta.
    support = dfb * (1.0 if beta >= 0 else -1.0)
    is_carrier = (matched["_match_variant"] == 1).to_numpy()
    cand = pd.DataFrame({"id": matched["id"].to_numpy()[is_carrier], "support": support[is_carrier]})
    cand = cand.sort_values("support", ascending=False, kind="mergesort", na_position="last")
    out["ranked_ids"] = [str(i) for i in cand["id"].tolist()]
    return out


def _classify(base_beta: float | None, res: dict | None, cfg) -> tuple[str, list[str]]:
    """Decision rule applied to the result of process_single_variant on the reduced data."""
    if res is None:
        return NOT_EVALUABLE, ["no result returned by the test"]

    obs = res.get("obs_coef")
    if obs is None:
        smd = res.get("max_smd")
        if smd is not None and smd > cfg.max_smd:
            return NOT_EVALUABLE, [f"reduced sample rejected by the SMD gate ({smd:.3f} > MAX_SMD {cfg.max_smd})"]
        return NOT_EVALUABLE, ["reduced sample not testable (too few units / matching / model failed)"]
    if abs(obs) > modeling.MAX_ABS_OBS_COEF:
        return NOT_EVALUABLE, [f"|beta|={abs(obs):.1f} > {modeling.MAX_ABS_OBS_COEF:g}: unstable fit, no evidence either way"]

    p = res.get("p_emp")
    if p is None:
        return NOT_EVALUABLE, ["no valid permutation"]

    fails = []
    if abs(obs) < cfg.min_obs_coef:
        # the pipeline does not run permutations below MIN_OBS_COEF and stores p_emp = 1 by default
        fails.append(f"|beta|={abs(obs):.3f} < MIN_OBS_COEF {cfg.min_obs_coef:g} (effect too small, no permutation run)")
    elif p > cfg.pvalue_threshold:
        fails.append(f"p_emp={p:.4f} > {cfg.pvalue_threshold:g}")
    if base_beta is not None and np.sign(obs) != np.sign(base_beta):
        fails.append(f"sign inverted (beta {base_beta:.3f} -> {obs:.3f})")
    if fails:
        return FAIL, fails
    return PASS, [f"p_emp={p:.4f}, beta={obs:.3f}"]


# ============================================================================
# Worker
# ============================================================================
def _init_worker(df_path: str, log_dir: str, covariate_cols: list[str], n_perm: int) -> None:
    configure_logging(log_dir)
    cfg = get_config()
    # process_single_variant reads get_config() (the singleton), so the number of
    # permutations has to be set HERE, not on a copy. n_perm == n_perm_high => the
    # light phase is the whole test (no extra "HIGH" phase): exactly n_perm permutations.
    object.__setattr__(cfg, "n_perm", n_perm)
    object.__setattr__(cfg, "n_perm_high", n_perm)
    global _BASE_DF
    with open(df_path, "rb") as f:
        _BASE_DF = pickle.load(f)
    modeling.global_df = _BASE_DF
    modeling.global_covariate_cols = covariate_cols


def _run_test(df: pd.DataFrame, col: str, label: str, Ecols: list[str]) -> dict | None:
    """Run the pipeline's test on `df` (swapped in as modeling.global_df, then restored)."""
    modeling.global_df = df
    try:
        return modeling.process_single_variant(col, label, Ecols, False)
    finally:
        modeling.global_df = _BASE_DF


def _task(col: str, label: str, exposure: str, generation: int, Ecols: list[str], Ccols: list[str],
          frac: float, baseline: bool) -> dict:
    cfg = get_config()
    out = {"variant": label, "exposure": exposure, "generation": generation, "col": col,
           "status": ERROR, "reasons": ""}
    t0 = time.time()
    try:
        rk = _rank_carriers(_BASE_DF, col, Ecols, Ccols, cfg)
        out.update(n_carriers=rk["n_carriers"], n_non_carriers=rk["n_non_carriers"], base_beta=rk["base_beta"])
        if rk["problem"]:
            out.update(status=NOT_EVALUABLE, reasons=rk["problem"])
            return out

        # ---- optional control: the SAME test, same settings, nothing removed ----
        if baseline:
            b = _run_test(_BASE_DF, col, label, Ecols)
            cs, cr = _classify(rk["base_beta"], b, cfg)
            out.update(ctrl_status=cs, ctrl_reasons="; ".join(cr))
            if b is not None:
                out.update(ctrl_p_emp=b.get("p_emp"), ctrl_beta=b.get("obs_coef"))
                if b.get("obs_coef") is not None and abs(b["obs_coef"] - rk["base_beta"]) > 1e-6 * max(1.0, abs(rk["base_beta"])):
                    log.warning("[%s %s] observed beta from the test (%.6f) differs from the replicated fit (%.6f)",
                                exposure, label, b["obs_coef"], rk["base_beta"])

        n_remove = max(1, math.ceil(frac * rk["n_carriers"]))
        removed = rk["ranked_ids"][:n_remove]
        left = rk["n_carriers"] - len(removed)
        out.update(n_removed=len(removed), n_carriers_left=left, removed_ids=";".join(removed))

        if left < cfg.min_treated:
            out.update(status=NOT_EVALUABLE,
                       reasons=f"only {left} carriers would be left (< MIN_TREATED {cfg.min_treated})")
            return out

        cols = list(dict.fromkeys(["id", cfg.target_col, col] + list(Ecols) + list(Ccols)))
        sub = _BASE_DF[cols].copy()
        sub[col] = sub[col].astype(object)
        sub.loc[sub["id"].astype(str).isin(removed), col] = "."

        res = _run_test(sub, col, label, Ecols)
        status, reasons = _classify(rk["base_beta"], res, cfg)
        out.update(status=status, reasons="; ".join(reasons))
        if res is not None:
            out.update(trim_beta=res.get("obs_coef"), trim_p_emp=res.get("p_emp"),
                       trim_iterations=res.get("iterations"), trim_n_treated=res.get("n_treated"),
                       trim_n_control=res.get("n_control"), trim_max_smd=res.get("max_smd"))
    except Exception as exc:  # keep the run going, report it
        log.exception("[%s %s] unexpected error", exposure, label)
        out.update(status=ERROR, reasons=f"{type(exc).__name__}: {exc}")
    finally:
        out["seconds"] = round(time.time() - t0, 1)
    return out


def _run_tasks(df_path: str, tasks: list[tuple], workers: int, covariate_cols: list[str], n_perm: int, cfg) -> list[dict]:
    results: list[dict] = []
    n = len(tasks)

    def _log_one(r: dict, i: int) -> None:
        log.info("[%d/%d] %s | %s | %s | %s | removed %s/%s carriers | %s",
                 i, n, r.get("exposure"), r.get("variant"), f"gen{r.get('generation')}", r["status"],
                 r.get("n_removed", "?"), r.get("n_carriers", "?"), r.get("reasons", ""))

    if workers <= 1:
        _init_worker(df_path, cfg.log_dir, covariate_cols, n_perm)
        for i, t in enumerate(tasks, 1):
            r = _task(*t)
            results.append(r)
            _log_one(r, i)
        return results

    with ProcessPoolExecutor(max_workers=min(workers, n), initializer=_init_worker,
                             initargs=(df_path, cfg.log_dir, covariate_cols, n_perm)) as ex:
        futs = {ex.submit(_task, *t): t for t in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            t = futs[fut]
            try:
                r = fut.result()
            except Exception as exc:  # worker crashed
                log.exception("worker failed on %s %s", t[2], t[1])
                r = {"variant": t[1], "exposure": t[2], "generation": t[3], "status": ERROR,
                     "reasons": f"worker failed: {type(exc).__name__}: {exc}"}
            results.append(r)
            _log_one(r, i)
    return results


# ============================================================================
# DB update
# ============================================================================
def _sibling_exposures(variant: str, exposure: str) -> list[str]:
    """Every exposure of the same land use (all radii) that the variant has in variant_results_significant."""
    with get_connection() as conn:
        with cursor_scope(conn) as cur:
            cur.execute("SELECT DISTINCT exposure FROM variant_results_significant WHERE variant=%s", (variant,))
            found = [r[0] for r in cur.fetchall()]
    fam = _land_use(exposure)
    return sorted({e for e in found if _land_use(e) == fam} | {exposure})


def _rows_in_use(variant: str, exposures: list[str]) -> int:
    fmt = ",".join(["%s"] * len(exposures))
    with get_connection() as conn:
        with cursor_scope(conn) as cur:
            cur.execute(
                f"SELECT COUNT(*) FROM variant_results_significant WHERE variant=%s AND exposure IN ({fmt}) AND to_use=1",
                (variant, *exposures),
            )
            return int(cur.fetchone()[0])


def _set_not_used(variant: str, exposures: list[str]) -> int:
    fmt = ",".join(["%s"] * len(exposures))
    with get_connection() as conn:
        with cursor_scope(conn) as cur:
            cur.execute(
                f"UPDATE variant_results_significant SET to_use=0, reason=%s "
                f"WHERE variant=%s AND exposure IN ({fmt}) AND to_use=1",
                (REASON, variant, *exposures),
            )
            return int(cur.rowcount)


# ============================================================================
# Reports
# ============================================================================
def _write_recap(res: pd.DataFrame, path: Path, info: dict) -> None:
    L = []
    L.append(f"CONTROLLO LEAVE-K-OUT (k_in_out) - generazione {info['generation']} - {info['timestamp']}")
    L.append("=" * 78)
    L.append(f"Quota di portatori rimossi: {info['frac']:.0%} (almeno 1), scelti per DFBETAS (i piu' a favore dell'effetto)")
    L.append(f"Permutazioni: {info['n_perm']}   soglia p: {info['pvalue_threshold']}   MIN_TREATED: {info['min_treated']}   "
             f"MIN_OBS_COEF: {info['min_obs_coef']}   MAX_SMD: {info['max_smd']}")
    L.append(f"RAW_FILE: {info['raw_file']}")
    L.append(f"ENV_FILE: {info['env_file']}")
    L.append(f"Modalita': {'DRY-RUN (nessun UPDATE al DB)' if info['dry_run'] else 'UPDATE al DB attivo'}")
    L.append("")
    L.append("Esito per esposizione:")
    if res.empty:
        L.append("  (nessuna riga)")
    else:
        tab = res.groupby(["exposure", "status"]).size().unstack(fill_value=0)
        for s in (PASS, FAIL, NOT_EVALUABLE, SKIP, ERROR):
            if s not in tab.columns:
                tab[s] = 0
        tab = tab[[PASS, FAIL, NOT_EVALUABLE, SKIP, ERROR]]
        L.append(tab.to_string())
    L.append("")
    L.append("Regola: FAIL se p empirico > soglia, oppure segno di beta invertito, oppure |beta| < MIN_OBS_COEF,")
    L.append("dopo la rimozione. NOT_EVALUABLE (variante non toccata) se restano meno di MIN_TREATED portatori o")
    L.append("se i controlli della pipeline rifiutano il campione ridotto (SMD, matching, stima instabile).")
    L.append("")

    for title, status in (("VARIANTI INVALIDATE (FAIL) -> to_use = 0, reason = 'k_in_out'", FAIL),
                          ("NON VALUTABILI (non toccate)", NOT_EVALUABLE),
                          ("ERRORI / NON PRESENTI NEL FILE", None)):
        sub = res[res["status"] == status] if status else res[res["status"].isin([ERROR, SKIP])]
        L.append(title + f"  [{len(sub)}]")
        L.append("-" * 78)
        for _, r in sub.iterrows():
            extra = ""
            if status == FAIL or status == NOT_EVALUABLE:
                nr, nc = r.get("n_removed"), r.get("n_carriers")
                if pd.notna(nr) and pd.notna(nc):
                    extra = f"  (rimossi {int(nr)}/{int(nc)} portatori)"
            L.append(f"  {r['exposure']:<18} {r['variant']:<24} {r.get('gene_name') or '':<16} {r['reasons']}{extra}")
        L.append("")

    upd = info.get("db_updates") or []
    L.append(f"AGGIORNAMENTI DB [{len(upd)} varianti]" + ("  (simulati: dry-run)" if info["dry_run"] else ""))
    L.append("-" * 78)
    for u in upd:
        L.append(f"  {u['variant']:<24} esposizioni {', '.join(u['exposures'])}  righe to_use=1 coinvolte: {u['rows']}")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


# ============================================================================
# Entry point
# ============================================================================
def run_kinout_check(frac: float = DEFAULT_FRAC, n_perm: int = DEFAULT_N_PERM, dry_run: bool = False,
                     workers: int | None = None, baseline: bool = True) -> pd.DataFrame:
    if not 0.0 < frac < 1.0:
        raise ValueError(f"--frac must be in (0, 1), got {frac}")
    if n_perm < 100:
        raise ValueError(f"--n-perm too small to be meaningful ({n_perm})")

    cfg = get_config()
    configure_logging(cfg.log_dir)
    t0 = time.time()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gen = int(cfg.generation)
    workers = int(workers or cfg.max_workers)
    # the in-process path (workers == 1) and the parent also read the singleton
    object.__setattr__(cfg, "n_perm", n_perm)
    object.__setattr__(cfg, "n_perm_high", n_perm)

    out_dir = Path(getattr(cfg, "kinout_check_dir", "./output/kinout_check")) / f"gen{gen}"
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=== k_in_out check: generation=%d, frac=%.2f, n_perm=%d, workers=%d, dry_run=%s, baseline_control=%s ===",
             gen, frac, n_perm, workers, dry_run, baseline)
    log.info("RAW_FILE=%s | ENV_FILE=%s | sample_generation_map=%s | PCA template=%s",
             cfg.raw_file, cfg.env_file, cfg.sample_generation_map or "<OUTPUT_FOLDER default>",
             cfg.pca_covariates_path_template)
    log.info("pipeline settings: match_k=%s min_treated=%s min_sample_size=%s max_smd=%s min_obs_coef=%s "
             "pvalue_threshold=%s standardize=%s use_pca=%s random_state=%s",
             cfg.match_k, cfg.min_treated, cfg.min_sample_size, cfg.max_smd, cfg.min_obs_coef,
             cfg.pvalue_threshold, cfg.standardize, cfg.use_pca_covariates, cfg.random_state)

    # ---- 1. significant variants (stored procedure) ----
    sig = get_significant_results()
    if sig.empty:
        log.info("get_significant_results() returned no rows. Nothing to do.")
        return pd.DataFrame()
    pairs = sig.drop_duplicates(["variant", "exposure"]).reset_index(drop=True)
    log.info("get_significant_results(): %d rows, %d unique variant x exposure pairs, %d unique variants; per exposure: %s",
             len(sig), len(pairs), pairs["variant"].nunique(), pairs["exposure"].value_counts().to_dict())
    p_col, b_col = f"empirical_p_g{gen}", f"obs_coef_g{gen}"
    ref = pairs.set_index(["variant", "exposure"])

    # ---- 2. genetic data (only the needed variant columns are kept) ----
    df_gen, _, mapping, _ = _load_genetic_data(cfg)
    orig_to_safe = {v: k for k, v in mapping.items()}
    wanted = list(dict.fromkeys(pairs["variant"]))
    needed = [orig_to_safe[v] for v in wanted if v in orig_to_safe]
    log.info("genetic file: %d samples, %d/%d requested variants present", len(df_gen), len(needed), len(wanted))
    df_small = df_gen[["id"] + needed].copy()
    del df_gen
    gc.collect()

    # ---- 3. per exposure: dataset -> tests ----
    rows: list[dict] = []
    base_pkl, ext = os.path.splitext(cfg.temp_df_path)
    for exposure, sub in pairs.groupby("exposure", sort=False):
        labels = list(sub["variant"])
        log.info("--- exposure %s: %d variants ---", exposure, len(labels))
        cfg_e = dataclasses.replace(cfg, exposure=exposure)
        try:
            cov, Ecols, Ccols = _quiet(_build_narrow_covariates, cfg_e, df_small["id"])
            df = pd.merge(cov, df_small, on="id", how="inner")
        except Exception as exc:
            log.exception("exposure %s: dataset could not be built", exposure)
            rows += [{"generation": gen, "exposure": exposure, "variant": v, "status": ERROR,
                      "reasons": f"dataset could not be built: {exc}"} for v in labels]
            continue
        nas = df[Ecols + Ccols].isna().sum()
        log.info("exposure %s: n=%d samples, Ecols=%s, covariates=%s, missing per column: %s",
                 exposure, len(df), Ecols, Ccols, {k: int(v) for k, v in nas.items() if v})

        tasks = []
        for lab in labels:
            col = orig_to_safe.get(lab)
            if col is None:
                rows.append({"generation": gen, "exposure": exposure, "variant": lab, "status": SKIP,
                             "reasons": f"variant not in the genetic file ({os.path.basename(cfg.raw_file)})"})
                log.warning("exposure %s: variant %s not present in the genetic file: skipped", exposure, lab)
                continue
            tasks.append((col, lab, exposure, gen, Ecols, Ccols, frac, baseline))
        if not tasks:
            continue

        df_path = f"{base_pkl}_kinout_gen{gen}_{exposure}{ext}"
        with open(df_path, "wb") as f:
            pickle.dump(df[list(dict.fromkeys(["id", cfg.target_col] + Ecols + Ccols + [t[0] for t in tasks]))], f)
        del df, cov
        gc.collect()
        try:
            rows += _run_tasks(df_path, tasks, workers, Ccols, n_perm, cfg)
        finally:
            with contextlib.suppress(OSError):
                os.remove(df_path)

    res = pd.DataFrame(rows)
    res["generation"] = gen
    res["gene_name"] = [ref["gene_name"].get((v, e)) if (v, e) in ref.index else None
                        for v, e in zip(res["variant"], res["exposure"])]
    res["db_obs_coef"] = [ref[b_col].get((v, e)) if b_col in ref.columns and (v, e) in ref.index else None
                          for v, e in zip(res["variant"], res["exposure"])]
    res["db_empirical_p"] = [ref[p_col].get((v, e)) if p_col in ref.columns and (v, e) in ref.index else None
                             for v, e in zip(res["variant"], res["exposure"])]
    lead = ["generation", "exposure", "variant", "gene_name", "status", "reasons", "n_carriers", "n_removed",
            "n_carriers_left", "base_beta", "trim_beta", "trim_p_emp", "trim_iterations", "trim_n_treated",
            "trim_n_control", "trim_max_smd", "ctrl_status", "ctrl_beta", "ctrl_p_emp", "ctrl_reasons", "db_obs_coef", "db_empirical_p",
            "removed_ids", "seconds"]
    res = res[[c for c in lead if c in res.columns] + [c for c in res.columns if c not in lead and c != "col"]]
    res = res.sort_values(["exposure", "status", "variant"]).reset_index(drop=True)

    # control vs trimmed: a FAIL whose un-trimmed control is not a PASS is not (only) caused by the removal
    if baseline and "ctrl_status" in res.columns:
        weak = res[(res["status"] == FAIL) & (res["ctrl_status"].notna()) & (res["ctrl_status"] != PASS)]
        for _, r in weak.iterrows():
            log.warning("[%s %s] FAIL after the removal, but the SAME test with nothing removed is also %s at %d "
                        "permutations (%s): the reason 'k_in_out' is not clean for this variant, check it by hand",
                        r["exposure"], r["variant"], r["ctrl_status"], n_perm, r.get("ctrl_reasons"))

    csv_path = out_dir / f"kinout_results_gen{gen}_{stamp}.csv"
    res.to_csv(csv_path, index=False)
    log.info("results saved to %s", csv_path)

    # ---- 4. DB update ----
    db_updates = []
    failed = res[res["status"] == FAIL]
    for variant, grp in failed.groupby("variant"):
        exposures = sorted(set().union(*[set(_sibling_exposures(variant, e)) for e in grp["exposure"]]))
        rows_n = _rows_in_use(variant, exposures)
        if dry_run:
            log.info("[dry-run] %s: would set to_use=0, reason='%s' on exposures %s (%d rows with to_use=1)",
                     variant, REASON, exposures, rows_n)
        else:
            rows_n = _set_not_used(variant, exposures)
            log.info("%s: to_use=0, reason='%s' on exposures %s (%d rows updated)", variant, REASON, exposures, rows_n)
        db_updates.append({"variant": variant, "exposures": exposures, "rows": rows_n})

    counts = res["status"].value_counts().to_dict() if not res.empty else {}
    info = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"), "generation": gen, "frac": frac, "n_perm": n_perm,
        "dry_run": dry_run, "baseline_control": baseline, "reason": REASON,
        "pvalue_threshold": cfg.pvalue_threshold, "min_treated": cfg.min_treated, "min_obs_coef": cfg.min_obs_coef,
        "max_smd": cfg.max_smd, "match_k": cfg.match_k, "raw_file": cfg.raw_file, "env_file": cfg.env_file,
        "n_pairs": int(len(pairs)), "status_counts": counts, "db_updates": db_updates,
        "seconds": round(time.time() - t0, 1),
    }
    (out_dir / f"kinout_run_info_gen{gen}_{stamp}.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    _write_recap(res, out_dir / f"kinout_recap_gen{gen}_{stamp}.txt", info)

    log.info("=== k_in_out check finished in %.0fs: %s; %d variants set to to_use=0%s ===",
             time.time() - t0, counts, len(db_updates), " (DRY-RUN: nothing written)" if dry_run else "")
    return res
