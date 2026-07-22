"""
Sostituto in-memory di vqtl/db/repository.py, con la STESSA interfaccia
pubblica (stessi nomi di funzione, stessi parametri, stesso rename delle
colonne in output) ma senza alcun MySQL/MariaDB dietro: ogni "tabella" e' un
dict Python {chiave: riga}, popolato/letto con la stessa semantica di
INSERT IGNORE / UPDATE / DELETE che il vero repository esprime in SQL.

USO (vedi test_vqtl_pipeline.py):
    import sys
    import fake_vqtl_repository as fake_repo
    sys.modules["vqtl.db.repository"] = fake_repo

Va fatto PRIMA di chiamare qualunque funzione di vqtl.core.* che faccia
"from vqtl.db import repository as repo" al suo interno (import locale,
quindi risolto al momento della chiamata, non all'import del modulo) --
Python trova "vqtl.db.repository" gia' in sys.modules e non esegue mai il
vero db/repository.py (che altrimenti fallirebbe all'import perche' importa
gene_environment.db.connection, non presente in questo ambiente di test).

Perche' questo approccio e non chiamate dirette alle funzioni pure
(_beta_qi_and_asymptotic_se, fit_interaction, ecc.): permette di testare il
codice di orchestrazione REALE (run_vqtl_scan, filter_candidates,
run_interaction_tests, run_rge_het, run_robustness_and_permutation) cosi'
com'e', non una sua reimplementazione parallela nello script di test -- e'
proprio l'orchestrazione (fingerprint, placeholder, resume, short-circuit,
rename di colonne fra uno step e l'altro) il punto in cui si sono trovati i
bug reali (vedi run_vqtl_scan/variant_subset, load_vqtl_dataset/PCA).
"""
from __future__ import annotations

import pandas as pd

# ============================================================
# Storage: un dict per "tabella", stesso nome della tabella SQL reale.
# Chiave = tupla della primary key (stessa composizione di db/schema.sql).
# ============================================================

_scan_results: dict[tuple, dict] = {}
_scan_results_significant: dict[tuple, dict] = {}
_scan_runs: dict[int, dict] = {}
_interaction_results: dict[tuple, dict] = {}
_interaction_results_significant: dict[tuple, dict] = {}
_rge_het_results: dict[tuple, dict] = {}
_robustness_results: dict[tuple, dict] = {}
_permutation_results: dict[tuple, dict] = {}

_KEYED_STORES = {
    "interaction": (_interaction_results, []),
    "rge_het": (_rge_het_results, []),
    "robustness": (_robustness_results, ["phenotype_variant"]),
    "permutation": (_permutation_results, []),
}


def reset_all() -> None:
    """Svuota tutte le tabelle finte -- utile fra un run e l'altro dello
    script di test se si vuole ripartire da zero senza riavviare il processo."""
    for d in (
        _scan_results, _scan_results_significant, _scan_runs,
        _interaction_results, _interaction_results_significant,
        _rge_het_results, _robustness_results, _permutation_results,
    ):
        d.clear()


# ============================================================
# Step 3+4: vqtl_scan_results / vqtl_scan_results_significant / vqtl_scan_runs
# ============================================================

def get_scan_fingerprint(generation: int) -> dict | None:
    row = _scan_runs.get(generation)
    return row["fingerprint"] if row else None


def reset_scan_run(generation: int, fingerprint: dict) -> None:
    for k in [k for k in _scan_results if k[0] == generation]:
        del _scan_results[k]
    _scan_runs[generation] = {"generation": generation, "fingerprint": fingerprint}


def ensure_scan_placeholders(generation: int, variants: list[dict], chunk_size: int = 5000) -> int:
    n = 0
    for v in variants:
        key = (generation, v["variant"])
        if key not in _scan_results:
            _scan_results[key] = {
                "generation": generation, "variant": v["variant"],
                "chromosome": v.get("chromosome"), "position": v.get("position"),
                "status": "pending", "n": None, "maf": None, "beta_qi": None, "se": None,
                "z": None, "p": None, "p_gc": None, "fdr_gc": None, "is_candidate": 0,
                "error_message": None,
            }
            n += 1
    return n


def get_done_scan_variants(generation: int) -> set[str]:
    return {v for (g, v), row in _scan_results.items() if g == generation and row["status"] == "done"}


def mark_scan_in_progress(generation: int, variant_list: list[str]) -> None:
    for v in variant_list:
        key = (generation, v)
        if key in _scan_results:
            _scan_results[key]["status"] = "in_progress"


def save_scan_chunk_results(generation: int, rows: list[dict]) -> None:
    for r in rows:
        key = (generation, r["variant"])
        if key not in _scan_results:
            continue
        _scan_results[key].update({
            "status": r.get("status", "done"), "n": r.get("n"), "maf": r.get("maf"),
            "beta_qi": r.get("beta_qi"), "se": r.get("se"), "z": r.get("z"), "p": r.get("p"),
            "error_message": r.get("error_message"),
        })


def update_gc_correction(generation: int, rows: list[dict]) -> None:
    for r in rows:
        key = (generation, r["variant"])
        if key in _scan_results:
            _scan_results[key]["p_gc"] = r["p_gc"]
            _scan_results[key]["fdr_gc"] = r["fdr_gc"]


def mark_candidates(generation: int, variant_list: list[str]) -> None:
    wanted = set(variant_list)
    for (g, v), row in _scan_results.items():
        if g == generation:
            row["is_candidate"] = 1 if v in wanted else 0


_SCAN_SIG_COLS = ["variant", "chromosome", "position", "n", "maf", "beta_qi", "se", "z", "p", "p_gc", "fdr_gc"]


def count_significant_scan(generation: int) -> int:
    return sum(1 for (g, _v) in _scan_results_significant if g == generation)


def sync_scan_significant(generation: int) -> int:
    for k in [k for k in _scan_results_significant if k[0] == generation]:
        del _scan_results_significant[k]
    n = 0
    for (g, v), row in _scan_results.items():
        if g == generation and row.get("is_candidate") == 1 and row["status"] == "done":
            _scan_results_significant[(g, v)] = {"generation": g, "variant": v, **{c: row[c] for c in _SCAN_SIG_COLS}}
            n += 1
    return n


_SCAN_COLUMNS = ["variant", "chromosome", "position", "n", "maf", "beta_qi", "se", "z", "p", "p_gc", "fdr_gc", "is_candidate"]
_SCAN_RENAME = {"chromosome": "CHR", "position": "POS", "n": "N", "beta_qi": "beta_QI", "maf": "MAF", "se": "SE", "z": "Z", "p": "P", "p_gc": "P_gc", "fdr_gc": "fdr_gc"}


def get_scan_results(generation: int, only_done: bool = True) -> pd.DataFrame:
    rows = []
    for (g, _v), row in _scan_results.items():
        if g != generation:
            continue
        if only_done and not (row["status"] == "done" and row["p"] is not None):
            continue
        rows.append({c: row[c] for c in _SCAN_COLUMNS})
    df = pd.DataFrame(rows, columns=_SCAN_COLUMNS)
    df = df.rename(columns={"variant": "SNP", **_SCAN_RENAME})
    if not df.empty:
        df = df.sort_values("P").reset_index(drop=True)
    return df


def get_candidates(generation: int) -> pd.DataFrame:
    df = get_scan_results(generation, only_done=True)
    if df.empty:
        return df
    return df[df["is_candidate"] == 1].reset_index(drop=True)


# ============================================================
# Tabelle "keyed" generiche: interaction / rge_het / robustness / permutation
# ============================================================

_STAT_COLS = {
    "interaction": ["beta_i", "se", "pval", "n", "maf"],
    "rge_het": [
        "rge_beta_exposure_on_snp", "rge_se", "rge_pval", "rge_flag",
        "het_bp_lm_stat", "het_bp_lm_pvalue", "het_bp_f_stat", "het_bp_f_pvalue",
        "heteroscedasticity_flag",
    ],
    "robustness": ["beta_i", "se", "pval", "n", "maf"],
    "permutation": [
        "beta_i_observed", "n_perm_valid", "empirical_pval", "asymptotic_pval",
        "levene_stat_observed", "levene_pval", "levene_n_perm_valid",
    ],
}


def _key(name: str, generation: int, r: dict) -> tuple:
    _store, extra = _KEYED_STORES[name]
    return (generation, r["variant"], r["exposure"], *[r[k] for k in extra])


def ensure_placeholders(name: str, generation: int, rows: list[dict], chunk_size: int = 2000) -> int:
    store, extra = _KEYED_STORES[name]
    n = 0
    for r in rows:
        key = _key(name, generation, r)
        if key not in store:
            row = {
                "generation": generation, "variant": r["variant"], "exposure": r["exposure"],
                "chromosome": r.get("chromosome"), "position": r.get("position"),
                "status": "pending", "error_message": None,
            }
            for k in extra:
                row[k] = r[k]
            for c in _STAT_COLS[name]:
                row[c] = None
            store[key] = row
            n += 1
    return n


def get_done_keys(name: str, generation: int) -> set[tuple]:
    store, extra = _KEYED_STORES[name]
    out = set()
    for key, row in store.items():
        if key[0] == generation and row["status"] == "done":
            out.add(tuple([row["variant"], row["exposure"]] + [row[k] for k in extra]))
    return out


def bulk_update_status(name: str, generation: int, rows: list[dict]) -> None:
    store, extra = _KEYED_STORES[name]
    for r in rows:
        key = _key(name, generation, r)
        if key not in store:
            continue
        store[key]["status"] = r.get("status", "done")
        store[key]["error_message"] = r.get("error_message")
        for c in _STAT_COLS[name]:
            store[key][c] = r.get(c)


def fetch_results(name: str, generation: int, only_done: bool = True) -> pd.DataFrame:
    store, extra = _KEYED_STORES[name]
    cols = ["variant", "exposure", "chromosome", "position"] + extra + _STAT_COLS[name]
    rows = []
    for key, row in store.items():
        if key[0] != generation:
            continue
        if only_done and row["status"] != "done":
            continue
        rows.append({c: row[c] for c in cols})
    df = pd.DataFrame(rows, columns=cols)
    return df.rename(columns={"variant": "SNP", "chromosome": "CHR", "position": "POS"})


def clear_downstream_for_variants(generation: int, variant_list: list[str]) -> None:
    wanted = set(variant_list)
    for store, _extra in _KEYED_STORES.values():
        for key in [k for k in store if k[0] == generation and k[1] in wanted]:
            del store[key]


# ============================================================
# vqtl_interaction_results_significant
# ============================================================

_INTERACTION_SIG_COLS = ["variant", "exposure", "chromosome", "position", "beta_i", "se", "pval", "n", "maf"]


def sync_interaction_significant(generation: int, p_threshold: float) -> int:
    for k in [k for k in _interaction_results_significant if k[0] == generation]:
        del _interaction_results_significant[k]
    n = 0
    for (g, v, e), row in _interaction_results.items():
        if g == generation and row["status"] == "done" and row.get("pval") is not None and row["pval"] < p_threshold:
            _interaction_results_significant[(g, v, e)] = {
                "generation": g, **{c: row[c] for c in _INTERACTION_SIG_COLS}
            }
            n += 1
    return n


def get_interaction_significant(generation: int) -> pd.DataFrame:
    rows = [
        {c: row[c] for c in _INTERACTION_SIG_COLS}
        for (g, _v, _e), row in _interaction_results_significant.items() if g == generation
    ]
    df = pd.DataFrame(rows, columns=_INTERACTION_SIG_COLS)
    return df.rename(columns={"variant": "SNP", "chromosome": "CHR", "position": "POS"})
