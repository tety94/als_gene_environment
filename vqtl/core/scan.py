"""
Step 3 - scan vQTL genome-wide (stile QUAIL), invariato nella logica
statistica rispetto al progetto originale (vedi README storico, sezione 4):

  1. Residualizza il fenotipo standardizzato sulle covariate (OLS) ->
     R = Y - covariate*b.
  2. Per ogni SNP e ogni tau in cfg.taus: quantile regression di R su
     dosage a tau e a tau+0.5; beta_diff(tau) = beta_{tau+0.5} - beta_{tau}.
  3. beta_QI = media_tau(beta_diff(tau)) (effetto quantile-integrato).
  4. SE(beta_QI): due modalita' disponibili via cfg.se_method.

UNICA differenza rispetto all'originale: il dosaggio arriva da una colonna
del DataFrame gia' costruito da `core.data.load_vqtl_dataset` (quindi gia'
allineato per campione, gia' filtrato MAF/LD a monte) invece che da un file
VCF per cromosoma letto con cyvcf2 -- niente piu' manifest, niente logica di
iterazione sui VCF qui dentro. Le colonne vengono processate in chunk e
parallelizzate con joblib esattamente come prima (`cfg.n_jobs`,
`cfg.chunk_size`), passando ai worker solo il sotto-blocco numpy del chunk
(non l'intero DataFrame) per lo stesso motivo di efficienza dell'originale.

=== NOTA SU se_method (rivisto dopo audit, vedi CHANGELOG_VQTL_BUGFIX.md) ===

In origine "asymptotic" combinava le SE per-tau di QuantReg con una formula
delta-method (mean(ses), o mean(ses)/sqrt(n_taus)): ENTRAMBE le varianti
si sono rivelate sbagliate, perche' le n_taus stime beta_diff(tau) NON sono
indipendenti -- sono tutte fit sugli STESSI identici dati (stesso dosaggio,
stesso residuo), solo su quantili diversi e spesso ravvicinati (default
0.05,0.10,...,0.45: 9 tau, passo 0.05). Trattarle come indipendenti (dividere
per sqrt(n_taus)) sottostima la SE (lambda_GC >> 1, troppo liberale);
ignorare la correlazione senza dividere affatto (mean(ses)) la sovrastima
(lambda_GC << 1, troppo conservativo). Verificato empiricamente su dati
sintetici con ground truth nota: un bootstrap (resampling per-soggetto, che
non fa alcuna assunzione di indipendenza fra tau) da' lambda_GC ~1 sullo
stesso identico pool nullo, a conferma che il problema era nella formula,
non nei dati.

Non esiste una formula chiusa semplice per la SE corretta senza stimare la
covarianza reale fra le stime per-tau (richiederebbe il sandwich asintotico
congiunto del processo di quantile regression, non banale da implementare
qui). La soluzione adottata: se_method="asymptotic" ora esegue un
mini-bootstrap (VQTL_ASYMPTOTIC_BOOTSTRAP_K repliche, default 50 -- molto
meno del bootstrap "pieno" usato per la validazione allo Step 7, ma
sufficiente per una SE calibrata correttamente) invece della vecchia
formula delta-method. Il nome del parametro resta "asymptotic" per non
rompere la configurazione esistente (VQTL_SE_METHOD nel .env), ma la
computazione sottostante e' cambiata. se_method="bootstrap" resta
disponibile per un bootstrap piu' robusto (VQTL_BOOTSTRAP_K, default 200,
usato tipicamente solo sui top loci allo Step 7 per costo computazionale).
"""
from __future__ import annotations

import threading
import time
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed
from scipy import stats
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.tools.sm_exceptions import IterationLimitWarning

from gene_environment.logging_utils import get_logger

from vqtl.config import VqtlConfig
from vqtl.core.data import VqtlDataset, dosage_matrix, variant_chrom_pos

log = get_logger(__name__)

# Il dosaggio e' un predittore discreto (0/1/2): il solver simplex di
# QuantReg puo' oscillare vicino al limite di iterazioni su soluzioni
# quasi-degeneri. Cap volutamente piu' basso del default statsmodels
# (500/1e-6): ~5x piu' veloce genoma-wide per una perdita di precisione
# trascurabile (invariato dall'audit del progetto originale).
QR_MAX_ITER = 100
QR_P_TOL = 1e-3

# ============================================================
# Contatori di convergenza (diagnostica). ATTENZIONE: sono in-process --
# con Parallel(backend="loky") ogni worker e' un processo separato, quindi
# questi contatori nel processo principale NON riflettono cio' che accade
# nei worker paralleli. Affidabili solo con n_jobs=1 (uso previsto: script
# di debug), oppure se in futuro vengono aggregati esplicitamente dal
# valore di ritorno di _process_chunk invece che da uno stato globale.
# ============================================================

_convergence_lock = threading.Lock()
_convergence_stats = {"tau_fits_attempted": 0, "tau_fits_discarded": 0, "variants_all_nan": 0}


def reset_convergence_stats() -> None:
    with _convergence_lock:
        for k in _convergence_stats:
            _convergence_stats[k] = 0


def get_convergence_stats() -> dict:
    with _convergence_lock:
        return dict(_convergence_stats)


def _record_tau_fit(discarded: bool) -> None:
    with _convergence_lock:
        _convergence_stats["tau_fits_attempted"] += 1
        if discarded:
            _convergence_stats["tau_fits_discarded"] += 1


def _record_variant_all_nan() -> None:
    with _convergence_lock:
        _convergence_stats["variants_all_nan"] += 1


def residualize(y: np.ndarray, covariates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Residualizza y sulle covariate (OLS). Pubblica (non piu' _residualize)
    perche' riusata anche da vqtl.core.permutation per il test di Levene
    permutazionale sul locus (stessa logica di correzione per covariate
    dello scan Step 3, niente da duplicare)."""
    X = sm.add_constant(covariates.astype(float))
    ok = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
    model = sm.OLS(y[ok], X[ok]).fit()
    resid = np.full_like(y, np.nan, dtype=float)
    resid[ok] = model.resid
    return resid, ok


def _beta_qi_single(dosage: np.ndarray, resid: np.ndarray, taus: list[float]) -> tuple[float, int]:
    """Solo beta_QI: un fit per tau e uno per tau+0.5, media dei diffs.
    Usata sia dal bootstrap "pieno" (resampling per-soggetto, K ripetizioni
    di questa funzione) sia dal mini-bootstrap di se_method="asymptotic"
    (vedi _beta_qi_and_asymptotic_se) -- stessa routine, contesti diversi
    di chiamata, nessuna duplicazione della logica di fit."""
    ok = ~np.isnan(dosage) & ~np.isnan(resid)
    d, r = dosage[ok], resid[ok]
    if len(np.unique(d)) < 2 or len(d) < 20:
        return np.nan, int(ok.sum())
    X = sm.add_constant(d)
    diffs = []
    for tau in taus:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", IterationLimitWarning)
                fit_lo = QuantReg(r, X).fit(q=tau, max_iter=QR_MAX_ITER, p_tol=QR_P_TOL)
                fit_hi = QuantReg(r, X).fit(q=tau + 0.5, max_iter=QR_MAX_ITER, p_tol=QR_P_TOL)
            diffs.append(fit_hi.params[1] - fit_lo.params[1])
            _record_tau_fit(discarded=False)
        except (IterationLimitWarning, Exception):
            _record_tau_fit(discarded=True)
            continue
    if not diffs:
        _record_variant_all_nan()
        return np.nan, int(ok.sum())
    return float(np.mean(diffs)), int(ok.sum())


def _beta_qi_and_asymptotic_se(
    dosage: np.ndarray, resid: np.ndarray, taus: list[float], mini_bootstrap_k: int, seed: int,
) -> tuple[float, float, int]:
    """beta_QI (fit sui dati originali, un fit per tau e uno per tau+0.5 --
    invariato) + SE stimata via mini-bootstrap (mini_bootstrap_k repliche di
    _beta_qi_single su resampling per-soggetto), non piu' via combinazione
    delle SE asintotiche per-tau di QuantReg -- vedi nota nel docstring del
    modulo sul perche' la vecchia formula delta-method era sistematicamente
    distorta (le stime per-tau sono correlate, non indipendenti)."""
    ok = ~np.isnan(dosage) & ~np.isnan(resid)
    d, r = dosage[ok], resid[ok]
    if len(np.unique(d)) < 2 or len(d) < 20:
        return np.nan, np.nan, int(ok.sum())
    X = sm.add_constant(d)
    diffs = []
    for tau in taus:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", IterationLimitWarning)
                fit_lo = QuantReg(r, X).fit(q=tau, max_iter=QR_MAX_ITER, p_tol=QR_P_TOL)
                fit_hi = QuantReg(r, X).fit(q=tau + 0.5, max_iter=QR_MAX_ITER, p_tol=QR_P_TOL)
            diffs.append(fit_hi.params[1] - fit_lo.params[1])
            _record_tau_fit(discarded=False)
        except (IterationLimitWarning, Exception):
            _record_tau_fit(discarded=True)
            continue
    if not diffs:
        _record_variant_all_nan()
        return np.nan, np.nan, int(ok.sum())
    beta_qi = float(np.mean(diffs))

    # SE via mini-bootstrap (resampling per-soggetto, K piccolo per restare
    # veloce genoma-wide) invece della vecchia combinazione delle SE
    # asintotiche per-tau.
    n = len(d)
    rng = np.random.default_rng(seed)
    boot = np.empty(mini_bootstrap_k)
    for b in range(mini_bootstrap_k):
        idx = rng.integers(0, n, n)
        bqi, _ = _beta_qi_single(d[idx], r[idx], taus)
        boot[b] = bqi if not np.isnan(bqi) else 0.0
    se = float(np.nanstd(boot, ddof=1))

    return beta_qi, se, int(ok.sum())


def _process_chunk(
    chunk_idx: int, dosage_chunk: np.ndarray, col_names: list[str], resid: np.ndarray, taus: list[float],
    se_method: str, bootstrap_k: int, asymptotic_bootstrap_k: int, min_maf: float, min_call_rate: float, seed: int,
) -> tuple[int, list[dict]]:
    """Ritorna SEMPRE una riga per ogni variante del chunk (mai un "buco"):
    una variante scartata dai filtri MAF/call-rate o per cui QuantReg non
    converge e' comunque status='done' (statistiche a None), non assente --
    altrimenti a DB resterebbe 'pending' per sempre e un run successivo la
    ritenterebbe all'infinito. status='failed' solo per eccezioni vere."""
    rng = np.random.default_rng(seed)
    rows = []
    for j, col in enumerate(col_names):
        try:
            dosage = dosage_chunk[:, j]

            call_rate = 1 - np.isnan(dosage).mean()
            maf = np.nanmean(dosage) / 2 if not np.isnan(np.nanmean(dosage)) else np.nan
            maf = min(maf, 1 - maf) if not np.isnan(maf) else np.nan
            if call_rate < min_call_rate or np.isnan(maf) or maf < min_maf:
                rows.append({"variant_safe": col, "status": "done", "n": None, "maf": None,
                             "beta_qi": None, "se": None, "z": None, "p": None, "error_message": None})
                continue

            ok = ~np.isnan(dosage) & ~np.isnan(resid)
            d, r = dosage[ok], resid[ok]

            if se_method == "bootstrap":
                beta_qi, n_used = _beta_qi_single(dosage, resid, taus)
                se = np.nan
                if not np.isnan(beta_qi):
                    n = len(d)
                    boot = np.empty(bootstrap_k)
                    for b in range(bootstrap_k):
                        idx = rng.integers(0, n, n)
                        bqi, _ = _beta_qi_single(d[idx], r[idx], taus)
                        boot[b] = bqi if not np.isnan(bqi) else 0.0
                    se = float(np.nanstd(boot, ddof=1))
            else:
                # "asymptotic": mini-bootstrap interno, vedi
                # _beta_qi_and_asymptotic_se. seed derivato da (seed, j) per
                # restare deterministico per variante indipendentemente
                # dall'ordine di elaborazione del chunk.
                beta_qi, se, n_used = _beta_qi_and_asymptotic_se(
                    dosage, resid, taus, asymptotic_bootstrap_k, seed=seed * 100_003 + j,
                )

            if np.isnan(beta_qi) or se is None or np.isnan(se) or se == 0:
                rows.append({"variant_safe": col, "status": "done", "n": int(n_used) if n_used else None,
                             "maf": round(float(maf), 4), "beta_qi": None, "se": None, "z": None, "p": None,
                             "error_message": None})
                continue

            z = beta_qi / se
            p = 2 * (1 - stats.norm.cdf(abs(z)))
            rows.append({"variant_safe": col, "status": "done", "n": int(n_used), "maf": round(float(maf), 4),
                         "beta_qi": beta_qi, "se": se, "z": z, "p": p, "error_message": None})
        except Exception as e:
            rows.append({"variant_safe": col, "status": "failed", "n": None, "maf": None, "beta_qi": None,
                         "se": None, "z": None, "p": None, "error_message": str(e)[:500]})
    return chunk_idx, rows


def _scan_fingerprint(vcfg: VqtlConfig, cols: list[str]) -> dict:
    """Firma della configurazione di questo scan: se cambia (varianti,
    parametri statistici...) le righe gia' salvate a DB per questa
    generazione non sono piu' valide e vengono scartate (vqtl_scan_results
    ripulita, si riparte da zero) invece di essere riusate per sbaglio."""
    return {
        "n_variants": len(cols),
        "first_variant": cols[0] if cols else None,
        "last_variant": cols[-1] if cols else None,
        "chunk_size": vcfg.chunk_size,
        "taus": vcfg.taus,
        "se_method": vcfg.se_method,
        "bootstrap_k": vcfg.bootstrap_k,
        "asymptotic_bootstrap_k": getattr(vcfg, "asymptotic_bootstrap_k", None),
        "min_maf": vcfg.min_maf,
        "min_call_rate": vcfg.min_call_rate,
    }


def run_vqtl_scan(
    dataset: VqtlDataset, vcfg: VqtlConfig, target_col: str, generation: int, force: bool = False,
    variant_subset: list[str] | None = None,
) -> pd.DataFrame:
    """Scan vQTL genoma-wide, con stato persistito a DB (vqtl_scan_results,
    vedi vqtl/db/repository.py e db/schema.sql) invece che su file: un
    placeholder status='pending' viene inserito per ogni variante prima di
    iniziare, e ogni chunk completato viene aggiornato (bulk UPDATE) SUBITO,
    non alla fine dello scan. Se il processo viene interrotto (Ctrl+C, OOM,
    job del cluster ucciso, ecc.), al riavvio le varianti gia' 'done' non
    vengono ripetute -- si riparte da dove si era interrotto, non da zero.
    `force=True` (da --force in cli.py) ripulisce tutto e riparte da capo
    anche se la fingerprint corrisponde.

    `variant_subset` (colonne "safe", vedi VqtlDataset.variant_cols):
    restringe lo scan a un elenco esplicito di varianti invece che a
    dataset.variant_cols per intero -- e' il parametro che cli.py passa da
    `--significant-only` (via select_variants_from_significant_results).
    None (default) = scan genoma-wide completo, comportamento invariato."""
    from vqtl.db import repository as repo

    y_col = f"{target_col}_z"
    cols = dataset.variant_cols if variant_subset is None else variant_subset
    if variant_subset is not None:
        log.info("Scan vQTL limitato a un subset esplicito di %d varianti (--significant-only).", len(cols))

    asymptotic_bootstrap_k = getattr(vcfg, "asymptotic_bootstrap_k", 50)
    log.info(
        "Step 3 - scan vQTL: %d varianti, taus=%s, se_method=%s (asymptotic_bootstrap_k=%s, bootstrap_k=%s), "
        "n_jobs=%s, chunk_size=%s",
        len(cols), vcfg.taus, vcfg.se_method, asymptotic_bootstrap_k, vcfg.bootstrap_k, vcfg.n_jobs, vcfg.chunk_size,
    )

    covariates = dataset.df[dataset.covariate_cols].to_numpy(dtype=float) if dataset.covariate_cols else np.zeros((len(dataset.df), 0))
    resid, ok_mask = residualize(dataset.df[y_col].to_numpy(dtype=float), covariates)
    log.info("Residualizzazione su covariate %s: %d/%d campioni completi", dataset.covariate_cols, int(ok_mask.sum()), len(dataset.df))

    # ---- Fingerprint: se e' cambiata (o e' il primo run), ripulisce
    # vqtl_scan_results per questa generazione e reinserisce i placeholder ----
    fingerprint = _scan_fingerprint(vcfg, cols)
    cached_fp = None if force else repo.get_scan_fingerprint(generation)
    if cached_fp != fingerprint:
        if force:
            log.info("--force: riparto da zero (ignoro eventuale fingerprint gia' salvata).")
        else:
            log.info("Nessuna fingerprint valida trovata (o parametri cambiati): scan da zero.")
        repo.reset_scan_run(generation, fingerprint)

    variants_meta = []
    for c in cols:
        real = dataset.mapping[c]
        chrom, pos = variant_chrom_pos(real)
        variants_meta.append({"variant": real, "chromosome": chrom, "position": pos})
    repo.ensure_scan_placeholders(generation, variants_meta)

    done_variants = repo.get_done_scan_variants(generation)
    inv_mapping = {v: k for k, v in dataset.mapping.items()}
    done_safe = {inv_mapping[v] for v in done_variants if v in inv_mapping}
    todo_cols = [c for c in cols if c not in done_safe]

    if done_safe:
        log.info(
            "Ripresa da DB: %d/%d varianti gia' completate in un run precedente, "
            "ricalcolo solo le restanti %d.", len(done_safe), len(cols), len(todo_cols),
        )

    chunks = [todo_cols[i:i + vcfg.chunk_size] for i in range(0, len(todo_cols), vcfg.chunk_size)]
    if todo_cols:
        # marca subito tutte le varianti in coda come 'in_progress': permette
        # di vedere lo stato di avanzamento live da un'altra sessione con
        # "SELECT status, COUNT(*) FROM vqtl_scan_results WHERE generation=N GROUP BY status"
        # mentre lo scan e' ancora in corso, senza aspettare che i chunk finiscano.
        repo.mark_scan_in_progress(generation, [dataset.mapping[c] for c in todo_cols])
    n_chunks = len(chunks)
    if n_chunks:
        log.info("%d chunk da fino a %d varianti", n_chunks, vcfg.chunk_size)

    t0 = time.monotonic()
    if chunks:
        gen = Parallel(n_jobs=vcfg.n_jobs, backend="loky", return_as="generator_unordered")(
            delayed(_process_chunk)(
                i, dosage_matrix(dataset, chunk_cols), chunk_cols, resid, vcfg.taus,
                vcfg.se_method, vcfg.bootstrap_k, asymptotic_bootstrap_k, vcfg.min_maf, vcfg.min_call_rate, seed=i,
            )
            for i, chunk_cols in enumerate(chunks)
        )
        n_done_this_run = 0
        log_every = max(1, n_chunks // 10)
        for chunk_idx, rows in gen:
            n_done_this_run += 1
            for r in rows:
                r["variant"] = dataset.mapping[r.pop("variant_safe")]
            repo.save_scan_chunk_results(generation, rows)
            if n_done_this_run % log_every == 0 or n_done_this_run == n_chunks:
                log.info(
                    "Scan vQTL: %d/%d varianti completate (%d chunk fatti in questo run)",
                    len(done_safe) + n_done_this_run * vcfg.chunk_size, len(cols), n_done_this_run,
                )
    elapsed = time.monotonic() - t0
    log.info("Scan vQTL: elaborazione completata in %.1fs.", elapsed)

    df_res = repo.get_scan_results(generation, only_done=True)
    log.info("Scan vQTL completato: %d varianti con un risultato valido (su %d testate).", len(df_res), len(cols))
    return df_res