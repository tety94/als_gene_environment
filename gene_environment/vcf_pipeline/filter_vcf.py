"""
Filtraggio VCF -> PLINK binario -> MAF filter -> LD pruning -> VCF filtrato.
(ex gene_reduction.py)
"""
from __future__ import annotations

import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

from gene_environment.config import get_config
from gene_environment.logging_utils import configure_logging, get_logger

log = get_logger(__name__)

OUTPUT_SUBFOLDER = "vcf_filtered"


def _run(cmd: list[str], log_prefix: str) -> None:
    log.debug("%s: eseguo: %s", log_prefix, " ".join(cmd))
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def filter_single_vcf(input_path: str, output_vcf_folder: str, cfg_dict: dict) -> tuple[str, bool, str | None]:
    """Esegue l'intera catena plink2 per un singolo VCF. Ritorna
    (nome_file, successo, messaggio_errore)."""
    from gene_environment.logging_utils import configure_logging as _cfg_log
    _cfg_log(cfg_dict["log_dir"])  # necessario nei worker separati

    vcf_file = os.path.basename(input_path)
    base_name = os.path.splitext(os.path.splitext(vcf_file)[0])[0]  # rimuove .vcf.gz

    final_vcf_path = os.path.join(output_vcf_folder, base_name + "_filtered.vcf")
    if os.path.exists(final_vcf_path):
        log.info("[%s] output già presente, salto: %s", vcf_file, final_vcf_path)
        return vcf_file, True, None

    try:
        plink_prefix = os.path.join(output_vcf_folder, base_name + "_plink")
        _run(["plink2", "--vcf", input_path, "--make-bed", "--out", plink_prefix], vcf_file)

        fam_file = plink_prefix + ".fam"
        remove_file = plink_prefix + "_remove.txt"
        exclude_prefixes = tuple(cfg_dict["exclude_id_prefixes"])

        n_removed = 0
        with open(fam_file) as f, open(remove_file, "w") as out:
            for line in f:
                fid, iid = line.strip().split()[:2]
                if exclude_prefixes and iid.startswith(exclude_prefixes):
                    out.write(f"{fid} {iid}\n")
                    n_removed += 1
        if n_removed:
            log.info("[%s] %d campioni esclusi per prefisso id (%s)", vcf_file, n_removed, exclude_prefixes)

        plink_maf_prefix = os.path.join(output_vcf_folder, base_name + "_maf")
        remove_args = ["--remove", remove_file] if n_removed else []
        _run(
            ["plink2", "--bfile", plink_prefix, *remove_args,
             "--maf", str(cfg_dict["maf_threshold"]), "--make-bed", "--out", plink_maf_prefix],
            vcf_file,
        )

        plink_prune_prefix = os.path.join(output_vcf_folder, base_name + "_pruned")
        _run(
            ["plink2", "--bfile", plink_maf_prefix, "--indep-pairwise",
             str(cfg_dict["ld_window_size"]), str(cfg_dict["ld_step"]), str(cfg_dict["ld_r2_threshold"]),
             "--out", plink_prune_prefix],
            vcf_file,
        )

        _run(
            ["plink2", "--bfile", plink_maf_prefix, "--extract", plink_prune_prefix + ".prune.in",
             "--recode", "vcf", "--out", os.path.splitext(final_vcf_path)[0]],
            vcf_file,
        )

        log.info("[%s] filtrato con successo -> %s", vcf_file, final_vcf_path)
        return vcf_file, True, None

    except subprocess.CalledProcessError as e:
        err = f"{e}\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}"
        log.error("[%s] fallito: %s", vcf_file, err)
        return vcf_file, False, err


def run_filter_vcf(exclude_id_prefixes: list[str] | None = None) -> None:
    cfg = get_config()
    configure_logging(cfg.log_dir)
    exclude_id_prefixes = exclude_id_prefixes or []

    cfg_dict = {
        "maf_threshold": cfg.maf_threshold,
        "ld_window_size": cfg.ld_window_size,
        "ld_step": cfg.ld_step,
        "ld_r2_threshold": cfg.ld_r2_threshold,
        "exclude_id_prefixes": exclude_id_prefixes,
        "log_dir": cfg.log_dir,
    }

    jobs = []
    for input_folder in cfg.vcf_folders:
        output_vcf_folder = os.path.join(input_folder, OUTPUT_SUBFOLDER)
        os.makedirs(output_vcf_folder, exist_ok=True)
        vcf_files = [
            f for f in os.listdir(input_folder)
            if f.endswith(".vcf.gz") and not f.startswith("._")
        ]
        for vcf_file in vcf_files:
            jobs.append((os.path.join(input_folder, vcf_file), output_vcf_folder))

    log.info("Filtraggio VCF: %d file da processare con %d worker", len(jobs), cfg.max_workers)

    failed = []
    with ProcessPoolExecutor(max_workers=cfg.max_workers) as ex:
        futures = {ex.submit(filter_single_vcf, inp, outp, cfg_dict): inp for inp, outp in jobs}
        for fut in as_completed(futures):
            vcf_file, ok, err = fut.result()
            if not ok:
                failed.append((vcf_file, err))

    if failed:
        log.error("Filtraggio completato con %d errori su %d file: %s", len(failed), len(jobs), [f for f, _ in failed])
    else:
        log.info("Filtraggio VCF completato senza errori (%d file).", len(jobs))


if __name__ == "__main__":
    run_filter_vcf()
