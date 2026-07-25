"""
Synthetic dataset generator (genetics + environment + population PCs),
PARAMETRIZED REFACTOR of the original version.

Everything that used to be a module-level constant (N_PATIENTS,
CAUSAL_VARIANTS, PURE_VARIANCE_VARIANTS, prop_unexposed, missing rate,
RNG_SEED, etc.) is now an argument of `generate_dataset(...)`, so the same
generator can be called in a loop from a multi-scenario orchestration script
(see run_scenarios.py) without duplicating code or hand-editing constants
for every run.

When launched from the command line it behaves like the original script
(same defaults), but accepts two optional arguments:
  --out-dir       output folder (default: <script_dir>/fake_data)
  --config-json   path to a JSON file with a subset of generate_dataset()
                   kwargs to override relative to the defaults (used by the
                   multi-scenario orchestrator)

NEW STRESS-TEST PARAMETERS (absent from the original version, all disabled
by default -> identical behaviour to the original if not specified):

  - subpop_frac / subpop_onset_shift / subpop_maf_shift: simulate a real
    population structure (two subpopulations with different MAF and
    baseline onset_age) that the PCs written to output do NOT capture (the
    PCs remain independent Gaussian noise, as in the original) -- used to
    stress-test the correction for stratification: if lambda_GC rises
    noticeably relative to the "no stratification" scenario, correction via
    (here, uninformative) PCs is not enough, exactly as can happen in
    production if the loaded PCs are not informative.

  - nonrandom_missing_carrier_rate: if different from None, the probability
    of missing genotype for CARRIERS uses this value instead of
    `missing_rate` (non-carriers stay at missing_rate) -- simulates
    informative missingness (correlated with genotype), which a plain
    downstream dropna() does not handle specially.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_CAUSAL_VARIANTS = {
    # original, unchanged (backward compatibility with scenarios already run)
    "1_1000001_A_G": (-4.5, -1.0),
    "2_2000002_C_T": (4.0, 0.5),
    "3_3000003_G_A": (-3.5, 0.0),
    "4_4000004_T_C": (3.2, -0.5),
    "5_5000005_A_T": (-5.0, 1.0),

    # balanced grid: 16 magnitudes (1.0 -> 8.5, step 0.5), each with a +/-
    # pair, main effect alternated/decorrelated from the interaction sign
    "1_1300000_A_G": (1.0, 0.0),
    "1_1300001_C_T": (-1.0, 0.0),
    "2_1300010_A_G": (1.5, -0.3),
    "2_1300011_C_T": (-1.5, 0.3),
    "3_1300020_A_G": (2.0, 0.5),
    "3_1300021_C_T": (-2.0, -0.5),
    "4_1300030_A_G": (2.5, -0.5),
    "4_1300031_C_T": (-2.5, 0.5),
    "5_1300040_A_G": (3.0, 0.3),
    "5_1300041_C_T": (-3.0, -0.3),
    "1_1300050_A_G": (3.5, -1.0),
    "1_1300051_C_T": (-3.5, 1.0),
    "2_1300060_A_G": (4.0, 1.0),
    "2_1300061_C_T": (-4.0, -1.0),
    "3_1300070_A_G": (4.5, 0.0),
    "3_1300071_C_T": (-4.5, 0.0),
    "4_1300080_A_G": (5.0, -0.6),
    "4_1300081_C_T": (-5.0, 0.6),
    "5_1300090_A_G": (5.5, 0.6),
    "5_1300091_C_T": (-5.5, -0.6),
    "1_1300100_A_G": (6.0, -0.2),
    "1_1300101_C_T": (-6.0, 0.2),
    "2_1300110_A_G": (6.5, 0.2),
    "2_1300111_C_T": (-6.5, -0.2),
    "3_1300120_A_G": (7.0, -0.8),
    "3_1300121_C_T": (-7.0, 0.8),
    "4_1300130_A_G": (7.5, 0.8),
    "4_1300131_C_T": (-7.5, -0.8),
    "5_1300140_A_G": (8.0, -0.4),
    "5_1300141_C_T": (-8.0, 0.4),
    "1_1300150_A_G": (8.5, 0.4),
    "1_1300151_C_T": (-8.5, -0.4),
    "1_1300200_A_G": (0.1, -0.02),
    "1_1300201_C_T": (-0.1, 0.02),
    "2_1300210_A_G": (0.2, 0.0),
    "2_1300211_C_T": (-0.2, 0.0),
    "3_1300220_A_G": (0.3, 0.0),
    "3_1300221_C_T": (-0.3, 0.0),
    "4_1300230_A_G": (0.4, -0.13),
    "4_1300231_C_T": (-0.4, 0.13),
    "5_1300240_A_G": (0.5, -0.34),
    "5_1300241_C_T": (-0.5, 0.34),
    "1_1300250_A_G": (0.6, 0.13),
    "1_1300251_C_T": (-0.6, -0.13),
    "2_1300260_A_G": (0.7, -0.46),
    "2_1300261_C_T": (-0.7, 0.46),
    "3_1300270_A_G": (0.8, 0.2),
    "3_1300271_C_T": (-0.8, -0.2),
    "4_1300280_A_G": (0.9, 0.06),
    "4_1300281_C_T": (-0.9, -0.06),
    "5_1300290_A_G": (1.25, 0.51),
    "5_1300291_C_T": (-1.25, -0.51),
    "1_1300300_A_G": (1.75, -0.44),
    "1_1300301_C_T": (-1.75, 0.44),
    "2_1300310_A_G": (2.25, 0.05),
    "2_1300311_C_T": (-2.25, -0.05),
    "3_1300320_A_G": (2.75, 0.35),
    "3_1300321_C_T": (-2.75, -0.35),
    "4_1300330_A_G": (3.25, -0.4),
    "4_1300331_C_T": (-3.25, 0.4),
    "5_1300340_A_G": (3.75, -0.51),
    "5_1300341_C_T": (-3.75, 0.51),
    "1_1300350_A_G": (4.25, 0.17),
    "1_1300351_C_T": (-4.25, -0.17),
    "2_1300360_A_G": (4.75, 0.1),
    "2_1300361_C_T": (-4.75, -0.1),
    "3_1300370_A_G": (5.25, 0.42),
    "3_1300371_C_T": (-5.25, -0.42),
    "4_1300380_A_G": (5.75, -0.37),
    "4_1300381_C_T": (-5.75, 0.37),

    # main effect only, zero interaction (false-positive control)
    "5_5200000_A_G": (0.0, 2.0),
    "5_5200001_A_G": (0.0, -2.0),
    "5_5200002_A_G": (0.0, 3.0),
    "5_5200003_A_G": (0.0, -3.0),
    "5_5200004_A_G": (0.0, 4.0),
    "5_5200005_A_G": (0.0, -4.0),
    "5_5200006_A_G": (0.0, 5.0),
    "5_5200007_A_G": (0.0, -5.0),
    "5_5200008_A_G": (0.0, 6.0),
}

DEFAULT_PURE_VARIANCE_VARIANTS = {
    # original
    "7_7000001_A_G": {0: 3.0, 1: 12.0},
    "7_7000002_C_T": {0: 12.0, 1: 3.0},

    # 7 additional pairs, different variance ratios, balanced lo/hi and hi/lo
    "7_7400000_A_G": {0: 4.0, 1: 10.0},
    "7_7400001_C_T": {0: 10.0, 1: 4.0},
    "7_7400010_A_G": {0: 5.0, 1: 8.0},
    "7_7400011_C_T": {0: 8.0, 1: 5.0},
    "7_7400020_A_G": {0: 2.0, 1: 14.0},
    "7_7400021_C_T": {0: 14.0, 1: 2.0},
    "7_7400030_A_G": {0: 2.0, 1: 10.0},
    "7_7400031_C_T": {0: 10.0, 1: 2.0},
    "7_7400040_A_G": {0: 3.0, 1: 10.0},
    "7_7400041_C_T": {0: 10.0, 1: 3.0},
    "7_7400050_A_G": {0: 6.0, 1: 12.0},
    "7_7400051_C_T": {0: 12.0, 1: 6.0},
    "7_7400060_A_G": {0: 1.0, 1: 10.0},
    "7_7400061_C_T": {0: 10.0, 1: 1.0},
}


def _stream(rng_seed: int, label: str) -> np.random.Generator:
    """Deterministic, independent random generator per label, seeded on
    (rng_seed, label) -- see the original docstring for why md5 instead of
    the built-in hash()."""
    digest = hashlib.md5(f"{rng_seed}:{label}".encode("utf-8")).hexdigest()
    seed = int(digest[:8], 16)
    return np.random.default_rng(seed)


def _sd_from_dosage(dosage_binary: np.ndarray, sd_by_dosage: dict[int, float]) -> np.ndarray:
    return np.array([sd_by_dosage[int(round(d))] for d in dosage_binary])


def generate_dataset(
    out_dir: str,
    rng_seed: int = 12345,
    n_patients: int = 1000,
    n_null_variants: int = 393,
    causal_variants: dict | None = None,
    pure_variance_variants: dict | None = None,
    maf_range: tuple[float, float] = (0.05, 0.35),
    prop_unexposed: float = 0.40,
    exposure_gamma_shape: float = 2.0,
    exposure_gamma_scale: float = 15.0,
    exposure_max: float = 100.0,
    missing_rate: float = 0.02,
    nonrandom_missing_carrier_rate: float | None = None,
    baseline_onset: float = 61.0,
    beta_sex: float = -1.2,
    beta_exposure_main: float = -0.03,
    noise_sd: float = 8.5,
    onset_clip: tuple[float, float] = (30.0, 85.0),
    n_pcs: int = 10,
    explained_variance_range: tuple[float, float] = (9.0, 5.0),
    subpop_frac: float = 0.0,
    subpop_onset_shift: float = 0.0,
    subpop_maf_shift: float = 0.0,
    verbose: bool = True,
) -> dict:
    """Generates env.csv, genetic.csv, pca_covariates_gen1.csv, ground_truth.csv
    in `out_dir`. Returns a small summary dict (useful for logging/for the
    multi-scenario orchestrator's report, without having to re-read the
    CSVs)."""

    causal_variants = dict(DEFAULT_CAUSAL_VARIANTS if causal_variants is None else causal_variants)
    pure_variance_variants = dict(
        DEFAULT_PURE_VARIANCE_VARIANTS if pure_variance_variants is None else pure_variance_variants
    )

    os.makedirs(out_dir, exist_ok=True)
    n = n_patients
    ids = [f"S{str(i).zfill(4)}" for i in range(1, n + 1)]

    rng_sex = _stream(rng_seed, "sex")
    sex = rng_sex.choice(["M", "F"], size=n)
    sex_num = (sex == "M").astype(float)

    # ---- latent subpopulation (stratification stress-test) ----
    # Binary assignment NOT written to output (it is not a covariate
    # observable by the pipeline) -- simulates a real population structure
    # that the PCs (independent noise, see below) do not capture.
    if subpop_frac > 0:
        rng_subpop = _stream(rng_seed, "subpop")
        subpop_b = (rng_subpop.random(n) < subpop_frac).astype(float)
    else:
        subpop_b = np.zeros(n)

    # ---- exposure ----
    rng_exposure = _stream(rng_seed, "exposure")
    unexposed_mask = rng_exposure.random(n) < prop_unexposed
    exposure = np.empty(n, dtype=float)
    exposure[unexposed_mask] = 0.0
    n_exposed = int((~unexposed_mask).sum())
    exposed_vals = rng_exposure.gamma(shape=exposure_gamma_shape, scale=exposure_gamma_scale, size=n_exposed)
    exposed_vals = np.clip(exposed_vals, 0.1, exposure_max)
    exposure[~unexposed_mask] = exposed_vals
    exposure = np.round(exposure, 2)
    exposure_std_approx = (exposure - exposure.mean()) / exposure.std()

    # ---- genotypes ----
    variant_labels = (
        list(causal_variants.keys())
        + list(pure_variance_variants.keys())
        + [f"{6 + (i // 200)}_{9_000_000 + i}_A_G" for i in range(n_null_variants)]
    )

    geno, geno_numeric, maf_used, carrier_freq_used = {}, {}, {}, {}

    for lab in variant_labels:
        rng_v = _stream(rng_seed, f"variant:{lab}")
        maf = rng_v.uniform(*maf_range)
        # MAF shift between subpopulations (population stratification):
        # applied to ALL variants, including null ones -- this is exactly
        # what inflates lambda_GC if left uncorrected.
        maf_eff = np.clip(maf + subpop_maf_shift * subpop_b, 0.001, 0.999)
        dosage_diploid = rng_v.binomial(2, maf_eff).astype(int)
        dosage_binary = (dosage_diploid >= 1).astype(int)
        maf_used[lab] = maf
        carrier_freq_used[lab] = float(dosage_binary.mean())

        rng_miss = _stream(rng_seed, f"missing:{lab}")
        if nonrandom_missing_carrier_rate is not None:
            miss_prob = np.where(dosage_binary == 1, nonrandom_missing_carrier_rate, missing_rate)
            miss_mask = rng_miss.random(n) < miss_prob
        else:
            miss_mask = rng_miss.random(n) < missing_rate

        dosage_col = dosage_binary.astype(object)
        dosage_col[miss_mask] = "."
        geno[lab] = dosage_col

        dosage_num = dosage_binary.astype(float)
        dosage_num[miss_mask] = np.nan
        geno_numeric[lab] = dosage_num

    # ---- phenotype ----
    rng_noise = _stream(rng_seed, "noise")
    noise = rng_noise.normal(0, noise_sd, size=n)

    onset_age = (
        baseline_onset
        + beta_sex * sex_num
        + beta_exposure_main * exposure
        + subpop_onset_shift * subpop_b
        + noise
    )

    for lab, (beta_inter, beta_main) in causal_variants.items():
        dosage_bin = np.nan_to_num(geno_numeric[lab], nan=0.0)
        onset_age = onset_age + beta_main * dosage_bin + beta_inter * dosage_bin * exposure_std_approx

    for lab, sd_by_dosage in pure_variance_variants.items():
        dosage_bin = np.nan_to_num(geno_numeric[lab], nan=0.0)
        rng_var_effect = _stream(rng_seed, f"variance_effect:{lab}")
        sd_i = _sd_from_dosage(dosage_bin, sd_by_dosage)
        onset_age = onset_age + rng_var_effect.normal(0, sd_i, size=n)

    onset_age = np.clip(onset_age, *onset_clip)

    # ---- output ----
    df_env = pd.DataFrame({
        "id": ids, "onset_age": onset_age.round(2),
        "exposure_env": exposure.round(3), "sex": sex,
    })
    df_env.to_csv(os.path.join(out_dir, "env.csv"), index=False)

    df_gen = pd.concat(
        [pd.Series(ids, name="id")] + [pd.Series(geno[lab], name=lab) for lab in variant_labels],
        axis=1,
    )
    df_gen.to_csv(os.path.join(out_dir, "genetic.csv"), index=False)

    rng_pca = _stream(rng_seed, "pca")
    explained_variance_pct = np.linspace(*explained_variance_range, n_pcs)
    pc_scale = np.sqrt(explained_variance_pct / explained_variance_pct.sum())
    pcs = rng_pca.normal(0, 1, size=(n, n_pcs)) * pc_scale
    df_pca = pd.DataFrame(pcs, columns=[f"PC{i+1}" for i in range(n_pcs)])
    df_pca.insert(0, "IID", ids)
    df_pca.to_csv(os.path.join(out_dir, "pca_covariates_gen1.csv"), index=False)

    truth_rows = []
    for lab in variant_labels:
        if lab in causal_variants:
            beta_inter, beta_main = causal_variants[lab]
            truth_rows.append({
                "variant": lab, "is_causal": True, "effect_type": "gxe_meanshift",
                "true_beta_interaction": beta_inter, "true_beta_main": beta_main,
                "sd_dosage0": np.nan, "sd_dosage1": np.nan, "sd_dosage2": np.nan,
                "maf": maf_used[lab], "carrier_freq": carrier_freq_used[lab],
            })
        elif lab in pure_variance_variants:
            sdd = pure_variance_variants[lab]
            truth_rows.append({
                "variant": lab, "is_causal": True, "effect_type": "pure_variance",
                "true_beta_interaction": 0.0, "true_beta_main": 0.0,
                "sd_dosage0": sdd[0], "sd_dosage1": sdd[1], "sd_dosage2": np.nan,
                "maf": maf_used[lab], "carrier_freq": carrier_freq_used[lab],
            })
        else:
            truth_rows.append({
                "variant": lab, "is_causal": False, "effect_type": "no_effect",
                "true_beta_interaction": 0.0, "true_beta_main": 0.0,
                "sd_dosage0": np.nan, "sd_dosage1": np.nan, "sd_dosage2": np.nan,
                "maf": maf_used[lab], "carrier_freq": carrier_freq_used[lab],
            })
    pd.DataFrame(truth_rows).to_csv(os.path.join(out_dir, "ground_truth.csv"), index=False)

    summary = {
        "out_dir": out_dir,
        "n_patients": n,
        "n_variants": len(variant_labels),
        "n_causal_gxe": len(causal_variants),
        "n_pure_variance": len(pure_variance_variants),
        "n_null": n_null_variants,
        "onset_mean": float(onset_age.mean()),
        "onset_sd": float(onset_age.std()),
    }
    if verbose:
        print(f"[gen_fake_data] {out_dir}: N={n}, variants={len(variant_labels)} "
              f"({len(causal_variants)} G×E, {len(pure_variance_variants)} vQTL pure, "
              f"{n_null_variants} null), onset={onset_age.mean():.1f}±{onset_age.std():.1f}")
    return summary


def _cli():
    parser = argparse.ArgumentParser(description="Generate a synthetic dataset for the gene_environment/vqtl pipeline test.")
    parser.add_argument("--out-dir", default=os.path.join(SCRIPT_DIR, "fake_data"))
    parser.add_argument("--config-json", default=None, help="JSON with kwargs to pass to generate_dataset()")
    args = parser.parse_args()

    kwargs = {}
    if args.config_json:
        with open(args.config_json) as f:
            kwargs = json.load(f)
    kwargs["out_dir"] = args.out_dir
    generate_dataset(**kwargs)


if __name__ == "__main__":
    _cli()
