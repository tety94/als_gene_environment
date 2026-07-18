"""
interpret_plink_output.py
==========================

Prende gli output STANDARD di plink2 (generati da 00_run_plink_qc.sh) e
risponde alle 3 domande operative:
  1) Serve un LMM per relatedness?
  2) Le PC vanno nel modello principale o basta la sensitivity?
  3) C'e' inflazione genomica nel tuo scan (se fornisci p-value)?

A differenza dello script precedente, qui il kinship e la PCA sono gia'
stati calcolati da plink2 (strumento standard, validato) -- questo script
fa solo l'interpretazione statistica rispetto alla tua esposizione.

INPUT
-----
--kin0        file king.kin0 prodotto da: plink2 --make-king-table
--eigenvec    file pca.eigenvec prodotto da: plink2 --pca 10
--metadata    csv/parquet con colonne: id, <exposure-col>
              (id deve combaciare con la colonna IID di plink2)
--exposure-col nome della colonna esposizione nei metadati
--pvalues     opzionale: csv/parquet con colonna p-value da un tuo scan
--pvalue-col  nome colonna p-value (default: p)
--pi-hat-threshold  soglia PI_HAT (default 0.125); nota: KINSHIP di plink2
              e' gia' il phi di KING-robust, quindi PI_HAT = 2 * KINSHIP

USO
---
python interpret_plink_output.py \
    --kin0 qc_output/king.kin0 \
    --eigenvec qc_output/pca.eigenvec \
    --metadata sample_metadata.csv \
    --exposure-col exposure_agri_score \
    --pvaluespvalues gwas_results.csv --pvalue-col p \
    --out-dir diagnostics_output
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def load_kin0(path):
    """
    Formato king.kin0 di plink2 (tab-separated), colonne tipiche:
    #FID1 IID1 FID2 IID2 NSNP HETHET IBS0 KINSHIP
    (i nomi esatti delle colonne possono variare leggermente per versione
    plink2; lo script normalizza cercando 'KINSHIP', 'IID1', 'IID2').
    """
    df = pd.read_csv(path, sep=r"\s+")
    df.columns = [c.lstrip("#") for c in df.columns]
    id1_col = [c for c in df.columns if c.upper() in ("IID1", "ID1")][0]
    id2_col = [c for c in df.columns if c.upper() in ("IID2", "ID2")][0]
    kin_col = [c for c in df.columns if c.upper() == "KINSHIP"][0]
    df = df.rename(columns={id1_col: "IID1", id2_col: "IID2", kin_col: "KINSHIP"})
    return df[["IID1", "IID2", "KINSHIP"]]


def load_eigenvec(path):
    """
    Formato pca.eigenvec di plink2: #FID IID PC1 PC2 ... PCk
    """
    df = pd.read_csv(path, sep=r"\s+")
    df.columns = [c.lstrip("#") for c in df.columns]
    return df


def summarize_relatedness(kin_df, pi_hat_threshold):
    kin_df = kin_df.copy()
    kin_df["PI_HAT"] = 2 * kin_df["KINSHIP"]
    n_pairs = len(kin_df)
    flagged = kin_df[kin_df["PI_HAT"] > pi_hat_threshold].sort_values("PI_HAT", ascending=False)
    n_above = len(flagged)
    pct_above = 100 * n_above / n_pairs if n_pairs > 0 else np.nan

    # segnala separatamente le coppie "quasi-identiche" (PI_HAT > 0.9):
    # probabile duplicato tecnico tra batch (gen1/gen2/gen3), non parentela
    likely_duplicates = kin_df[kin_df["PI_HAT"] > 0.9]

    return kin_df, flagged, n_pairs, n_above, pct_above, likely_duplicates


def correlate_pcs_with_exposure(eigenvec_df, metadata_df, exposure_col, n_pcs):
    from scipy.stats import pearsonr

    eigenvec_df["IID_clean"] = eigenvec_df["IID"].str.split("_").str[0]
    merged = eigenvec_df.merge(metadata_df, left_on="IID_clean", right_on="id", how="inner")

    n_matched = len(merged)
    n_meta = len(metadata_df)
    n_eig = len(eigenvec_df)

    pc_cols = [c for c in merged.columns if c.startswith("PC")][:n_pcs]
    print("NA PCs:", merged[pc_cols].isna().sum().sum())
    exposure = merged[exposure_col].astype(float).values

    valid = ~np.isnan(exposure)
    exposure_v = exposure[valid]
    pcs_v = merged.loc[valid, pc_cols].values

    corr_per_pc = []
    for i, col in enumerate(pc_cols):
        r, p_val = pearsonr(pcs_v[:, i], exposure_v)
        corr_per_pc.append({"PC": col, "pearson_r": round(float(r), 4), "p_value": float(p_val)})

    from numpy.linalg import lstsq
    X = np.column_stack([pcs_v, np.ones(pcs_v.shape[0])])
    beta, _, _, _ = lstsq(X, exposure_v, rcond=None)
    pred = X @ beta
    ss_res = np.sum((exposure_v - pred) ** 2)
    ss_tot = np.sum((exposure_v - exposure_v.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return corr_per_pc, r_squared, n_matched, n_meta, n_eig, merged, pc_cols


def genomic_inflation_factor(pvalues):
    from scipy.stats import chi2
    p = np.asarray(pvalues, dtype=float)
    p = p[~np.isnan(p)]
    p = np.clip(p, 1e-300, 1)
    chi2_obs = chi2.isf(p, df=1)
    return float(np.median(chi2_obs) / 0.4549364)


def qq_plot(pvalues, out_path):
    """
    QQ-plot -log10(p) osservati vs attesi sotto H0 (uniforme). Complemento
    visivo del lambda GC: un numero solo puo' nascondere inflazione
    concentrata in coda (poche varianti fortemente devianti) che il QQ-plot
    rende visibile a colpo d'occhio, come richiesto tipicamente in un paper.
    """
    if plt is None:
        return None
    p = np.asarray(pvalues, dtype=float)
    p = p[~np.isnan(p)]
    p = np.clip(p, 1e-300, 1)
    p_sorted = np.sort(p)
    n = len(p_sorted)
    if n == 0:
        return None
    expected = -np.log10(np.arange(1, n + 1) / (n + 1))
    observed = -np.log10(p_sorted)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(expected, observed, s=6, alpha=0.5, color="steelblue")
    max_val = max(expected.max(), observed.max())
    ax.plot([0, max_val], [0, max_val], color="red", linestyle="--", linewidth=1, label="expected (H0)")
    ax.set_xlabel("expected -log10(p)")
    ax.set_ylabel("observed -log10(p)")
    ax.set_title("QQ-plot")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kin0", required=True)
    parser.add_argument("--eigenvec", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--exposure-col", default="exposure")
    parser.add_argument("--pvalues", default=None)
    parser.add_argument("--pvalue-col", default="p")
    parser.add_argument("--pi-hat-threshold", type=float, default=0.125)
    parser.add_argument("--n-pcs", type=int, default=10)
    parser.add_argument("--out-dir", default="diagnostics_output")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = []

    def log(msg):
        print(msg)
        lines.append(msg)

    log("=" * 70)
    log("INTERPRETAZIONE OUTPUT PLINK2 - screening pre-analisi G x E")
    log("=" * 70)

    # ---------------- metadata ----------------
    if args.metadata.endswith(".csv"):
        metadata_df = pd.read_csv(args.metadata)
    else:
        metadata_df = pd.read_parquet(args.metadata)
    if "id" not in metadata_df.columns:
        log(f"ERRORE: colonna 'id' non trovata in {args.metadata}. "
            f"Colonne trovate: {list(metadata_df.columns)}")
        return

    # ---------------- 1. relatedness ----------------
    log("\n" + "-" * 70)
    log("1) RELATEDNESS (da king.kin0, calcolato da plink2)")
    log("-" * 70)

    kin_raw = load_kin0(args.kin0)
    kin_df, flagged, n_pairs, n_above, pct_above, likely_dup = summarize_relatedness(
        kin_raw, args.pi_hat_threshold
    )

    log(f"Coppie totali riportate da plink2: {n_pairs}")
    log(f"Soglia PI_HAT: {args.pi_hat_threshold}")
    log(f"Coppie sopra soglia: {n_above} ({pct_above:.4f}% del totale)")

    if len(likely_dup) > 0:
        log(f"\n>>> ATTENZIONE: {len(likely_dup)} coppie hanno PI_HAT > 0.9 "
            "(quasi-identici). Probabile DUPLICATO TECNICO tra batch "
            "gen1/gen2/gen3, non vera parentela biologica. Controlla questi "
            "ID prima di procedere:")
        for _, row in likely_dup.head(20).iterrows():
            log(f"    {row['IID1']} - {row['IID2']}: PI_HAT={row['PI_HAT']:.4f}")

    real_relatedness = flagged[flagged["PI_HAT"] <= 0.9]
    if len(real_relatedness) > 0:
        log(f"\nCoppie con parentela plausibile (0.125 < PI_HAT <= 0.9): {len(real_relatedness)}")
        for _, row in real_relatedness.head(20).iterrows():
            log(f"    {row['IID1']} - {row['IID2']}: PI_HAT={row['PI_HAT']:.4f}")

    if pct_above < 1.0:
        log("\n>>> VERDETTO: <1% delle coppie sopra soglia. LMM probabilmente "
            "non necessario nella main analysis (dopo aver rimosso eventuali "
            "duplicati tecnici tra batch).")
    else:
        log("\n>>> VERDETTO: >=1% delle coppie sopra soglia. Valuta seriamente "
            "un LMM, o quantomeno l'esclusione di un membro per ogni coppia "
            "flaggata dalla main analysis.")

    if plt is not None and n_pairs > 0:
        plt.figure(figsize=(6, 4))
        plt.hist(kin_df["PI_HAT"], bins=80)
        plt.axvline(args.pi_hat_threshold, color="red", linestyle="--", label="threshold")
        plt.xlabel("PI_HAT (2 x plink2 KINSHIP)")
        plt.ylabel("N pairs")
        plt.title("PI_HAT distribution")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "pi_hat_distribution.png", dpi=150)
        plt.close()
        log(f"[grafico: {out_dir / 'pi_hat_distribution.png'}]")

    # ---------------- 2. PCA vs esposizione ----------------
    log("\n" + "-" * 70)
    log("2) PC vs ESPOSIZIONE")
    log("-" * 70)

    eigenvec_df = load_eigenvec(args.eigenvec)
    corr_per_pc, r_squared, n_matched, n_meta, n_eig, merged, pc_cols = correlate_pcs_with_exposure(
        eigenvec_df, metadata_df, args.exposure_col, args.n_pcs
    )



    log(f"Campioni in eigenvec: {n_eig} | in metadata: {n_meta} | matchati: {n_matched}")
    if n_matched < min(n_eig, n_meta) * 0.9:
        log("  ATTENZIONE: match basso tra id metadata e IID plink2. "
            "Controlla la convenzione di naming (es. FID_IID vs IID puro).")

    log("\nCorrelazione (Pearson) PC vs esposizione:")
    for row in corr_per_pc:
        flag = "  <-- notevole" if abs(row["pearson_r"]) > 0.2 and row["p_value"] < 0.05 else ""
        log(f"    {row['PC']}: r={row['pearson_r']:.4f}, p={row['p_value']:.2e}{flag}")

    # Tabella supplementare per il paper (non solo testo di log): una riga
    # per PC con r, p-value e flag di notevolezza, pronta da includere come
    # tabella supplementare.
    corr_table = pd.DataFrame(corr_per_pc)
    corr_table["notevole"] = (corr_table["pearson_r"].abs() > 0.2) & (corr_table["p_value"] < 0.05)
    corr_csv_path = out_dir / "pc_exposure_correlation.csv"
    corr_table.to_csv(corr_csv_path, index=False)
    log(f"[tabella: {corr_csv_path}]")

    log(f"\nR^2 (esposizione ~ {' + '.join(pc_cols)}): {r_squared:.4f} "
        f"({r_squared:.1%} varianza esposizione spiegata da ancestry)")

    if r_squared > 0.10:
        log("\n>>> VERDETTO: R^2 > 10%. Confounding ancestry-esposizione concreto. "
            "PC vanno nel MODELLO PRINCIPALE, non solo sensitivity.")
    elif r_squared > 0.03:
        log("\n>>> VERDETTO: confounding moderato (3-10%). Consigliato includerle "
            "nel modello base; tienile comunque anche in sensitivity.")
    else:
        log("\n>>> VERDETTO: R^2 < 3%. Confounding trascurabile. PC come "
            "sensitivity e' una scelta difendibile.")

    if plt is not None:
        plt.figure(figsize=(6, 5))
        sc = plt.scatter(merged[pc_cols[0]], merged[pc_cols[1]],
                          c=merged[args.exposure_col], cmap="viridis", s=25)
        plt.colorbar(sc, label=args.exposure_col)
        plt.xlabel(pc_cols[0])
        plt.ylabel(pc_cols[1])
        plt.title("PC1 vs PC2 colored by exposure")
        plt.tight_layout()
        plt.savefig(out_dir / "pca_vs_exposure.png", dpi=150)
        plt.close()
        log(f"[grafico: {out_dir / 'pca_vs_exposure.png'}]")

    # ---------------- 3. lambda GC ----------------
    log("\n" + "-" * 70)
    log("3) LAMBDA GC")
    log("-" * 70)
    if args.pvalues:
        if args.pvalues.endswith(".csv"):
            pval_df = pd.read_csv(args.pvalues)
        else:
            pval_df = pd.read_parquet(args.pvalues)
        pvals = pval_df[args.pvalue_col].values
        lam = genomic_inflation_factor(pvals)
        log(f"N test: {(~np.isnan(pvals)).sum()} | Lambda GC = {lam:.4f}")
        if lam > 1.05:
            log(">>> VERDETTO: lambda > 1.05, possibile stratificazione residua.")
        else:
            log(">>> VERDETTO: lambda entro range accettabile.")

        qq_path = out_dir / "qq_plot.png"
        saved = qq_plot(pvals, qq_path)
        if saved:
            log(f"[grafico: {qq_path}]")
        elif plt is None:
            log("  (matplotlib non disponibile, QQ-plot saltato)")
    else:
        log("Nessun --pvalues fornito, salto questo controllo.")

    log("\n" + "=" * 70)
    log("SOMMARIO")
    log("=" * 70)
    log(f"- LMM: {'non necessario' if pct_above < 1.0 else 'da valutare'}")
    log(f"- PC nel modello base: {'SI' if r_squared > 0.10 else ('consigliato' if r_squared > 0.03 else 'sensitivity ok')}")

    (out_dir / "diagnostics_report.txt").write_text("\n".join(lines), encoding="utf-8")
    log(f"\nReport salvato in: {out_dir / 'diagnostics_report.txt'}")


if __name__ == "__main__":
    main()