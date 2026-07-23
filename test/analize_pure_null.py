"""
Analisi mirata sulla zona vqtl_zone == "pure_null" del test isolato
(isolated_power_curve.csv), come discusso: prima l'intervallo di confidenza
binomiale esatto sul tasso di falsi positivi osservato, e SOLO se il 5%
nominale resta fuori (o vicino al bordo) si procede al confronto
P_asym (K=50) vs P_boot (K=200), gia' salvato per ogni variante nel run
isolato (colonne p_vqtl / p_vqtl_boot in isolated_summary.json, quindi
zero costo di ricalcolo).

USO (da lanciare TU, stessa cartella di run_isolated_causal_test.py, dopo
aver girato quest'ultimo -- eventualmente con --extra-replicates per
allargare n nella zona pure_null):

    python analyze_pure_null.py
    python analyze_pure_null.py --alpha 0.10          # CI al 90% invece che 95%
    python analyze_pure_null.py --csv path/altro.csv  # default: isolated/isolated_power_curve.csv

Legge isolated/isolated_power_curve.csv (scritto da run_isolated_causal_test.py).
Scrive isolated/pure_null_ci_report.json con il risultato completo.
"""
from __future__ import annotations

import argparse
import json
import os

import pandas as pd
from scipy.stats import binomtest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ISOLATED_ROOT = os.path.join(SCRIPT_DIR, "isolated")
DEFAULT_CSV = os.path.join(ISOLATED_ROOT, "isolated_power_curve.csv")
NOMINAL_RATE = 0.05


def _ci_block(n_sig: int, n_tot: int, alpha: float) -> dict:
    if n_tot == 0:
        return {"n": 0, "n_significant": 0, "observed_rate": None,
                "ci_low": None, "ci_high": None, "nominal_in_ci": None}
    res = binomtest(n_sig, n_tot, p=NOMINAL_RATE, alternative="two-sided")
    ci = res.proportion_ci(confidence_level=1 - alpha, method="exact")
    observed = n_sig / n_tot
    return {
        "n": n_tot,
        "n_significant": n_sig,
        "observed_rate": round(observed, 4),
        "ci_low": round(ci.low, 4),
        "ci_high": round(ci.high, 4),
        "confidence_level": 1 - alpha,
        "nominal_in_ci": bool(ci.low <= NOMINAL_RATE <= ci.high),
        "binom_test_pvalue": round(res.pvalue, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="CI binomiale sulla zona pure_null + confronto asym/boot condizionale")
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--alpha", type=float, default=0.05, help="1 - livello di confidenza (default 0.05 -> CI 95%)")
    parser.add_argument("--near-edge-margin", type=float, default=0.02,
                         help="se il 5% nominale e' DENTRO il CI ma a meno di questo margine dal bordo, "
                              "procedi comunque al confronto asym/boot per prudenza (default 0.02)")
    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        raise SystemExit(
            f"Non trovo {args.csv}. Lancia prima run_isolated_causal_test.py "
            f"(eventualmente con --extra-replicates) nella stessa cartella."
        )

    df = pd.read_csv(args.csv)
    if "vqtl_zone" not in df.columns:
        raise SystemExit(
            "Il CSV non ha la colonna vqtl_zone: e' stato generato da una versione "
            "vecchia dello script? Rilancia run_isolated_causal_test.py."
        )

    pure_null = df[(df["vqtl_zone"] == "pure_null") & (df["status"] == "ok")].copy()
    n_tot = len(pure_null)
    n_sig = int(pure_null["significant_vqtl"].sum()) if n_tot else 0

    print(f"Zona pure_null: n={n_tot} varianti, {n_sig} significative su P_asym "
          f"(tasso osservato {100 * n_sig / n_tot:.1f}%)." if n_tot else "Zona pure_null vuota.")

    ci_asym = _ci_block(n_sig, n_tot, args.alpha)
    print(f"IC esatto {int(ci_asym.get('confidence_level', 0) * 100) if n_tot else '-'}% "
          f"su P_asym: [{ci_asym['ci_low']}, {ci_asym['ci_high']}]"
          if n_tot else "")

    report = {
        "csv_source": args.csv,
        "alpha": args.alpha,
        "n_pure_null_total_rows": n_tot,
        "asym": ci_asym,
    }

    proceed_to_boot = False
    if n_tot == 0:
        print("Nessuna variante in pure_null: niente da valutare.")
    elif not ci_asym["nominal_in_ci"]:
        print(f"Il 5% nominale è FUORI dall'IC su P_asym -> procedo al confronto con P_boot.")
        proceed_to_boot = True
    else:
        dist_to_edge = min(abs(NOMINAL_RATE - ci_asym["ci_low"]), abs(ci_asym["ci_high"] - NOMINAL_RATE))
        if dist_to_edge < args.near_edge_margin:
            print(f"Il 5% nominale è DENTRO l'IC ma vicino al bordo (distanza {dist_to_edge:.3f} "
                  f"< margine {args.near_edge_margin}) -> procedo comunque al confronto con P_boot per prudenza.")
            proceed_to_boot = True
        else:
            print("Il 5% nominale è comodamente dentro l'IC su P_asym: nessuna evidenza di problema. "
                  "Non serve il confronto con P_boot.")

    report["proceeded_to_boot_comparison"] = proceed_to_boot

    if proceed_to_boot:
        has_boot = "p_vqtl_boot" in pure_null.columns and pure_null["p_vqtl_boot"].notna().any()
        if not has_boot:
            print(
                "\n[ATTENZIONE] Il CSV non contiene p_vqtl_boot per la zona pure_null "
                "(probabilmente generato con una versione di run_isolated_causal_test.py "
                "precedente alla patch che salva anche P_boot). Rilancia lo script isolato "
                "aggiornato -- anche solo con --force sulle varianti pure_null è sufficiente "
                "a ripopolare p_vqtl_boot senza rigenerare i dataset da zero, visto che "
                "run_vqtl_debug esegue comunque entrambi i se_method a ogni run."
            )
            report["boot"] = None
        else:
            if "significant_vqtl_boot" in pure_null.columns:
                n_sig_boot = int(pure_null["significant_vqtl_boot"].fillna(False).sum())
            else:
                n_sig_boot = int((pure_null["p_vqtl_boot"] < NOMINAL_RATE).sum())
            ci_boot = _ci_block(n_sig_boot, n_tot, args.alpha)
            print(f"Stesso subset su P_boot (K=200): {n_sig_boot}/{n_tot} significative "
                  f"(tasso osservato {100 * n_sig_boot / n_tot:.1f}%), "
                  f"IC: [{ci_boot['ci_low']}, {ci_boot['ci_high']}]")
            report["boot"] = ci_boot

            if ci_asym["observed_rate"] is not None and ci_boot["observed_rate"] is not None:
                if abs(ci_asym["observed_rate"] - ci_boot["observed_rate"]) < 0.03 and not ci_boot["nominal_in_ci"] == True:
                    pass
                if ci_boot["nominal_in_ci"] and not ci_asym["nominal_in_ci"]:
                    print(
                        "\n-> P_boot (K=200) rientra nel nominale mentre P_asym (K=50) no: "
                        "coerente con rumore di stima della SE a K=50 basso. Vale la pena "
                        "valutare se alzare VQTL_ASYMPTOTIC_BOOTSTRAP_K di default, o usare "
                        "se_method='bootstrap' per lo screening dei falsi positivi."
                    )
                elif not ci_boot["nominal_in_ci"] and not ci_asym["nominal_in_ci"]:
                    print(
                        "\n-> Anche P_boot (K=200, il metodo di produzione) mostra il 5% fuori "
                        "dall'IC: NON è un problema di K. Probabile bias sistematico nel test di "
                        "varianza stesso, indipendente dal numero di repliche bootstrap -- "
                        "servirebbe un controllo aggiuntivo (es. QQ-plot dei p-value vQTL sotto "
                        "H0 pura, senza alcuna componente G×E) per isolare la causa."
                    )
                else:
                    print(
                        "\n-> Entrambi i metodi rientrano nel nominale su questo subset: "
                        "il segnale visto inizialmente su P_asym era compatibile col rumore "
                        "campionario di questo n."
                    )

    out_path = os.path.join(ISOLATED_ROOT, "pure_null_ci_report.json")
    os.makedirs(ISOLATED_ROOT, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n[export] {out_path}")


if __name__ == "__main__":
    main()