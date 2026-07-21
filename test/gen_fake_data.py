"""
Genera un dataset sintetico (genetica + ambiente + PC di popolazione) con un
effetto di interazione gene x esposizione INIETTATO A PRIORI su un piccolo
sottoinsieme di varianti "causali", e nessun effetto sulle varianti "nulle".

onset_age: tarata per assomigliare a una distribuzione di età d'esordio SLA
sporadica (mediana ~60 anni, SD ~10, range 30-85).

exposure_env: ~40% dei pazienti non esposti (valore 0), i restanti con un
punteggio di esposizione cumulativa continuo, right-skewed, in (0, 100].

Obiettivo: verificare che la pipeline (matching + OLS + permutation test)
recuperi correttamente le varianti causali (basso p_emp, segno del
coefficiente coerente con quello iniettato) e NON segnali sistematicamente
le varianti nulle come significative.
"""
from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

RNG_SEED = 12345

N_PATIENTS = 600
N_NULL_VARIANTS = 60

# variant_label -> (beta_interaction, beta_main)  [effetto vero iniettato]
# beta_interaction è l'effetto per unità di esposizione STANDARDIZZATA, per
# copia allelica (dosaggio 0/1/2). Segni misti per verificare che la
# pipeline recuperi correttamente sia effetti protettivi che di rischio.
CAUSAL_VARIANTS = {
    "1_1000001_A_G": (-4.5, -1.0),   # interazione forte, negativa (anticipa l'esordio con esposizione)
    "2_2000002_C_T": (4.0, 0.5),     # interazione forte, positiva (ritarda l'esordio con esposizione)
    "3_3000003_G_A": (-3.5, 0.0),    # interazione moderata, negativa, nessun effetto principale
    "4_4000004_T_C": (3.2, -0.5),    # interazione moderata, positiva
    "5_5000005_A_T": (-5.0, 1.0),    # interazione forte, negativa
}

# Cartella di output: relativa a questo file, non alla cwd da cui lo lanci.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "fake_data")


def stream(label: str) -> np.random.Generator:
    """Generatore casuale indipendente e deterministico per etichetta.

    Seedato sull'hash MD5 dell'etichetta (stabile fra esecuzioni ed
    indipendente da N_PATIENTS e dall'ordine di chiamata) -- NON sull'hash()
    di Python, che è randomizzato ad ogni processo (PEP 456). Ogni "stream"
    (sesso, esposizione, rumore, ogni singola variante, PC) è così isolato
    dagli altri: cambiare N_PATIENTS o aggiungere/togliere varianti non fa
    slittare i numeri casuali consumati dalle altre componenti.
    """
    digest = hashlib.md5(f"{RNG_SEED}:{label}".encode("utf-8")).hexdigest()
    seed = int(digest[:8], 16)
    return np.random.default_rng(seed)



def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    n = N_PATIENTS

    ids = [f"S{str(i).zfill(4)}" for i in range(1, n + 1)]

    rng_sex = stream("sex")
    sex = rng_sex.choice(["M", "F"], size=n)
    sex_num = (sex == "M").astype(float)

    # Esposizione ambientale: ~40% dei pazienti NON esposti (0), i restanti
    # con un punteggio di esposizione cumulativa continuo e right-skewed
    # (molti valori bassi, coda verso l'alto), come tipico per punteggi di
    # esposizione ambientale cumulativa (es. anni-dose a un fattore di
    # rischio), limitato a un massimo plausibile di 100.
    rng_exposure = stream("exposure")
    prop_unexposed = 0.40
    unexposed_mask = rng_exposure.random(n) < prop_unexposed
    exposure = np.empty(n, dtype=float)
    exposure[unexposed_mask] = 0.0
    n_exposed = int((~unexposed_mask).sum())
    exposed_vals = rng_exposure.gamma(shape=2.0, scale=15.0, size=n_exposed)
    exposed_vals = np.clip(exposed_vals, 0.1, 100.0)  # >0 e <=100 per costruzione
    exposure[~unexposed_mask] = exposed_vals
    exposure = np.round(exposure, 2)
    # standardizzazione "manuale" solo per generare l'effetto coerente con
    # quello che la pipeline calcolerà internamente (StandardScaler) -- i
    # due non saranno identici al bit, ma abbastanza vicini da iniettare
    # l'effetto nella scala giusta.
    exposure_std_approx = (exposure - exposure.mean()) / exposure.std()

    # ---- genotipi: uno stream indipendente PER VARIANTE, seedato sul suo
    # nome -- la MAF e il genotipo di ciascuna variante (comprese le
    # causali) non dipendono da N_PATIENTS ne' dalle altre varianti. ----
    variant_labels = list(CAUSAL_VARIANTS.keys()) + [
        f"{6 + (i // 200)}_{9_000_000 + i}_A_G" for i in range(N_NULL_VARIANTS)
    ]

    geno = {}
    maf_used = {}
    for lab in variant_labels:
        rng_v = stream(f"variant:{lab}")
        maf = rng_v.uniform(0.05, 0.35)
        maf_used[lab] = maf
        dosage = rng_v.binomial(2, maf, size=n).astype(int)
        # ~2% di missing genotype, codificato con "." come nei VCF filtrati
        # reali, per testare anche quel path del codice (df[col] != ".").
        miss_mask = rng_v.random(n) < 0.02
        dosage_col = dosage.astype(object)
        dosage_col[miss_mask] = "."
        geno[lab] = dosage_col

    # ---- fenotipo (onset_age) ----
    # Parametri tarati per assomigliare a una distribuzione di età d'esordio
    # SLA sporadica: mediana ~61 anni, SD ~9-10 anni, range plausibile
    # ~30-85 (casi giovanili/familiari <30 e >85 sono rari e qui esclusi per
    # semplicità, essendo un dataset sintetico).
    baseline = 61.0
    beta_sex = -1.2   # M leggermente più precoce, giusto per avere una covariata "reale"
    beta_exposure_main = -0.03  # lieve effetto ambientale diretto (per punto di punteggio)
    rng_noise = stream("noise")
    noise = rng_noise.normal(0, 8.5, size=n)

    onset_age = (
        baseline
        + beta_sex * sex_num
        + beta_exposure_main * exposure
        + noise
    )

    for lab, (beta_inter, beta_main) in CAUSAL_VARIANTS.items():
        dosage = geno[lab]
        dosage_numeric = np.array([0.0 if v == "." else float(v) for v in dosage])
        onset_age = onset_age + beta_main * dosage_numeric + beta_inter * dosage_numeric * exposure_std_approx

    onset_age = np.clip(onset_age, 30, 85)  # range d'esordio SLA plausibile

    # ---- file ambientale ----
    df_env = pd.DataFrame({
        "id": ids,
        "onset_age": onset_age.round(2),
        "exposure_env": exposure.round(3),
        "sex": sex,
    })
    df_env.to_csv(os.path.join(OUT_DIR, "env.csv"), index=False)

    # ---- file genetico ----
    df_gen = pd.DataFrame({"id": ids})
    for lab in variant_labels:
        df_gen[lab] = geno[lab]
    df_gen.to_csv(os.path.join(OUT_DIR, "genetic.csv"), index=False)

    # ---- PC di popolazione (covariate di correzione, non di interazione) ----
    # Qui NON derivano da una vera decomposizione PCA di una matrice genotipica
    # (come farebbe plink2 --pca nella pipeline reale): sono simulate come
    # variabili latenti indipendenti dal fenotipo e dal genotipo, con varianza
    # decrescente per imitare il profilo tipico di uno scree plot di PCA
    # genetica (PC1 la piu' "informativa", l'ultima la meno). Le frazioni
    # sotto sono varianza spiegata SUL TOTALE DELLE PC (sommano a 1),
    # riscalate nel range 9% -> 5%.
    rng_pca = stream("pca")
    n_pcs = 10
    explained_variance_pct = np.linspace(9.0, 5.0, n_pcs)  # PC1=9%, ..., ultima=5%
    pc_scale = np.sqrt(explained_variance_pct / explained_variance_pct.sum())
    pcs = rng_pca.normal(0, 1, size=(n, n_pcs)) * pc_scale
    df_pca = pd.DataFrame(pcs, columns=[f"PC{i+1}" for i in range(n_pcs)])
    df_pca.insert(0, "IID", ids)
    df_pca.to_csv(os.path.join(OUT_DIR, "pca_covariates_gen1.csv"), index=False)
    print("PC di popolazione (varianza spiegata simulata):")
    for i, pct in enumerate(explained_variance_pct, start=1):
        print(f"  PC{i}: {pct:.1f}%")

    # ---- ground truth per il confronto a valle ----
    truth_rows = []
    for lab in variant_labels:
        beta_inter, beta_main = CAUSAL_VARIANTS.get(lab, (0.0, 0.0))
        truth_rows.append({
            "variant": lab,
            "is_causal": lab in CAUSAL_VARIANTS,
            "true_beta_interaction": beta_inter,
            "true_beta_main": beta_main,
            "maf": maf_used[lab],
        })
    pd.DataFrame(truth_rows).to_csv(os.path.join(OUT_DIR, "ground_truth.csv"), index=False)

    print(f"Pazienti: {n}")
    print(f"Varianti totali: {len(variant_labels)} ({len(CAUSAL_VARIANTS)} causali, {N_NULL_VARIANTS} nulle)")
    print(f"onset_age: media={onset_age.mean():.1f}, sd={onset_age.std():.1f}")
    print(f"File scritti in {OUT_DIR}/")


if __name__ == "__main__":
    main()