class NeuroScore:

    @staticmethod
    def compute(d: dict) -> float:
        score = 0
        score += 1 if d.get("expressed_brain") else 0
        score += 1 if d.get("expressed_neurons") else 0
        score += 1 if d.get("expressed_glia") else 0

        # GO disattivato (vedi GeneAnnotator) -> non contribuisce allo score.
        # CTD e' invece attivo, con TRE segnali distinti pesati diversamente:
        #
        #   - ctd_neuro_disease_direct: associazione CURATA direttamente in
        #     letteratura, stessa "famiglia" di evidenza di PanelApp/Open
        #     Targets -> peso comparabile a quello.
        #
        #   - ctd_neuro_disease_pesticide_mediated: CTD stessa infierisce un
        #     collegamento a malattia neuro passando per un pesticida
        #     specifico. E' il segnale piu' specifico per questo studio,
        #     perche' conferma, da una fonte indipendente, esattamente
        #     l'ipotesi gene-ambiente che si sta testando -> peso alto,
        #     paragonabile o superiore all'evidenza diretta.
        #
        #   - ctd_chemicals (pesticidi, senza legame a malattia riportato):
        #     il gene e' influenzato da un'esposizione ambientale nota, ma
        #     senza che CTD colleghi esplicitamente quel chimico alla SLA.
        #     Plausibilita' meccanicistica piu' debole -> peso minore.
        if d.get("ctd_neuro_disease_pesticide_mediated"):
            score += 3
        elif d.get("ctd_neuro_disease_direct"):
            score += 2

        if d.get("ctd_chemicals"):
            score += 1

        # Segnali di evidenza SLA specifici, pesati in base all'affidabilita'
        # della fonte:
        #   - PanelApp "green" = evidenza diagnostic-grade, il segnale piu' forte
        #   - PanelApp "amber" = evidenza moderata
        #   - Open Targets score = evidenza aggregata continua (letteratura + GWAS + altro)
        confidence = d.get("als_panelapp_confidence")
        if confidence == "3":       # green
            score += 3
        elif confidence == "2":     # amber
            score += 1.5

        ot_score = d.get("als_opentargets_score")
        if ot_score:
            score += ot_score * 2   # score Open Targets e' 0.0-1.0, riscalato per pesare quanto gli altri segnali

        return score