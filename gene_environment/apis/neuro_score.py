class NeuroScore:

    @staticmethod
    def compute(d: dict) -> float:
        score = 0
        score += 1 if d.get("expressed_brain") else 0
        score += 1 if d.get("expressed_neurons") else 0
        score += 1 if d.get("expressed_glia") else 0

        # GO disattivato (vedi GeneAnnotator) -> non contribuisce allo score.
        # CTD e' invece attivo: contribuisce con due segnali distinti, pesati
        # diversamente perche' rispondono a due domande diverse:
        #   - ctd_neuro_diseases: il gene e' associato (letteratura curata) a
        #     una malattia neuro/SLA? Stessa "famiglia" di evidenza di
        #     PanelApp/Open Targets, quindi peso comparabile a quello.
        #   - ctd_chemicals (pesticidi): il gene e' influenzato da
        #     un'esposizione ambientale nota? Non e' evidenza di causalita'
        #     con la SLA di per se', ma e' centrale per lo studio
        #     gene-ambiente: un gene sia SLA-rilevante SIA responsivo a
        #     pesticidi e' il caso piu' interessante. Peso minore perche'
        #     e' un segnale di plausibilita' meccanicistica, non di malattia.
        if d.get("ctd_neuro_diseases"):
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