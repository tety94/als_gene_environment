class NeuroScore:

    @staticmethod
    def compute(d: dict) -> float:
        score = 0
        score += 1 if d.get("expressed_brain") else 0
        score += 1 if d.get("expressed_neurons") else 0
        score += 1 if d.get("expressed_glia") else 0

        # GO/CTD disattivati (vedi GeneAnnotator) -> non contribuiscono più allo score.
        # Sostituiti da segnali di evidenza SLA specifici, pesati in base
        # all'affidabilità della fonte:
        #   - PanelApp "green" = evidenza diagnostic-grade, il segnale più forte
        #   - PanelApp "amber" = evidenza moderata
        #   - Open Targets score = evidenza aggregata continua (letteratura + GWAS + altro)
        confidence = d.get("als_panelapp_confidence")
        if confidence == "3":       # green
            score += 3
        elif confidence == "2":     # amber
            score += 1.5

        ot_score = d.get("als_opentargets_score")
        if ot_score:
            score += ot_score * 2   # score Open Targets è 0.0-1.0, riscalato per pesare quanto gli altri segnali

        return score