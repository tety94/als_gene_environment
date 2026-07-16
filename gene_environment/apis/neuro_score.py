class NeuroScore:

    @staticmethod
    def compute(d: dict) -> float:
        score = 0
        score += 2 if d["expressed_brain"] else 0
        score += 2 if d["expressed_neurons"] else 0
        score += 1 if d["expressed_glia"] else 0
        score += 2 if d["go_neuro_processes"] else 0
        score += 2 if d["ctd_chemicals"] else 0
        return score
