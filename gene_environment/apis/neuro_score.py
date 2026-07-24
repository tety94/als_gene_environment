# gene_environment/apis/neuro_score.py
"""
Calcolo dello score di "plausibilita' neuro" per geni candidati SLA.

Lo score e' una combinazione pesata di tre famiglie di evidenza:

    1. Espressione nel sistema nervoso centrale (GTEx / HPA).
    2. Evidenza CTD (Comparative Toxicogenomics Database) su
       malattia neuro-motoria e/o esposizione a pesticidi.
    3. Evidenza SLA-specifica curata (PanelApp, Open Targets).

Il GO (Gene Ontology) e' attualmente disattivato a monte (vedi
GeneAnnotator) e non contribuisce allo score.

Lo score non ha un limite superiore fisso ed e' pensato per il
RANKING relativo dei geni candidati all'interno di uno stesso run,
non come probabilita' o metrica normalizzata.

IMPORTANTE: questo modulo mantiene invariate le regole di scoring
originali (pesi, ordine di combinazione, segnali mutuamente esclusivi
vs. additivi). Le uniche modifiche rispetto alla versione precedente
sono di organizzazione del codice, documentazione e robustezza di
fronte a campi mancanti o malformati: nessun peso e nessuna
interpretazione dei campi e' stata alterata.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

# --- Pesi dello score -------------------------------------------------
# Costanti isolate solo per leggibilita' e per evitare "numeri magici"
# sparsi nel codice: i valori sono identici alla versione originale.

WEIGHT_EXPRESSED_BRAIN: float = 1
WEIGHT_EXPRESSED_NEURONS: float = 1
WEIGHT_EXPRESSED_GLIA: float = 1

WEIGHT_CTD_NEURO_DISEASE_PESTICIDE_MEDIATED: float = 3
WEIGHT_CTD_NEURO_DISEASE_DIRECT: float = 2
WEIGHT_CTD_CHEMICALS: float = 1

WEIGHT_PANELAPP_GREEN: float = 3
WEIGHT_PANELAPP_AMBER: float = 1.5

PANELAPP_CONFIDENCE_GREEN: str = "3"
PANELAPP_CONFIDENCE_AMBER: str = "2"

# Lo score Open Targets e' 0.0-1.0: viene riscalato per pesare quanto
# gli altri segnali (invariato rispetto all'originale).
OPENTARGETS_SCALING_FACTOR: float = 2


@dataclass(frozen=True)
class NeuroScoreInput:
    """
    Vista tipizzata sui soli campi del dizionario di annotazione del
    gene (prodotto da ``GeneAnnotator.annotate``) effettivamente usati
    da ``NeuroScore``. Gli altri campi (``gene_id``, ``gene_symbol``,
    ``gene_type``, ``go_*``, ...) non partecipano allo scoring e sono
    volutamente omessi qui.

    Attributes:
        expressed_brain: espressione nel tessuto cerebrale (GTEx).
        expressed_neurons: espressione nei neuroni (HPA single-cell).
        expressed_glia: espressione nelle cellule gliali (HPA single-cell).
        ctd_neuro_disease_direct: CTD riporta un'associazione diretta,
            curata in letteratura, tra il gene e una malattia
            SLA/motoneuronale.
        ctd_neuro_disease_pesticide_mediated: CTD riporta
            un'associazione mediata da pesticidi tra il gene e una
            malattia SLA/motoneuronale (gene -> chimico -> malattia).
        ctd_chemicals: il gene e' associato in CTD a un'esposizione a
            pesticidi, senza un legame a malattia riportato da CTD
            (es. stringa di nomi chimici separati da virgola, o valore
            falsy se assente).
        als_panelapp_confidence: livello di confidenza del pannello SLA
            PanelApp, come stringa: "3" = green (diagnostic-grade),
            "2" = amber (moderata); altri valori/None = nessun peso
            aggiuntivo.
        als_opentargets_score: score di associazione SLA da Open
            Targets, nell'intervallo 0.0-1.0, oppure None/0 se non
            disponibile.
    """

    expressed_brain: bool = False
    expressed_neurons: bool = False
    expressed_glia: bool = False
    ctd_neuro_disease_direct: bool = False
    ctd_neuro_disease_pesticide_mediated: bool = False
    ctd_chemicals: Optional[str] = None
    als_panelapp_confidence: Optional[str] = None
    als_opentargets_score: Optional[float] = None

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "NeuroScoreInput":
        """
        Costruisce un ``NeuroScoreInput`` a partire dal dizionario di
        annotazione grezzo, tollerando chiavi mancanti esattamente come
        faceva l'implementazione originale basata su ``dict.get()``
        (chiave assente == nessuna evidenza == falsy).
        """
        return cls(
            expressed_brain=bool(d.get("expressed_brain")),
            expressed_neurons=bool(d.get("expressed_neurons")),
            expressed_glia=bool(d.get("expressed_glia")),
            ctd_neuro_disease_direct=bool(d.get("ctd_neuro_disease_direct")),
            ctd_neuro_disease_pesticide_mediated=bool(
                d.get("ctd_neuro_disease_pesticide_mediated")
            ),
            ctd_chemicals=d.get("ctd_chemicals"),
            als_panelapp_confidence=d.get("als_panelapp_confidence"),
            als_opentargets_score=d.get("als_opentargets_score"),
        )


def _safe_numeric(value: Any) -> float:
    """
    Converte ``value`` in ``float`` in modo robusto, per proteggere lo
    score da campi malformati (es. stringa non numerica) senza far
    sollevare un'eccezione all'intera pipeline di annotazione.

    Restituisce 0.0 se ``value`` e' None/falsy o non convertibile.
    Per input validi (int/float, incluso 0.0) il comportamento e'
    identico all'originale: un valore falsy (None, 0, 0.0) contribuisce
    comunque 0 allo score, con o senza questa funzione di sicurezza.
    """
    if not value:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


class NeuroScore:
    """
    Calcola lo score composito di plausibilita' neuro per un gene
    candidato.

    Vedi il docstring di modulo per il razionale dello scoring. Tutti i
    pesi e la logica di combinazione sono invariati rispetto
    all'implementazione originale; questa classe si limita a
    riorganizzare il codice in componenti piu' leggibili e testabili.
    """

    @staticmethod
    def compute(d: Mapping[str, Any]) -> float:
        """
        Calcola lo score di plausibilita' neuro per un singolo gene.

        Args:
            d: dizionario di annotazione del gene (come costruito da
                ``GeneAnnotator.annotate``). Vengono letti solo i campi
                descritti in ``NeuroScoreInput``; le altre chiavi sono
                ignorate. Campi mancanti o None sono trattati come
                "nessuna evidenza" e contribuiscono 0 allo score,
                esattamente come nell'implementazione originale basata
                su ``dict.get()``.

        Returns:
            Lo score totale di plausibilita' neuro (float, senza limite
            superiore fisso; piu' alto = piu' plausibile). Pensato per
            confrontare geni tra loro all'interno dello stesso run, non
            come misura assoluta.
        """
        inputs = NeuroScoreInput.from_dict(d)

        score = 0.0
        score += NeuroScore._expression_score(inputs)
        score += NeuroScore._ctd_score(inputs)
        score += NeuroScore._als_evidence_score(inputs)
        return score

    @staticmethod
    def _expression_score(inputs: NeuroScoreInput) -> float:
        """
        Evidenza di espressione nel SNC: +1 per ciascuno tra espressione
        cerebrale / neuronale / gliale presente. Segnali indipendenti e
        additivi (massimo 3 punti).
        """
        score = 0.0
        if inputs.expressed_brain:
            score += WEIGHT_EXPRESSED_BRAIN
        if inputs.expressed_neurons:
            score += WEIGHT_EXPRESSED_NEURONS
        if inputs.expressed_glia:
            score += WEIGHT_EXPRESSED_GLIA
        return score

    @staticmethod
    def _ctd_score(inputs: NeuroScoreInput) -> float:
        """
        Evidenza CTD (Comparative Toxicogenomics Database).

        Il GO e' disattivato a monte (vedi GeneAnnotator) e non
        contribuisce allo score. CTD fornisce invece tre segnali
        distinti, pesati in base a quanto direttamente supportano
        l'ipotesi gene-ambiente in esame:

          - ``ctd_neuro_disease_pesticide_mediated``: CTD stessa
            collega il gene a una malattia SLA/motoneuronale
            PASSANDO per uno specifico pesticida. E' il segnale piu'
            specifico per questo studio, perche' conferma da una fonte
            indipendente esattamente l'ipotesi gene-ambiente testata
            -> peso piu' alto.

          - ``ctd_neuro_disease_direct``: associazione diretta,
            curata in letteratura, tra gene e malattia -- stessa
            "famiglia" di evidenza di PanelApp/Open Targets -> peso
            comparabile a quello.

            (Questi due segnali di malattia sono mutuamente esclusivi
            nello scoring: conta solo il piu' forte tra i due presenti,
            come nell'if/elif originale.)

          - ``ctd_chemicals``: il gene e' influenzato da
            un'esposizione ambientale nota (pesticidi), ma CTD non
            collega esplicitamente quel chimico alla SLA. Plausibilita'
            meccanicistica piu' debole -> peso minore. E' indipendente
            dai due segnali di malattia sopra e puo' sommarsi a
            entrambi.
        """
        score = 0.0
        if inputs.ctd_neuro_disease_pesticide_mediated:
            score += WEIGHT_CTD_NEURO_DISEASE_PESTICIDE_MEDIATED
        elif inputs.ctd_neuro_disease_direct:
            score += WEIGHT_CTD_NEURO_DISEASE_DIRECT

        if inputs.ctd_chemicals:
            score += WEIGHT_CTD_CHEMICALS

        return score

    @staticmethod
    def _als_evidence_score(inputs: NeuroScoreInput) -> float:
        """
        Evidenza SLA-specifica curata, pesata in base all'affidabilita'
        della fonte:

          - PanelApp "green" (confidence "3") = evidenza
            diagnostic-grade, il segnale piu' forte.
          - PanelApp "amber" (confidence "2") = evidenza moderata.
          - Score Open Targets = evidenza aggregata continua
            (letteratura + GWAS + altro), riscalata per pesare quanto
            gli altri segnali.
        """
        score = 0.0

        if inputs.als_panelapp_confidence == PANELAPP_CONFIDENCE_GREEN:
            score += WEIGHT_PANELAPP_GREEN
        elif inputs.als_panelapp_confidence == PANELAPP_CONFIDENCE_AMBER:
            score += WEIGHT_PANELAPP_AMBER

        ot_score = _safe_numeric(inputs.als_opentargets_score)
        if ot_score:
            score += ot_score * OPENTARGETS_SCALING_FACTOR

        return score