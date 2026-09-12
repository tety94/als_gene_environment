# gene_environment/apis/neuro_score.py
"""Computes a "neuro plausibility" score for ALS candidate genes.

The score is a weighted combination of three families of evidence:

    1. Expression in the central nervous system (GTEx / HPA).
    2. CTD (Comparative Toxicogenomics Database) evidence on
       neuro-motor disease and/or pesticide exposure.
    3. Curated ALS-specific evidence (PanelApp, Open Targets).

GO (Gene Ontology) is currently disabled upstream (see GeneAnnotator)
and does not contribute to the score.

The score has no fixed upper bound and is meant for RANKING candidate
genes relative to each other within the same run, not as a probability
or normalized metric.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

# --- Score weights ------------------------------------------------
# Isolated as constants purely for readability, to avoid magic numbers
# scattered through the code.

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

# The Open Targets score is 0.0-1.0: rescaled to weigh comparably to the other signals.
OPENTARGETS_SCALING_FACTOR: float = 2


@dataclass(frozen=True)
class NeuroScoreInput:
    """Typed view over just the fields of the gene annotation dict
    (produced by ``GeneAnnotator.annotate``) that are actually used by
    ``NeuroScore``. Other fields (``gene_id``, ``gene_symbol``,
    ``gene_type``, ``go_*``, ...) don't participate in scoring and are
    deliberately omitted here.

    Attributes:
        expressed_brain: expression in brain tissue (GTEx).
        expressed_neurons: expression in neurons (HPA single-cell).
        expressed_glia: expression in glial cells (HPA single-cell).
        ctd_neuro_disease_direct: CTD reports a direct, literature-curated
            association between the gene and an ALS/motor-neuron disease.
        ctd_neuro_disease_pesticide_mediated: CTD reports a
            pesticide-mediated association between the gene and an
            ALS/motor-neuron disease (gene -> chemical -> disease).
        ctd_chemicals: the gene is associated in CTD with a pesticide
            exposure, with no disease link reported by CTD (e.g. a
            comma-separated string of chemical names, or a falsy value
            if absent).
        als_panelapp_confidence: PanelApp ALS panel confidence level, as
            a string: "3" = green (diagnostic-grade), "2" = amber
            (moderate); other values/None = no additional weight.
        als_opentargets_score: ALS association score from Open Targets,
            in the 0.0-1.0 range, or None/0 if unavailable.
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
        """Build a ``NeuroScoreInput`` from the raw annotation dict,
        tolerating missing keys (missing key == no evidence == falsy)."""
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
    """Robustly convert ``value`` to ``float``, to protect the score from
    malformed fields (e.g. a non-numeric string) without raising an
    exception in the whole annotation pipeline.

    Returns 0.0 if ``value`` is None/falsy or not convertible."""
    if not value:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


class NeuroScore:
    """Computes the composite neuro-plausibility score for a candidate gene.

    See the module docstring for the scoring rationale."""

    @staticmethod
    def compute(d: Mapping[str, Any]) -> float:
        """Compute the neuro-plausibility score for a single gene.

        Args:
            d: gene annotation dict (as built by
                ``GeneAnnotator.annotate``). Only the fields described in
                ``NeuroScoreInput`` are read; other keys are ignored.
                Missing or None fields are treated as "no evidence" and
                contribute 0 to the score.

        Returns:
            The total neuro-plausibility score (float, no fixed upper
            bound; higher = more plausible). Meant for comparing genes
            against each other within the same run, not as an absolute
            measure.
        """
        inputs = NeuroScoreInput.from_dict(d)

        score = 0.0
        score += NeuroScore._expression_score(inputs)
        score += NeuroScore._ctd_score(inputs)
        score += NeuroScore._als_evidence_score(inputs)
        return score

    @staticmethod
    def _expression_score(inputs: NeuroScoreInput) -> float:
        """CNS expression evidence: +1 for each of brain / neuronal / glial
        expression present. Independent, additive signals (max 3 points)."""
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
        """CTD (Comparative Toxicogenomics Database) evidence.

        GO is disabled upstream (see GeneAnnotator) and does not
        contribute to the score. CTD instead provides three distinct
        signals, weighted by how directly they support the gene x
        environment hypothesis under study:

          - ``ctd_neuro_disease_pesticide_mediated``: CTD itself links the
            gene to an ALS/motor-neuron disease THROUGH a specific
            pesticide. This is the most specific signal for this study,
            because it confirms the exact gene x environment hypothesis
            being tested from an independent source -> highest weight.

          - ``ctd_neuro_disease_direct``: a direct, literature-curated
            association between gene and disease -- the same "family" of
            evidence as PanelApp/Open Targets -> comparable weight.

            (These two disease signals are mutually exclusive in the
            scoring: only the stronger of the two present counts.)

          - ``ctd_chemicals``: the gene is affected by a known
            environmental exposure (pesticides), but CTD does not
            explicitly link that chemical to ALS. Weaker mechanistic
            plausibility -> lower weight. Independent of the two disease
            signals above and can add to either.
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
        """Curated ALS-specific evidence, weighted by source reliability:

          - PanelApp "green" (confidence "3") = diagnostic-grade evidence,
            the strongest signal.
          - PanelApp "amber" (confidence "2") = moderate evidence.
          - Open Targets score = continuous aggregated evidence
            (literature + GWAS + other), rescaled to weigh comparably to
            the other signals.
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
