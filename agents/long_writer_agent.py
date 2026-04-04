from __future__ import annotations

from agents.base_agent import BaseAgent
from src.protocols import AgentStatus, UrgencyLevel


class LongWriterAgent(BaseAgent):
    """Stub agent simulating long-form technical document writing.

    Represents writing a multi-section technical paper or whitepaper.
    Decision points occur at section transitions where framing, depth,
    and structure choices are made.

    The agent's context history grows fastest of all agent types — each
    "written" section adds a summary to output history — maximally stressing
    the flat-context baseline over many steering events.

    Default structure: 5 section checkpoints, each triggering a steering request.
    Ground-truth answers (for methodology section of a clinical trial paper):
      1. study design framing   → "randomised" / "rct" / "controlled"
      2. statistical depth      → "power" / "sample size" / "power calculation"
      3. primary outcome        → "primary endpoint" / "primary outcome"
      4. control group          → "placebo" / "standard of care"
      5. limitations placement  → "limitations" / "discussion"
    """

    _STEPS: list[tuple[str, bool, str, UrgencyLevel]] = [
        # Section 1 — Introduction / study design overview
        (
            "Drafting section 1: background and rationale for the trial",
            False, "", UrgencyLevel.LOW,
        ),
        (
            "Section 1 complete; section 2 (study design) needs framing decision",
            True,
            (
                "For the study design framing in section 2 of this clinical trial paper: "
                "should the methodology open by establishing this as a randomised controlled trial (RCT) "
                "with double-blind allocation, a prospective cohort design, or a quasi-experimental design? "
                "The trial used randomisation with 1:1 allocation. "
                "RCT framing is expected by reviewers for this journal."
            ),
            UrgencyLevel.HIGH,
        ),
        # Section 2 — Statistical analysis plan
        (
            "Section 2 drafted with RCT framing; writing statistical analysis section",
            False, "", UrgencyLevel.LOW,
        ),
        (
            "Statistical depth decision needed: how much detail on power calculation in methodology",
            True,
            (
                "For the statistical analysis section: should the methodology describe the power calculation "
                "and sample size derivation inline (standard for clinical papers — reviewers expect it), "
                "put it in a supplementary methods appendix to save space, "
                "or cite a separate statistical analysis plan document? "
                "Power calculations and sample size justification are expected inline by CONSORT guidelines."
            ),
            UrgencyLevel.MEDIUM,
        ),
        # Section 3 — Primary and secondary outcomes
        (
            "Statistical section written with power calc inline; drafting outcomes section",
            False, "", UrgencyLevel.LOW,
        ),
        (
            "Need to decide primary endpoint framing to lead the outcomes section",
            True,
            (
                "The outcomes section needs to lead with the primary endpoint. "
                "Options: frame the primary outcome as a composite endpoint (combines multiple events), "
                "single pre-specified primary endpoint (cleaner, preferred by regulatory reviewers), "
                "or list co-primary endpoints? "
                "Single pre-specified primary endpoints are standard and reduce multiple comparison concerns."
            ),
            UrgencyLevel.MEDIUM,
        ),
        # Section 4 — Control group description
        (
            "Outcomes section complete; writing control group and comparator description",
            False, "", UrgencyLevel.LOW,
        ),
        (
            "Control group description: placebo or standard of care comparator framing",
            True,
            (
                "How should the control group be described in the methods? "
                "As a placebo control (inactive comparator, maximises blinding), "
                "standard of care (active comparator, more ethical but harder to blind), "
                "or wait-list control (common for behavioral interventions)? "
                "The trial used an inert placebo with matched appearance."
            ),
            UrgencyLevel.MEDIUM,
        ),
        # Section 5 — Limitations and conclusion framing
        (
            "Control group section written; nearing final section — limitations",
            False, "", UrgencyLevel.LOW,
        ),
        (
            "Where to place limitations: end of methodology section or start of discussion",
            True,
            (
                "For placement of study limitations: include them at the end of the methodology section "
                "as a prospective limitations paragraph, or defer all limitations to the discussion section "
                "where they can be contextualised against the results? "
                "CONSORT and ICMJE guidelines recommend limitations in the discussion section."
            ),
            UrgencyLevel.LOW,
        ),
    ]

    async def _execute(self) -> None:
        for summary, needs_steering, question, urgency in self._STEPS:
            self._push_update(AgentStatus.RUNNING, summary, urgency)
            if needs_steering:
                response = await self._request_steering(
                    relevant_context=self._recent_output(),
                    question=question,
                    blocking=True,
                    urgency=urgency,
                )
                self._push_update(
                    AgentStatus.RUNNING,
                    f"guidance: {response.response_text[:80]}",
                    UrgencyLevel.LOW,
                )
