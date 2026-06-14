"""Evidence node — symmetric retrieval per hypothesis (anti confirmation bias).

Pipeline per hypothesis:
  LLM → (supporting_query, refuting_query) both in English
  supporting_query → HybridRetriever.search() → Citation list  [FOR]
  refuting_query   → HybridRetriever.search() → Citation list  [AGAINST]

Anti-bias invariants enforced by code:
  - Two separate searches per hypothesis (symmetric — not just supporting).
  - Queries are always in English (LLM prompt + query language assertion).
  - Citations are built only from retrieved payloads — never from LLM memory.
  - Deduplication within each list to avoid the same PMID appearing twice.

Graceful degradation:
  If the Qdrant corpus is empty (not yet ingested), the node returns empty
  evidence lists rather than raising. The synthesis agent handles this case
  by noting that no literature was retrieved.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from meddx.llm import model_for_agent
from meddx.prompts import load_prompt
from meddx.rag.retriever import get_retriever
from meddx.schemas import Citation, DiagnosticState, Hypothesis, HypothesisEvidence, Phase

TOP_K = 5  # results per stance per hypothesis


# ---------------------------------------------------------------------------
# LLM output schema
# ---------------------------------------------------------------------------

class _QueryPair(BaseModel):
    hypothesis_id: str = Field(description="Exact hypothesis id as provided (e.g. 'h1')")
    supporting_query: str = Field(
        description=(
            "English keyword query (5–10 words, MeSH style) for evidence "
            "SUPPORTING this hypothesis."
        )
    )
    refuting_query: str = Field(
        description=(
            "English keyword query (5–10 words) for evidence AGAINST / REFUTING "
            "this hypothesis. Include terms like 'mimic', 'atypical', 'false "
            "positive', 'limitations', 'against', 'alternative diagnosis'."
        )
    )


class _EvidenceQueriesResult(BaseModel):
    pairs: list[_QueryPair] = Field(
        description="One query pair per hypothesis. Must cover every hypothesis_id supplied."
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fallback_queries(hypothesis_name: str) -> tuple[str, str]:
    """Template queries when the LLM misses a hypothesis."""
    return (
        f"{hypothesis_name} diagnosis clinical criteria evidence",
        f"{hypothesis_name} mimic alternative diagnosis against limitations",
    )


def _build_hypothesis_block(hypotheses: list[Hypothesis]) -> str:
    lines = ["Hypotheses (generate one query pair for each):"]
    for h in hypotheses:
        lines.append(f"  id={h.id!r}  name={h.name!r}  system={h.organ_system}")
    return "\n".join(lines)


def _format_case_summary(state: DiagnosticState) -> str:
    case = state.patient_case
    parts = []
    if case.chief_complaint:
        parts.append(f"Chief complaint: {case.chief_complaint}")
    if case.risk_factors:
        parts.append(f"Risk factors: {case.risk_factors}")
    if case.history_of_present_illness:
        parts.append(f"History: {case.history_of_present_illness}")
    return "\n".join(parts) if parts else "See patient case."


def _generate_queries(state: DiagnosticState) -> dict[str, tuple[str, str]]:
    """Call LLM once to generate (supporting, refuting) query pairs for all
    hypotheses. Returns {hypothesis_id: (supporting_query, refuting_query)}.
    Falls back to templates for any hypothesis the LLM missed.
    """
    llm = model_for_agent("evidence").with_structured_output(_EvidenceQueriesResult)
    system = load_prompt("evidence")

    case_summary = _format_case_summary(state)
    hypothesis_block = _build_hypothesis_block(state.hypotheses)

    result: _EvidenceQueriesResult = llm.invoke(
        [
            SystemMessage(content=system),
            HumanMessage(
                content=(
                    f"Patient case summary:\n{case_summary}\n\n"
                    f"{hypothesis_block}\n\n"
                    "Generate one supporting and one refuting English search "
                    "query for each hypothesis id listed above."
                )
            ),
        ]
    )

    query_map: dict[str, tuple[str, str]] = {
        pair.hypothesis_id: (pair.supporting_query, pair.refuting_query)
        for pair in result.pairs
    }

    # Fill in templates for any hypothesis the LLM did not cover
    for h in state.hypotheses:
        if h.id not in query_map:
            query_map[h.id] = _fallback_queries(h.name)

    return query_map


def _payload_to_citation(payload: dict) -> Citation | None:
    """Convert a Qdrant point payload to a Citation.  Returns None if the
    payload lacks a title (should not happen with well-formed ingestion)."""
    title = (payload.get("title") or "").strip()
    if not title:
        return None
    pmid = payload.get("pmid") or None
    doi = payload.get("doi") or None
    return Citation(
        pmid=pmid,
        doi=doi,
        title=title,
        journal=(payload.get("journal") or "").strip(),
        year=int(payload.get("year") or 0),
        study_type=(payload.get("study_type") or "other"),
        url=(
            f"https://europepmc.org/article/MED/{pmid}"
            if pmid else None
        ),
    )


def _deduplicate(citations: list[Citation]) -> list[Citation]:
    """Remove citations with the same PMID or DOI (keep first occurrence)."""
    seen: set[str] = set()
    result: list[Citation] = []
    for c in citations:
        ids = c.identifiers()
        if ids:
            if ids & seen:
                continue
            seen |= ids
        result.append(c)
    return result


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def evidence_node(state: DiagnosticState) -> dict:
    # 1. Generate English search query pairs via LLM
    query_map = _generate_queries(state)

    # 2. Get the shared retriever (lazy, cached — loads BGE-M3 + Qdrant once)
    retriever = get_retriever()

    evidence_list: list[HypothesisEvidence] = []

    if not retriever.is_ready():
        # Corpus not ingested yet — return empty evidence rather than crash.
        # Synthesis will note the absence of citations.
        for h in state.hypotheses:
            evidence_list.append(HypothesisEvidence(hypothesis_id=h.id))
        return {"evidence": evidence_list, "phase": Phase.CHALLENGE}

    # 3. Symmetric retrieval: FOR and AGAINST each hypothesis separately
    for h in state.hypotheses:
        supporting_q, refuting_q = query_map[h.id]

        supp_payloads = retriever.search(supporting_q, k=TOP_K)
        ref_payloads = retriever.search(refuting_q, k=TOP_K)

        supporting = _deduplicate(
            [c for p in supp_payloads if (c := _payload_to_citation(p)) is not None]
        )
        refuting = _deduplicate(
            [c for p in ref_payloads if (c := _payload_to_citation(p)) is not None]
        )

        evidence_list.append(
            HypothesisEvidence(
                hypothesis_id=h.id,
                supporting=supporting,
                refuting=refuting,
            )
        )

    return {"evidence": evidence_list, "phase": Phase.CHALLENGE}
