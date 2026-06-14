"""Programmatic citation validation (anti-hallucination invariant).

Every PMID/DOI used in an answer must exist in the retrieved context. This
is a code-level check — we never rely on the LLM to police its own citations.
"""

from meddx.schemas import Citation


def unknown_citations(used: list[Citation], retrieved: list[Citation]) -> list[Citation]:
    """Return citations from `used` whose identifiers are absent from
    `retrieved`. An empty result means the answer is citation-clean."""
    allowed: set[str] = set()
    for c in retrieved:
        allowed |= c.identifiers()
    return [c for c in used if not (c.identifiers() & allowed)]


def assert_citations_grounded(used: list[Citation], retrieved: list[Citation]) -> None:
    """Raise ValueError if any used citation is not grounded in retrieval."""
    unknown = unknown_citations(used, retrieved)
    if unknown:
        ids = ", ".join(sorted(i for c in unknown for i in c.identifiers()) or ["<no id>"])
        raise ValueError(f"Citations not present in retrieved context: {ids}")
