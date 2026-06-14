"""Europe PMC REST client — primary corpus source (ADR: docs/adr/0002, 0007).

Covers: PubMed/MEDLINE, PMC Open Access, BMC, PLOS, Cureus via one interface.
Uses cursor-based pagination; no API key required for read access.

Evidence-level priority follows EBM hierarchy, not country of origin — the
`study_type` and `evidence_level` payload fields in Qdrant enable filtering
by evidence strength rather than protocol source (see architecture §4).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import httpx

EPMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
_DEFAULT_PAGE = 25
_INTER_PAGE_SLEEP = 0.5  # polite rate-limiting: 2 req/s

# Map Europe PMC pubType strings to our evidence taxonomy
_PUBTYPE_TO_STUDY_TYPE: dict[str, str] = {
    "Meta-Analysis": "meta-analysis",
    "Systematic Review": "systematic-review",
    "Randomized Controlled Trial": "rct",
    "Clinical Trial": "clinical-trial",
    "Clinical Study": "clinical-trial",
    "Observational Study": "cohort",
    "Comparative Study": "cohort",
    "Multicenter Study": "cohort",
    "Case Reports": "case-report",
    "Case Report": "case-report",
    "Review": "review",
    "Literature Review": "review",
    "Practice Guideline": "guideline",
    "Guideline": "guideline",
}

# Evidence level: lower = stronger (used for Qdrant payload filtering)
STUDY_TYPE_LEVEL: dict[str, int] = {
    "meta-analysis": 1,
    "systematic-review": 2,
    "rct": 3,
    "clinical-trial": 4,
    "cohort": 5,
    "case-report": 6,
    "review": 7,
    "guideline": 7,
    "other": 8,
}


@dataclass
class ArticleRecord:
    pmid: str | None
    doi: str | None
    pmcid: str | None
    title: str
    journal: str
    year: int
    abstract: str
    study_type: str
    evidence_level: int  # 1 (meta-analysis) … 8 (other)

    @property
    def has_usable_text(self) -> bool:
        return bool(self.abstract and len(self.abstract.split()) >= 30)


def classify_study_type(pub_types: list[str]) -> tuple[str, int]:
    """Return (study_type, evidence_level) from a list of pubType strings.

    Multiple types may be present (e.g. ['Journal Article', 'RCT']); we pick
    the strongest (lowest level number).
    """
    best_type = "other"
    best_level = STUDY_TYPE_LEVEL["other"]
    for pt in pub_types:
        st = _PUBTYPE_TO_STUDY_TYPE.get(pt)
        if st is None:
            continue
        level = STUDY_TYPE_LEVEL[st]
        if level < best_level:
            best_type, best_level = st, level
    return best_type, best_level


def _parse_article(raw: dict) -> ArticleRecord | None:
    abstract = raw.get("abstractText", "").strip()
    if not abstract:
        return None

    title = raw.get("title", "").strip().rstrip(".")
    if not title:
        return None

    pub_types: list[str] = []
    pt_block = raw.get("pubTypeList") or {}
    raw_types = pt_block.get("pubType", [])
    if isinstance(raw_types, str):
        raw_types = [raw_types]
    pub_types = [str(t) for t in raw_types]

    study_type, evidence_level = classify_study_type(pub_types)

    try:
        year = int(raw.get("pubYear", 0))
    except (ValueError, TypeError):
        year = 0

    return ArticleRecord(
        pmid=raw.get("id") or raw.get("pmid") or None,
        doi=raw.get("doi") or None,
        pmcid=raw.get("pmcid") or None,
        title=title,
        journal=(raw.get("journalTitle") or raw.get("journalAbbreviation") or "").strip(),
        year=year,
        abstract=abstract,
        study_type=study_type,
        evidence_level=evidence_level,
    )


class EuropePMCClient:
    """Synchronous Europe PMC REST client with cursor-based pagination."""

    def __init__(self, timeout: float = 30.0):
        self._client = httpx.Client(timeout=timeout)

    def search(
        self,
        query: str,
        limit: int = 100,
        min_year: int | None = None,
        max_year: int | None = None,
    ) -> list[ArticleRecord]:
        """Fetch articles for *query* (English MeSH / keywords), up to *limit*."""
        q = query
        if min_year and max_year:
            q += f" AND (FIRST_PDATE:[{min_year}-01-01 TO {max_year}-12-31])"
        elif min_year:
            q += f" AND (FIRST_PDATE:[{min_year}-01-01 TO 3000-01-01])"

        articles: list[ArticleRecord] = []
        cursor = "*"

        while len(articles) < limit:
            batch = min(_DEFAULT_PAGE, limit - len(articles))
            params = {
                "query": q,
                "format": "json",
                "pageSize": batch,
                "resultType": "core",
                "cursorMark": cursor,
            }
            response = self._client.get(EPMC_SEARCH, params=params)
            response.raise_for_status()
            payload = response.json()

            result_list = payload.get("resultList", {}).get("result", [])
            if not result_list:
                break

            for raw in result_list:
                record = _parse_article(raw)
                if record is not None:
                    articles.append(record)

            next_cursor = payload.get("nextCursorMark")
            if not next_cursor or next_cursor == cursor:
                break
            cursor = next_cursor

            if len(result_list) < batch:
                break

            time.sleep(_INTER_PAGE_SLEEP)

        return articles[:limit]

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
