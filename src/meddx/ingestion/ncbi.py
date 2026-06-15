"""NCBI E-utilities client — alternative / fallback corpus source.

Covers PubMed/MEDLINE (same article universe as Europe PMC but different host).
Uses a two-step pipeline: esearch (PMIDs) → efetch (full records with abstracts).
No API key required; rate limit is 3 req/s unauthenticated, 10 req/s with key.
Set NCBI_API_KEY in .env for higher throughput on large ingests.

Returns the same ArticleRecord type as europe_pmc.py so the chunker/embedder
pipeline is unchanged.
"""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET

import httpx

from meddx.ingestion.europe_pmc import ArticleRecord, classify_study_type

NCBI_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
NCBI_EFETCH  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

_EFETCH_BATCH  = 100   # max 200 per request; 100 is safe under timeouts
_SLEEP_BETWEEN = 0.4   # 2.5 req/s — under the 3 req/s unauthenticated cap


class NCBIClient:
    """Two-step PubMed client: esearch (PMIDs) → efetch (XML) → ArticleRecord."""

    def __init__(self, api_key: str | None = None, timeout: float = 30.0) -> None:
        self._http = httpx.Client(timeout=timeout)
        self._api_key = api_key

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        limit: int = 100,
        min_year: int | None = None,
        max_year: int | None = None,
    ) -> list[ArticleRecord]:
        """Return up to *limit* PubMed articles matching *query*."""
        q = _build_query(query, min_year, max_year)
        pmids = self._esearch(q, limit)
        if not pmids:
            return []

        articles: list[ArticleRecord] = []
        for i in range(0, len(pmids), _EFETCH_BATCH):
            batch = pmids[i : i + _EFETCH_BATCH]
            articles.extend(self._efetch(batch))
            if i + _EFETCH_BATCH < len(pmids):
                time.sleep(_SLEEP_BETWEEN)

        return articles

    def close(self) -> None:
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _base_params(self) -> dict:
        p: dict = {"db": "pubmed"}
        if self._api_key:
            p["api_key"] = self._api_key
        return p

    def _esearch(self, query: str, limit: int) -> list[str]:
        resp = self._http.get(
            NCBI_ESEARCH,
            params={**self._base_params(), "term": query, "retmax": limit,
                    "retmode": "json", "sort": "relevance"},
        )
        resp.raise_for_status()
        time.sleep(_SLEEP_BETWEEN)
        return resp.json().get("esearchresult", {}).get("idlist", [])

    def _efetch(self, pmids: list[str]) -> list[ArticleRecord]:
        data = {**self._base_params(), "id": ",".join(pmids),
                "rettype": "abstract", "retmode": "xml"}
        # POST avoids URL-length limits for large batches
        resp = self._http.post(NCBI_EFETCH, data=data)
        resp.raise_for_status()
        time.sleep(_SLEEP_BETWEEN)
        return _parse_pubmed_xml(resp.text)


# ---------------------------------------------------------------------------
# Query builder
# ---------------------------------------------------------------------------

def _build_query(query: str, min_year: int | None, max_year: int | None) -> str:
    q = query
    if min_year and max_year:
        q += f' AND ("{min_year}"[PDAT] : "{max_year}"[PDAT])'
    elif min_year:
        q += f' AND ("{min_year}"[PDAT] : "3000"[PDAT])'
    return q


# ---------------------------------------------------------------------------
# PubMed XML parser
# ---------------------------------------------------------------------------

def _parse_pubmed_xml(xml_text: str) -> list[ArticleRecord]:
    """Parse a PubmedArticleSet XML string into ArticleRecord list."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []

    records: list[ArticleRecord] = []
    for pub_article in root.findall(".//PubmedArticle"):
        rec = _parse_one(pub_article)
        if rec is not None:
            records.append(rec)
    return records


def _parse_one(pub_article: ET.Element) -> ArticleRecord | None:
    medline = pub_article.find("MedlineCitation")
    if medline is None:
        return None
    article = medline.find("Article")
    if article is None:
        return None

    abstract = _abstract_text(article)
    if not abstract:
        return None

    title_node = article.find("ArticleTitle")
    title = "".join(title_node.itertext()).strip().rstrip(".") if title_node is not None else ""
    if not title:
        return None

    pub_types = [
        pt.text or ""
        for pt in article.findall(".//PublicationTypeList/PublicationType")
    ]
    study_type, evidence_level = classify_study_type(pub_types)

    pmid_node = medline.find("PMID")
    pmid = pmid_node.text.strip() if pmid_node is not None else None

    doi: str | None = None
    pmcid: str | None = None
    for aid in pub_article.findall(".//PubmedData/ArticleIdList/ArticleId"):
        id_type = aid.get("IdType", "")
        if id_type == "doi":
            doi = aid.text
        elif id_type == "pmc":
            pmcid = aid.text

    journal_node = article.find(".//Journal/Title")
    journal = journal_node.text.strip() if journal_node is not None else ""

    return ArticleRecord(
        pmid=pmid,
        doi=doi,
        pmcid=pmcid,
        title=title,
        journal=journal,
        year=_pub_year(article),
        abstract=abstract,
        study_type=study_type,
        evidence_level=evidence_level,
    )


def _abstract_text(article: ET.Element) -> str:
    """Join structured abstract sections (with/without Label) into one string."""
    parts: list[str] = []
    for node in article.findall(".//Abstract/AbstractText"):
        label = node.get("Label", "")
        text = "".join(node.itertext()).strip()
        if not text:
            continue
        parts.append(f"{label}: {text}" if label else text)
    return " ".join(parts).strip()


def _pub_year(article: ET.Element) -> int:
    """Extract year from PubDate/Year or MedlineDate ('2022 Jan-Feb' → 2022)."""
    year_node = article.find(".//Journal/JournalIssue/PubDate/Year")
    if year_node is not None and year_node.text:
        try:
            return int(year_node.text)
        except ValueError:
            pass
    ml_node = article.find(".//Journal/JournalIssue/PubDate/MedlineDate")
    if ml_node is not None and ml_node.text:
        try:
            return int(ml_node.text[:4])
        except ValueError:
            pass
    return 0
