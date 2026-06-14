"""Tests for the ingestion pipeline: Europe PMC parsing + chunker.

No network, no GPU, no Qdrant required — we mock httpx and provide fixture
article records to test parsing and chunking logic in isolation.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

# ── Provide fake heavy modules before they are imported ──────────────────────
# This allows running tests without torch / FlagEmbedding / qdrant-client.
for _mod in ("FlagEmbedding", "qdrant_client", "qdrant_client.models"):
    sys.modules.setdefault(_mod, MagicMock())

from meddx.ingestion.chunker import (
    ArticleChunk,
    _normalise_section_name,
    _sentence_windows,
    _split_structured,
    chunk_article,
)
from meddx.ingestion.europe_pmc import (
    ArticleRecord,
    EuropePMCClient,
    classify_study_type,
)


# ─────────────────────────────────────────────────────────────────────────────
# Evidence level classification
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("pub_types,expected_type,expected_level", [
    (["Meta-Analysis"],                          "meta-analysis",    1),
    (["Systematic Review"],                      "systematic-review", 2),
    (["Journal Article", "Randomized Controlled Trial"], "rct",      3),
    (["Clinical Trial"],                         "clinical-trial",   4),
    (["Observational Study"],                    "cohort",           5),
    (["Case Reports"],                           "case-report",      6),
    (["Review"],                                 "review",           7),
    (["Journal Article"],                        "other",            8),
    ([],                                         "other",            8),
])
def test_classify_study_type(pub_types, expected_type, expected_level):
    st, level = classify_study_type(pub_types)
    assert st == expected_type
    assert level == expected_level


def test_classify_picks_strongest_when_multiple_types():
    """Journal Article + RCT → RCT (level 3), not Journal Article (level 8)."""
    st, level = classify_study_type(["Journal Article", "Randomized Controlled Trial"])
    assert st == "rct"
    assert level == 3


def test_classify_picks_strongest_meta_analysis_over_review():
    st, level = classify_study_type(["Review", "Meta-Analysis"])
    assert st == "meta-analysis"
    assert level == 1


# ─────────────────────────────────────────────────────────────────────────────
# ArticleRecord
# ─────────────────────────────────────────────────────────────────────────────

def _make_article(**kwargs) -> ArticleRecord:
    defaults = dict(
        pmid="12345678",
        doi="10.1000/test",
        pmcid=None,
        title="Test Article",
        journal="Test Journal",
        year=2024,
        abstract="This is the abstract. " * 10,
        study_type="rct",
        evidence_level=3,
    )
    return ArticleRecord(**{**defaults, **kwargs})


def test_has_usable_text_true_for_long_abstract():
    article = _make_article(abstract="Word " * 50)
    assert article.has_usable_text is True


def test_has_usable_text_false_for_short_abstract():
    article = _make_article(abstract="Short.")
    assert article.has_usable_text is False


def test_has_usable_text_false_for_empty():
    article = _make_article(abstract="")
    assert article.has_usable_text is False


# ─────────────────────────────────────────────────────────────────────────────
# Section name normalisation
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("header,expected", [
    ("BACKGROUND", "background"),
    ("OBJECTIVES", "background"),
    ("AIM", "background"),
    ("METHODS", "methods"),
    ("PATIENTS AND METHODS", "methods"),
    ("RESULTS", "results"),
    ("FINDINGS", "results"),
    ("CONCLUSIONS", "conclusions"),
    ("DISCUSSION", "conclusions"),
    ("SUMMARY", "conclusions"),
    ("UNKNOWN_HEADER", "other"),
])
def test_normalise_section_name(header, expected):
    assert _normalise_section_name(header) == expected


# ─────────────────────────────────────────────────────────────────────────────
# Structured abstract detection
# ─────────────────────────────────────────────────────────────────────────────

_STRUCTURED_ABSTRACT = """\
BACKGROUND: Chest pain is a common presenting complaint in emergency departments.
METHODS: We conducted a randomised trial across 12 centres.
RESULTS: High-sensitivity troponin had sensitivity of 99.3%.
CONCLUSIONS: Early rule-out protocols are safe and reduce unnecessary admissions.\
"""

_UNSTRUCTURED_ABSTRACT = (
    "Chest pain is a common presenting complaint. "
    "Multiple aetiologies must be considered. "
    "This review summarises current evidence."
)


def test_split_structured_detects_all_sections():
    sections = _split_structured(_STRUCTURED_ABSTRACT)
    names = [name for name, _ in sections]
    assert "background" in names
    assert "methods" in names
    assert "results" in names
    assert "conclusions" in names


def test_split_structured_returns_empty_for_unstructured():
    assert _split_structured(_UNSTRUCTURED_ABSTRACT) == []


def test_split_structured_body_contains_text():
    sections = dict(_split_structured(_STRUCTURED_ABSTRACT))
    assert "troponin" in sections["results"]
    assert "randomised" in sections["methods"]


# ─────────────────────────────────────────────────────────────────────────────
# Sentence window splitting
# ─────────────────────────────────────────────────────────────────────────────

def test_sentence_windows_short_text_gives_single_window():
    text = "Short sentence. Another short sentence."
    windows = _sentence_windows(text)
    assert len(windows) == 1


def test_sentence_windows_long_text_gives_multiple_windows():
    # Build a text clearly > MAX_WORDS (400)
    sentence = "This is a sentence with ten words in it here."
    text = " ".join([sentence] * 50)  # 500+ words
    windows = _sentence_windows(text)
    assert len(windows) > 1


def test_sentence_windows_overlap_preserved():
    """Words from the end of window N appear at the start of window N+1."""
    sentence = "Word " * 10  # 10 words per sentence
    text = " ".join([sentence.strip()] * 50)
    windows = _sentence_windows(text)
    if len(windows) < 2:
        pytest.skip("text too short for overlap test")
    last_words_of_w0 = set(windows[0].split()[-10:])
    first_words_of_w1 = set(windows[1].split()[:10])
    assert last_words_of_w0 & first_words_of_w1  # some overlap


# ─────────────────────────────────────────────────────────────────────────────
# chunk_article — end-to-end
# ─────────────────────────────────────────────────────────────────────────────

def test_chunk_structured_abstract_produces_multiple_chunks():
    article = _make_article(abstract=_STRUCTURED_ABSTRACT)
    chunks = chunk_article(article)
    assert len(chunks) >= 3
    sections = {c.section for c in chunks}
    assert "background" in sections
    assert "results" in sections


def test_chunk_short_unstructured_gives_single_chunk():
    article = _make_article(abstract=_UNSTRUCTURED_ABSTRACT)
    chunks = chunk_article(article)
    assert len(chunks) == 1
    assert chunks[0].section == "abstract"


def test_chunk_metadata_propagated():
    article = _make_article()
    chunks = chunk_article(article)
    for chunk in chunks:
        assert chunk.pmid == article.pmid
        assert chunk.doi == article.doi
        assert chunk.year == article.year
        assert chunk.study_type == article.study_type
        assert chunk.evidence_level == article.evidence_level
        assert chunk.journal == article.journal


def test_chunk_text_prefixed_with_title():
    article = _make_article(title="Troponin in ACS")
    chunks = chunk_article(article)
    assert all("Troponin in ACS" in c.text for c in chunks)


def test_chunk_index_sequential():
    article = _make_article(abstract=_STRUCTURED_ABSTRACT)
    chunks = chunk_article(article)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_point_id_deterministic():
    article = _make_article()
    chunks_a = chunk_article(article)
    chunks_b = chunk_article(article)
    assert [c.point_id() for c in chunks_a] == [c.point_id() for c in chunks_b]


def test_point_id_differs_across_chunks():
    article = _make_article(abstract=_STRUCTURED_ABSTRACT)
    chunks = chunk_article(article)
    ids = [c.point_id() for c in chunks]
    assert len(ids) == len(set(ids))


def test_payload_contains_required_fields():
    article = _make_article()
    chunk = chunk_article(article)[0]
    payload = chunk.payload()
    for key in ("pmid", "doi", "title", "journal", "year",
                "study_type", "evidence_level", "section", "text", "chunk_index"):
        assert key in payload


def test_empty_abstract_yields_no_chunks():
    article = _make_article(abstract="")
    assert chunk_article(article) == []


# ─────────────────────────────────────────────────────────────────────────────
# Europe PMC client — response parsing (mocked httpx)
# ─────────────────────────────────────────────────────────────────────────────

_EPMC_RESPONSE = {
    "resultList": {
        "result": [
            {
                "id": "12345678",
                "doi": "10.1000/test",
                "pmcid": None,
                "title": "High-sensitivity troponin for ACS rule-out.",
                "journalTitle": "NEJM",
                "pubYear": "2023",
                "abstractText": (
                    "BACKGROUND: Acute coronary syndrome rule-out is critical in "
                    "the emergency department setting and requires rapid assessment. "
                    "METHODS: We conducted a prospective cohort study across twelve "
                    "tertiary care centres enrolling consecutive chest pain patients. "
                    "RESULTS: High-sensitivity troponin had a sensitivity of 99.3% "
                    "and specificity of 92.1% for NSTEMI at presentation. "
                    "CONCLUSIONS: An early rule-out protocol using high-sensitivity "
                    "troponin enables safe discharge within two hours of presentation."
                ),
                "pubTypeList": {"pubType": ["Journal Article", "Randomized Controlled Trial"]},
            },
            {
                "id": "99999999",
                "title": "Article without abstract.",
                "journalTitle": "BMJ",
                "pubYear": "2022",
                "pubTypeList": {"pubType": ["Journal Article"]},
                # no abstractText
            },
        ]
    },
    "nextCursorMark": "AoE=",
}

_EPMC_EMPTY_RESPONSE = {
    "resultList": {"result": []},
    "nextCursorMark": "AoE=",
}


def test_epmc_client_parses_articles(monkeypatch):
    mock_resp = MagicMock()
    mock_resp.json.return_value = _EPMC_RESPONSE
    mock_resp.raise_for_status = MagicMock()

    # Second call returns empty to stop pagination
    mock_empty = MagicMock()
    mock_empty.json.return_value = _EPMC_EMPTY_RESPONSE
    mock_empty.raise_for_status = MagicMock()

    call_count = 0

    def fake_get(self_, url, **kwargs):
        nonlocal call_count
        call_count += 1
        return mock_resp if call_count == 1 else mock_empty

    monkeypatch.setattr("meddx.ingestion.europe_pmc.httpx.Client.get", fake_get)

    client = EuropePMCClient()
    articles = client.search("troponin ACS", limit=10)

    # One article parsed (the one without abstract is skipped by _parse_article)
    assert len(articles) == 1
    art = articles[0]
    assert art.pmid == "12345678"
    assert art.doi == "10.1000/test"
    assert art.study_type == "rct"
    assert art.evidence_level == 3
    assert art.year == 2023
    assert art.has_usable_text


def test_epmc_client_skips_articles_without_abstract(monkeypatch):
    mock_resp = MagicMock()
    mock_resp.json.return_value = _EPMC_RESPONSE
    mock_resp.raise_for_status = MagicMock()
    mock_empty = MagicMock()
    mock_empty.json.return_value = _EPMC_EMPTY_RESPONSE
    mock_empty.raise_for_status = MagicMock()
    call_count = 0

    def fake_get(self_, url, **kwargs):
        nonlocal call_count
        call_count += 1
        return mock_resp if call_count == 1 else mock_empty

    monkeypatch.setattr("meddx.ingestion.europe_pmc.httpx.Client.get", fake_get)

    client = EuropePMCClient()
    articles = client.search("test", limit=10)
    # Article 99999999 has no abstract and must be filtered
    assert all(a.pmid != "99999999" for a in articles)
