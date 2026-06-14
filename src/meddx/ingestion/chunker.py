"""Section-aware chunking for medical literature.

Strategy (in priority order):
  1. Detect structured abstract sections (BACKGROUND / METHODS / RESULTS /
     CONCLUSIONS and their variants) — each section becomes one chunk.
  2. For long unstructured text (> MAX_WORDS): sliding sentence-window with
     OVERLAP_WORDS of overlap so retrieval context is not cut at boundaries.
  3. Short text (≤ MAX_WORDS): single chunk.

Each chunk carries the full article metadata so Qdrant payloads enable
filtering by year, study_type, and evidence_level (evidence-level prioritisation
over country-specific protocols — see architecture §4).
"""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass, field

from meddx.ingestion.europe_pmc import ArticleRecord

MAX_WORDS = 400
OVERLAP_WORDS = 50

# Common structured-abstract section headers.
_SECTION_RE = re.compile(
    r"(?m)^\s*("
    r"BACKGROUND|CONTEXT|INTRODUCTION|AIMS?|OBJECTIVES?|PURPOSE|"
    r"PATIENTS?\s+AND\s+METHODS?|MATERIALS?\s+AND\s+METHODS?|METHODS?|"
    r"STUDY\s+DESIGN|"
    r"RESULTS?|FINDINGS|OUTCOMES?|"
    r"CONCLUSIONS?|SUMMARY|DISCUSSION|INTERPRETATION|IMPLICATIONS|"
    r"SIGNIFICANCE|WHAT\s+IS\s+NEW|HIGHLIGHTS?"
    r")\s*:",
    re.IGNORECASE,
)


@dataclass
class ArticleChunk:
    # Article identity
    pmid: str | None
    doi: str | None
    pmcid: str | None
    title: str
    journal: str
    year: int
    # Evidence metadata (Qdrant payload filters)
    study_type: str
    evidence_level: int
    # Chunk content
    section: str   # "abstract", "background", "methods", "results", "conclusions", "other"
    text: str      # prefixed with title for richer embedding context
    chunk_index: int

    def point_id(self) -> str:
        """Deterministic UUID — idempotent on repeated ingestion runs."""
        source = f"{self.pmid or self.doi or self.title}:{self.chunk_index}"
        return str(uuid.UUID(hashlib.md5(source.encode()).hexdigest()))

    def payload(self) -> dict:
        return {
            "pmid": self.pmid,
            "doi": self.doi,
            "pmcid": self.pmcid,
            "title": self.title,
            "journal": self.journal,
            "year": self.year,
            "study_type": self.study_type,
            "evidence_level": self.evidence_level,
            "section": self.section,
            "text": self.text,
            "chunk_index": self.chunk_index,
        }


def _normalise_section_name(header: str) -> str:
    h = header.lower().strip().rstrip(":")
    if any(k in h for k in ("background", "context", "introduction", "aim", "objective", "purpose")):
        return "background"
    if any(k in h for k in ("method", "patient", "material", "design")):
        return "methods"
    if any(k in h for k in ("result", "finding", "outcome")):
        return "results"
    if any(k in h for k in ("conclusion", "summary", "interpretation", "implication",
                            "significance", "highlight", "discussion")):
        return "conclusions"
    return "other"


def _split_structured(text: str) -> list[tuple[str, str]]:
    """Return [(section_name, section_text), …] for structured abstracts."""
    matches = list(_SECTION_RE.finditer(text))
    if not matches:
        return []

    sections = []
    for i, match in enumerate(matches):
        section_name = _normalise_section_name(match.group(1))
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append((section_name, body))
    return sections


def _sentence_windows(text: str) -> list[str]:
    """Split long text into overlapping word windows, breaking on sentences."""
    # Simple sentence split: keep delimiter with the preceding sentence.
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    windows: list[str] = []
    current_words: list[str] = []
    overlap_buffer: list[str] = []

    for sentence in sentences:
        words = sentence.split()
        current_words.extend(words)
        if len(current_words) >= MAX_WORDS:
            windows.append(" ".join(current_words))
            # Keep the last OVERLAP_WORDS as the start of the next window
            overlap_buffer = current_words[-OVERLAP_WORDS:]
            current_words = list(overlap_buffer)

    if current_words:
        windows.append(" ".join(current_words))
    return windows


def _make_chunk(
    article: ArticleRecord,
    section: str,
    raw_text: str,
    chunk_index: int,
) -> ArticleChunk:
    # Prefix with title so the dense embedding has article-level context.
    text = f"Title: {article.title}\n\n{raw_text}"
    return ArticleChunk(
        pmid=article.pmid,
        doi=article.doi,
        pmcid=article.pmcid,
        title=article.title,
        journal=article.journal,
        year=article.year,
        study_type=article.study_type,
        evidence_level=article.evidence_level,
        section=section,
        text=text,
        chunk_index=chunk_index,
    )


def chunk_article(article: ArticleRecord) -> list[ArticleChunk]:
    """Convert one ArticleRecord into one or more ArticleChunks."""
    text = article.abstract.strip()
    if not text:
        return []

    chunks: list[ArticleChunk] = []
    idx = 0

    # 1. Try structured section detection
    sections = _split_structured(text)
    if sections:
        for section_name, body in sections:
            chunks.append(_make_chunk(article, section_name, body, idx))
            idx += 1
        return chunks

    # 2. Long unstructured → sliding windows
    if len(text.split()) > MAX_WORDS:
        for window in _sentence_windows(text):
            chunks.append(_make_chunk(article, "abstract", window, idx))
            idx += 1
        return chunks

    # 3. Short text → single chunk
    chunks.append(_make_chunk(article, "abstract", text, 0))
    return chunks
