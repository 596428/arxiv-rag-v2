"""
arXiv RAG v1 - Section Filter

Filter out noise sections that don't contribute to semantic search.
Sections like acknowledgments, references, author contributions are excluded.
"""

import re
from typing import Optional

from .models import Section, ParsedDocument


# Sections to exclude from chunking/embedding (case-insensitive patterns)
EXCLUDED_SECTION_PATTERNS: list[str] = [
    # References & Citations
    r"^references?$",
    r"^bibliography$",
    r"^citations?$",
    r"^works?\s+cited$",

    # Acknowledgments
    r"^acknowledgm?ents?$",
    r"^acknowledgements?$",

    # Author Information
    r"^author\s+contributions?$",
    r"^contributions?$",
    r"^author\s+information$",
    r"^authors?$",

    # Ethics & Compliance
    r"^ethics\s+statement$",
    r"^ethical\s+considerations?$",
    r"^broader\s+impact$",
    r"^societal\s+impact$",

    # Funding & Disclosure
    r"^funding$",
    r"^funding\s+statement$",
    r"^financial\s+support$",
    r"^competing\s+interests?$",
    r"^conflict\s+of\s+interest$",
    r"^declaration\s+of\s+interests?$",
    r"^disclosures?$",

    # Supplementary
    r"^supplementary\s+materials?$",
    r"^supplemental\s+materials?$",
    r"^appendix$",
    r"^appendices$",

    # Data & Code (often just links)
    r"^data\s+availability$",
    r"^code\s+availability$",
    r"^reproducibility$",
]

# Compile patterns for efficiency
_EXCLUDED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in EXCLUDED_SECTION_PATTERNS]


def is_excluded_section(title: str) -> bool:
    """
    Check if a section title should be excluded.

    Args:
        title: Section title to check

    Returns:
        True if section should be excluded
    """
    if not title:
        return False

    # Normalize: strip whitespace and numbering
    normalized = re.sub(r"^[\d.]+\s*", "", title.strip())
    normalized = re.sub(r"\s+", " ", normalized)

    for pattern in _EXCLUDED_PATTERNS:
        if pattern.match(normalized):
            return True

    return False


def filter_sections(sections: list[Section]) -> list[Section]:
    """
    Filter out excluded sections from a list.

    Args:
        sections: List of sections to filter

    Returns:
        Filtered list with excluded sections removed
    """
    filtered = []
    for section in sections:
        if not is_excluded_section(section.title):
            # Also filter subsections recursively
            if section.subsections:
                section.subsections = filter_sections(section.subsections)
            filtered.append(section)
    return filtered


def filter_document(doc: ParsedDocument) -> ParsedDocument:
    """
    Filter excluded sections from a parsed document.

    Args:
        doc: Parsed document

    Returns:
        Document with filtered sections (original is not modified)
    """
    filtered_sections = filter_sections(doc.sections)

    # Create filtered copy
    return ParsedDocument(
        arxiv_id=doc.arxiv_id,
        title=doc.title,
        abstract=doc.abstract,
        sections=filtered_sections,
        equations=doc.equations,
        figures=doc.figures,
        tables=doc.tables,
        parse_method=doc.parse_method,
        source_file=doc.source_file,
        parsed_at=doc.parsed_at,
        has_quality_issues=doc.has_quality_issues,
        quality_issues=doc.quality_issues,
    )


def get_section_stats(doc: ParsedDocument) -> dict:
    """
    Get statistics about sections before/after filtering.

    Args:
        doc: Parsed document

    Returns:
        Dictionary with section statistics
    """
    def count_sections(sections: list[Section]) -> int:
        count = len(sections)
        for s in sections:
            count += count_sections(s.subsections)
        return count

    def count_excluded(sections: list[Section]) -> int:
        count = 0
        for s in sections:
            if is_excluded_section(s.title):
                count += 1
            count += count_excluded(s.subsections)
        return count

    total = count_sections(doc.sections)
    excluded = count_excluded(doc.sections)

    return {
        "total_sections": total,
        "excluded_sections": excluded,
        "retained_sections": total - excluded,
        "exclusion_rate": excluded / max(1, total),
    }


# Section importance weights (for potential future use in ranking)
SECTION_IMPORTANCE: dict[str, float] = {
    "abstract": 1.0,
    "introduction": 0.9,
    "method": 0.85,
    "methodology": 0.85,
    "methods": 0.85,
    "approach": 0.85,
    "model": 0.85,
    "architecture": 0.85,
    "experiments": 0.8,
    "experimental setup": 0.75,
    "results": 0.85,
    "evaluation": 0.8,
    "analysis": 0.8,
    "discussion": 0.75,
    "conclusion": 0.7,
    "conclusions": 0.7,
    "related work": 0.6,
    "background": 0.65,
    "preliminaries": 0.6,
    "limitations": 0.7,
    "future work": 0.5,
}


def get_section_importance(title: str) -> float:
    """
    Get importance weight for a section.

    Args:
        title: Section title

    Returns:
        Importance weight (0.0-1.0), default 0.7
    """
    if not title:
        return 0.7

    normalized = re.sub(r"^[\d.]+\s*", "", title.strip().lower())
    normalized = re.sub(r"\s+", " ", normalized)

    return SECTION_IMPORTANCE.get(normalized, 0.7)
