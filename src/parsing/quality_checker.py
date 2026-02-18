"""
arXiv RAG v1 - Text Quality Checker

Validate parsed text quality and detect issues.
"""

import re
from dataclasses import dataclass
from typing import Optional

from .models import ParsedDocument, Section, Paragraph


@dataclass
class QualityIssue:
    """A detected quality issue."""
    issue_type: str
    severity: str  # "warning" or "error"
    location: str  # e.g., "section:introduction", "paragraph:3"
    description: str
    sample: Optional[str] = None  # Sample of problematic content


@dataclass
class QualityReport:
    """Quality check report for a document."""
    arxiv_id: str
    passed: bool
    issues: list[QualityIssue]
    metrics: dict

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")

    def summary(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        return (
            f"Quality Report for {self.arxiv_id}: {status}\n"
            f"  Errors: {self.error_count}, Warnings: {self.warning_count}\n"
            f"  Metrics: {self.metrics}"
        )


# Quality thresholds
MIN_ALPHANUMERIC_RATIO = 0.5  # At least 50% alphanumeric
MAX_SPECIAL_CHAR_RUN = 10  # Max consecutive special characters
MIN_PARAGRAPH_LENGTH = 20  # Minimum characters for a paragraph
MAX_UNCONVERTED_COMMANDS = 0.05  # Max 5% unconverted LaTeX commands


def check_encoding_issues(text: str) -> list[QualityIssue]:
    """Check for encoding problems."""
    issues = []

    # Unicode replacement character
    if "\ufffd" in text:
        count = text.count("\ufffd")
        issues.append(QualityIssue(
            issue_type="encoding",
            severity="error",
            location="text",
            description=f"Found {count} Unicode replacement characters (encoding errors)",
            sample=text[max(0, text.find("\ufffd") - 20):text.find("\ufffd") + 20],
        ))

    # NULL characters
    if "\x00" in text:
        issues.append(QualityIssue(
            issue_type="encoding",
            severity="error",
            location="text",
            description="Found NULL characters in text",
        ))

    return issues


def check_alphanumeric_ratio(text: str, location: str = "text") -> list[QualityIssue]:
    """Check if text has sufficient alphanumeric content."""
    issues = []

    if not text or len(text) < 50:  # Skip short texts
        return issues

    alnum_count = sum(1 for c in text if c.isalnum())
    ratio = alnum_count / len(text)

    if ratio < MIN_ALPHANUMERIC_RATIO:
        issues.append(QualityIssue(
            issue_type="content_ratio",
            severity="warning",
            location=location,
            description=f"Low alphanumeric ratio ({ratio:.1%}), may indicate parsing issues",
            sample=text[:100] if len(text) > 100 else text,
        ))

    return issues


def check_special_char_runs(text: str, location: str = "text") -> list[QualityIssue]:
    """Check for long runs of special characters."""
    issues = []

    # Pattern for runs of non-alphanumeric, non-space characters
    pattern = r"[^\w\s]{" + str(MAX_SPECIAL_CHAR_RUN) + r",}"
    matches = re.findall(pattern, text)

    if matches:
        issues.append(QualityIssue(
            issue_type="special_chars",
            severity="warning",
            location=location,
            description=f"Found {len(matches)} long runs of special characters",
            sample=matches[0][:50] if matches else None,
        ))

    return issues


def check_unconverted_latex(text: str, location: str = "text") -> list[QualityIssue]:
    """Check for unconverted LaTeX commands."""
    issues = []

    # Count potential LaTeX commands
    command_pattern = r"\\[a-zA-Z]+(?:\{[^}]*\})?"
    commands = re.findall(command_pattern, text)

    # Filter out legitimate escaped characters
    legitimate = {r"\%", r"\&", r"\#", r"\_", r"\$", r"\{", r"\}"}
    unconverted = [c for c in commands if c not in legitimate]

    words = len(text.split())
    if words > 0 and len(unconverted) > 0:
        ratio = len(unconverted) / words
        if ratio > MAX_UNCONVERTED_COMMANDS:
            issues.append(QualityIssue(
                issue_type="unconverted_latex",
                severity="warning",
                location=location,
                description=f"Found {len(unconverted)} unconverted LaTeX commands ({ratio:.1%})",
                sample=", ".join(unconverted[:5]),
            ))

    return issues


def check_paragraph_quality(paragraph: Paragraph) -> list[QualityIssue]:
    """Check quality of a single paragraph."""
    issues = []
    text = paragraph.content

    if len(text) < MIN_PARAGRAPH_LENGTH:
        return issues  # Skip very short paragraphs

    # Run all checks
    issues.extend(check_encoding_issues(text))
    issues.extend(check_alphanumeric_ratio(text, f"paragraph:{paragraph.paragraph_id}"))
    issues.extend(check_special_char_runs(text, f"paragraph:{paragraph.paragraph_id}"))
    issues.extend(check_unconverted_latex(text, f"paragraph:{paragraph.paragraph_id}"))

    return issues


def check_section_quality(section: Section) -> list[QualityIssue]:
    """Check quality of a section."""
    issues = []

    # Check section title
    if section.title:
        title_issues = check_unconverted_latex(section.title, f"section:{section.section_id}:title")
        issues.extend(title_issues)

    # Check paragraphs
    for para in section.paragraphs:
        issues.extend(check_paragraph_quality(para))

    # Recursively check subsections
    for subsec in section.subsections:
        issues.extend(check_section_quality(subsec))

    return issues


def check_document_quality(doc: ParsedDocument) -> QualityReport:
    """
    Perform comprehensive quality check on a parsed document.

    Args:
        doc: Parsed document to check

    Returns:
        Quality report with issues and metrics
    """
    issues = []

    # Check abstract
    if doc.abstract:
        issues.extend(check_encoding_issues(doc.abstract))
        issues.extend(check_alphanumeric_ratio(doc.abstract, "abstract"))
        issues.extend(check_unconverted_latex(doc.abstract, "abstract"))

    # Check all sections
    for section in doc.sections:
        issues.extend(check_section_quality(section))

    # Calculate metrics
    total_text = doc.full_text
    metrics = {
        "total_characters": len(total_text),
        "total_words": len(total_text.split()),
        "section_count": doc.total_sections,
        "paragraph_count": doc.total_paragraphs,
        "equation_count": doc.total_equations,
        "figure_count": doc.total_figures,
        "table_count": doc.total_tables,
    }

    # Determine pass/fail
    error_count = sum(1 for i in issues if i.severity == "error")
    passed = error_count == 0

    return QualityReport(
        arxiv_id=doc.arxiv_id,
        passed=passed,
        issues=issues,
        metrics=metrics,
    )


def fix_common_issues(text: str) -> str:
    """
    Apply automatic fixes for common quality issues.

    Args:
        text: Text to fix

    Returns:
        Fixed text
    """
    result = text

    # Remove Unicode replacement characters
    result = result.replace("\ufffd", "")

    # Remove NULL characters
    result = result.replace("\x00", "")

    # Normalize whitespace
    result = re.sub(r"[ \t]+", " ", result)
    result = re.sub(r"\n{3,}", "\n\n", result)

    # Remove isolated special characters
    result = re.sub(r"(?<!\w)[^\w\s]{5,}(?!\w)", " ", result)

    return result.strip()


def is_valid_text(text: str, min_words: int = 10) -> bool:
    """
    Quick check if text is valid for processing.

    Args:
        text: Text to validate
        min_words: Minimum word count

    Returns:
        True if text appears valid
    """
    if not text:
        return False

    # Check word count
    words = text.split()
    if len(words) < min_words:
        return False

    # Check for encoding issues
    if "\ufffd" in text or "\x00" in text:
        return False

    # Check alphanumeric ratio
    alnum = sum(1 for c in text if c.isalnum())
    if alnum / len(text) < 0.3:
        return False

    return True
