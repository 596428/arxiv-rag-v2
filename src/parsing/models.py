"""
arXiv RAG v1 - Parsing Data Models

Parsed document structure and related data types.
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ParseMethod(str, Enum):
    """Document parsing method used."""
    LATEX = "latex"
    MARKER = "marker"
    HYBRID = "hybrid"  # LaTeX + Marker fallback for some sections


class ContentType(str, Enum):
    """Type of content block."""
    TEXT = "text"
    EQUATION = "equation"
    FIGURE = "figure"
    TABLE = "table"
    CODE = "code"
    ALGORITHM = "algorithm"


class Equation(BaseModel):
    """Extracted equation from a paper."""
    equation_id: str = Field(..., description="Unique equation ID (paper_id_eq_N)")
    latex: str = Field(..., description="Original LaTeX source")
    text_description: Optional[str] = Field(None, description="Natural language description (Gemini)")
    is_inline: bool = Field(default=False, description="Inline vs display equation")
    label: Optional[str] = Field(None, description="LaTeX label (e.g., eq:attention)")
    section_id: Optional[str] = Field(None, description="Parent section ID")
    context_before: Optional[str] = Field(None, description="Text before equation (for context)")
    context_after: Optional[str] = Field(None, description="Text after equation (for context)")


class Figure(BaseModel):
    """Extracted figure from a paper."""
    figure_id: str = Field(..., description="Unique figure ID (paper_id_fig_N)")
    image_path: Optional[str] = Field(None, description="Local path to extracted image")
    caption: Optional[str] = Field(None, description="Figure caption text")
    label: Optional[str] = Field(None, description="LaTeX label (e.g., fig:architecture)")
    section_id: Optional[str] = Field(None, description="Parent section ID")
    figure_number: Optional[int] = Field(None, description="Figure number in paper")


class Table(BaseModel):
    """Extracted table from a paper."""
    table_id: str = Field(..., description="Unique table ID (paper_id_tab_N)")
    content: str = Field(..., description="Table content (markdown format)")
    caption: Optional[str] = Field(None, description="Table caption text")
    label: Optional[str] = Field(None, description="LaTeX label (e.g., tab:results)")
    section_id: Optional[str] = Field(None, description="Parent section ID")
    table_number: Optional[int] = Field(None, description="Table number in paper")
    headers: list[str] = Field(default_factory=list, description="Column headers")
    row_count: int = Field(default=0, description="Number of data rows")


class Paragraph(BaseModel):
    """A paragraph of text within a section."""
    paragraph_id: str = Field(..., description="Unique paragraph ID")
    content: str = Field(..., description="Paragraph text content")
    content_type: ContentType = Field(default=ContentType.TEXT, description="Content type")
    order: int = Field(..., description="Order within section")

    # References to embedded elements
    equation_refs: list[str] = Field(default_factory=list, description="Referenced equation IDs")
    figure_refs: list[str] = Field(default_factory=list, description="Referenced figure IDs")
    table_refs: list[str] = Field(default_factory=list, description="Referenced table IDs")


class Section(BaseModel):
    """A section of the parsed document."""
    section_id: str = Field(..., description="Unique section ID (paper_id_sec_N)")
    title: str = Field(..., description="Section title")
    level: int = Field(default=1, ge=1, le=6, description="Section level (1=top level)")
    order: int = Field(..., description="Order in document")

    # Content
    paragraphs: list[Paragraph] = Field(default_factory=list, description="Section paragraphs")
    subsections: list["Section"] = Field(default_factory=list, description="Nested subsections")

    # Extracted elements
    equations: list[Equation] = Field(default_factory=list, description="Equations in section")
    figures: list[Figure] = Field(default_factory=list, description="Figures in section")
    tables: list[Table] = Field(default_factory=list, description="Tables in section")

    @property
    def full_text(self) -> str:
        """Get all paragraph text concatenated."""
        return "\n\n".join(p.content for p in self.paragraphs)

    @property
    def word_count(self) -> int:
        """Approximate word count."""
        return len(self.full_text.split())


# Enable forward reference for nested sections
Section.model_rebuild()


class ParsedDocument(BaseModel):
    """Complete parsed document structure."""
    arxiv_id: str = Field(..., description="arXiv paper ID")
    title: str = Field(..., description="Paper title")

    # Abstract is special - not in sections
    abstract: str = Field(default="", description="Paper abstract")

    # Document structure
    sections: list[Section] = Field(default_factory=list, description="Document sections")

    # All extracted elements (flattened for easy access)
    equations: list[Equation] = Field(default_factory=list, description="All equations")
    figures: list[Figure] = Field(default_factory=list, description="All figures")
    tables: list[Table] = Field(default_factory=list, description="All tables")

    # Parsing metadata
    parse_method: ParseMethod = Field(..., description="Method used to parse")
    source_file: str = Field(..., description="Source file path (PDF or LaTeX)")
    parsed_at: datetime = Field(default_factory=datetime.now, description="Parsing timestamp")

    # Quality metrics
    total_sections: int = Field(default=0, description="Total section count")
    total_paragraphs: int = Field(default=0, description="Total paragraph count")
    total_equations: int = Field(default=0, description="Total equation count")
    total_figures: int = Field(default=0, description="Total figure count")
    total_tables: int = Field(default=0, description="Total table count")

    # Quality flags
    has_quality_issues: bool = Field(default=False, description="Quality issues detected")
    quality_issues: list[str] = Field(default_factory=list, description="List of quality issues")

    def update_counts(self) -> None:
        """Update all count fields from actual content."""
        self.total_sections = len(self.sections)
        self.total_paragraphs = sum(len(s.paragraphs) for s in self.sections)
        self.total_equations = len(self.equations)
        self.total_figures = len(self.figures)
        self.total_tables = len(self.tables)

    @property
    def full_text(self) -> str:
        """Get all document text (abstract + sections)."""
        parts = [self.abstract] if self.abstract else []
        for section in self.sections:
            if section.title:
                parts.append(f"## {section.title}")
            parts.append(section.full_text)
        return "\n\n".join(parts)

    def to_json_file(self, path: str) -> None:
        """Save parsed document to JSON file."""
        import json
        from pathlib import Path

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(mode="json"), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_json_file(cls, path: str) -> "ParsedDocument":
        """Load parsed document from JSON file."""
        import json

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.model_validate(data)


class ParseResult(BaseModel):
    """Result of a parsing operation."""
    success: bool = Field(..., description="Whether parsing succeeded")
    arxiv_id: str = Field(..., description="Paper ID")
    document: Optional[ParsedDocument] = Field(None, description="Parsed document if successful")
    error: Optional[str] = Field(None, description="Error message if failed")
    parse_method: Optional[ParseMethod] = Field(None, description="Method attempted")
    duration_seconds: float = Field(default=0.0, description="Parsing duration")

    @classmethod
    def failure(cls, arxiv_id: str, error: str, method: ParseMethod) -> "ParseResult":
        """Create a failure result."""
        return cls(
            success=False,
            arxiv_id=arxiv_id,
            error=error,
            parse_method=method,
        )


class ParsingStats(BaseModel):
    """Statistics for the parsing process."""
    total_papers: int = 0
    latex_success: int = 0
    latex_failed: int = 0
    marker_success: int = 0
    marker_failed: int = 0

    total_sections: int = 0
    total_equations: int = 0
    total_figures: int = 0
    total_tables: int = 0

    quality_issues_count: int = 0

    def summary(self) -> str:
        """Get human-readable summary."""
        total_success = self.latex_success + self.marker_success
        total_failed = self.latex_failed + self.marker_failed
        return (
            f"Parsing Stats:\n"
            f"  Total: {self.total_papers} papers\n"
            f"  Success: {total_success} ({total_success/max(1,self.total_papers)*100:.1f}%)\n"
            f"    - LaTeX: {self.latex_success}\n"
            f"    - Marker: {self.marker_success}\n"
            f"  Failed: {total_failed}\n"
            f"  Extracted:\n"
            f"    - Sections: {self.total_sections}\n"
            f"    - Equations: {self.total_equations}\n"
            f"    - Figures: {self.total_figures}\n"
            f"    - Tables: {self.total_tables}\n"
            f"  Quality Issues: {self.quality_issues_count}"
        )
