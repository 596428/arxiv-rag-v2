"""
arXiv RAG v1 - Parsing Module

Document parsing with LaTeX priority and Marker PDF fallback.
"""

from .models import (
    ParseMethod,
    ContentType,
    Equation,
    Figure,
    Table,
    Paragraph,
    Section,
    ParsedDocument,
    ParseResult,
    ParsingStats,
)

from .latex_parser import (
    LatexParser,
    LatexParseError,
    parse_latex_archive,
)

from .marker_parser import (
    MarkerParser,
    MarkerParseError,
    get_marker_parser,
    parse_pdf,
)

from .section_filter import (
    is_excluded_section,
    filter_sections,
    filter_document,
    get_section_stats,
    get_section_importance,
)

from .latex_cleaner import (
    clean_latex_text,
    clean_section_title,
    clean_paper_title,
    clean_equation_latex,
    extract_text_content,
    is_math_heavy,
)

from .quality_checker import (
    QualityIssue,
    QualityReport,
    check_document_quality,
    fix_common_issues,
    is_valid_text,
)

from .equation_processor import (
    EquationProcessor,
    AsyncEquationProcessor,
    get_equation_processor,
)

from .figure_processor import (
    FigureProcessor,
    get_figure_processor,
    extract_caption_text,
)


__all__ = [
    # Models
    "ParseMethod",
    "ContentType",
    "Equation",
    "Figure",
    "Table",
    "Paragraph",
    "Section",
    "ParsedDocument",
    "ParseResult",
    "ParsingStats",
    # LaTeX Parser
    "LatexParser",
    "LatexParseError",
    "parse_latex_archive",
    # Marker Parser
    "MarkerParser",
    "MarkerParseError",
    "get_marker_parser",
    "parse_pdf",
    # Section Filter
    "is_excluded_section",
    "filter_sections",
    "filter_document",
    "get_section_stats",
    "get_section_importance",
    # LaTeX Cleaner
    "clean_latex_text",
    "clean_section_title",
    "clean_paper_title",
    "clean_equation_latex",
    "extract_text_content",
    "is_math_heavy",
    # Quality Checker
    "QualityIssue",
    "QualityReport",
    "check_document_quality",
    "fix_common_issues",
    "is_valid_text",
    # Equation Processor
    "EquationProcessor",
    "AsyncEquationProcessor",
    "get_equation_processor",
    # Figure Processor
    "FigureProcessor",
    "get_figure_processor",
    "extract_caption_text",
]
