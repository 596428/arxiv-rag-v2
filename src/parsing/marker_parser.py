"""
arXiv RAG v1 - Marker PDF Parser

Parse PDF files using Marker library (GPU-accelerated).
Used as fallback when LaTeX source is unavailable or parsing fails.
"""

import logging
import os
import re
from pathlib import Path
from typing import Optional

from .models import (
    ParsedDocument,
    ParseMethod,
    Section,
    Paragraph,
    Equation,
    Figure,
    Table,
    ContentType,
)
from .latex_cleaner import clean_latex_text

logger = logging.getLogger(__name__)


class MarkerParseError(Exception):
    """Error during Marker PDF parsing."""
    pass


class MarkerParser:
    """
    PDF parser using Marker library.

    Marker extracts structured content from PDFs including:
    - Text with section hierarchy
    - Equations (as LaTeX)
    - Tables (as markdown)
    - Figures (as images)
    """

    def __init__(
        self,
        figures_dir: Optional[Path] = None,
        device: str = "cuda",
        batch_size: int = 1,
    ):
        """
        Initialize Marker parser.

        Args:
            figures_dir: Directory to save extracted figures
            device: Device for model inference ("cuda" or "cpu")
            batch_size: Batch size for processing
        """
        self.figures_dir = figures_dir
        self.device = device
        self.batch_size = batch_size
        self._converter = None
        self._models = None

        # Counters for ID generation
        self._equation_counter = 0
        self._figure_counter = 0
        self._table_counter = 0
        self._section_counter = 0
        self._paragraph_counter = 0

    def _ensure_models_loaded(self):
        """Lazy load Marker models."""
        if self._converter is not None:
            return

        try:
            import torch
            from marker.converters.pdf import PdfConverter
            from marker.models import create_model_dict

            # Set GPU memory fraction for 4060ti
            if torch.cuda.is_available() and self.device == "cuda":
                torch.cuda.set_per_process_memory_fraction(0.8)

            logger.info("Loading Marker models...")
            self._models = create_model_dict()
            self._converter = PdfConverter(artifact_dict=self._models)
            logger.info("Marker models loaded successfully")

        except ImportError as e:
            raise MarkerParseError(
                f"Marker library not available: {e}. "
                "Install with: pip install marker-pdf"
            )
        except Exception as e:
            raise MarkerParseError(f"Failed to load Marker models: {e}")

    def _reset_counters(self):
        """Reset all counters for a new document."""
        self._equation_counter = 0
        self._figure_counter = 0
        self._table_counter = 0
        self._section_counter = 0
        self._paragraph_counter = 0

    def parse_pdf(self, pdf_path: Path, arxiv_id: str) -> ParsedDocument:
        """
        Parse a PDF file.

        Args:
            pdf_path: Path to the PDF file
            arxiv_id: arXiv paper ID

        Returns:
            ParsedDocument
        """
        self._ensure_models_loaded()
        self._reset_counters()

        if not pdf_path.exists():
            raise MarkerParseError(f"PDF file not found: {pdf_path}")

        try:
            # Run Marker conversion
            result = self._converter(str(pdf_path))

            # Extract markdown content
            markdown = result.markdown if hasattr(result, "markdown") else str(result)

            # Extract images if available
            images = {}
            if hasattr(result, "images") and result.images:
                images = result.images

            # Parse the markdown into structured document
            return self._parse_markdown(markdown, arxiv_id, str(pdf_path), images)

        except Exception as e:
            raise MarkerParseError(f"Marker conversion failed: {e}")

    def _parse_markdown(
        self,
        markdown: str,
        arxiv_id: str,
        source_file: str,
        images: dict,
    ) -> ParsedDocument:
        """Parse Marker's markdown output into structured document."""

        # Extract title (first # heading)
        title = ""
        title_match = re.search(r"^#\s+(.+)$", markdown, re.MULTILINE)
        if title_match:
            title = title_match.group(1).strip()

        # Extract abstract
        abstract = self._extract_abstract_from_markdown(markdown)

        # Parse sections
        sections = self._parse_markdown_sections(markdown, arxiv_id)

        # Extract equations
        equations = self._extract_equations_from_markdown(markdown, arxiv_id)

        # Extract and save figures
        figures = self._extract_figures_from_markdown(markdown, arxiv_id, images)

        # Extract tables
        tables = self._extract_tables_from_markdown(markdown, arxiv_id)

        doc = ParsedDocument(
            arxiv_id=arxiv_id,
            title=title,
            abstract=abstract,
            sections=sections,
            equations=equations,
            figures=figures,
            tables=tables,
            parse_method=ParseMethod.MARKER,
            source_file=source_file,
        )
        doc.update_counts()

        return doc

    def _extract_abstract_from_markdown(self, markdown: str) -> str:
        """Extract abstract from markdown."""
        # Look for "Abstract" section
        match = re.search(
            r"(?:^|\n)#+\s*Abstract\s*\n(.*?)(?=\n#|\Z)",
            markdown,
            re.IGNORECASE | re.DOTALL,
        )
        if match:
            return clean_latex_text(match.group(1).strip())

        # Look for text between title and first section
        parts = re.split(r"\n##?\s+", markdown, maxsplit=2)
        if len(parts) > 1:
            # First part after title, before first section
            potential_abstract = parts[1] if len(parts) > 2 else ""
            if potential_abstract and len(potential_abstract) > 100:
                return clean_latex_text(potential_abstract[:2000])

        return ""

    def _parse_markdown_sections(self, markdown: str, arxiv_id: str) -> list[Section]:
        """Parse markdown headings into sections."""
        sections = []

        # Split by section headings
        # Pattern: ## Heading or ### Heading
        pattern = r"(?:^|\n)(#{1,4})\s+(.+?)(?=\n)"
        matches = list(re.finditer(pattern, markdown))

        for i, match in enumerate(matches):
            level = len(match.group(1))
            title = match.group(2).strip()

            # Skip title (level 1) and abstract
            if level == 1 or title.lower() == "abstract":
                continue

            # Get section content
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
            content = markdown[start:end].strip()

            self._section_counter += 1
            section_id = f"{arxiv_id}_sec_{self._section_counter}"

            # Parse paragraphs
            paragraphs = self._parse_paragraphs_from_text(content, arxiv_id)

            section = Section(
                section_id=section_id,
                title=title,
                level=level - 1,  # Adjust level (## = level 1)
                order=self._section_counter,
                paragraphs=paragraphs,
            )
            sections.append(section)

        return sections

    def _parse_paragraphs_from_text(self, text: str, arxiv_id: str) -> list[Paragraph]:
        """Parse text into paragraphs."""
        paragraphs = []

        # Split by double newlines
        raw_paragraphs = re.split(r"\n\s*\n", text)

        for i, raw in enumerate(raw_paragraphs):
            # Skip headings, code blocks, etc.
            if raw.strip().startswith("#") or raw.strip().startswith("```"):
                continue

            content = clean_latex_text(raw)

            # Skip empty or very short
            if len(content.strip()) < 20:
                continue

            self._paragraph_counter += 1
            para_id = f"{arxiv_id}_para_{self._paragraph_counter}"

            paragraph = Paragraph(
                paragraph_id=para_id,
                content=content,
                content_type=ContentType.TEXT,
                order=i + 1,
            )
            paragraphs.append(paragraph)

        return paragraphs

    def _extract_equations_from_markdown(
        self, markdown: str, arxiv_id: str
    ) -> list[Equation]:
        """Extract equations from markdown."""
        equations = []

        # Display equations: $$ ... $$
        display_pattern = r"\$\$(.+?)\$\$"
        for match in re.finditer(display_pattern, markdown, re.DOTALL):
            self._equation_counter += 1
            eq_id = f"{arxiv_id}_eq_{self._equation_counter}"

            latex = match.group(1).strip()

            # Get context
            ctx_start = max(0, match.start() - 200)
            ctx_end = min(len(markdown), match.end() + 200)

            equation = Equation(
                equation_id=eq_id,
                latex=latex,
                is_inline=False,
                context_before=markdown[ctx_start:match.start()].strip()[-100:],
                context_after=markdown[match.end():ctx_end].strip()[:100],
            )
            equations.append(equation)

        return equations

    def _extract_figures_from_markdown(
        self,
        markdown: str,
        arxiv_id: str,
        images: dict,
    ) -> list[Figure]:
        """Extract figures from markdown and save images."""
        figures = []

        # Image pattern: ![caption](path) or just image references
        img_pattern = r"!\[([^\]]*)\]\(([^)]+)\)"

        for match in re.finditer(img_pattern, markdown):
            self._figure_counter += 1
            fig_id = f"{arxiv_id}_fig_{self._figure_counter}"

            caption = match.group(1) or None
            img_ref = match.group(2)

            # Save image if available
            image_path = None
            if self.figures_dir and img_ref in images:
                try:
                    img = images[img_ref]
                    # Determine format
                    img_format = "png"
                    dst_path = self.figures_dir / f"{fig_id}.{img_format}"
                    self.figures_dir.mkdir(parents=True, exist_ok=True)

                    if hasattr(img, "save"):
                        img.save(dst_path)
                        image_path = str(dst_path)
                except Exception as e:
                    logger.warning(f"Failed to save figure {fig_id}: {e}")

            figure = Figure(
                figure_id=fig_id,
                image_path=image_path,
                caption=caption,
                figure_number=self._figure_counter,
            )
            figures.append(figure)

        return figures

    def _extract_tables_from_markdown(
        self, markdown: str, arxiv_id: str
    ) -> list[Table]:
        """Extract tables from markdown."""
        tables = []

        # Markdown table pattern (simplified)
        # Tables have | separators and --- header rows
        table_pattern = r"(\|[^\n]+\|\n\|[-: |]+\|\n(?:\|[^\n]+\|\n?)+)"

        for match in re.finditer(table_pattern, markdown):
            self._table_counter += 1
            tab_id = f"{arxiv_id}_tab_{self._table_counter}"

            content = match.group(1).strip()

            # Parse headers
            lines = content.split("\n")
            headers = []
            if lines:
                header_line = lines[0]
                headers = [
                    cell.strip()
                    for cell in header_line.split("|")
                    if cell.strip()
                ]

            # Count data rows (excluding header and separator)
            row_count = len(lines) - 2 if len(lines) > 2 else 0

            # Look for caption (text immediately before table)
            caption_match = re.search(
                r"(?:Table|Tab\.?)\s*\d*[.:]\s*([^\n]+)\n\s*" + re.escape(content[:50]),
                markdown,
                re.IGNORECASE,
            )
            caption = caption_match.group(1) if caption_match else None

            table = Table(
                table_id=tab_id,
                content=content,
                caption=caption,
                headers=headers,
                row_count=row_count,
                table_number=self._table_counter,
            )
            tables.append(table)

        return tables


# Global parser instance (lazy loaded)
_marker_parser: Optional[MarkerParser] = None


def get_marker_parser(
    figures_dir: Optional[Path] = None,
    device: str = "cuda",
) -> MarkerParser:
    """
    Get or create Marker parser instance.

    Args:
        figures_dir: Directory to save extracted figures
        device: Device for model inference

    Returns:
        MarkerParser instance
    """
    global _marker_parser

    if _marker_parser is None:
        _marker_parser = MarkerParser(figures_dir=figures_dir, device=device)

    return _marker_parser


def parse_pdf(
    pdf_path: Path,
    arxiv_id: str,
    figures_dir: Optional[Path] = None,
    device: str = "cuda",
) -> ParsedDocument:
    """
    Convenience function to parse a PDF file.

    Args:
        pdf_path: Path to PDF file
        arxiv_id: arXiv paper ID
        figures_dir: Directory to save extracted figures
        device: Device for model inference

    Returns:
        ParsedDocument
    """
    parser = get_marker_parser(figures_dir=figures_dir, device=device)
    return parser.parse_pdf(pdf_path, arxiv_id)
