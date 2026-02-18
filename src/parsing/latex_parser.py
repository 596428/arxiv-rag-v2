"""
arXiv RAG v1 - LaTeX Parser

Parse LaTeX source files to extract structured document content.
Uses pylatexenc for LaTeX parsing.
"""

import logging
import os
import re
import tarfile
import tempfile
from pathlib import Path
from typing import Optional

from pylatexenc.latex2text import LatexNodes2Text
from pylatexenc.latexwalker import (
    LatexWalker,
    LatexEnvironmentNode,
    LatexGroupNode,
    LatexMacroNode,
    LatexCharsNode,
    LatexMathNode,
)

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
from .latex_cleaner import clean_latex_text, clean_section_title, clean_equation_latex, clean_paper_title

logger = logging.getLogger(__name__)


class LatexParseError(Exception):
    """Error during LaTeX parsing."""
    pass


class LatexParser:
    """
    Parser for LaTeX source files.

    Extracts document structure, text, equations, figures, and tables.
    """

    def __init__(
        self,
        figures_dir: Optional[Path] = None,
        inline_math_min_length: int = 20,
    ):
        """
        Initialize LaTeX parser.

        Args:
            figures_dir: Directory to save extracted figures
            inline_math_min_length: Minimum character length to extract inline math.
                                   Set to 0 to disable inline math extraction.
                                   Default 20 filters out simple variables like $x$, $\\theta$.
        """
        self.figures_dir = figures_dir
        self.inline_math_min_length = inline_math_min_length
        self.l2t = LatexNodes2Text()

        # Counters for ID generation
        self._equation_counter = 0
        self._figure_counter = 0
        self._table_counter = 0
        self._section_counter = 0
        self._paragraph_counter = 0

    def _reset_counters(self):
        """Reset all counters for a new document."""
        self._equation_counter = 0
        self._figure_counter = 0
        self._table_counter = 0
        self._section_counter = 0
        self._paragraph_counter = 0

    def parse_archive(self, archive_path: Path, arxiv_id: str) -> ParsedDocument:
        """
        Parse a LaTeX archive (.tar.gz).

        Args:
            archive_path: Path to the archive
            arxiv_id: arXiv paper ID

        Returns:
            ParsedDocument
        """
        self._reset_counters()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Extract archive
            try:
                with tarfile.open(archive_path, "r:gz") as tar:
                    tar.extractall(tmpdir, filter="data")
            except Exception as e:
                raise LatexParseError(f"Failed to extract archive: {e}")

            # Find main .tex file
            tex_file = self._find_main_tex(Path(tmpdir))
            if not tex_file:
                raise LatexParseError("No main .tex file found in archive")

            # Parse the main file
            return self._parse_tex_file(tex_file, arxiv_id, str(archive_path))

    def _find_main_tex(self, directory: Path) -> Optional[Path]:
        """
        Find the main .tex file in a directory.

        Priority:
        1. File containing \\documentclass
        2. main.tex
        3. paper.tex
        4. First .tex file found
        """
        tex_files = list(directory.rglob("*.tex"))

        if not tex_files:
            return None

        # Look for documentclass
        for tex_file in tex_files:
            try:
                content = tex_file.read_text(encoding="utf-8", errors="ignore")
                if r"\documentclass" in content or r"\begin{document}" in content:
                    return tex_file
            except Exception:
                continue

        # Fallback names
        for name in ["main.tex", "paper.tex", "article.tex"]:
            for tex_file in tex_files:
                if tex_file.name.lower() == name:
                    return tex_file

        # Return first .tex file
        return tex_files[0] if tex_files else None

    def _parse_tex_file(
        self, tex_path: Path, arxiv_id: str, source_file: str
    ) -> ParsedDocument:
        """Parse a single .tex file."""
        try:
            content = tex_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            raise LatexParseError(f"Failed to read tex file: {e}")

        # Extract document body
        body = self._extract_document_body(content)
        if not body:
            body = content  # Use full content if no document environment

        # Extract title
        title = self._extract_title(content)

        # Extract abstract
        abstract = self._extract_abstract(content)

        # Parse sections
        sections = self._parse_sections(body, arxiv_id)

        # Extract equations, figures, tables
        equations = self._extract_equations(body, arxiv_id)
        figures = self._extract_figures(body, arxiv_id, tex_path.parent)
        tables = self._extract_tables(body, arxiv_id)

        doc = ParsedDocument(
            arxiv_id=arxiv_id,
            title=title,
            abstract=abstract,
            sections=sections,
            equations=equations,
            figures=figures,
            tables=tables,
            parse_method=ParseMethod.LATEX,
            source_file=source_file,
        )
        doc.update_counts()

        return doc

    def _extract_document_body(self, content: str) -> Optional[str]:
        """Extract content between \\begin{document} and \\end{document}."""
        match = re.search(
            r"\\begin\{document\}(.*?)\\end\{document\}",
            content,
            re.DOTALL,
        )
        return match.group(1) if match else None

    def _extract_title(self, content: str) -> str:
        """Extract paper title with nested brace support.

        Supports multiple title command formats:
        - Standard: \\title{}
        - ICML: \\icmltitle{}
        - NeurIPS: \\neuripstitle{}
        - ICLR: \\iclrtitle{}
        - ACL: \\acltitle{}
        - Running titles: \\icmltitlerunning{}, etc.
        """
        # Title command patterns in priority order (prefer main title over running title)
        title_patterns = [
            r"\\title\s*\{",
            r"\\icmltitle\s*\{",
            r"\\neuripstitle\s*\{",
            r"\\iclrtitle\s*\{",
            r"\\acltitle\s*\{",
            r"\\Title\s*\{",
            # Running titles as fallback
            r"\\icmltitlerunning\s*\{",
            r"\\titlerunning\s*\{",
        ]

        for pattern in title_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                # Extract content with balanced braces
                start = match.end()
                brace_count = 1
                pos = start

                while pos < len(content) and brace_count > 0:
                    if content[pos] == "{":
                        brace_count += 1
                    elif content[pos] == "}":
                        brace_count -= 1
                    pos += 1

                if brace_count == 0:
                    raw_title = content[start:pos - 1]
                    title = clean_paper_title(raw_title)
                    if title:  # Only return if non-empty
                        return title

        return ""

    def _extract_abstract(self, content: str) -> str:
        """Extract paper abstract."""
        # Try abstract environment
        match = re.search(
            r"\\begin\{abstract\}(.*?)\\end\{abstract\}",
            content,
            re.DOTALL,
        )
        if match:
            return clean_latex_text(match.group(1))

        # Try \abstract{} command
        match = re.search(r"\\abstract\{([^}]+)\}", content)
        if match:
            return clean_latex_text(match.group(1))

        return ""

    def _parse_sections(self, content: str, arxiv_id: str) -> list[Section]:
        """Parse document sections."""
        sections = []

        # Section patterns with levels
        section_patterns = [
            (r"\\section\*?\{([^}]+)\}", 1),
            (r"\\subsection\*?\{([^}]+)\}", 2),
            (r"\\subsubsection\*?\{([^}]+)\}", 3),
        ]

        # Find all section markers with positions
        markers = []
        for pattern, level in section_patterns:
            for match in re.finditer(pattern, content):
                markers.append({
                    "title": match.group(1),
                    "level": level,
                    "start": match.end(),
                    "match_start": match.start(),
                })

        # Sort by position
        markers.sort(key=lambda x: x["match_start"])

        # Extract content for each section
        for i, marker in enumerate(markers):
            # Content ends at next section or end of document
            end_pos = markers[i + 1]["match_start"] if i + 1 < len(markers) else len(content)
            section_content = content[marker["start"]:end_pos]

            self._section_counter += 1
            section_id = f"{arxiv_id}_sec_{self._section_counter}"

            # Parse paragraphs from section content
            paragraphs = self._parse_paragraphs(section_content, arxiv_id)

            section = Section(
                section_id=section_id,
                title=clean_section_title(marker["title"]),
                level=marker["level"],
                order=i + 1,
                paragraphs=paragraphs,
            )
            sections.append(section)

        return sections

    def _parse_paragraphs(self, content: str, arxiv_id: str) -> list[Paragraph]:
        """Parse paragraphs from section content."""
        paragraphs = []

        # Split by double newlines or \par
        raw_paragraphs = re.split(r"\n\s*\n|\\par\b", content)

        for i, raw in enumerate(raw_paragraphs):
            text = clean_latex_text(raw)

            # Skip empty or very short paragraphs
            if len(text.strip()) < 20:
                continue

            self._paragraph_counter += 1
            para_id = f"{arxiv_id}_para_{self._paragraph_counter}"

            paragraph = Paragraph(
                paragraph_id=para_id,
                content=text,
                content_type=ContentType.TEXT,
                order=i + 1,
            )
            paragraphs.append(paragraph)

        return paragraphs

    def _extract_equations(self, content: str, arxiv_id: str) -> list[Equation]:
        """Extract equations from content.

        Extracts:
        - Display math environments (equation, align, gather, multline, $$, \\[\\])
        - Significant inline math ($...$) if inline_math_min_length > 0
        """
        equations = []

        # Display math environments (always extracted)
        display_patterns = [
            r"\\begin\{equation\*?\}(.*?)\\end\{equation\*?\}",
            r"\\begin\{align\*?\}(.*?)\\end\{align\*?\}",
            r"\\begin\{gather\*?\}(.*?)\\end\{gather\*?\}",
            r"\\begin\{multline\*?\}(.*?)\\end\{multline\*?\}",
            r"\$\$([^$]+)\$\$",  # Display math
            r"\\\[(.+?)\\\]",     # Display math
        ]

        for pattern in display_patterns:
            for match in re.finditer(pattern, content, re.DOTALL):
                self._equation_counter += 1
                eq_id = f"{arxiv_id}_eq_{self._equation_counter}"

                latex = clean_equation_latex(match.group(1))

                # Extract label if present
                label_match = re.search(r"\\label\{([^}]+)\}", match.group(0))
                label = label_match.group(1) if label_match else None

                # Get context
                ctx_start = max(0, match.start() - 200)
                ctx_end = min(len(content), match.end() + 200)

                equation = Equation(
                    equation_id=eq_id,
                    latex=latex,
                    is_inline=False,
                    label=label,
                    context_before=content[ctx_start:match.start()].strip()[-100:],
                    context_after=content[match.end():ctx_end].strip()[:100],
                )
                equations.append(equation)

        # Inline math ($...$) - only extract significant ones
        if self.inline_math_min_length > 0:
            # Match single $ not preceded/followed by $ (avoid $$)
            inline_pattern = r"(?<!\$)\$(?!\$)([^\$\n]+)\$(?!\$)"
            for match in re.finditer(inline_pattern, content):
                latex_content = match.group(1).strip()

                # Skip short/trivial inline math
                if len(latex_content) < self.inline_math_min_length:
                    continue

                self._equation_counter += 1
                eq_id = f"{arxiv_id}_eq_{self._equation_counter}"

                latex = clean_equation_latex(latex_content)

                # Get context
                ctx_start = max(0, match.start() - 200)
                ctx_end = min(len(content), match.end() + 200)

                equation = Equation(
                    equation_id=eq_id,
                    latex=latex,
                    is_inline=True,
                    label=None,  # Inline math doesn't have labels
                    context_before=content[ctx_start:match.start()].strip()[-100:],
                    context_after=content[match.end():ctx_end].strip()[:100],
                )
                equations.append(equation)

        return equations

    def _extract_figures(
        self, content: str, arxiv_id: str, base_dir: Path
    ) -> list[Figure]:
        """Extract figures from content."""
        figures = []

        # Figure environment pattern
        pattern = r"\\begin\{figure\*?\}(.*?)\\end\{figure\*?\}"

        for match in re.finditer(pattern, content, re.DOTALL):
            fig_content = match.group(1)
            self._figure_counter += 1
            fig_id = f"{arxiv_id}_fig_{self._figure_counter}"

            # Extract caption
            caption_match = re.search(r"\\caption\{([^}]+)\}", fig_content)
            caption = clean_latex_text(caption_match.group(1)) if caption_match else None

            # Extract label
            label_match = re.search(r"\\label\{([^}]+)\}", fig_content)
            label = label_match.group(1) if label_match else None

            # Extract image path
            img_match = re.search(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", fig_content)
            image_path = None
            if img_match and self.figures_dir:
                src_path = base_dir / img_match.group(1)
                # Handle path resolution (add extensions if needed)
                for ext in ["", ".pdf", ".png", ".jpg", ".eps"]:
                    test_path = Path(str(src_path) + ext)
                    if test_path.exists():
                        # Copy to figures directory
                        dst_path = self.figures_dir / f"{fig_id}{test_path.suffix}"
                        try:
                            import shutil
                            shutil.copy(test_path, dst_path)
                            image_path = str(dst_path)
                        except Exception as e:
                            logger.warning(f"Failed to copy figure: {e}")
                        break

            figure = Figure(
                figure_id=fig_id,
                image_path=image_path,
                caption=caption,
                label=label,
                figure_number=self._figure_counter,
            )
            figures.append(figure)

        return figures

    def _extract_tables(self, content: str, arxiv_id: str) -> list[Table]:
        """Extract tables from content."""
        tables = []

        # Table environment pattern
        pattern = r"\\begin\{table\*?\}(.*?)\\end\{table\*?\}"

        for match in re.finditer(pattern, content, re.DOTALL):
            tab_content = match.group(1)
            self._table_counter += 1
            tab_id = f"{arxiv_id}_tab_{self._table_counter}"

            # Extract caption
            caption_match = re.search(r"\\caption\{([^}]+)\}", tab_content)
            caption = clean_latex_text(caption_match.group(1)) if caption_match else None

            # Extract label
            label_match = re.search(r"\\label\{([^}]+)\}", tab_content)
            label = label_match.group(1) if label_match else None

            # Convert tabular to markdown (simplified)
            table_md = self._convert_tabular_to_markdown(tab_content)

            # Count rows
            row_count = table_md.count("\n") - 1 if table_md else 0

            table = Table(
                table_id=tab_id,
                content=table_md,
                caption=caption,
                label=label,
                table_number=self._table_counter,
                row_count=row_count,
            )
            tables.append(table)

        return tables

    def _convert_tabular_to_markdown(self, content: str) -> str:
        """Convert LaTeX tabular to markdown table."""
        # Find tabular content
        match = re.search(
            r"\\begin\{tabular\}(?:\{[^}]*\})?(.*?)\\end\{tabular\}",
            content,
            re.DOTALL,
        )
        if not match:
            return ""

        tabular = match.group(1)

        # Split by \\ (row separator)
        rows = re.split(r"\\\\", tabular)

        md_rows = []
        for i, row in enumerate(rows):
            # Skip empty rows and hlines
            row = row.strip()
            if not row or row.startswith(r"\hline") or row.startswith(r"\cline"):
                continue

            # Split by & (column separator)
            cells = row.split("&")
            cells = [clean_latex_text(c.strip()) for c in cells]

            md_row = "| " + " | ".join(cells) + " |"
            md_rows.append(md_row)

            # Add header separator after first row
            if i == 0 and len(md_rows) == 1:
                sep = "| " + " | ".join(["---"] * len(cells)) + " |"
                md_rows.append(sep)

        return "\n".join(md_rows)


def parse_latex_archive(
    archive_path: Path,
    arxiv_id: str,
    figures_dir: Optional[Path] = None,
    inline_math_min_length: int = 20,
) -> ParsedDocument:
    """
    Convenience function to parse a LaTeX archive.

    Args:
        archive_path: Path to .tar.gz archive
        arxiv_id: arXiv paper ID
        figures_dir: Directory to save extracted figures
        inline_math_min_length: Minimum character length to extract inline math.
                               Set to 0 to disable inline math extraction.

    Returns:
        ParsedDocument
    """
    parser = LatexParser(
        figures_dir=figures_dir,
        inline_math_min_length=inline_math_min_length,
    )
    return parser.parse_archive(archive_path, arxiv_id)
