"""
Tests for parsing module models.
"""

import pytest

from src.parsing.models import (
    ParsedDocument,
    ParseMethod,
    Section,
    Paragraph,
    Equation,
    Figure,
    Table,
)


class TestParsedDocument:
    """Test ParsedDocument model."""

    def test_document_creation(self):
        """Test basic document creation."""
        doc = ParsedDocument(
            arxiv_id="2501.12345v1",
            title="Test Paper Title",
            abstract="This is the abstract.",
            parse_method=ParseMethod.LATEX,
            source_file="data/latex/2501.12345v1/main.tex",
        )

        assert doc.arxiv_id == "2501.12345v1"
        assert doc.title == "Test Paper Title"
        assert len(doc.sections) == 0

    def test_document_with_sections(self):
        """Test document with sections."""
        section = Section(
            section_id="sec_1",
            title="Introduction",
            level=1,
            order=0,
            paragraphs=[
                Paragraph(
                    paragraph_id="p_1",
                    content="First paragraph content.",
                    order=0,
                )
            ],
        )

        doc = ParsedDocument(
            arxiv_id="2501.12345v1",
            title="Test Paper",
            parse_method=ParseMethod.LATEX,
            source_file="data/latex/2501.12345v1/main.tex",
            sections=[section],
        )

        assert len(doc.sections) == 1
        assert doc.sections[0].title == "Introduction"

    def test_document_update_counts(self):
        """Test document update_counts method."""
        doc = ParsedDocument(
            arxiv_id="2501.12345v1",
            title="Test Paper",
            abstract="Abstract text here.",
            parse_method=ParseMethod.LATEX,
            source_file="data/latex/2501.12345v1/main.tex",
            sections=[
                Section(
                    section_id="sec_1",
                    title="Section 1",
                    order=0,
                    paragraphs=[
                        Paragraph(paragraph_id="p1", content="Content 1", order=0),
                        Paragraph(paragraph_id="p2", content="Content 2", order=1),
                    ],
                ),
            ],
            equations=[
                Equation(equation_id="eq_1", latex="E=mc^2"),
            ],
        )

        doc.update_counts()

        assert doc.total_sections == 1
        assert doc.total_paragraphs == 2
        assert doc.total_equations == 1


class TestSection:
    """Test Section model."""

    def test_section_creation(self):
        """Test basic section creation."""
        section = Section(
            section_id="sec_1",
            title="Methods",
            level=1,
            order=0,
        )

        assert section.title == "Methods"
        assert section.level == 1
        assert len(section.paragraphs) == 0

    def test_nested_sections(self):
        """Test nested subsections."""
        subsection = Section(
            section_id="sec_1_1",
            title="Data Collection",
            level=2,
            order=0,
        )

        section = Section(
            section_id="sec_1",
            title="Methods",
            level=1,
            order=0,
            subsections=[subsection],
        )

        assert len(section.subsections) == 1
        assert section.subsections[0].level == 2


class TestEquation:
    """Test Equation model."""

    def test_equation_creation(self):
        """Test basic equation creation."""
        eq = Equation(
            equation_id="eq_1",
            latex=r"\frac{d}{dx}f(x) = f'(x)",
        )

        assert eq.equation_id == "eq_1"
        assert "frac" in eq.latex

    def test_equation_with_description(self):
        """Test equation with text description."""
        eq = Equation(
            equation_id="eq_1",
            latex="E = mc^2",
            text_description="Energy equals mass times the speed of light squared.",
        )

        assert eq.text_description is not None
        assert "energy" in eq.text_description.lower()


class TestFigure:
    """Test Figure model."""

    def test_figure_creation(self):
        """Test basic figure creation."""
        fig = Figure(
            figure_id="fig_1",
            caption="Model architecture diagram.",
        )

        assert fig.figure_id == "fig_1"
        assert "architecture" in fig.caption

    def test_figure_with_path(self):
        """Test figure with image path."""
        fig = Figure(
            figure_id="fig_1",
            caption="Results plot",
            image_path="data/figures/2501.12345v1_fig1.png",
        )

        assert fig.image_path is not None
        assert fig.image_path.endswith(".png")


class TestTable:
    """Test Table model."""

    def test_table_creation(self):
        """Test basic table creation."""
        table = Table(
            table_id="tab_1",
            caption="Benchmark results",
            content="| Model | Accuracy |\n|-------|----------|\n| GPT-4 | 95.2% |",
        )

        assert table.table_id == "tab_1"
        assert "Benchmark" in table.caption
        assert "GPT-4" in table.content
