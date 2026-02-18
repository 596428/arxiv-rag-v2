"""
Tests for collection module models.
"""

import pytest
from datetime import date

from src.collection.models import Paper, PaperStatus, ParseMethod


class TestPaper:
    """Test Paper model."""

    def test_paper_creation(self):
        """Test basic paper creation."""
        paper = Paper(
            arxiv_id="2501.12345v1",
            title="Test Paper Title",
            authors=["Author One", "Author Two"],
            abstract="This is a test abstract.",
            categories=["cs.CL", "cs.LG"],
            published_date=date(2025, 1, 15),
        )

        assert paper.arxiv_id == "2501.12345v1"
        assert paper.title == "Test Paper Title"
        assert len(paper.authors) == 2
        assert paper.status == PaperStatus.PENDING

    def test_paper_status_transitions(self):
        """Test paper status values."""
        assert PaperStatus.PENDING.value == "pending"
        assert PaperStatus.COLLECTED.value == "collected"
        assert PaperStatus.PARSED.value == "parsed"
        assert PaperStatus.EMBEDDED.value == "embedded"
        assert PaperStatus.FAILED.value == "failed"

    def test_parse_method_values(self):
        """Test parse method enum values."""
        assert ParseMethod.LATEX.value == "latex"
        assert ParseMethod.MARKER.value == "marker"
        assert ParseMethod.NONE.value == "none"

    def test_paper_to_db_dict(self):
        """Test paper serialization to database format."""
        paper = Paper(
            arxiv_id="2501.12345v1",
            title="Test Paper",
            authors=["Author One"],
            abstract="Abstract text",
            categories=["cs.CL"],
            published_date=date(2025, 1, 15),
            citation_count=100,
        )

        db_dict = paper.to_db_dict()

        assert db_dict["arxiv_id"] == "2501.12345v1"
        assert db_dict["title"] == "Test Paper"
        assert db_dict["citation_count"] == 100
        assert "parse_status" in db_dict

    def test_paper_default_values(self):
        """Test paper default values."""
        paper = Paper(
            arxiv_id="2501.00001v1",
            title="Minimal Paper",
        )

        assert paper.citation_count == 0
        assert paper.download_count == 0
        assert paper.pdf_path is None
        assert paper.latex_path is None
        assert paper.status == PaperStatus.PENDING


class TestPaperStatus:
    """Test PaperStatus enum."""

    def test_status_from_string(self):
        """Test creating status from string."""
        status = PaperStatus("pending")
        assert status == PaperStatus.PENDING

    def test_invalid_status_raises(self):
        """Test invalid status raises ValueError."""
        with pytest.raises(ValueError):
            PaperStatus("invalid_status")
