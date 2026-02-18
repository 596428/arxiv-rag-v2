"""
arXiv RAG v1 - Data Models for Collection Module

Paper metadata and related data structures.
"""

from datetime import date, datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class PaperStatus(str, Enum):
    """Paper processing status."""
    PENDING = "pending"
    COLLECTED = "collected"
    PARSED = "parsed"
    EMBEDDED = "embedded"
    FAILED = "failed"


class ParseMethod(str, Enum):
    """Document parsing method used."""
    LATEX = "latex"
    MARKER = "marker"
    NONE = "none"


class Paper(BaseModel):
    """
    arXiv paper metadata model.

    Maps to the 'papers' table in Supabase.
    """
    arxiv_id: str = Field(..., description="arXiv paper ID (e.g., '2401.12345')")
    title: str = Field(..., description="Paper title")
    authors: list[str] = Field(default_factory=list, description="List of author names")
    abstract: str = Field(default="", description="Paper abstract")
    categories: list[str] = Field(default_factory=list, description="arXiv categories")
    published_date: Optional[date] = Field(None, description="Publication date")
    updated_date: Optional[date] = Field(None, description="Last update date")

    # External metrics
    citation_count: int = Field(default=0, description="Citation count from Semantic Scholar")
    download_count: int = Field(default=0, description="Download count (if available)")

    # URLs and paths
    pdf_url: Optional[str] = Field(None, description="PDF download URL")
    latex_url: Optional[str] = Field(None, description="LaTeX source URL")
    pdf_path: Optional[str] = Field(None, description="Local PDF file path")
    latex_path: Optional[str] = Field(None, description="Local LaTeX archive path")

    # Processing status
    status: PaperStatus = Field(default=PaperStatus.PENDING, description="Processing status")
    parse_method: Optional[ParseMethod] = Field(None, description="Parsing method used")

    # Relevance filtering
    is_llm_relevant: Optional[bool] = Field(None, description="LLM relevance verification result")
    relevance_reason: Optional[str] = Field(None, description="Reason for relevance decision")

    # Timestamps
    created_at: Optional[datetime] = Field(None, description="Record creation time")
    updated_at: Optional[datetime] = Field(None, description="Last update time")

    @field_validator("arxiv_id")
    @classmethod
    def normalize_arxiv_id(cls, v: str) -> str:
        """Normalize arXiv ID format (remove 'arXiv:' prefix if present)."""
        if v.lower().startswith("arxiv:"):
            return v[6:]
        return v

    @property
    def arxiv_url(self) -> str:
        """Get the arXiv abstract page URL."""
        return f"https://arxiv.org/abs/{self.arxiv_id}"

    @property
    def default_pdf_url(self) -> str:
        """Get the default PDF download URL."""
        return f"https://arxiv.org/pdf/{self.arxiv_id}.pdf"

    @property
    def default_latex_url(self) -> str:
        """Get the default LaTeX source URL (e-print)."""
        return f"https://arxiv.org/e-print/{self.arxiv_id}"

    def to_db_dict(self) -> dict:
        """Convert to dictionary for database insertion."""
        return {
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "categories": self.categories,
            "published_date": self.published_date.isoformat() if self.published_date else None,
            "citation_count": self.citation_count,
            "download_count": self.download_count,
            "pdf_path": self.pdf_path,
            "latex_path": self.latex_path,
            "parse_status": self.status.value,
            "parse_method": self.parse_method.value if self.parse_method else None,
        }


class SearchQuery(BaseModel):
    """arXiv search query parameters."""
    keywords: list[str] = Field(default_factory=list, description="Search keywords")
    categories: list[str] = Field(
        default=["cs.CL", "cs.LG", "cs.AI", "stat.ML"],
        description="arXiv categories to search"
    )
    start_date: Optional[date] = Field(None, description="Start date filter")
    end_date: Optional[date] = Field(None, description="End date filter")
    max_results: int = Field(default=100, ge=1, le=10000, description="Maximum results to fetch")

    def build_query_string(self) -> str:
        """Build arXiv API query string."""
        parts = []

        # Category filter
        if self.categories:
            cat_query = " OR ".join(f"cat:{cat}" for cat in self.categories)
            parts.append(f"({cat_query})")

        # Keyword filter (search in title and abstract)
        if self.keywords:
            kw_query = " OR ".join(
                f'(ti:"{kw}" OR abs:"{kw}")'
                for kw in self.keywords
            )
            parts.append(f"({kw_query})")

        return " AND ".join(parts) if parts else "cat:cs.CL"


class CollectionConfig(BaseModel):
    """Configuration for date windowing collection."""
    start_date: date = Field(..., description="Start date for collection")
    end_date: date = Field(..., description="End date for collection")
    window_days: int = Field(default=14, ge=1, le=365, description="Window size in days")
    max_per_window: int = Field(default=10000, ge=100, description="Max results per window")
    page_size: int = Field(default=2000, ge=100, le=2000, description="Page size for pagination")

    @property
    def window_count(self) -> int:
        """Calculate number of windows needed."""
        from datetime import timedelta
        total_days = (self.end_date - self.start_date).days
        return (total_days + self.window_days - 1) // self.window_days


class CollectionState(BaseModel):
    """Checkpoint state for resumable collection."""
    started_at: datetime = Field(default_factory=datetime.now, description="Collection start time")
    windows_completed: list[str] = Field(default_factory=list, description="Completed window keys")
    current_window: Optional[str] = Field(None, description="Current window being processed")
    papers_collected: int = Field(default=0, description="Total papers collected so far")
    last_updated: datetime = Field(default_factory=datetime.now, description="Last update time")

    def mark_window_complete(self, window_key: str, paper_count: int) -> None:
        """Mark a window as completed."""
        if window_key not in self.windows_completed:
            self.windows_completed.append(window_key)
        self.papers_collected += paper_count
        self.current_window = None
        self.last_updated = datetime.now()

    def is_window_completed(self, window_key: str) -> bool:
        """Check if a window has been completed."""
        return window_key in self.windows_completed


class CollectionStats(BaseModel):
    """Statistics for the collection process."""
    total_fetched: int = 0
    total_filtered: int = 0
    total_downloaded: int = 0
    total_failed: int = 0

    # Stage-specific counts
    stage1_count: int = 0  # Broad recall
    stage2a_passed: int = 0  # Rule-based filter passed
    stage2b_verified: int = 0  # LLM verification passed
    stage2b_rejected: int = 0  # LLM verification rejected

    # Windowing stats
    windows_processed: int = 0
    windows_total: int = 0

    def summary(self) -> str:
        """Get a human-readable summary."""
        window_info = ""
        if self.windows_total > 0:
            window_info = f"  Windows: {self.windows_processed}/{self.windows_total}\n"
        return (
            f"Collection Stats:\n"
            f"{window_info}"
            f"  Stage 1 (Broad Recall): {self.stage1_count}\n"
            f"  Stage 2a (Rule-based): {self.stage2a_passed}\n"
            f"  Stage 2b (LLM Verified): {self.stage2b_verified}\n"
            f"  Stage 2b (Rejected): {self.stage2b_rejected}\n"
            f"  Downloaded: {self.total_downloaded}\n"
            f"  Failed: {self.total_failed}"
        )
