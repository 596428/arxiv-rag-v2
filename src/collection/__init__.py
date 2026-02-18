"""
arXiv RAG v1 - Collection Module

Data collection from arXiv and related sources.
"""

from .models import (
    Paper,
    PaperStatus,
    ParseMethod,
    SearchQuery,
    CollectionStats,
    CollectionConfig,
    CollectionState,
)
from .arxiv_client import (
    ArxivClient,
    ALL_POSITIVE_KEYWORDS,
    PRIMARY_KEYWORDS,
    MODEL_FAMILIES_OPENSOURCE,
    MODEL_FAMILIES_BIGTECH,
    TECHNIQUE_KEYWORDS,
    REFINED_KEYWORDS,
    NEGATIVE_KEYWORDS,
    quick_search,
    generate_date_windows,
)
from .semantic_scholar import (
    SemanticScholarClient,
    get_semantic_scholar_client,
    close_client as close_semantic_scholar,
)
from .downloader import (
    Downloader,
    get_downloader,
    close_downloader,
)

__all__ = [
    # Models
    "Paper",
    "PaperStatus",
    "ParseMethod",
    "SearchQuery",
    "CollectionStats",
    "CollectionConfig",
    "CollectionState",
    # arXiv
    "ArxivClient",
    "ALL_POSITIVE_KEYWORDS",
    "PRIMARY_KEYWORDS",
    "MODEL_FAMILIES_OPENSOURCE",
    "MODEL_FAMILIES_BIGTECH",
    "TECHNIQUE_KEYWORDS",
    "REFINED_KEYWORDS",
    "NEGATIVE_KEYWORDS",
    "quick_search",
    "generate_date_windows",
    # Semantic Scholar
    "SemanticScholarClient",
    "get_semantic_scholar_client",
    "close_semantic_scholar",
    # Downloader
    "Downloader",
    "get_downloader",
    "close_downloader",
]
