"""
arXiv RAG v1 - NG Keywords Manager

Manages negative keywords for filtering non-LLM papers.
Supports loading, saving, and updating keywords with changelog tracking.
"""

import json
from datetime import date
from pathlib import Path
from typing import Optional

from ..utils.logging import get_logger

logger = get_logger("ng_keywords")

# Default path for NG keywords
DEFAULT_NG_KEYWORDS_PATH = Path(__file__).parent.parent.parent / "data" / "ng_keywords.json"


class NGKeywordsManager:
    """
    Manager for negative (NG) keywords used to filter non-LLM papers.

    Features:
    - Categorized keywords (biomedical, chemistry, etc.)
    - Flat keyword list for efficient matching
    - Version tracking and changelog
    - Add/remove with audit trail
    """

    def __init__(self, filepath: Path = None):
        self.filepath = filepath or DEFAULT_NG_KEYWORDS_PATH
        self._data: Optional[dict] = None

    @property
    def data(self) -> dict:
        """Lazy load data from file."""
        if self._data is None:
            self._data = self._load()
        return self._data

    def _load(self) -> dict:
        """Load keywords from JSON file."""
        if not self.filepath.exists():
            logger.warning(f"NG keywords file not found: {self.filepath}")
            return self._empty_data()

        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data.get('flat_keywords', []))} NG keywords")
            return data
        except Exception as e:
            logger.error(f"Failed to load NG keywords: {e}")
            return self._empty_data()

    def _empty_data(self) -> dict:
        """Return empty data structure."""
        return {
            "version": date.today().isoformat(),
            "description": "Negative keywords for filtering non-LLM papers",
            "categories": {},
            "flat_keywords": [],
            "changelog": [],
        }

    def save(self) -> bool:
        """Save keywords to JSON file."""
        try:
            # Ensure directory exists
            self.filepath.parent.mkdir(parents=True, exist_ok=True)

            # Update version
            self._data["version"] = date.today().isoformat()

            with open(self.filepath, 'w', encoding='utf-8') as f:
                json.dump(self._data, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved NG keywords to {self.filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save NG keywords: {e}")
            return False

    @property
    def keywords(self) -> list[str]:
        """Get flat list of all keywords (lowercase)."""
        return [kw.lower() for kw in self.data.get("flat_keywords", [])]

    @property
    def categories(self) -> dict[str, list[str]]:
        """Get keywords by category."""
        return self.data.get("categories", {})

    def contains(self, text: str) -> tuple[bool, list[str]]:
        """
        Check if text contains any NG keywords.

        Args:
            text: Text to check (title + abstract)

        Returns:
            Tuple of (has_match, matched_keywords)
        """
        text_lower = text.lower()
        matched = []

        for kw in self.keywords:
            if kw in text_lower:
                matched.append(kw)

        return len(matched) > 0, matched

    def add_keywords(
        self,
        keywords: list[str],
        category: str = "uncategorized",
        reason: str = None,
    ) -> int:
        """
        Add new keywords.

        Args:
            keywords: Keywords to add
            category: Category to add them to
            reason: Reason for adding (for changelog)

        Returns:
            Number of keywords added (excluding duplicates)
        """
        existing = set(kw.lower() for kw in self.data.get("flat_keywords", []))
        new_keywords = [kw.lower().strip() for kw in keywords if kw.lower().strip() not in existing]

        if not new_keywords:
            return 0

        # Add to category
        if category not in self.data["categories"]:
            self.data["categories"][category] = []
        self.data["categories"][category].extend(new_keywords)

        # Add to flat list
        self.data["flat_keywords"].extend(new_keywords)

        # Add changelog entry
        self.data["changelog"].append({
            "date": date.today().isoformat(),
            "action": "add",
            "keywords": new_keywords,
            "category": category,
            "reason": reason,
        })

        logger.info(f"Added {len(new_keywords)} keywords to category '{category}'")
        return len(new_keywords)

    def remove_keywords(
        self,
        keywords: list[str],
        reason: str = None,
    ) -> int:
        """
        Remove keywords.

        Args:
            keywords: Keywords to remove
            reason: Reason for removal (for changelog)

        Returns:
            Number of keywords removed
        """
        to_remove = set(kw.lower().strip() for kw in keywords)
        removed = []

        # Remove from flat list
        new_flat = []
        for kw in self.data.get("flat_keywords", []):
            if kw.lower() in to_remove:
                removed.append(kw)
            else:
                new_flat.append(kw)
        self.data["flat_keywords"] = new_flat

        # Remove from categories
        for category in self.data.get("categories", {}).values():
            category[:] = [kw for kw in category if kw.lower() not in to_remove]

        if removed:
            self.data["changelog"].append({
                "date": date.today().isoformat(),
                "action": "remove",
                "keywords": removed,
                "reason": reason,
            })
            logger.info(f"Removed {len(removed)} keywords")

        return len(removed)

    def get_stats(self) -> dict:
        """Get keyword statistics."""
        categories = self.data.get("categories", {})
        return {
            "total_keywords": len(self.data.get("flat_keywords", [])),
            "categories": len(categories),
            "by_category": {cat: len(kws) for cat, kws in categories.items()},
            "version": self.data.get("version"),
            "changelog_entries": len(self.data.get("changelog", [])),
        }

    def reload(self):
        """Reload keywords from file."""
        self._data = None
        _ = self.data  # Trigger reload


# Singleton instance
_manager: Optional[NGKeywordsManager] = None


def get_ng_keywords_manager() -> NGKeywordsManager:
    """Get or create the NG keywords manager singleton."""
    global _manager
    if _manager is None:
        _manager = NGKeywordsManager()
    return _manager


def filter_by_ng_keywords(papers: list, text_key: str = None) -> tuple[list, list]:
    """
    Filter papers using NG keywords.

    Args:
        papers: List of papers (dicts or objects with title/abstract)
        text_key: If provided, use this key for text; otherwise use title+abstract

    Returns:
        Tuple of (filtered_out, passed)
    """
    manager = get_ng_keywords_manager()
    filtered_out = []
    passed = []

    for paper in papers:
        # Get text to check
        if text_key:
            text = str(paper.get(text_key, "")) if isinstance(paper, dict) else str(getattr(paper, text_key, ""))
        else:
            if isinstance(paper, dict):
                text = f"{paper.get('title', '')} {paper.get('abstract', '')}"
            else:
                text = f"{getattr(paper, 'title', '')} {getattr(paper, 'abstract', '')}"

        has_ng, matched = manager.contains(text)

        if has_ng:
            # Add matched keywords to paper for debugging
            if isinstance(paper, dict):
                paper["_ng_matched"] = matched
            filtered_out.append(paper)
        else:
            passed.append(paper)

    logger.info(f"NG filter: {len(filtered_out)} filtered, {len(passed)} passed")
    return filtered_out, passed
