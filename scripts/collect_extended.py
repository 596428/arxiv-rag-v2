#!/usr/bin/env python3
"""
arXiv RAG v1 - Extended Data Collection (14 Months)

Collects papers over a 14-month date range with iterative NG keyword learning.

Pipeline:
1. Category + Positive Keyword filtering (arXiv API)
2. NG Keyword auto-filtering
3. Gemini classification + NG keyword extraction
4. Human review checkpoint
5. Repeat for each month

Usage:
    python scripts/collect_extended.py                           # Start from beginning
    python scripts/collect_extended.py --resume                  # Resume from last checkpoint
    python scripts/collect_extended.py --month 2025-01           # Specific month
    python scripts/collect_extended.py --dry-run                 # Preview only
"""

import argparse
import asyncio
import json
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.collection.arxiv_client import ArxivClient
from src.collection.ng_keywords import get_ng_keywords_manager, filter_by_ng_keywords
from src.collection.models import Paper
from src.storage.supabase_client import get_supabase_client
from src.utils.logging import get_logger

logger = get_logger("collect_extended")

# Date range for collection (14 months)
START_DATE = date(2024, 1, 1)
END_DATE = date(2025, 2, 28)

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "collection"


class CollectionState:
    """Tracks collection progress across months."""

    def __init__(self, filepath: Path = None):
        self.filepath = filepath or (OUTPUT_DIR / "collection_state.json")
        self._state: Optional[dict] = None

    @property
    def state(self) -> dict:
        if self._state is None:
            self._state = self._load()
        return self._state

    def _load(self) -> dict:
        if self.filepath.exists():
            with open(self.filepath, 'r') as f:
                return json.load(f)
        return {
            "completed_months": [],
            "current_month": None,
            "total_collected": 0,
            "total_filtered": 0,
            "papers_by_month": {},
        }

    def save(self):
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(self.filepath, 'w') as f:
            json.dump(self._state, f, indent=2)

    def mark_month_complete(self, month: str, stats: dict):
        self.state["completed_months"].append(month)
        self.state["papers_by_month"][month] = stats
        self.state["total_collected"] += stats.get("collected", 0)
        self.state["total_filtered"] += stats.get("filtered", 0)
        self.state["current_month"] = None
        self.save()

    def is_month_complete(self, month: str) -> bool:
        return month in self.state.get("completed_months", [])


def generate_months(start: date, end: date) -> list[tuple[str, date, date]]:
    """Generate list of (month_key, start_date, end_date) tuples."""
    months = []
    current = start.replace(day=1)

    while current <= end:
        # Get last day of month
        if current.month == 12:
            next_month = current.replace(year=current.year + 1, month=1)
        else:
            next_month = current.replace(month=current.month + 1)
        last_day = next_month - timedelta(days=1)

        month_key = current.strftime("%Y-%m")
        months.append((month_key, current, min(last_day, end)))

        current = next_month

    return months


async def collect_month(
    month_key: str,
    start_date: date,
    end_date: date,
    dry_run: bool = False,
) -> dict:
    """
    Collect papers for a single month.

    Returns:
        Stats dict with collected, filtered, edge_cases counts
    """
    stats = {
        "month": month_key,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "stage1_count": 0,  # From arXiv API
        "stage2a_clearly_llm": 0,
        "stage2a_filtered": 0,
        "stage2a_edge_cases": 0,
        "ng_filtered": 0,
        "collected": 0,
        "filtered": 0,
    }

    logger.info(f"\n{'='*60}")
    logger.info(f"Collecting: {month_key} ({start_date} to {end_date})")
    logger.info(f"{'='*60}")

    # Stage 1: arXiv API search
    client = ArxivClient()
    all_papers = await client.search_paginated(
        start_date=start_date,
        end_date=end_date,
        max_results=10000,  # arXiv limit per window
    )

    stats["stage1_count"] = len(all_papers)
    logger.info(f"Stage 1 (arXiv API): {len(all_papers)} papers")

    if dry_run:
        logger.info("DRY RUN - stopping here")
        return stats

    # Stage 2a: Rule-based filtering
    clearly_llm, edge_cases = client.filter_stage2a(all_papers)
    stats["stage2a_clearly_llm"] = len(clearly_llm)
    stats["stage2a_edge_cases"] = len(edge_cases)
    stats["stage2a_filtered"] = len(all_papers) - len(clearly_llm) - len(edge_cases)

    logger.info(f"Stage 2a: {len(clearly_llm)} clearly LLM, {len(edge_cases)} edge cases")

    # Stage 2b: NG keyword filtering (on edge cases)
    ng_filtered, passed_edge_cases = filter_by_ng_keywords(edge_cases)
    stats["ng_filtered"] = len(ng_filtered)

    logger.info(f"NG filter: {len(ng_filtered)} filtered, {len(passed_edge_cases)} edge cases remain")

    # Combine clearly LLM + passed edge cases
    final_papers = clearly_llm + passed_edge_cases
    stats["collected"] = len(final_papers)
    stats["filtered"] = len(all_papers) - len(final_papers)

    logger.info(f"Final: {len(final_papers)} papers collected, {stats['filtered']} filtered")

    # Save to database
    if final_papers:
        supabase = get_supabase_client()
        inserted = supabase.batch_insert_papers(final_papers)
        logger.info(f"Inserted {inserted} papers to database")

    # Save edge cases for review
    if passed_edge_cases:
        edge_cases_file = OUTPUT_DIR / f"edge_cases_{month_key}.json"
        edge_cases_file.parent.mkdir(parents=True, exist_ok=True)

        edge_cases_data = [
            {
                "arxiv_id": p.arxiv_id,
                "title": p.title,
                "abstract": p.abstract[:500],
                "categories": p.categories,
            }
            for p in passed_edge_cases[:100]  # Limit for review
        ]

        with open(edge_cases_file, 'w') as f:
            json.dump(edge_cases_data, f, indent=2)

        logger.info(f"Saved {len(edge_cases_data)} edge cases for review: {edge_cases_file}")

    return stats


async def collect_all(
    resume: bool = False,
    specific_month: str = None,
    dry_run: bool = False,
):
    """
    Collect papers for all months.

    Args:
        resume: Resume from last checkpoint
        specific_month: Collect only this month (format: YYYY-MM)
        dry_run: Preview without writing
    """
    state = CollectionState()
    months = generate_months(START_DATE, END_DATE)

    logger.info(f"Collection range: {START_DATE} to {END_DATE}")
    logger.info(f"Total months: {len(months)}")

    if specific_month:
        months = [(m, s, e) for m, s, e in months if m == specific_month]
        if not months:
            logger.error(f"Month not found in range: {specific_month}")
            return

    for month_key, start_date, end_date in months:
        # Skip completed months if resuming
        if resume and state.is_month_complete(month_key):
            logger.info(f"Skipping completed month: {month_key}")
            continue

        stats = await collect_month(month_key, start_date, end_date, dry_run)

        if not dry_run:
            state.mark_month_complete(month_key, stats)

        # Progress report
        if not dry_run:
            logger.info(f"\nProgress: {len(state.state['completed_months'])}/{len(months)} months")
            logger.info(f"Total collected: {state.state['total_collected']}")
            logger.info(f"Total filtered: {state.state['total_filtered']}")

    # Final summary
    print("\n" + "=" * 60)
    print("COLLECTION SUMMARY")
    print("=" * 60)
    print(f"Months processed: {len(state.state.get('completed_months', []))}")
    print(f"Total papers collected: {state.state.get('total_collected', 0)}")
    print(f"Total papers filtered: {state.state.get('total_filtered', 0)}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Extended data collection over 14 months"
    )
    parser.add_argument(
        "--resume", "-r",
        action="store_true",
        help="Resume from last checkpoint"
    )
    parser.add_argument(
        "--month", "-m",
        type=str,
        help="Collect specific month only (format: YYYY-MM)"
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Preview without writing to database"
    )
    parser.add_argument(
        "--show-state",
        action="store_true",
        help="Show current collection state and exit"
    )

    args = parser.parse_args()

    if args.show_state:
        state = CollectionState()
        print(json.dumps(state.state, indent=2))
        return 0

    # Run collection
    asyncio.run(collect_all(
        resume=args.resume,
        specific_month=args.month,
        dry_run=args.dry_run,
    ))

    return 0


if __name__ == "__main__":
    exit(main())
