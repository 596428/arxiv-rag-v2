#!/usr/bin/env python3
"""
arXiv RAG v1 - Generate Valid Queries

Filters synthetic queries to only include those where at least one
relevant paper exists in the database (has embeddings).

Usage:
    python scripts/generate_valid_queries.py                    # Generate valid queries
    python scripts/generate_valid_queries.py --dry-run          # Preview only
    python scripts/generate_valid_queries.py --input custom.json # Use custom input file
    python scripts/generate_valid_queries.py --output valid.json # Custom output path
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import get_settings
from src.utils.logging import setup_logging

load_dotenv()
logger = logging.getLogger(__name__)


def get_db_connection():
    """Get direct PostgreSQL connection."""
    db_url = os.getenv('DATABASE_URL') or os.getenv('SUPABASE_DB_URL')
    if not db_url:
        raise ValueError("DATABASE_URL not set")

    conn = psycopg2.connect(db_url)
    return conn


def get_papers_in_db(conn) -> set[str]:
    """Get set of arxiv_ids that have chunks with embeddings in the database."""
    with conn.cursor() as cur:
        # Get papers that have at least one chunk with BGE-M3 embedding (primary)
        # Note: chunks.paper_id is TEXT (arxiv_id), not INTEGER
        cur.execute("""
            SELECT DISTINCT c.paper_id
            FROM chunks c
            WHERE c.embedding_dense IS NOT NULL
        """)
        papers_with_bge = {row[0] for row in cur.fetchall()}

        # Also get papers with OpenAI embeddings as alternative
        cur.execute("""
            SELECT DISTINCT c.paper_id
            FROM chunks c
            WHERE c.embedding_openai IS NOT NULL
        """)
        papers_with_openai = {row[0] for row in cur.fetchall()}

    # Union of both sets
    return papers_with_bge | papers_with_openai


def get_all_papers_in_db(conn) -> set[str]:
    """Get all paper arxiv_ids in the database (regardless of embeddings)."""
    with conn.cursor() as cur:
        # Get from chunks table (paper_id is arxiv_id)
        cur.execute("SELECT DISTINCT paper_id FROM chunks")
        return {row[0] for row in cur.fetchall()}


def filter_queries(
    queries: list[dict],
    papers_in_db: set[str],
) -> tuple[list[dict], list[dict]]:
    """
    Filter queries to only include those with at least one relevant paper in DB.

    Returns:
        (valid_queries, invalid_queries)
    """
    valid_queries = []
    invalid_queries = []

    for query in queries:
        relevant_papers = query.get("relevant_papers", [])

        # Check if any relevant paper is in DB
        papers_found = [p for p in relevant_papers if p in papers_in_db]

        if papers_found:
            # Create a copy with only the papers that are in DB
            valid_query = {
                **query,
                "relevant_papers": papers_found,
                "original_relevant_count": len(relevant_papers),
            }
            valid_queries.append(valid_query)
        else:
            invalid_queries.append({
                **query,
                "missing_papers": relevant_papers,
            })

    return valid_queries, invalid_queries


def main():
    parser = argparse.ArgumentParser(
        description="Filter queries to only those with papers in database"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/eval/synthetic_queries.json",
        help="Input queries JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/eval/valid_queries.json",
        help="Output valid queries JSON file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview only, don't save results",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--check-embeddings",
        action="store_true",
        default=True,
        help="Only include papers that have embeddings (default: True)",
    )
    parser.add_argument(
        "--no-check-embeddings",
        action="store_true",
        help="Include papers in DB regardless of embedding status",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)

    settings = get_settings()

    # Load input queries
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    with open(input_path, "r") as f:
        queries = json.load(f)

    logger.info(f"Loaded {len(queries)} queries from {input_path}")

    # Connect to database
    try:
        conn = get_db_connection()
    except ValueError as e:
        logger.error(str(e))
        print("\nERROR: DATABASE_URL not set")
        print("Set DATABASE_URL in .env file to connect to the database")
        sys.exit(1)

    # Get papers in DB
    if args.no_check_embeddings:
        papers_in_db = get_all_papers_in_db(conn)
        logger.info(f"Found {len(papers_in_db)} papers in database (any status)")
    else:
        papers_in_db = get_papers_in_db(conn)
        logger.info(f"Found {len(papers_in_db)} papers with embeddings in database")

    conn.close()

    # Get unique relevant papers from queries
    all_relevant = set()
    for q in queries:
        all_relevant.update(q.get("relevant_papers", []))
    logger.info(f"Queries reference {len(all_relevant)} unique papers")

    # Check coverage
    covered = all_relevant & papers_in_db
    missing = all_relevant - papers_in_db
    logger.info(f"Papers in DB: {len(covered)}/{len(all_relevant)} ({100*len(covered)/len(all_relevant):.1f}%)")

    # Filter queries
    valid_queries, invalid_queries = filter_queries(queries, papers_in_db)

    # Statistics
    stats = {
        "total_queries": len(queries),
        "valid_queries": len(valid_queries),
        "invalid_queries": len(invalid_queries),
        "coverage_percent": 100 * len(valid_queries) / len(queries) if queries else 0,
        "total_relevant_papers": len(all_relevant),
        "papers_in_db": len(covered),
        "papers_missing": len(missing),
        "sample_missing_papers": list(missing)[:20],
    }

    # Print summary
    print("\n" + "=" * 60)
    print("QUERY VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total queries:     {stats['total_queries']}")
    print(f"Valid queries:     {stats['valid_queries']} ({stats['coverage_percent']:.1f}%)")
    print(f"Invalid queries:   {stats['invalid_queries']}")
    print("-" * 60)
    print(f"Relevant papers:   {stats['total_relevant_papers']}")
    print(f"Papers in DB:      {stats['papers_in_db']}")
    print(f"Papers missing:    {stats['papers_missing']}")
    print("=" * 60)

    if stats['papers_missing'] > 0 and args.verbose:
        print("\nSample missing papers:")
        for p in stats['sample_missing_papers']:
            print(f"  - {p}")

    if args.dry_run:
        print("\n=== DRY RUN - No files saved ===")
        return

    # Save valid queries
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(valid_queries, f, indent=2)

    print(f"\nValid queries saved to: {output_path}")

    # Save statistics
    stats_path = output_path.with_suffix(".stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Statistics saved to: {stats_path}")

    # Optionally save invalid queries for debugging
    if invalid_queries:
        invalid_path = output_path.parent / "invalid_queries.json"
        with open(invalid_path, "w") as f:
            json.dump(invalid_queries[:100], f, indent=2)  # Limit to 100
        print(f"Invalid queries sample saved to: {invalid_path}")


if __name__ == "__main__":
    main()
