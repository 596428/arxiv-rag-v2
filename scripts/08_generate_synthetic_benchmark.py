#!/usr/bin/env python3
"""
arXiv RAG v1 - Synthetic Benchmark Generation (Gemini API)

Generates high-quality search queries from paper abstracts using Gemini API.
Each generated query has the source paper as ground truth.

Usage:
    python scripts/08_generate_synthetic_benchmark.py
    python scripts/08_generate_synthetic_benchmark.py --limit 100 --output data/eval/synthetic_queries.json
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential

from src.storage.supabase_client import get_supabase_client
from src.utils.config import settings
from src.utils.logging import get_logger

logger = get_logger("synthetic_benchmark")


QUERY_GENERATION_PROMPT = """You are generating search queries for an academic paper retrieval system.

Given this paper's title and abstract, generate 1-2 natural search queries that a researcher might use to find this paper.

Title: {title}
Abstract: {abstract}

Requirements:
- Queries should be natural questions or keyword phrases
- Focus on the paper's main contribution or methodology
- Avoid using the exact title as a query
- Each query should be 10-30 words
- Queries should be in English

Output format (JSON only, no markdown):
{{"queries": ["query1", "query2"]}}"""


class SyntheticBenchmarkGenerator:
    """Generate synthetic benchmark queries using Gemini API."""

    def __init__(self, model_name: str = None):
        self.api_key = settings.gemini_api_key
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not set in environment")

        genai.configure(api_key=self.api_key)
        self.model_name = model_name or settings.gemini_model
        self.model = genai.GenerativeModel(self.model_name)
        logger.info(f"Initialized Gemini model: {self.model_name}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def generate_queries(self, title: str, abstract: str) -> list[str]:
        """Generate search queries for a paper."""
        if not abstract or len(abstract.strip()) < 50:
            logger.warning(f"Abstract too short for: {title[:50]}...")
            return []

        prompt = QUERY_GENERATION_PROMPT.format(
            title=title,
            abstract=abstract[:2000]  # Truncate long abstracts
        )

        try:
            response = self.model.generate_content(prompt)
            text = response.text.strip()

            # Clean up response - remove markdown code blocks if present
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            text = text.strip()

            # Parse JSON response
            data = json.loads(text)
            queries = data.get("queries", [])

            # Validate queries
            valid_queries = []
            for q in queries:
                if isinstance(q, str) and 10 <= len(q) <= 300:
                    valid_queries.append(q.strip())

            return valid_queries

        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error for {title[:30]}: {e}")
            return []
        except Exception as e:
            logger.error(f"Generation failed for {title[:30]}: {e}")
            raise

    def process_papers(
        self,
        papers: list[dict],
        batch_size: int = 10,
        delay: float = 0.5,
    ) -> list[dict]:
        """Process papers and generate queries with rate limiting."""
        eval_queries = []
        total = len(papers)

        for i, paper in enumerate(papers):
            arxiv_id = paper.get("arxiv_id", "")
            title = paper.get("title", "")
            abstract = paper.get("abstract", "")
            categories = paper.get("categories", [])

            if not title or not abstract:
                logger.debug(f"Skipping paper without title/abstract: {arxiv_id}")
                continue

            try:
                queries = self.generate_queries(title, abstract)

                for query in queries:
                    eval_queries.append({
                        "query": query,
                        "relevant_papers": [arxiv_id],
                        "category": categories[0] if categories else "unknown",
                    })

                if queries:
                    logger.info(f"[{i+1}/{total}] {arxiv_id}: {len(queries)} queries")
                else:
                    logger.debug(f"[{i+1}/{total}] {arxiv_id}: no queries generated")

            except Exception as e:
                logger.error(f"[{i+1}/{total}] {arxiv_id} failed: {e}")
                continue

            # Rate limiting
            if (i + 1) % batch_size == 0:
                logger.info(f"Progress: {i+1}/{total} papers processed")
                time.sleep(delay)

        return eval_queries


def get_papers_with_abstracts(client, limit: int = 1000) -> list[dict]:
    """Fetch papers with title and abstract from Supabase."""
    try:
        result = (
            client.client.table("papers")
            .select("arxiv_id, title, abstract, categories")
            .not_.is_("abstract", "null")
            .order("citation_count", desc=True)
            .limit(limit)
            .execute()
        )
        return result.data or []
    except Exception as e:
        logger.error(f"Failed to fetch papers: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description="Generate Synthetic Benchmark")
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Maximum number of papers to process (default: 1000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/eval/synthetic_queries.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Gemini model name (default: from config)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for rate limiting (default: 10)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay between batches in seconds (default: 0.5)",
    )

    args = parser.parse_args()

    # Initialize
    logger.info("Initializing...")
    client = get_supabase_client()
    generator = SyntheticBenchmarkGenerator(model_name=args.model)

    # Fetch papers
    logger.info(f"Fetching papers (limit: {args.limit})...")
    papers = get_papers_with_abstracts(client, limit=args.limit)
    logger.info(f"Found {len(papers)} papers with abstracts")

    if not papers:
        print("No papers found with abstracts")
        return

    # Generate queries
    print(f"\n=== Synthetic Benchmark Generation ===")
    print(f"Papers: {len(papers)}")
    print(f"Model: {generator.model_name}")
    print(f"Batch size: {args.batch_size}")
    print()

    start_time = time.time()
    eval_queries = generator.process_papers(
        papers,
        batch_size=args.batch_size,
        delay=args.delay,
    )
    elapsed = time.time() - start_time

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(eval_queries, f, indent=2, ensure_ascii=False)

    # Report
    print(f"\n=== Results ===")
    print(f"Papers processed: {len(papers)}")
    print(f"Queries generated: {len(eval_queries)}")
    print(f"Avg queries/paper: {len(eval_queries)/len(papers):.2f}")
    print(f"Time: {elapsed:.1f}s")
    print(f"Output: {output_path}")

    # Category distribution
    categories = {}
    for q in eval_queries:
        cat = q.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1

    print(f"\nCategory distribution:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1])[:10]:
        print(f"  {cat}: {count}")

    return eval_queries


if __name__ == "__main__":
    main()
