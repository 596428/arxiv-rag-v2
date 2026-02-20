#!/usr/bin/env python3
"""
arXiv RAG v1 - LLM Paper Classification

Classifies edge case papers using Gemini and extracts new NG keywords.

Pipeline:
1. Load edge cases from previous collection stage
2. Classify using gemini-3-flash-preview
3. Extract NG keywords from rejected papers
4. Output: suitable papers + new NG keywords for review

Usage:
    python scripts/classify_papers.py --month 2025-01
    python scripts/classify_papers.py --input data/collection/edge_cases_2025-01.json
    python scripts/classify_papers.py --batch-size 20  # Concurrent requests
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.collection.ng_keywords import get_ng_keywords_manager
from src.utils.logging import get_logger

logger = get_logger("classify_papers")

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "collection"

# Classification prompt
CLASSIFICATION_PROMPT = """
You are an expert in LLM/NLP research. Classify whether this paper is suitable for an LLM research dataset.

Paper Title: {title}
Abstract: {abstract}

Classification criteria:
SUITABLE papers focus on:
- Large Language Models (LLM), foundation models, pretraining
- Model architecture, transformer improvements
- Alignment, RLHF, safety, reasoning
- Multimodal models with language focus
- RAG, retrieval-augmented generation
- Prompt engineering, in-context learning
- Model compression, quantization, efficiency for LLMs

NOT SUITABLE papers focus on:
- Domain-specific applications (medical, biology, chemistry, robotics, climate, finance)
- Pure computer vision without language (image classification, object detection)
- Traditional ML without LLM connection
- Hardware, systems, networking
- Signal processing, time series without NLP
- Other non-LLM deep learning

Output format (JSON only, no markdown):
{{
  "suitable": true/false,
  "confidence": "high"/"medium"/"low",
  "reason": "Brief explanation",
  "ng_keywords": ["keyword1", "keyword2"] // Only if NOT suitable - extract 2-3 domain keywords
}}

Respond with JSON only, no other text.
"""


async def classify_paper_gemini(
    title: str,
    abstract: str,
    api_key: str,
) -> Optional[dict]:
    """
    Classify a single paper using Gemini.

    Returns:
        Classification result dict or None if failed
    """
    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")

        prompt = CLASSIFICATION_PROMPT.format(
            title=title,
            abstract=abstract[:2000],  # Truncate long abstracts
        )

        response = model.generate_content(prompt)
        text = response.text.strip()

        # Parse JSON response
        # Handle potential markdown code blocks
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        result = json.loads(text)
        return result

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse Gemini response: {e}")
        return None
    except Exception as e:
        logger.error(f"Gemini classification failed: {e}")
        return None


async def classify_batch(
    papers: list[dict],
    api_key: str,
    batch_size: int = 10,
) -> tuple[list[dict], list[dict], set[str]]:
    """
    Classify a batch of papers.

    Args:
        papers: List of paper dicts with title, abstract
        api_key: Gemini API key
        batch_size: Concurrent requests

    Returns:
        Tuple of (suitable_papers, unsuitable_papers, new_ng_keywords)
    """
    suitable = []
    unsuitable = []
    new_ng_keywords = set()

    # Process in batches with rate limiting
    for i in range(0, len(papers), batch_size):
        batch = papers[i:i + batch_size]

        tasks = [
            classify_paper_gemini(
                p.get("title", ""),
                p.get("abstract", ""),
                api_key,
            )
            for p in batch
        ]

        results = await asyncio.gather(*tasks)

        for paper, result in zip(batch, results):
            if result is None:
                # Classification failed - treat as edge case
                unsuitable.append(paper)
                continue

            paper["classification"] = result

            if result.get("suitable", False):
                suitable.append(paper)
            else:
                unsuitable.append(paper)

                # Extract NG keywords
                ng_kws = result.get("ng_keywords", [])
                for kw in ng_kws:
                    if kw and len(kw) > 2:
                        new_ng_keywords.add(kw.lower().strip())

        # Progress
        logger.info(f"Classified {min(i + batch_size, len(papers))}/{len(papers)} papers")

        # Rate limiting
        await asyncio.sleep(1)

    return suitable, unsuitable, new_ng_keywords


async def main_async(args):
    """Async main function."""
    # Get API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY not set")
        return 1

    # Load edge cases
    if args.input:
        input_file = Path(args.input)
    elif args.month:
        input_file = OUTPUT_DIR / f"edge_cases_{args.month}.json"
    else:
        logger.error("Specify --input or --month")
        return 1

    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return 1

    with open(input_file, 'r') as f:
        papers = json.load(f)

    logger.info(f"Loaded {len(papers)} edge cases from {input_file}")

    if args.dry_run:
        logger.info("DRY RUN - would classify these papers:")
        for p in papers[:5]:
            logger.info(f"  - {p.get('arxiv_id')}: {p.get('title', '')[:60]}...")
        return 0

    # Classify
    suitable, unsuitable, new_ng_keywords = await classify_batch(
        papers,
        api_key,
        batch_size=args.batch_size,
    )

    # Output results
    month = args.month or input_file.stem.replace("edge_cases_", "")

    # Save suitable papers
    suitable_file = OUTPUT_DIR / f"suitable_{month}.json"
    with open(suitable_file, 'w') as f:
        json.dump(suitable, f, indent=2)
    logger.info(f"Saved {len(suitable)} suitable papers to {suitable_file}")

    # Save unsuitable papers (for review)
    unsuitable_file = OUTPUT_DIR / f"unsuitable_{month}.json"
    with open(unsuitable_file, 'w') as f:
        json.dump(unsuitable, f, indent=2)
    logger.info(f"Saved {len(unsuitable)} unsuitable papers to {unsuitable_file}")

    # Save new NG keywords for review
    if new_ng_keywords:
        ng_file = OUTPUT_DIR / f"new_ng_keywords_{month}.json"
        with open(ng_file, 'w') as f:
            json.dump({
                "month": month,
                "keywords": sorted(new_ng_keywords),
                "count": len(new_ng_keywords),
                "status": "pending_review",
            }, f, indent=2)
        logger.info(f"Saved {len(new_ng_keywords)} new NG keywords for review: {ng_file}")

        # Show keywords
        print("\n" + "=" * 50)
        print("NEW NG KEYWORDS (pending review)")
        print("=" * 50)
        for kw in sorted(new_ng_keywords):
            print(f"  - {kw}")
        print("=" * 50)
        print(f"\nReview and add to data/ng_keywords.json using:")
        print(f"  python -c \"from src.collection.ng_keywords import *; "
              f"m = get_ng_keywords_manager(); "
              f"m.add_keywords({list(new_ng_keywords)}, 'llm_extracted'); m.save()\"")

    # Summary
    print("\n" + "=" * 50)
    print("CLASSIFICATION SUMMARY")
    print("=" * 50)
    print(f"Total edge cases:    {len(papers)}")
    print(f"Suitable (keep):     {len(suitable)}")
    print(f"Unsuitable (filter): {len(unsuitable)}")
    print(f"New NG keywords:     {len(new_ng_keywords)}")
    print("=" * 50)

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Classify edge case papers using Gemini"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="Input JSON file with edge cases"
    )
    parser.add_argument(
        "--month", "-m",
        type=str,
        help="Month to classify (format: YYYY-MM)"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=10,
        help="Concurrent classification requests (default: 10)"
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Preview without classifying"
    )

    args = parser.parse_args()

    return asyncio.run(main_async(args))


if __name__ == "__main__":
    exit(main())
