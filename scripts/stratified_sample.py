#!/usr/bin/env python3
"""
arXiv RAG v1 - Stratified Sampling

Selects top papers while maintaining topic diversity.

Target distribution:
- LLM Pretraining: 20%
- Reasoning/Alignment: 20%
- Multimodal: 15%
- Retrieval/RAG: 15%
- Architecture: 10%
- Optimization: 10%
- Evaluation: 10%

Usage:
    python scripts/stratified_sample.py --input data/collection/scored_papers.json
    python scripts/stratified_sample.py --target 1000
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging import get_logger

logger = get_logger("stratified_sample")

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "collection"

# Target distribution (percentages)
TARGET_DISTRIBUTION = {
    "llm_pretraining": 0.20,
    "reasoning_alignment": 0.20,
    "multimodal": 0.15,
    "retrieval_rag": 0.15,
    "architecture": 0.10,
    "optimization": 0.10,
    "evaluation": 0.10,
}


def stratified_sample(
    papers: list[dict],
    target_count: int,
    distribution: dict[str, float] = None,
) -> list[dict]:
    """
    Select papers with stratified sampling by topic.

    Args:
        papers: Sorted papers with score_components.topic
        target_count: Target number of papers
        distribution: Target topic distribution (default: TARGET_DISTRIBUTION)

    Returns:
        Selected papers
    """
    distribution = distribution or TARGET_DISTRIBUTION

    # Group papers by topic
    by_topic = defaultdict(list)
    for paper in papers:
        topic = paper.get("score_components", {}).get("topic", "other")
        by_topic[topic].append(paper)

    # Calculate target counts per topic
    target_per_topic = {}
    total_pct = sum(distribution.values())

    for topic, pct in distribution.items():
        target_per_topic[topic] = int(target_count * pct / total_pct)

    # Handle 'other' category
    assigned = sum(target_per_topic.values())
    target_per_topic["other"] = target_count - assigned

    logger.info(f"Target per topic: {target_per_topic}")

    # Select papers
    selected = []

    for topic in list(distribution.keys()) + ["other"]:
        available = by_topic.get(topic, [])
        target = target_per_topic.get(topic, 0)

        # Take top papers from this topic (already sorted by score)
        selected.extend(available[:target])

        actual = min(len(available), target)
        if actual < target:
            logger.warning(f"Topic '{topic}': only {actual}/{target} papers available")

    # If we didn't reach target, fill from highest-scored remaining papers
    if len(selected) < target_count:
        selected_ids = {p.get("arxiv_id") for p in selected}
        remaining = [p for p in papers if p.get("arxiv_id") not in selected_ids]

        need = target_count - len(selected)
        selected.extend(remaining[:need])
        logger.info(f"Added {min(need, len(remaining))} papers to reach target")

    # Re-sort by score
    selected.sort(key=lambda x: x.get("final_score", 0), reverse=True)

    # Update ranks
    for i, paper in enumerate(selected):
        paper["final_rank"] = i + 1

    return selected


def validate_diversity(papers: list[dict]) -> dict:
    """
    Validate diversity metrics of selected papers.

    Returns:
        Validation metrics dict
    """
    from collections import Counter
    import math

    # Topic distribution
    topics = [p.get("score_components", {}).get("topic", "other") for p in papers]
    topic_counts = Counter(topics)

    # Calculate entropy
    total = len(papers)
    entropy = 0
    for count in topic_counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)

    # Temporal distribution (by month)
    months = []
    for p in papers:
        pub_date = p.get("published_date", "")
        if pub_date:
            months.append(pub_date[:7])  # YYYY-MM
    month_counts = Counter(months)

    # Temporal coefficient of variation
    if month_counts:
        values = list(month_counts.values())
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        cv = math.sqrt(variance) / mean if mean > 0 else 0
    else:
        cv = 0

    return {
        "topic_distribution": dict(topic_counts),
        "topic_entropy": round(entropy, 3),
        "temporal_cv": round(cv, 3),
        "month_distribution": dict(month_counts),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Stratified sampling of top papers"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=str(OUTPUT_DIR / "scored_papers.json"),
        help="Input JSON file with scored papers"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=str(OUTPUT_DIR / "final_papers.json"),
        help="Output JSON file"
    )
    parser.add_argument(
        "--target", "-t",
        type=int,
        default=1000,
        help="Target number of papers (default: 1000)"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run diversity validation"
    )

    args = parser.parse_args()

    # Load papers
    input_file = Path(args.input)
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return 1

    with open(input_file, 'r') as f:
        papers = json.load(f)

    logger.info(f"Loaded {len(papers)} scored papers from {input_file}")

    # Perform stratified sampling
    selected = stratified_sample(papers, args.target)

    logger.info(f"Selected {len(selected)} papers")

    # Validate diversity
    metrics = validate_diversity(selected)

    # Check validation criteria
    print("\n" + "=" * 50)
    print("DIVERSITY VALIDATION")
    print("=" * 50)

    # Topic entropy > 3.5 bits
    entropy_ok = metrics["topic_entropy"] >= 3.5
    print(f"Topic entropy:      {metrics['topic_entropy']:.3f} (target: >= 3.5) {'PASS' if entropy_ok else 'FAIL'}")

    # Temporal CV < 0.3
    cv_ok = metrics["temporal_cv"] <= 0.3
    print(f"Temporal CV:        {metrics['temporal_cv']:.3f} (target: <= 0.3) {'PASS' if cv_ok else 'FAIL'}")

    # Category coverage >= 5%
    total = len(selected)
    coverage_ok = True
    print(f"\nTopic coverage (target: >= 5% each):")
    for topic, count in sorted(metrics["topic_distribution"].items(), key=lambda x: -x[1]):
        pct = count / total * 100
        ok = pct >= 5
        if not ok:
            coverage_ok = False
        print(f"  {topic:20s}: {count:4d} ({pct:5.1f}%) {'PASS' if ok else 'FAIL'}")

    print("=" * 50)

    all_pass = entropy_ok and cv_ok and coverage_ok
    print(f"Overall validation: {'PASS' if all_pass else 'FAIL'}")
    print("=" * 50)

    # Save results
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "papers": selected,
        "metadata": {
            "total_selected": len(selected),
            "target": args.target,
            "validation": metrics,
            "validation_passed": all_pass,
        }
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Saved {len(selected)} papers to {output_file}")

    # Summary
    print(f"\nFinal output: {output_file}")
    print(f"Total papers: {len(selected)}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    exit(main())
