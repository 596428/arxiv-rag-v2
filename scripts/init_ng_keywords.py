#!/usr/bin/env python3
"""
arXiv RAG v1 - Initialize NG Keywords

Converts NGwords_gpt.txt to data/ng_keywords.json format.
This script is idempotent and can be re-run to reset keywords.

Usage:
    python scripts/init_ng_keywords.py
    python scripts/init_ng_keywords.py --input custom_ngwords.txt
"""

import argparse
import json
import re
from datetime import date
from pathlib import Path


def parse_ngwords_file(filepath: Path) -> dict[str, list[str]]:
    """
    Parse NGwords_gpt.txt into categorized keywords.

    Format expected:
    * Category Name
    keyword1
    keyword2
    ...

    * Another Category
    ...
    """
    categories = {}
    current_category = None
    current_keywords = []

    # Category name normalization map
    category_map = {
        "biomedical / bioinformatics / healthcare": "biomedical",
        "biomedical": "biomedical",
        "chemistry / material science / physics simulation": "chemistry_materials",
        "chemistry": "chemistry_materials",
        "earth science / climate / remote sensing / geoscience": "earth_science",
        "earth science": "earth_science",
        "robotics / control / signal processing": "robotics_control",
        "robotics": "robotics_control",
        "finance / economics / social science": "finance_social",
        "finance": "finance_social",
        "domain-specific computer vision": "domain_cv",
        "domain specific computer vision": "domain_cv",
        "education / psychology / cognitive science": "education_psychology",
        "education": "education_psychology",
        "low-level hardware / systems": "hardware_systems",
        "hardware": "hardware_systems",
    }

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Check for category header (starts with *)
            if line.startswith('*'):
                # Save previous category
                if current_category and current_keywords:
                    categories[current_category] = current_keywords

                # Parse new category
                category_name = line.lstrip('*').strip().lower()

                # Normalize category name
                for pattern, normalized in category_map.items():
                    if pattern in category_name:
                        current_category = normalized
                        break
                else:
                    # Use a cleaned version as fallback
                    current_category = re.sub(r'[^a-z0-9]+', '_', category_name).strip('_')

                current_keywords = []
            else:
                # It's a keyword
                keyword = line.lower().strip()
                if keyword and keyword not in current_keywords:
                    current_keywords.append(keyword)

    # Save last category
    if current_category and current_keywords:
        categories[current_category] = current_keywords

    return categories


def create_ng_keywords_json(categories: dict[str, list[str]]) -> dict:
    """Create the full ng_keywords.json structure."""
    # Flatten all keywords
    flat_keywords = []
    for keywords in categories.values():
        flat_keywords.extend(keywords)

    # Remove duplicates while preserving order
    seen = set()
    unique_keywords = []
    for kw in flat_keywords:
        if kw not in seen:
            seen.add(kw)
            unique_keywords.append(kw)

    return {
        "version": date.today().isoformat(),
        "description": "Negative keywords for filtering non-LLM papers. Source: NGwords_gpt.txt",
        "categories": categories,
        "flat_keywords": unique_keywords,
        "changelog": [
            {
                "date": date.today().isoformat(),
                "action": "initial",
                "description": f"Initialized from NGwords_gpt.txt ({len(categories)} categories, {len(unique_keywords)} keywords)"
            }
        ]
    }


def main():
    parser = argparse.ArgumentParser(description="Initialize NG keywords from NGwords_gpt.txt")
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=Path("NGwords_gpt.txt"),
        help="Input file (default: NGwords_gpt.txt)"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("data/ng_keywords.json"),
        help="Output file (default: data/ng_keywords.json)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print output without writing file"
    )

    args = parser.parse_args()

    # Resolve paths relative to project root
    project_root = Path(__file__).parent.parent
    input_path = project_root / args.input if not args.input.is_absolute() else args.input
    output_path = project_root / args.output if not args.output.is_absolute() else args.output

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1

    print(f"Reading: {input_path}")
    categories = parse_ngwords_file(input_path)

    print(f"Found {len(categories)} categories:")
    for cat, keywords in categories.items():
        print(f"  - {cat}: {len(keywords)} keywords")

    ng_keywords = create_ng_keywords_json(categories)
    total_keywords = len(ng_keywords["flat_keywords"])
    print(f"\nTotal unique keywords: {total_keywords}")

    if args.dry_run:
        print("\n--- DRY RUN OUTPUT ---")
        print(json.dumps(ng_keywords, indent=2))
    else:
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(ng_keywords, f, indent=2)

        print(f"\nWritten to: {output_path}")

    return 0


if __name__ == "__main__":
    exit(main())
