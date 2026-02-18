#!/usr/bin/env python3
"""
arXiv RAG v1 - OpenAI Embedding for Sample Papers

Embeds a sample of papers with OpenAI text-embedding-3-large for comparison.

Usage:
    python scripts/07_openai_embed_sample.py --sample-size 100
    python scripts/07_openai_embed_sample.py --paper-ids 2501.00879v3,2501.01031v3
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embedding.openai_embedder import OpenAIEmbedder
from src.embedding.models import EmbeddingConfig
from src.storage.supabase_client import get_supabase_client
from src.utils.logging import get_logger

logger = get_logger("openai_embed")


def get_papers_with_chunks(client, limit: int = 1000) -> list[str]:
    """Get paper IDs that have chunks."""
    paper_ids = set()
    offset = 0
    page_size = 1000

    while len(paper_ids) < limit:
        result = client.client.from_('chunks').select(
            'paper_id'
        ).range(offset, offset + page_size - 1).execute()

        if not result.data:
            break

        for row in result.data:
            paper_ids.add(row['paper_id'])

        offset += page_size

        if len(result.data) < page_size:
            break

    return list(paper_ids)


def get_chunks_for_papers(client, paper_ids: list[str]) -> list[dict]:
    """Get chunks for specified papers that don't have OpenAI embeddings."""
    chunks = []

    for paper_id in paper_ids:
        result = client.client.from_('chunks').select(
            'chunk_id, paper_id, content'
        ).eq('paper_id', paper_id).is_('embedding_openai', 'null').execute()

        if result.data:
            chunks.extend(result.data)

    return chunks


def update_chunk_openai_embedding(client, chunk_id: str, embedding: list[float]) -> bool:
    """Update a chunk with OpenAI embedding."""
    try:
        client.client.from_('chunks').update({
            'embedding_openai': embedding
        }).eq('chunk_id', chunk_id).execute()
        return True
    except Exception as e:
        logger.error(f"Failed to update chunk {chunk_id}: {e}")
        return False


def embed_chunks_openai(
    chunks: list[dict],
    client,
    batch_size: int = 100,
) -> dict:
    """Embed chunks with OpenAI and update database."""
    config = EmbeddingConfig(
        use_openai=True,
        openai_batch_size=batch_size,
    )
    embedder = OpenAIEmbedder(config)

    stats = {
        'total': len(chunks),
        'embedded': 0,
        'failed': 0,
        'cost_estimate': 0.0,
    }

    # Process in batches
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [c['content'] for c in batch]

        try:
            logger.info(f"Embedding batch {i // batch_size + 1}/{(len(chunks) + batch_size - 1) // batch_size}")
            embeddings = embedder.embed_texts(texts)

            # Update each chunk
            for j, chunk in enumerate(batch):
                success = update_chunk_openai_embedding(
                    client,
                    chunk['chunk_id'],
                    embeddings[j]
                )
                if success:
                    stats['embedded'] += 1
                else:
                    stats['failed'] += 1

            # Estimate cost: $0.00002 per 1K tokens, ~500 tokens per chunk avg
            stats['cost_estimate'] += len(batch) * 0.5 * 0.00002

            # Rate limiting
            time.sleep(0.5)

        except Exception as e:
            logger.error(f"Batch {i // batch_size} failed: {e}")
            stats['failed'] += len(batch)

    return stats


def main():
    parser = argparse.ArgumentParser(description="OpenAI Embedding for Sample Papers")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Number of papers to sample (default: 100)",
    )
    parser.add_argument(
        "--paper-ids",
        type=str,
        help="Comma-separated paper IDs to embed (overrides --sample-size)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for OpenAI API calls (default: 100)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only count chunks without embedding",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file for stats",
    )

    args = parser.parse_args()

    logger.info("Initializing...")
    client = get_supabase_client()

    # Get paper IDs
    if args.paper_ids:
        paper_ids = [p.strip() for p in args.paper_ids.split(',')]
        logger.info(f"Using specified paper IDs: {len(paper_ids)} papers")
    else:
        logger.info("Getting papers with chunks...")
        all_papers = get_papers_with_chunks(client, limit=10000)
        logger.info(f"Found {len(all_papers)} papers with chunks")

        # Random sample
        sample_size = min(args.sample_size, len(all_papers))
        paper_ids = random.sample(all_papers, sample_size)
        logger.info(f"Sampled {sample_size} papers for OpenAI embedding")

    # Get chunks
    logger.info("Getting chunks without OpenAI embeddings...")
    chunks = get_chunks_for_papers(client, paper_ids)
    logger.info(f"Found {len(chunks)} chunks to embed")

    if args.dry_run:
        print(f"\n=== DRY RUN ===")
        print(f"Papers: {len(paper_ids)}")
        print(f"Chunks to embed: {len(chunks)}")
        print(f"Estimated cost: ${len(chunks) * 0.5 * 0.00002:.4f}")
        print(f"(Based on ~500 tokens per chunk, $0.00002/1K tokens)")
        return

    if not chunks:
        print("No chunks to embed (all may already have OpenAI embeddings)")
        return

    # Embed
    print(f"\n=== OpenAI Embedding ===")
    print(f"Papers: {len(paper_ids)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Batch size: {args.batch_size}")
    print()

    start_time = time.time()
    stats = embed_chunks_openai(chunks, client, batch_size=args.batch_size)
    elapsed = time.time() - start_time

    # Report
    print(f"\n=== Results ===")
    print(f"Total chunks: {stats['total']}")
    print(f"Embedded: {stats['embedded']}")
    print(f"Failed: {stats['failed']}")
    print(f"Time: {elapsed:.1f}s")
    print(f"Estimated cost: ${stats['cost_estimate']:.4f}")

    # Save stats
    if args.output:
        stats['elapsed_seconds'] = elapsed
        stats['paper_count'] = len(paper_ids)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\nStats saved to: {output_path}")

    return stats


if __name__ == "__main__":
    main()
