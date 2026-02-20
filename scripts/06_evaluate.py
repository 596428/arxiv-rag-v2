#!/usr/bin/env python3
"""
arXiv RAG v1 - Search Quality Evaluation Script

Evaluates search quality using:
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (NDCG)
- Precision@K
- Recall@K

Usage:
    python scripts/06_evaluate.py                    # Run with default test queries
    python scripts/06_evaluate.py --queries queries.json  # Custom queries
    python scripts/06_evaluate.py --modes hybrid dense sparse  # Compare modes
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.retriever import (
    HybridRetriever,
    HybridFullRetriever,
    OpenAIRetriever,
    ColBERTRetriever,
    SearchResponse,
)
from src.rag.qdrant_retriever import (
    QdrantHybridRetriever,
)


@dataclass
class EvalQuery:
    """A single evaluation query with ground truth."""
    query: str
    relevant_papers: list[str] = field(default_factory=list)  # arxiv_ids
    relevant_chunks: list[str] = field(default_factory=list)  # chunk_ids
    category: str = ""  # Query category (e.g., "methodology", "results")
    original_relevant_count: int = 0  # Original count before filtering


@dataclass
class EvalMetrics:
    """Evaluation metrics for a single query."""
    query: str
    mrr: float = 0.0
    ndcg_at_5: float = 0.0
    ndcg_at_10: float = 0.0
    precision_at_5: float = 0.0
    precision_at_10: float = 0.0
    recall_at_10: float = 0.0
    search_time_ms: float = 0.0
    num_results: int = 0


def calculate_dcg(relevance_scores: list[float], k: int) -> float:
    """Calculate Discounted Cumulative Gain at k."""
    import math
    dcg = 0.0
    for i, rel in enumerate(relevance_scores[:k]):
        dcg += rel / math.log2(i + 2)  # i+2 because log2(1)=0
    return dcg


def calculate_ndcg(relevance_scores: list[float], k: int) -> float:
    """Calculate Normalized DCG at k."""
    dcg = calculate_dcg(relevance_scores, k)
    ideal_scores = sorted(relevance_scores, reverse=True)
    idcg = calculate_dcg(ideal_scores, k)
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_query(
    retriever: HybridRetriever,
    eval_query: EvalQuery,
    top_k: int = 10,
    mode: str = "hybrid",
    use_reranker: bool = False,
    lightweight_reranker: bool = False,
    rerank_top_k: int = 5,
) -> EvalMetrics:
    """Evaluate a single query."""
    # Perform search
    start_time = time.time()

    # Handle reranker suffix (e.g., "hybrid+rerank")
    base_mode = mode.replace("+rerank", "")
    use_reranker = use_reranker or "+rerank" in mode

    if base_mode == "hybrid":
        response = retriever.search(eval_query.query, top_k=top_k)
    elif base_mode == "dense":
        response = retriever.search_dense_only(eval_query.query, top_k=top_k)
    elif base_mode == "sparse":
        response = retriever.search_sparse_only(eval_query.query, top_k=top_k)
    elif base_mode == "openai":
        # Use OpenAI retriever for comparison
        openai_retriever = OpenAIRetriever()
        response = openai_retriever.search(eval_query.query, top_k=top_k)
    elif base_mode == "colbert":
        # Use ColBERT retriever
        colbert_retriever = ColBERTRetriever()
        import time as time_mod
        search_start = time_mod.time()
        results = colbert_retriever.search(eval_query.query, top_k=top_k)
        response = SearchResponse(
            query=eval_query.query,
            results=results,
            total_found=len(results),
            colbert_count=len(results),
            search_time_ms=(time_mod.time() - search_start) * 1000,
        )
    elif base_mode == "hybrid_full":
        # Use full hybrid retriever (dense + sparse + colbert)
        full_retriever = HybridFullRetriever()
        response = full_retriever.search(eval_query.query, top_k=top_k)
    elif base_mode == "qdrant_hybrid":
        # Use Qdrant hybrid retriever (optimized)
        qdrant_retriever = QdrantHybridRetriever()
        response = qdrant_retriever.search(eval_query.query, top_k=top_k)
    elif base_mode == "qdrant_dense":
        # Use Qdrant dense-only retriever
        qdrant_retriever = QdrantHybridRetriever()
        response = qdrant_retriever.search_dense_only(eval_query.query, top_k=top_k)
    elif base_mode == "qdrant_sparse":
        # Use Qdrant sparse-only retriever
        qdrant_retriever = QdrantHybridRetriever()
        response = qdrant_retriever.search_sparse_only(eval_query.query, top_k=top_k)
    elif base_mode == "qdrant_rerank":
        # Use Qdrant hybrid with reranker
        qdrant_retriever = QdrantHybridRetriever()
        response = qdrant_retriever.search(
            eval_query.query, top_k=top_k,
            use_reranker=True, rerank_top_k=top_k
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Apply reranking if enabled
    if use_reranker and response.results:
        from src.rag.reranker import BGEReranker, LightweightReranker

        # Free GPU memory before loading reranker (OOM prevention)
        try:
            if base_mode in ("hybrid", "dense", "sparse", "colbert", "hybrid_full"):
                retriever.unload_models()
            elif base_mode == "colbert":
                colbert_retriever.unload_models()
            elif base_mode == "hybrid_full":
                full_retriever.unload_models()

            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass  # Ignore if no GPU or already unloaded

        if lightweight_reranker:
            reranker = LightweightReranker()
        else:
            reranker = BGEReranker()

        response.results = reranker.rerank(
            eval_query.query,
            response.results,
            top_k=rerank_top_k
        )
        reranker.unload()

    search_time = (time.time() - start_time) * 1000

    # Calculate relevance scores (binary for now)
    relevance = []
    for result in response.results:
        is_relevant = (
            result.paper_id in eval_query.relevant_papers or
            result.chunk_id in eval_query.relevant_chunks
        )
        relevance.append(1.0 if is_relevant else 0.0)

    # MRR: reciprocal rank of first relevant result
    mrr = 0.0
    for i, rel in enumerate(relevance):
        if rel > 0:
            mrr = 1.0 / (i + 1)
            break

    # Precision@K
    precision_at_5 = sum(relevance[:5]) / 5 if len(relevance) >= 5 else sum(relevance) / max(len(relevance), 1)
    precision_at_10 = sum(relevance[:10]) / 10 if len(relevance) >= 10 else sum(relevance) / max(len(relevance), 1)

    # Recall@10
    total_relevant = len(eval_query.relevant_papers) + len(eval_query.relevant_chunks)
    recall_at_10 = sum(relevance[:10]) / total_relevant if total_relevant > 0 else 0.0

    # NDCG
    ndcg_at_5 = calculate_ndcg(relevance, 5)
    ndcg_at_10 = calculate_ndcg(relevance, 10)

    return EvalMetrics(
        query=eval_query.query,
        mrr=mrr,
        ndcg_at_5=ndcg_at_5,
        ndcg_at_10=ndcg_at_10,
        precision_at_5=precision_at_5,
        precision_at_10=precision_at_10,
        recall_at_10=recall_at_10,
        search_time_ms=search_time,
        num_results=len(response.results),
    )


def get_default_eval_queries() -> list[EvalQuery]:
    """Return default evaluation queries based on actually embedded papers."""
    return [
        EvalQuery(
            query="What is RLHF and how does reinforcement learning from human feedback improve language models?",
            relevant_papers=["2501.01031v3", "2501.01336v1"],
            category="alignment",
        ),
        EvalQuery(
            query="How does retrieval augmented generation (RAG) improve LLM responses?",
            relevant_papers=["2501.00879v3", "2501.01031v3"],
            category="rag",
        ),
        EvalQuery(
            query="What techniques are used for multi-tool reasoning in LLMs?",
            relevant_papers=["2501.01290v1", "2501.00830v2"],
            category="reasoning",
        ),
        EvalQuery(
            query="How do multimodal large language models process both text and images?",
            relevant_papers=["2501.00750v2", "2501.01645v3"],
            category="multimodal",
        ),
        EvalQuery(
            query="How are LLM agents used for autonomous problem solving?",
            relevant_papers=["2501.01205v1", "2501.00750v2"],
            category="agents",
        ),
        EvalQuery(
            query="What benchmarks are used to evaluate LLM capabilities?",
            relevant_papers=["2501.01243v3", "2501.01290v1"],
            category="evaluation",
        ),
        EvalQuery(
            query="How do attention mechanisms work in transformer models?",
            relevant_papers=["2501.00759v3", "2501.01073v2"],
            category="architecture",
        ),
        EvalQuery(
            query="What methods improve trustworthiness and robustness of RAG systems?",
            relevant_papers=["2501.00879v3", "2501.00888v1"],
            category="rag",
        ),
    ]


def run_evaluation(
    queries: list[EvalQuery],
    modes: list[str] = ["hybrid", "dense", "sparse"],
    top_k: int = 10,
    use_reranker: bool = False,
    lightweight_reranker: bool = False,
    rerank_top_k: int = 5,
) -> dict:
    """Run full evaluation across modes."""
    print("Initializing retriever...")
    retriever = HybridRetriever()

    results = {}

    for mode in modes:
        print(f"\n{'='*60}")
        print(f"Evaluating mode: {mode.upper()}")
        print("="*60)

        mode_metrics = []

        for i, eq in enumerate(queries):
            print(f"\n[{i+1}/{len(queries)}] {eq.query[:60]}...")

            try:
                metrics = evaluate_query(
                    retriever, eq, top_k=top_k, mode=mode,
                    use_reranker=use_reranker,
                    lightweight_reranker=lightweight_reranker,
                    rerank_top_k=rerank_top_k,
                )
                mode_metrics.append(metrics)

                print(f"  MRR: {metrics.mrr:.3f} | P@5: {metrics.precision_at_5:.3f} | "
                      f"NDCG@10: {metrics.ndcg_at_10:.3f} | Time: {metrics.search_time_ms:.0f}ms")

            except Exception as e:
                print(f"  ERROR: {e}")

        # Aggregate metrics
        if mode_metrics:
            avg_mrr = sum(m.mrr for m in mode_metrics) / len(mode_metrics)
            avg_ndcg_5 = sum(m.ndcg_at_5 for m in mode_metrics) / len(mode_metrics)
            avg_ndcg_10 = sum(m.ndcg_at_10 for m in mode_metrics) / len(mode_metrics)
            avg_p5 = sum(m.precision_at_5 for m in mode_metrics) / len(mode_metrics)
            avg_p10 = sum(m.precision_at_10 for m in mode_metrics) / len(mode_metrics)
            avg_time = sum(m.search_time_ms for m in mode_metrics) / len(mode_metrics)

            results[mode] = {
                "avg_mrr": avg_mrr,
                "avg_ndcg@5": avg_ndcg_5,
                "avg_ndcg@10": avg_ndcg_10,
                "avg_precision@5": avg_p5,
                "avg_precision@10": avg_p10,
                "avg_search_time_ms": avg_time,
                "num_queries": len(mode_metrics),
            }

            print(f"\n{mode.upper()} Summary:")
            print(f"  Avg MRR:      {avg_mrr:.3f}")
            print(f"  Avg NDCG@5:   {avg_ndcg_5:.3f}")
            print(f"  Avg NDCG@10:  {avg_ndcg_10:.3f}")
            print(f"  Avg P@5:      {avg_p5:.3f}")
            print(f"  Avg P@10:     {avg_p10:.3f}")
            print(f"  Avg Time:     {avg_time:.0f}ms")

    return results


def main():
    parser = argparse.ArgumentParser(description="Search Quality Evaluation")
    parser.add_argument(
        "--queries",
        type=str,
        help="Path to JSON file with evaluation queries",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["hybrid", "dense", "sparse"],
        help="Search modes to evaluate. Available: hybrid, dense, sparse, openai, colbert, hybrid_full, qdrant_hybrid, qdrant_dense, qdrant_sparse, qdrant_rerank. Add '+rerank' suffix for reranking (e.g., hybrid+rerank)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of results to retrieve (default: 10)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--use-reranker",
        action="store_true",
        default=False,
        help="Enable reranking (default: disabled)",
    )
    parser.add_argument(
        "--lightweight-reranker",
        action="store_true",
        default=False,
        help="Use lightweight reranker (bge-reranker-base) instead of full model",
    )
    parser.add_argument(
        "--rerank-top-k",
        type=int,
        default=5,
        help="Number of results after reranking (default: 5)",
    )

    args = parser.parse_args()

    # Load queries
    if args.queries:
        with open(args.queries) as f:
            query_data = json.load(f)
            queries = [EvalQuery(**q) for q in query_data]
    else:
        queries = get_default_eval_queries()

    print(f"Running evaluation with {len(queries)} queries")
    print(f"Modes: {args.modes}")
    print(f"Top-K: {args.top_k}")
    print(f"Reranker: {'lightweight' if args.lightweight_reranker else 'full' if args.use_reranker else 'disabled'}")

    # Run evaluation
    results = run_evaluation(
        queries,
        modes=args.modes,
        top_k=args.top_k,
        use_reranker=args.use_reranker,
        lightweight_reranker=args.lightweight_reranker,
        rerank_top_k=args.rerank_top_k,
    )

    # Print comparison
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"{'Mode':<10} {'MRR':>8} {'NDCG@10':>10} {'P@10':>8} {'Time(ms)':>10}")
    print("-"*60)
    for mode, metrics in results.items():
        print(f"{mode:<10} {metrics['avg_mrr']:>8.3f} {metrics['avg_ndcg@10']:>10.3f} "
              f"{metrics['avg_precision@10']:>8.3f} {metrics['avg_search_time_ms']:>10.0f}")

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    main()
