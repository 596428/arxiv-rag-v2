"""
arXiv RAG v1 - Reranker

Cross-encoder reranking using BGE-reranker-v2-m3 for improved precision.
"""

import time
from typing import Optional

from ..utils.logging import get_logger
from .retriever import SearchResult

logger = get_logger("reranker")


class BGEReranker:
    """
    Cross-encoder reranker using BGE-reranker-v2-m3.

    Reranks initial retrieval results for improved precision.
    Typical usage: Top-20 → Top-5 with better relevance ordering.

    Reference: https://huggingface.co/BAAI/bge-reranker-v2-m3
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "cuda",
        use_fp16: bool = True,
        batch_size: int = 32,
    ):
        """
        Initialize BGE reranker.

        Args:
            model_name: HuggingFace model name
            device: Device to use (cuda/cpu)
            use_fp16: Use half precision for GPU
            batch_size: Batch size for reranking
        """
        self.model_name = model_name
        self.device = self._get_device(device)
        self.use_fp16 = use_fp16 and self.device == "cuda"
        self.batch_size = batch_size
        self._model = None
        self._tokenizer = None

    def _get_device(self, device: str) -> str:
        """Determine device to use."""
        try:
            import torch
            if device == "cuda" and torch.cuda.is_available():
                return "cuda"
            elif device == "mps" and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def _get_safe_batch_size(self, num_pairs: int) -> int:
        """
        Dynamically adjust batch size based on available GPU memory.

        Args:
            num_pairs: Number of query-document pairs to process

        Returns:
            Safe batch size that fits in available memory
        """
        if self.device != "cuda":
            return min(num_pairs, 16)  # CPU is memory-limited

        try:
            import torch
            # Get available GPU memory
            total_memory = torch.cuda.get_device_properties(0).total_memory
            allocated_memory = torch.cuda.memory_allocated(0)
            free_memory = total_memory - allocated_memory

            # Estimate ~100MB per batch of 32 for reranker
            # Conservative: use 50% of free memory
            safe_memory = free_memory * 0.5
            estimated_batch = int(safe_memory / (100 * 1024 * 1024)) * 8

            # Clamp between 8 and 32
            safe_batch = max(8, min(32, estimated_batch, num_pairs))

            if safe_batch < self.batch_size:
                logger.debug(f"Adjusted batch size: {self.batch_size} -> {safe_batch} (free memory: {free_memory / 1e9:.1f}GB)")

            return safe_batch
        except Exception as e:
            logger.debug(f"Could not determine safe batch size: {e}")
            return min(16, num_pairs)  # Conservative fallback

    def _switch_to_cpu(self) -> None:
        """Switch model to CPU after GPU OOM."""
        logger.warning("Switching reranker to CPU due to GPU memory constraints")

        # Unload current model
        if self._model is not None:
            del self._model
            self._model = None

        self.clear_cache()

        # Update device settings
        self.device = "cpu"
        self.use_fp16 = False

        # Model will be reloaded on next access via lazy loading

    @property
    def model(self):
        """Lazy load reranker model."""
        if self._model is None:
            logger.info(f"Loading reranker model on {self.device}...")
            start = time.time()

            try:
                from FlagEmbedding import FlagReranker

                self._model = FlagReranker(
                    self.model_name,
                    use_fp16=self.use_fp16,
                    device=self.device,
                )
                logger.info(f"Reranker loaded in {time.time() - start:.1f}s")

            except ImportError:
                raise ImportError(
                    "FlagEmbedding not installed. Run: pip install FlagEmbedding"
                )

        return self._model

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """
        Rerank search results using cross-encoder.

        Args:
            query: Original search query
            results: Initial search results to rerank
            top_k: Number of results to return after reranking

        Returns:
            Reranked results with updated scores
        """
        if not results:
            return []

        if len(results) <= top_k:
            # No need to rerank if we have fewer results than requested
            return results

        start = time.time()

        # Clear GPU cache before loading reranker
        self.clear_cache()

        # Prepare query-document pairs
        pairs = [[query, r.content] for r in results]

        # Get safe batch size
        safe_batch = self._get_safe_batch_size(len(pairs))

        def _compute_scores_with_retry(pairs_to_score: list) -> list:
            """Compute scores with OOM retry logic."""
            try:
                scores = self.model.compute_score(pairs_to_score, normalize=True)
                if isinstance(scores, float):
                    scores = [scores]
                return scores
            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "CUDA" in str(e):
                    logger.warning(f"GPU OOM during reranking, switching to CPU: {e}")
                    self._switch_to_cpu()
                    # Retry on CPU
                    scores = self.model.compute_score(pairs_to_score, normalize=True)
                    if isinstance(scores, float):
                        scores = [scores]
                    return scores
                raise

        try:
            # Process in batches if needed
            if len(pairs) > safe_batch:
                logger.debug(f"Processing {len(pairs)} pairs in batches of {safe_batch}")
                all_scores = []
                for i in range(0, len(pairs), safe_batch):
                    batch = pairs[i:i + safe_batch]
                    batch_scores = _compute_scores_with_retry(batch)
                    all_scores.extend(batch_scores)
                scores = all_scores
            else:
                scores = _compute_scores_with_retry(pairs)

            # Combine with results and sort
            scored_results = list(zip(scores, results))
            scored_results.sort(key=lambda x: x[0], reverse=True)

            # Update scores and return top-k
            reranked = []
            for score, result in scored_results[:top_k]:
                # Preserve original scores, add reranker score
                result.metadata["reranker_score"] = float(score)
                result.metadata["original_score"] = result.score
                result.score = float(score)
                reranked.append(result)

            elapsed_ms = (time.time() - start) * 1000
            logger.debug(
                f"Reranked {len(results)} -> {len(reranked)} in {elapsed_ms:.0f}ms"
            )

            return reranked

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            # Return original results on failure
            return results[:top_k]

    def compute_score(
        self,
        query: str,
        document: str,
        normalize: bool = True,
    ) -> float:
        """
        Compute relevance score for a single query-document pair.

        Args:
            query: Query text
            document: Document text
            normalize: Whether to normalize score to [0, 1]

        Returns:
            Relevance score
        """
        try:
            score = self.model.compute_score([[query, document]], normalize=normalize)
            return float(score[0]) if isinstance(score, list) else float(score)
        except Exception as e:
            logger.error(f"Score computation failed: {e}")
            return 0.0

    def batch_compute_scores(
        self,
        pairs: list[tuple[str, str]],
        normalize: bool = True,
    ) -> list[float]:
        """
        Compute relevance scores for multiple query-document pairs.

        Args:
            pairs: List of (query, document) tuples
            normalize: Whether to normalize scores

        Returns:
            List of relevance scores
        """
        if not pairs:
            return []

        try:
            formatted_pairs = [[q, d] for q, d in pairs]
            scores = self.model.compute_score(formatted_pairs, normalize=normalize)

            if isinstance(scores, float):
                return [scores]
            return [float(s) for s in scores]

        except Exception as e:
            logger.error(f"Batch scoring failed: {e}")
            return [0.0] * len(pairs)

    def clear_cache(self) -> None:
        """Clear GPU memory cache."""
        if self.device == "cuda":
            try:
                import torch
                torch.cuda.empty_cache()
                logger.debug("GPU cache cleared")
            except ImportError:
                pass

    def unload(self) -> None:
        """Unload model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
            self.clear_cache()
            logger.info("Reranker model unloaded")


class LightweightReranker:
    """
    Lightweight reranker using smaller model for faster inference.

    Uses bge-reranker-base for speed-sensitive applications.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.device = device
        self._reranker = None

    @property
    def reranker(self) -> BGEReranker:
        """Lazy load lightweight reranker."""
        if self._reranker is None:
            self._reranker = BGEReranker(
                model_name=self.model_name,
                device=self.device,
                use_fp16=True,
            )
        return self._reranker

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Rerank using lightweight model."""
        return self.reranker.rerank(query, results, top_k)


# Convenience function
def rerank_results(
    query: str,
    results: list[SearchResult],
    top_k: int = 5,
    use_lightweight: bool = False,
) -> list[SearchResult]:
    """
    Rerank search results.

    Args:
        query: Search query
        results: Results to rerank
        top_k: Number of results to return
        use_lightweight: Use faster but less accurate model

    Returns:
        Reranked results
    """
    if use_lightweight:
        reranker = LightweightReranker()
    else:
        reranker = BGEReranker()

    return reranker.rerank(query, results, top_k)
