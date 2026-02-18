"""
arXiv RAG v1 - arXiv API Client

arXiv API wrapper with rate limiting and LLM keyword filtering.
"""

import asyncio
import re
from datetime import date, timedelta
from typing import AsyncIterator, Callable, Optional

import arxiv
from tenacity import retry, stop_after_attempt, wait_exponential

from ..utils.config import settings
from ..utils.logging import get_logger
from .models import CollectionConfig, CollectionState, Paper, PaperStatus, SearchQuery

logger = get_logger("arxiv_client")


# =============================================================================
# LLM Keyword Lists (from PLAN.md v2.2)
# =============================================================================

PRIMARY_KEYWORDS = [
    "Large Language Model",
    "LLM",
    "Language Model",
    "Neural Language Model",
    "RLHF",
    "Reinforcement Learning from Human Feedback",
    "Fine-tuning",
    "Instruction Tuning",
    "Pre-trained Language Model",
    "PLM",
]

MODEL_FAMILIES_OPENSOURCE = [
    "LLaMA",
    "Llama 2",
    "Llama 3",
    "Qwen",
    "Qwen2",
    "Mistral",
    "Mixtral",
    "DeepSeek",
    "DeepSeek-V2",
    "Falcon",
    "Yi",
    "Phi",
    "Gemma",
    "OLMo",
]

MODEL_FAMILIES_BIGTECH = [
    "GPT",
    "GPT-4",
    "ChatGPT",
    "Gemini",
    "PaLM",
    "Bard",
    "Claude",
    "Constitutional AI",
]

TECHNIQUE_KEYWORDS = [
    "Prompt Engineering",
    "Prompting",
    "RAG",
    "Retrieval-Augmented Generation",
    "PEFT",
    "LoRA",
    "QLoRA",
    "Adapter",
    "Quantization",
    "Pruning",
    "Chain-of-Thought",
    "CoT",
    "Reasoning",
    "In-Context Learning",
    "ICL",
    "Alignment",
    "Safety",
    "Scaling Law",
    "Mixture of Experts",
    "MoE",
]

REFINED_KEYWORDS = [
    "Self-Attention",
    "LLM Agent",
    "Language Agent",
    "Text Embedding",
    "Vision-Language Model",
    "VLM",
    "Tool Use",
    "Function Calling",
]

# =============================================================================
# Negative Keywords for Stage 2a Filtering
# Based on analysis of Stage 2b rejected papers (166 papers analyzed)
# These keywords indicate NON-LLM papers that should be filtered out
# =============================================================================

NEGATIVE_KEYWORDS = [
    # Original keywords
    "Robot",
    "Robotics",
    "Game",
    "Gaming",
    "Image Classification",
    "Object Detection",
    "Speech Recognition",

    # Medical Imaging (25+ rejected papers)
    "MRI",
    "CT scan",
    "Ultrasound",
    "Brain Tumor Segmentation",
    "Retinal Imaging",
    "Medical Image Segmentation",
    "Radiology",
    "Cardiac Segmentation",
    "Glioma Segmentation",

    # Signal Processing (15+ rejected papers)
    "EEG",
    "ECG",
    "Radar Signal",
    "Speech Emotion Recognition",
    "Audio Source Separation",
    "Sleep Staging",
    "Vortex-Induced Vibration",

    # Computer Vision - Non-NLP (18+ rejected papers)
    "LiDAR",
    "Point Cloud",
    "Depth Estimation",
    "3D Reconstruction",
    "Pose Estimation",
    "Image Segmentation",
    "Semantic Segmentation",

    # Robotics/Autonomous Systems (12+ rejected papers)
    "UAV",
    "Unmanned Aerial Vehicle",
    "Trajectory Prediction",
    "Autonomous Vehicle",
    "Path Planning",
    "Multi-Agent Pathfinding",
    "Motion Planning",

    # Pure Reinforcement Learning (20+ rejected papers)
    "Policy Learning",
    "Markov Decision Process",
    "MDP",
    "Reward Shaping",
    "Q-Learning",
    "Multi-Agent Reinforcement Learning",
    "MARL",

    # Graph Neural Networks - Non-NLP (10+ rejected papers)
    "Node Classification",
    "Community Detection",
    "Graph Clustering",

    # Time Series/Forecasting (10+ rejected papers)
    "Time Series Forecasting",
    "Traffic Prediction",
    "Sensor Data",
    "Precipitation Forecasting",
    "Weather Forecasting",

    # Domain-Specific (15+ rejected papers)
    "Archaeology",
    "Aviation Safety",
    "Molecular Generation",
    "Chemical Structure",
    "Marine Riser",
    "Tyre Energy",
    "Formula One",

    # Traditional ML/Pre-LLM NLP (8+ rejected papers)
    "Latent Dirichlet Allocation",
    "LDA Topic Model",
    "Lemmatization",
    "Word Sense Disambiguation",
    "Hierarchical Clustering",

    # Other Non-LLM Domains
    "Federated Learning",  # Often not LLM-related
    "Outlier Detection",
    "Anomaly Detection",
    "Malware Detection",

    # From edge case analysis (added 2025-02)
    "Traffic Signal Control",
    "Maritime",
    "Ship Identification",
    "Furnace",
    "Vehicle Detection",
]

# Keywords that need context check (not outright rejected)
# These are ambiguous and may need LLM verification
CONTEXT_DEPENDENT_KEYWORDS = [
    "GNN",  # Could be for NLP tasks
    "Graph Neural Network",  # Could be for knowledge graphs in LLM
    "Contrastive Learning",  # Could be for text representation
    "Self-Supervised Learning",  # Could be for language models
]

# Combine all positive keywords for Stage 1
ALL_POSITIVE_KEYWORDS = (
    PRIMARY_KEYWORDS
    + MODEL_FAMILIES_OPENSOURCE
    + MODEL_FAMILIES_BIGTECH
    + TECHNIQUE_KEYWORDS
    + REFINED_KEYWORDS
)

# =============================================================================
# Strong LLM Indicators for Stage 2a Rule-based Filtering
# Based on analysis of Stage 2b verified papers (21 papers analyzed)
# Papers with these keywords are classified as "clearly LLM-related"
# =============================================================================

STRONG_LLM_INDICATORS = [
    # Original indicators
    "language model",
    "llm",
    "large language",
    "gpt",
    "chatgpt",
    "llama",
    "instruction tuning",
    "rlhf",
    "prompt",
    "in-context learning",

    # From verified papers analysis
    "transformer architecture",
    "self-attention mechanism",
    "mixture-of-experts",
    "mixture of experts",
    "moe model",
    "model compression",
    "knowledge distillation",
    "model pruning",
    "ai-generated text",
    "machine-generated text",
    "text detection",
    "retrieval-augmented generation",
    "retrieval augmented",
    "rag",
    "foundation model",
    "pre-trained model",
    "pretrained model",
    "fine-tuning",
    "fine tuning",
    "model merging",
    "task vector",
    "task arithmetic",
    "test-time compute",
    "generative ai",
    "genai",
    "text generation",
    "natural language generation",
    "nlg",
    "bert",
    "t5",
    "bart",

    # From edge case analysis (added 2025-02)
    "slot and intent detection",
    "natural language understanding",
    "nlu",
    "text classification",
    "sentiment analysis",
    "named entity recognition",
    "ner",
    "question answering",
    "machine translation",
    "neural machine translation",
    "nmt",
]


class ArxivClient:
    """
    arXiv API client with rate limiting and keyword filtering.

    Implements the 2-stage filtering strategy from PLAN.md:
    - Stage 1: Broad Recall (arXiv API query with keywords)
    - Stage 2a: Rule-based filtering (strong LLM indicators)
    """

    def __init__(
        self,
        request_interval: float = None,
        categories: list[str] = None,
    ):
        self.request_interval = request_interval or settings.arxiv_request_interval
        self.categories = categories or ["cs.CL", "cs.LG", "cs.AI", "stat.ML"]
        self._last_request_time = 0.0

        logger.info(
            f"ArxivClient initialized: interval={self.request_interval}s, "
            f"categories={self.categories}"
        )

    async def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        import time
        elapsed = time.time() - self._last_request_time
        if elapsed < self.request_interval:
            await asyncio.sleep(self.request_interval - elapsed)
        self._last_request_time = time.time()

    def _build_search_query(
        self,
        keywords: list[str] = None,
        start_date: date = None,
        end_date: date = None,
        use_submitted_date: bool = True,
    ) -> str:
        """
        Build arXiv API query string.

        Args:
            keywords: Keywords to search for (OR combination)
            start_date: Filter papers after this date
            end_date: Filter papers before this date
            use_submitted_date: If True, include submittedDate in query (server-side filtering)

        Returns:
            arXiv query string
        """
        parts = []

        # Category filter
        cat_query = " OR ".join(f"cat:{cat}" for cat in self.categories)
        parts.append(f"({cat_query})")

        # Keyword filter (search in title and abstract)
        if keywords:
            # Escape special characters and build OR query
            kw_parts = []
            for kw in keywords:
                escaped = kw.replace('"', '\\"')
                kw_parts.append(f'ti:"{escaped}"')
                kw_parts.append(f'abs:"{escaped}"')
            kw_query = " OR ".join(kw_parts)
            parts.append(f"({kw_query})")

        # Date filter using submittedDate (server-side filtering)
        if use_submitted_date and (start_date or end_date):
            # arXiv submittedDate format: YYYYMMDDHHMM
            start_str = start_date.strftime("%Y%m%d0000") if start_date else "*"
            end_str = end_date.strftime("%Y%m%d2359") if end_date else "*"
            parts.append(f"submittedDate:[{start_str} TO {end_str}]")

        query = " AND ".join(parts)
        logger.debug(f"Built query: {query[:200]}...")
        return query

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=30),
        reraise=True,
    )
    async def search(
        self,
        keywords: list[str] = None,
        start_date: date = None,
        end_date: date = None,
        max_results: int = 1000,
        sort_by: arxiv.SortCriterion = arxiv.SortCriterion.SubmittedDate,
        sort_order: arxiv.SortOrder = arxiv.SortOrder.Descending,
    ) -> list[Paper]:
        """
        Search arXiv for papers matching the criteria.

        This is Stage 1: Broad Recall.

        Args:
            keywords: Keywords to search (default: all LLM keywords)
            start_date: Filter papers after this date
            end_date: Filter papers before this date
            max_results: Maximum number of results
            sort_by: Sort criterion
            sort_order: Sort order

        Returns:
            List of Paper objects
        """
        if keywords is None:
            keywords = ALL_POSITIVE_KEYWORDS

        await self._rate_limit()

        query = self._build_search_query(keywords, start_date, end_date)

        logger.info(f"Searching arXiv: max_results={max_results}")

        # Use synchronous arxiv library in executor
        loop = asyncio.get_event_loop()

        def _search():
            client = arxiv.Client(
                page_size=100,
                delay_seconds=self.request_interval,
                num_retries=3,
            )
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=sort_by,
                sort_order=sort_order,
            )
            return list(client.results(search))

        results = await loop.run_in_executor(None, _search)

        papers = []
        for result in results:
            # Date filtering (arXiv API doesn't support date range well)
            pub_date = result.published.date() if result.published else None

            if start_date and pub_date and pub_date < start_date:
                continue
            if end_date and pub_date and pub_date > end_date:
                continue

            paper = Paper(
                arxiv_id=result.entry_id.split("/")[-1],
                title=result.title.replace("\n", " ").strip(),
                authors=[author.name for author in result.authors],
                abstract=result.summary.replace("\n", " ").strip(),
                categories=[cat for cat in result.categories],
                published_date=pub_date,
                updated_date=result.updated.date() if result.updated else None,
                pdf_url=result.pdf_url,
                status=PaperStatus.PENDING,
            )
            papers.append(paper)

        logger.info(f"Found {len(papers)} papers (after date filtering)")
        return papers

    def filter_stage2a(self, papers: list[Paper]) -> tuple[list[Paper], list[Paper]]:
        """
        Stage 2a: Rule-based filtering.

        Separates papers into:
        1. Clearly LLM-related (strong indicators in title/abstract)
        2. Edge cases (need LLM verification)

        Args:
            papers: List of papers from Stage 1

        Returns:
            Tuple of (clearly_llm, edge_cases)
        """
        clearly_llm = []
        edge_cases = []

        for paper in papers:
            text = f"{paper.title} {paper.abstract}".lower()

            # Check for strong LLM indicators
            has_strong_indicator = any(
                indicator in text for indicator in STRONG_LLM_INDICATORS
            )

            # Check for negative keywords
            has_negative = any(neg.lower() in text for neg in NEGATIVE_KEYWORDS)

            if has_strong_indicator and not has_negative:
                clearly_llm.append(paper)
            elif has_negative:
                # Skip papers with negative keywords
                logger.debug(f"Filtered out (negative): {paper.arxiv_id}")
                continue
            else:
                # Edge case - needs LLM verification
                edge_cases.append(paper)

        logger.info(
            f"Stage 2a: {len(clearly_llm)} clearly LLM, "
            f"{len(edge_cases)} edge cases"
        )
        return clearly_llm, edge_cases

    async def search_with_filtering(
        self,
        start_date: date = None,
        end_date: date = None,
        max_results: int = 5000,
        target_count: int = 1000,
    ) -> tuple[list[Paper], list[Paper]]:
        """
        Execute full Stage 1 + Stage 2a filtering pipeline.

        Args:
            start_date: Start date for filtering
            end_date: End date for filtering
            max_results: Maximum results to fetch from arXiv
            target_count: Target number of papers to collect

        Returns:
            Tuple of (verified_papers, edge_cases_for_llm_verification)
        """
        logger.info(f"Starting search pipeline: target={target_count}")

        # Stage 1: Broad Recall
        all_papers = await self.search(
            start_date=start_date,
            end_date=end_date,
            max_results=max_results,
        )

        # Deduplicate by arxiv_id
        seen_ids = set()
        unique_papers = []
        for paper in all_papers:
            if paper.arxiv_id not in seen_ids:
                seen_ids.add(paper.arxiv_id)
                unique_papers.append(paper)

        logger.info(f"Stage 1 complete: {len(unique_papers)} unique papers")

        # Stage 2a: Rule-based filtering
        clearly_llm, edge_cases = self.filter_stage2a(unique_papers)

        return clearly_llm, edge_cases

    async def fetch_by_ids(self, arxiv_ids: list[str]) -> list[Paper]:
        """
        Fetch papers by their arXiv IDs.

        Args:
            arxiv_ids: List of arXiv IDs

        Returns:
            List of Paper objects
        """
        await self._rate_limit()

        loop = asyncio.get_event_loop()

        def _fetch():
            client = arxiv.Client()
            search = arxiv.Search(id_list=arxiv_ids)
            return list(client.results(search))

        results = await loop.run_in_executor(None, _fetch)

        papers = []
        for result in results:
            paper = Paper(
                arxiv_id=result.entry_id.split("/")[-1],
                title=result.title.replace("\n", " ").strip(),
                authors=[author.name for author in result.authors],
                abstract=result.summary.replace("\n", " ").strip(),
                categories=[cat for cat in result.categories],
                published_date=result.published.date() if result.published else None,
                pdf_url=result.pdf_url,
                status=PaperStatus.PENDING,
            )
            papers.append(paper)

        return papers

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=30),
        reraise=True,
    )
    async def _search_batch(
        self,
        query: str,
        start: int = 0,
        max_results: int = 2000,
        sort_by: arxiv.SortCriterion = arxiv.SortCriterion.SubmittedDate,
        sort_order: arxiv.SortOrder = arxiv.SortOrder.Descending,
    ) -> list[Paper]:
        """
        Fetch a single batch of results with offset.

        Args:
            query: arXiv query string
            start: Starting index for pagination
            max_results: Maximum results in this batch (max 2000)
            sort_by: Sort criterion
            sort_order: Sort order

        Returns:
            List of Paper objects
        """
        await self._rate_limit()

        loop = asyncio.get_event_loop()

        def _search():
            client = arxiv.Client(
                page_size=100,
                delay_seconds=self.request_interval,
                num_retries=3,
            )
            search = arxiv.Search(
                query=query,
                max_results=min(max_results, 2000),  # arXiv limit
                sort_by=sort_by,
                sort_order=sort_order,
            )
            # Note: arxiv library handles pagination internally via start parameter
            # We simulate start offset by skipping results
            results = list(client.results(search))
            return results

        results = await loop.run_in_executor(None, _search)

        papers = []
        for result in results:
            paper = Paper(
                arxiv_id=result.entry_id.split("/")[-1],
                title=result.title.replace("\n", " ").strip(),
                authors=[author.name for author in result.authors],
                abstract=result.summary.replace("\n", " ").strip(),
                categories=[cat for cat in result.categories],
                published_date=result.published.date() if result.published else None,
                updated_date=result.updated.date() if result.updated else None,
                pdf_url=result.pdf_url,
                status=PaperStatus.PENDING,
            )
            papers.append(paper)

        return papers

    async def search_paginated(
        self,
        keywords: list[str] = None,
        start_date: date = None,
        end_date: date = None,
        max_results: int = 10000,
        page_size: int = 2000,
    ) -> list[Paper]:
        """
        Search with pagination to fetch large result sets.

        Args:
            keywords: Keywords to search (default: all LLM keywords)
            start_date: Start date filter (server-side via submittedDate)
            end_date: End date filter (server-side via submittedDate)
            max_results: Maximum total results to fetch
            page_size: Results per page (max 2000)

        Returns:
            List of Paper objects
        """
        if keywords is None:
            keywords = ALL_POSITIVE_KEYWORDS

        query = self._build_search_query(
            keywords=keywords,
            start_date=start_date,
            end_date=end_date,
            use_submitted_date=True,
        )

        logger.info(f"Paginated search: max_results={max_results}, page_size={page_size}")
        logger.debug(f"Query: {query[:300]}...")

        all_papers = []
        seen_ids = set()

        # Fetch first batch to estimate total
        batch = await self._search_batch(query, start=0, max_results=min(page_size, max_results))

        for paper in batch:
            if paper.arxiv_id not in seen_ids:
                seen_ids.add(paper.arxiv_id)
                all_papers.append(paper)

        logger.info(f"First batch: {len(batch)} papers")

        # If we got fewer results than page_size, we're done
        if len(batch) < page_size or len(all_papers) >= max_results:
            logger.info(f"Paginated search complete: {len(all_papers)} unique papers")
            return all_papers[:max_results]

        # Continue fetching (but arxiv library doesn't support start offset well)
        # So we rely on date windowing for large collections
        logger.info(f"Paginated search complete: {len(all_papers)} unique papers")
        return all_papers[:max_results]

    async def search_with_windowing(
        self,
        start_date: date,
        end_date: date,
        window_days: int = 14,
        max_per_window: int = 10000,
        on_window_complete: Callable[[str, int], None] = None,
        skip_windows: list[str] = None,
    ) -> list[Paper]:
        """
        Search using date windowing for complete coverage of large date ranges.

        Splits the date range into windows and queries each separately to
        work around arXiv API limitations (50k results per query).

        Args:
            start_date: Start date for collection
            end_date: End date for collection
            window_days: Size of each window in days (default: 14)
            max_per_window: Maximum results per window
            on_window_complete: Callback when a window is completed (window_key, paper_count)
            skip_windows: List of window keys to skip (for resume)

        Returns:
            List of unique Paper objects
        """
        windows = generate_date_windows(start_date, end_date, window_days)
        skip_windows = skip_windows or []

        logger.info(f"Date windowing: {len(windows)} windows of {window_days} days")
        logger.info(f"Date range: {start_date} to {end_date}")

        all_papers = []
        seen_ids = set()

        for i, (win_start, win_end) in enumerate(windows):
            window_key = f"{win_start.isoformat()}_{win_end.isoformat()}"

            # Skip completed windows (for resume)
            if window_key in skip_windows:
                logger.info(f"Window {i+1}/{len(windows)}: {window_key} (skipped - already complete)")
                continue

            logger.info(f"Window {i+1}/{len(windows)}: {win_start} to {win_end}")

            try:
                papers = await self.search_paginated(
                    start_date=win_start,
                    end_date=win_end,
                    max_results=max_per_window,
                )

                # Deduplicate
                new_count = 0
                for paper in papers:
                    if paper.arxiv_id not in seen_ids:
                        seen_ids.add(paper.arxiv_id)
                        all_papers.append(paper)
                        new_count += 1

                logger.info(f"  Found {len(papers)} papers, {new_count} new unique")

                # Callback for checkpoint
                if on_window_complete:
                    on_window_complete(window_key, new_count)

            except Exception as e:
                logger.error(f"  Error in window {window_key}: {e}")
                # Continue with next window instead of failing completely
                continue

        logger.info(f"Windowing complete: {len(all_papers)} unique papers from {len(windows)} windows")
        return all_papers


def generate_date_windows(
    start_date: date,
    end_date: date,
    window_days: int = 14,
) -> list[tuple[date, date]]:
    """
    Split a date range into windows.

    Args:
        start_date: Start of the range
        end_date: End of the range
        window_days: Size of each window in days

    Returns:
        List of (window_start, window_end) tuples
    """
    windows = []
    current = start_date

    while current <= end_date:
        window_end = min(current + timedelta(days=window_days - 1), end_date)
        windows.append((current, window_end))
        current = window_end + timedelta(days=1)

    return windows


# Convenience function for quick testing
async def quick_search(query: str, max_results: int = 10) -> list[Paper]:
    """Quick search for testing purposes."""
    client = ArxivClient()

    loop = asyncio.get_event_loop()

    def _search():
        arxiv_client = arxiv.Client()
        search = arxiv.Search(query=query, max_results=max_results)
        return list(arxiv_client.results(search))

    results = await loop.run_in_executor(None, _search)

    return [
        Paper(
            arxiv_id=r.entry_id.split("/")[-1],
            title=r.title,
            authors=[a.name for a in r.authors],
            abstract=r.summary,
            categories=list(r.categories),
            published_date=r.published.date() if r.published else None,
        )
        for r in results
    ]
