"""
arXiv RAG v1 - Download Manager

Async download manager for PDF and LaTeX source files.
"""

import asyncio
from pathlib import Path
from typing import Optional

import aiofiles
import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from ..utils.config import settings
from ..utils.logging import get_logger, ProgressLogger
from .models import Paper, PaperStatus

logger = get_logger("downloader")


class DownloadError(Exception):
    """Download failed error."""
    pass


class Downloader:
    """
    Async download manager for arXiv papers.

    Features:
    - Concurrent PDF/LaTeX downloads
    - Progress tracking
    - Retry with exponential backoff
    - Resume support (skip existing files)
    """

    def __init__(
        self,
        pdf_dir: Path = None,
        latex_dir: Path = None,
        max_concurrent: int = None,
        request_interval: float = 1.0,
    ):
        self.pdf_dir = pdf_dir or settings.pdf_dir
        self.latex_dir = latex_dir or settings.latex_dir
        self.max_concurrent = max_concurrent or settings.max_concurrent_downloads
        self.request_interval = request_interval

        # Ensure directories exist
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.latex_dir.mkdir(parents=True, exist_ok=True)

        self._client: Optional[httpx.AsyncClient] = None
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._last_request_time = 0.0

        logger.info(
            f"Downloader initialized: "
            f"pdf_dir={self.pdf_dir}, "
            f"latex_dir={self.latex_dir}, "
            f"max_concurrent={self.max_concurrent}"
        )

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(60.0, connect=10.0),
                follow_redirects=True,
                headers={
                    "User-Agent": "arXiv-RAG-v1 Research Project (contact@example.com)"
                },
            )
        return self._client

    async def _get_semaphore(self) -> asyncio.Semaphore:
        """Get or create semaphore for concurrency control."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrent)
        return self._semaphore

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        import time
        elapsed = time.time() - self._last_request_time
        if elapsed < self.request_interval:
            await asyncio.sleep(self.request_interval - elapsed)
        self._last_request_time = time.time()

    def _get_pdf_path(self, arxiv_id: str) -> Path:
        """Get local path for PDF file."""
        # Sanitize arxiv_id for filename
        safe_id = arxiv_id.replace("/", "_").replace(":", "_")
        return self.pdf_dir / f"{safe_id}.pdf"

    def _get_latex_path(self, arxiv_id: str) -> Path:
        """Get local path for LaTeX archive."""
        safe_id = arxiv_id.replace("/", "_").replace(":", "_")
        return self.latex_dir / f"{safe_id}.tar.gz"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=30),
        retry=retry_if_exception_type((httpx.HTTPError, DownloadError)),
        reraise=True,
    )
    async def _download_file(
        self,
        url: str,
        dest_path: Path,
        expected_type: str = None,
    ) -> bool:
        """
        Download a file with retry logic.

        Args:
            url: URL to download
            dest_path: Destination file path
            expected_type: Expected content type prefix (e.g., "application/pdf")

        Returns:
            True if successful

        Raises:
            DownloadError: Download failed
        """
        await self._rate_limit()

        client = await self._get_client()

        try:
            response = await client.get(url)
            response.raise_for_status()

            # Check content type if specified
            content_type = response.headers.get("content-type", "")
            if expected_type and not content_type.startswith(expected_type):
                # Some arXiv responses are gzipped or have different content types
                # Log but don't fail
                logger.debug(
                    f"Unexpected content type for {url}: {content_type}"
                )

            # Write to file
            async with aiofiles.open(dest_path, "wb") as f:
                await f.write(response.content)

            logger.debug(f"Downloaded: {dest_path.name} ({len(response.content)} bytes)")
            return True

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.warning(f"Not found: {url}")
                return False
            raise DownloadError(f"HTTP error {e.response.status_code}: {url}")

        except httpx.RequestError as e:
            raise DownloadError(f"Request error: {e}")

    async def download_pdf(
        self,
        paper: Paper,
        skip_existing: bool = True,
    ) -> Optional[Path]:
        """
        Download PDF for a paper.

        Args:
            paper: Paper object
            skip_existing: Skip if file already exists

        Returns:
            Path to downloaded file, or None if failed
        """
        dest_path = self._get_pdf_path(paper.arxiv_id)

        if skip_existing and dest_path.exists():
            logger.debug(f"Skipping existing: {dest_path.name}")
            return dest_path

        pdf_url = paper.pdf_url or paper.default_pdf_url

        semaphore = await self._get_semaphore()
        async with semaphore:
            try:
                success = await self._download_file(
                    pdf_url,
                    dest_path,
                    expected_type="application/pdf",
                )
                return dest_path if success else None

            except DownloadError as e:
                logger.error(f"Failed to download PDF for {paper.arxiv_id}: {e}")
                return None

    async def download_latex(
        self,
        paper: Paper,
        skip_existing: bool = True,
    ) -> Optional[Path]:
        """
        Download LaTeX source for a paper.

        Args:
            paper: Paper object
            skip_existing: Skip if file already exists

        Returns:
            Path to downloaded file, or None if failed/not available
        """
        dest_path = self._get_latex_path(paper.arxiv_id)

        if skip_existing and dest_path.exists():
            logger.debug(f"Skipping existing: {dest_path.name}")
            return dest_path

        latex_url = paper.latex_url or paper.default_latex_url

        semaphore = await self._get_semaphore()
        async with semaphore:
            try:
                success = await self._download_file(
                    latex_url,
                    dest_path,
                )
                return dest_path if success else None

            except DownloadError as e:
                # LaTeX source might not be available for all papers
                logger.debug(f"LaTeX not available for {paper.arxiv_id}: {e}")
                return None

    async def download_paper(
        self,
        paper: Paper,
        download_pdf: bool = True,
        download_latex: bool = True,
        skip_existing: bool = True,
    ) -> Paper:
        """
        Download both PDF and LaTeX for a paper.

        Args:
            paper: Paper object
            download_pdf: Whether to download PDF
            download_latex: Whether to download LaTeX source
            skip_existing: Skip existing files

        Returns:
            Updated Paper object with file paths
        """
        tasks = []

        if download_pdf:
            tasks.append(self.download_pdf(paper, skip_existing))
        if download_latex:
            tasks.append(self.download_latex(paper, skip_existing))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        idx = 0
        if download_pdf:
            pdf_result = results[idx]
            if isinstance(pdf_result, Path):
                paper.pdf_path = str(pdf_result)
            idx += 1

        if download_latex:
            latex_result = results[idx]
            if isinstance(latex_result, Path):
                paper.latex_path = str(latex_result)

        # Update status
        if paper.pdf_path:
            paper.status = PaperStatus.COLLECTED

        return paper

    async def download_batch(
        self,
        papers: list[Paper],
        download_pdf: bool = True,
        download_latex: bool = True,
        skip_existing: bool = True,
        progress_callback: callable = None,
    ) -> list[Paper]:
        """
        Download files for multiple papers.

        Args:
            papers: List of Paper objects
            download_pdf: Whether to download PDFs
            download_latex: Whether to download LaTeX sources
            skip_existing: Skip existing files
            progress_callback: Optional callback(completed, total)

        Returns:
            List of updated Paper objects
        """
        logger.info(f"Starting batch download: {len(papers)} papers")

        progress = ProgressLogger(
            total=len(papers),
            name="papers",
            log_every=10,
        )

        results = []
        for i, paper in enumerate(papers):
            updated = await self.download_paper(
                paper,
                download_pdf=download_pdf,
                download_latex=download_latex,
                skip_existing=skip_existing,
            )
            results.append(updated)

            progress.update(1)
            if progress_callback:
                progress_callback(i + 1, len(papers))

        # Summary statistics
        pdf_count = sum(1 for p in results if p.pdf_path)
        latex_count = sum(1 for p in results if p.latex_path)

        logger.info(
            f"Batch download complete: "
            f"{pdf_count} PDFs, {latex_count} LaTeX sources"
        )

        return results


# Singleton downloader
_downloader: Optional[Downloader] = None


def get_downloader() -> Downloader:
    """Get or create the downloader singleton."""
    global _downloader
    if _downloader is None:
        _downloader = Downloader()
    return _downloader


async def close_downloader() -> None:
    """Close the singleton downloader."""
    global _downloader
    if _downloader:
        await _downloader.close()
        _downloader = None
