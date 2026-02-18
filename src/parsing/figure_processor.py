"""
arXiv RAG v1 - Figure Processor

Process extracted figures and manage image files.
V1: Caption-only extraction (no VLM description).
"""

import logging
import shutil
from pathlib import Path
from typing import Optional

from .models import Figure, ParsedDocument

logger = logging.getLogger(__name__)


class FigureProcessor:
    """
    Process figures from parsed documents.

    V1 Implementation:
    - Extract and save figure images
    - Extract captions (no VLM description)
    - Organize figures by paper
    """

    def __init__(self, figures_dir: Path):
        """
        Initialize figure processor.

        Args:
            figures_dir: Base directory for storing figures
        """
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def get_paper_figures_dir(self, arxiv_id: str) -> Path:
        """Get the figures directory for a specific paper."""
        paper_dir = self.figures_dir / arxiv_id.replace("/", "_")
        paper_dir.mkdir(parents=True, exist_ok=True)
        return paper_dir

    def save_figure(
        self,
        figure: Figure,
        arxiv_id: str,
        source_path: Optional[Path] = None,
        image_data: Optional[bytes] = None,
    ) -> Figure:
        """
        Save a figure image to the figures directory.

        Args:
            figure: Figure metadata
            arxiv_id: Paper ID
            source_path: Source image file path (if copying)
            image_data: Raw image bytes (if creating new)

        Returns:
            Figure with updated image_path
        """
        paper_dir = self.get_paper_figures_dir(arxiv_id)

        # Determine filename and extension
        if source_path:
            ext = source_path.suffix or ".png"
            dst_path = paper_dir / f"{figure.figure_id}{ext}"

            try:
                shutil.copy(source_path, dst_path)
                figure.image_path = str(dst_path)
                logger.debug(f"Copied figure: {source_path} -> {dst_path}")
            except Exception as e:
                logger.warning(f"Failed to copy figure {figure.figure_id}: {e}")

        elif image_data:
            # Detect format from magic bytes
            ext = self._detect_image_format(image_data)
            dst_path = paper_dir / f"{figure.figure_id}{ext}"

            try:
                dst_path.write_bytes(image_data)
                figure.image_path = str(dst_path)
                logger.debug(f"Saved figure: {dst_path}")
            except Exception as e:
                logger.warning(f"Failed to save figure {figure.figure_id}: {e}")

        return figure

    def _detect_image_format(self, data: bytes) -> str:
        """Detect image format from magic bytes."""
        if data[:8] == b"\x89PNG\r\n\x1a\n":
            return ".png"
        elif data[:2] == b"\xff\xd8":
            return ".jpg"
        elif data[:6] in (b"GIF87a", b"GIF89a"):
            return ".gif"
        elif data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            return ".webp"
        elif data[:4] == b"%PDF":
            return ".pdf"
        else:
            return ".png"  # Default

    def process_document_figures(
        self,
        doc: ParsedDocument,
        source_images: Optional[dict] = None,
    ) -> ParsedDocument:
        """
        Process all figures in a document.

        Args:
            doc: Parsed document with figures
            source_images: Optional dict of {figure_id: image_path or bytes}

        Returns:
            Document with processed figures
        """
        if not doc.figures:
            return doc

        source_images = source_images or {}
        processed_figures = []

        for figure in doc.figures:
            # Check if we have source image
            if figure.figure_id in source_images:
                source = source_images[figure.figure_id]
                if isinstance(source, (str, Path)):
                    figure = self.save_figure(
                        figure, doc.arxiv_id, source_path=Path(source)
                    )
                elif isinstance(source, bytes):
                    figure = self.save_figure(
                        figure, doc.arxiv_id, image_data=source
                    )
            elif figure.image_path and Path(figure.image_path).exists():
                # Already have a valid path
                pass
            else:
                # No image available
                logger.debug(f"No image source for {figure.figure_id}")

            processed_figures.append(figure)

        doc.figures = processed_figures
        return doc

    def cleanup_paper_figures(self, arxiv_id: str) -> int:
        """
        Remove all figures for a paper.

        Args:
            arxiv_id: Paper ID

        Returns:
            Number of files removed
        """
        paper_dir = self.get_paper_figures_dir(arxiv_id)

        if not paper_dir.exists():
            return 0

        count = 0
        for f in paper_dir.iterdir():
            if f.is_file():
                f.unlink()
                count += 1

        # Remove empty directory
        try:
            paper_dir.rmdir()
        except OSError:
            pass  # Directory not empty

        return count

    def get_figure_stats(self) -> dict:
        """Get statistics about stored figures."""
        total_files = 0
        total_size = 0
        papers_with_figures = 0
        format_counts = {}

        for paper_dir in self.figures_dir.iterdir():
            if not paper_dir.is_dir():
                continue

            paper_files = list(paper_dir.glob("*"))
            if paper_files:
                papers_with_figures += 1

            for f in paper_files:
                if f.is_file():
                    total_files += 1
                    total_size += f.stat().st_size
                    ext = f.suffix.lower()
                    format_counts[ext] = format_counts.get(ext, 0) + 1

        return {
            "total_files": total_files,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "papers_with_figures": papers_with_figures,
            "format_distribution": format_counts,
        }


def get_figure_processor(figures_dir: Optional[Path] = None) -> FigureProcessor:
    """
    Get figure processor instance.

    Args:
        figures_dir: Directory for storing figures

    Returns:
        FigureProcessor instance
    """
    if figures_dir is None:
        from src.utils.config import settings
        figures_dir = settings.figures_dir

    return FigureProcessor(figures_dir)


def extract_caption_text(caption: Optional[str]) -> str:
    """
    Clean and normalize figure caption text.

    Args:
        caption: Raw caption text

    Returns:
        Cleaned caption
    """
    if not caption:
        return ""

    from .latex_cleaner import clean_latex_text

    # Clean LaTeX artifacts
    cleaned = clean_latex_text(caption)

    # Remove "Figure X:" prefix
    import re
    cleaned = re.sub(r"^(?:Figure|Fig\.?)\s*\d+[.:]\s*", "", cleaned, flags=re.IGNORECASE)

    return cleaned.strip()
