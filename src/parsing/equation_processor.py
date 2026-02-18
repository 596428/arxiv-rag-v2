"""
arXiv RAG v1 - Equation Processor

Process extracted equations and generate text descriptions using Gemini API.
"""

import asyncio
import logging
from typing import Optional

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from .models import Equation, ParsedDocument

logger = logging.getLogger(__name__)


class EquationProcessor:
    """
    Process equations and generate text descriptions.

    Uses Gemini API to convert LaTeX equations to natural language.
    """

    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        model_name: str = "gemini-2.0-flash",
        batch_size: int = 10,
        max_retries: int = 3,
    ):
        """
        Initialize equation processor.

        Args:
            gemini_api_key: Gemini API key (uses env var if not provided)
            model_name: Gemini model to use
            batch_size: Number of equations to process in parallel
            max_retries: Maximum retry attempts for API calls
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_retries = max_retries
        self._client = None
        self._model = None

        # Initialize Gemini client
        if gemini_api_key:
            self._init_client(gemini_api_key)

    def _init_client(self, api_key: str):
        """Initialize Gemini client."""
        try:
            import google.generativeai as genai

            genai.configure(api_key=api_key)
            self._model = genai.GenerativeModel(self.model_name)
            logger.info(f"Gemini client initialized with model: {self.model_name}")
        except ImportError:
            logger.warning("google-generativeai not installed, equation descriptions disabled")
        except Exception as e:
            logger.warning(f"Failed to initialize Gemini client: {e}")

    def _ensure_client(self):
        """Ensure Gemini client is initialized."""
        if self._model is not None:
            return True

        # Try to initialize from environment
        import os
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            self._init_client(api_key)

        return self._model is not None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
    )
    def _generate_description(self, equation: Equation) -> str:
        """
        Generate text description for a single equation.

        Args:
            equation: Equation to describe

        Returns:
            Natural language description
        """
        if not self._ensure_client():
            return ""

        prompt = self._build_prompt(equation)

        try:
            response = self._model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.warning(f"Failed to generate description for {equation.equation_id}: {e}")
            raise

    def _build_prompt(self, equation: Equation) -> str:
        """Build prompt for equation description."""
        context = ""
        if equation.context_before:
            context += f"Context before: {equation.context_before}\n"
        if equation.context_after:
            context += f"Context after: {equation.context_after}\n"

        return f"""You are a technical writing assistant. Convert the following LaTeX equation to a clear, concise natural language description.

LaTeX equation:
```latex
{equation.latex}
```

{context}

Instructions:
1. Describe what the equation represents mathematically
2. Explain each variable/symbol if identifiable from context
3. Keep the description under 100 words
4. Use plain English, avoid jargon where possible
5. If it's a well-known equation (e.g., softmax, cross-entropy), name it

Response (natural language description only):"""

    def process_equation(self, equation: Equation) -> Equation:
        """
        Process a single equation and add text description.

        Args:
            equation: Equation to process

        Returns:
            Equation with text_description populated
        """
        if equation.text_description:
            return equation  # Already has description

        try:
            description = self._generate_description(equation)
            equation.text_description = description
        except Exception as e:
            logger.warning(f"Failed to process equation {equation.equation_id}: {e}")
            # Keep original equation without description

        return equation

    def process_equations(
        self,
        equations: list[Equation],
        progress_callback: Optional[callable] = None,
    ) -> list[Equation]:
        """
        Process multiple equations.

        Args:
            equations: List of equations to process
            progress_callback: Optional callback(processed, total)

        Returns:
            List of processed equations
        """
        if not equations:
            return []

        if not self._ensure_client():
            logger.warning("Gemini client not available, skipping equation descriptions")
            return equations

        processed = []
        total = len(equations)

        for i, equation in enumerate(equations):
            processed_eq = self.process_equation(equation)
            processed.append(processed_eq)

            if progress_callback:
                progress_callback(i + 1, total)

        return processed

    def process_document(
        self,
        doc: ParsedDocument,
        max_equations: Optional[int] = None,
    ) -> ParsedDocument:
        """
        Process all equations in a document.

        Args:
            doc: Parsed document
            max_equations: Maximum equations to process (for cost control)

        Returns:
            Document with equation descriptions added
        """
        if not doc.equations:
            return doc

        equations_to_process = doc.equations
        if max_equations:
            equations_to_process = equations_to_process[:max_equations]

        logger.info(f"Processing {len(equations_to_process)} equations for {doc.arxiv_id}")

        processed = self.process_equations(equations_to_process)

        # Update document equations
        doc.equations = processed
        return doc


# Async version for better throughput
class AsyncEquationProcessor(EquationProcessor):
    """Async version of equation processor for better throughput."""

    async def _generate_description_async(self, equation: Equation) -> str:
        """Generate description asynchronously."""
        # Run sync method in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._generate_description,
            equation,
        )

    async def process_equations_async(
        self,
        equations: list[Equation],
        progress_callback: Optional[callable] = None,
    ) -> list[Equation]:
        """Process equations asynchronously with concurrency control."""
        if not equations:
            return []

        if not self._ensure_client():
            return equations

        semaphore = asyncio.Semaphore(self.batch_size)
        processed = []
        total = len(equations)
        completed = 0

        async def process_one(eq: Equation) -> Equation:
            nonlocal completed
            async with semaphore:
                try:
                    desc = await self._generate_description_async(eq)
                    eq.text_description = desc
                except Exception as e:
                    logger.warning(f"Failed: {eq.equation_id}: {e}")

                completed += 1
                if progress_callback:
                    progress_callback(completed, total)

                return eq

        tasks = [process_one(eq) for eq in equations]
        processed = await asyncio.gather(*tasks)

        return list(processed)


def get_equation_processor(
    api_key: Optional[str] = None,
    async_mode: bool = False,
) -> EquationProcessor:
    """
    Get equation processor instance.

    Args:
        api_key: Gemini API key
        async_mode: Whether to use async processor

    Returns:
        EquationProcessor instance
    """
    if async_mode:
        return AsyncEquationProcessor(gemini_api_key=api_key)
    return EquationProcessor(gemini_api_key=api_key)
