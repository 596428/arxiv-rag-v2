"""
arXiv RAG v1 - Gemini API Wrapper

수식 설명 생성 및 논문 관련성 검증용 Gemini API 래퍼
"""

import asyncio
from typing import Optional

import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import settings
from .logging import get_logger

logger = get_logger("gemini")


class GeminiClient:
    """
    Gemini API client for equation description and paper relevance verification.

    Usage:
        client = GeminiClient()
        description = await client.describe_equation(r"E = mc^2", context="Einstein's equation")
        is_llm = await client.verify_llm_relevance(abstract)
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or settings.gemini_api_key
        self.model_name = model or settings.gemini_model

        if not self.api_key:
            raise ValueError("Gemini API key not configured. Set GEMINI_API_KEY in .env")

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)
        logger.info(f"Gemini client initialized with model: {self.model_name}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def _generate(self, prompt: str) -> str:
        """Generate content with retry logic."""
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, lambda: self.model.generate_content(prompt)
        )
        return response.text

    async def describe_equation(
        self,
        latex: str,
        context: Optional[str] = None,
    ) -> str:
        """
        Generate a natural language description of a LaTeX equation.

        Args:
            latex: LaTeX equation string
            context: Optional context from the paper

        Returns:
            Natural language description of the equation
        """
        prompt = f"""다음 LaTeX 수식을 자연어로 설명해주세요.

수식: {latex}
{f"논문 컨텍스트: {context}" if context else ""}

다음 형식으로 답변해주세요:
1. 수식 이름 (있는 경우)
2. 각 변수의 의미
3. 수식이 표현하는 관계
4. 관련 키워드 (검색용, 쉼표로 구분)

답변은 한국어로 작성하되, 기술 용어는 영어 병기 가능합니다.
"""
        try:
            description = await self._generate(prompt)
            logger.debug(f"Generated description for equation: {latex[:50]}...")
            return description.strip()
        except Exception as e:
            logger.error(f"Failed to describe equation: {e}")
            # Graceful degradation: return raw LaTeX
            return f"[수식] {latex}"

    async def verify_llm_relevance(self, abstract: str) -> tuple[bool, str]:
        """
        Verify if a paper is primarily about LLM.

        Args:
            abstract: Paper abstract text

        Returns:
            Tuple of (is_llm_paper: bool, reason: str)
        """
        prompt = f"""다음 논문 초록을 읽고, 이 논문이 주로 Large Language Model (LLM)에 관한 것인지 판단해주세요.

초록:
{abstract}

판단 기준:
- LLM의 학습, 아키텍처, 평가, 응용에 관한 논문 → Yes
- LLM을 도구로 사용하지만 주제가 다른 논문 → Uncertain
- LLM과 관련 없는 논문 (로봇, 게임, 이미지 분류 등) → No

다음 형식으로만 답변:
DECISION: [Yes/No/Uncertain]
REASON: [한 줄 이유]
"""
        try:
            response = await self._generate(prompt)
            lines = response.strip().split("\n")

            decision = "Uncertain"
            reason = ""

            for line in lines:
                if line.startswith("DECISION:"):
                    decision = line.replace("DECISION:", "").strip()
                elif line.startswith("REASON:"):
                    reason = line.replace("REASON:", "").strip()

            is_llm = decision.lower() == "yes"
            logger.debug(f"LLM relevance check: {decision} - {reason[:50]}...")
            return is_llm, reason

        except Exception as e:
            logger.error(f"Failed to verify LLM relevance: {e}")
            # Default to uncertain on error
            return False, f"Error: {str(e)}"

    async def batch_describe_equations(
        self,
        equations: list[tuple[str, Optional[str]]],
        concurrency: int = 5,
    ) -> list[str]:
        """
        Batch process multiple equations with concurrency control.

        Args:
            equations: List of (latex, context) tuples
            concurrency: Max concurrent requests

        Returns:
            List of descriptions
        """
        semaphore = asyncio.Semaphore(concurrency)

        async def describe_with_limit(latex: str, context: Optional[str]) -> str:
            async with semaphore:
                return await self.describe_equation(latex, context)

        tasks = [describe_with_limit(latex, ctx) for latex, ctx in equations]
        return await asyncio.gather(*tasks)


# Singleton instance
_client: Optional[GeminiClient] = None


def get_gemini_client() -> GeminiClient:
    """Get or create the Gemini client singleton."""
    global _client
    if _client is None:
        _client = GeminiClient()
    return _client
