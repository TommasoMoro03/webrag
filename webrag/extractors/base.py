"""Abstract base class for content extractors."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from webrag.schemas.document import ExtractionResult
from webrag.schemas.source_profile import SourceProfile


class BaseExtractor(ABC):
    """
    Abstract base class for extracting meaningful content from raw HTML/text.

    Extractors are responsible for parsing HTML, removing boilerplate,
    extracting main content, metadata, and structuring the data.

    Future implementations may include:
    - TrafilaturaExtractor (using trafilatura library)
    - BeautifulSoupExtractor (custom BS4-based extraction)
    - ReadabilityExtractor (using readability-lxml)
    - SemanticExtractor (LLM-based intelligent extraction)
    - CustomExtractor (user-defined CSS/XPath selectors)
    """

    def __init__(self, **kwargs):
        """
        Initialize the extractor.

        Args:
            **kwargs: Extractor-specific configuration
        """
        self.config = kwargs

    @abstractmethod
    def extract(
        self,
        raw_content: str,
        source: SourceProfile,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ExtractionResult:
        """
        Extract meaningful content from raw HTML/text.

        Args:
            raw_content: Raw HTML or text content to extract from
            source: SourceProfile containing hints and configuration
            metadata: Optional metadata from the fetch stage (headers, etc.)

        Returns:
            ExtractionResult containing extracted content and metadata

        Raises:
            ExtractionError: If extraction fails
            ContentNotFoundError: If no meaningful content found
            ParsingError: If HTML/content parsing fails
        """
        raise NotImplementedError

    @abstractmethod
    def extract_metadata(self, raw_content: str) -> Dict[str, Any]:
        """
        Extract metadata from raw content.

        Should extract common metadata like:
        - title
        - author
        - description
        - keywords
        - published_date
        - language
        - canonical_url

        Args:
            raw_content: Raw HTML or text content

        Returns:
            Dictionary of extracted metadata fields
        """
        raise NotImplementedError

    @abstractmethod
    def extract_links(self, raw_content: str, base_url: str) -> list[str]:
        """
        Extract links from content for potential crawling.

        Args:
            raw_content: Raw HTML content
            base_url: Base URL for resolving relative links

        Returns:
            List of absolute URLs found in the content
        """
        raise NotImplementedError

    def validate_content(self, content: str) -> bool:
        """
        Validate if extracted content meets quality criteria.

        Default implementation checks for minimum length.
        Can be overridden for more sophisticated validation.

        Args:
            content: Extracted content text

        Returns:
            True if content is valid, False otherwise
        """
        if not content or len(content.strip()) < 50:
            return False
        return True

    def calculate_confidence_score(
        self,
        raw_content: str,
        extracted_content: str
    ) -> float:
        """
        Calculate confidence score for extraction quality.

        Default implementation is a simple ratio-based heuristic.
        Can be overridden for more sophisticated scoring.

        Args:
            raw_content: Original raw content
            extracted_content: Extracted/cleaned content

        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not extracted_content:
            return 0.0

        # Simple heuristic: ratio of extracted to raw content
        # (More sophisticated implementations can consider other factors)
        ratio = len(extracted_content) / max(len(raw_content), 1)
        return min(max(ratio * 2, 0.0), 1.0)  # Scale and clamp to [0, 1]

    def apply_user_hints(
        self,
        raw_content: str,
        user_hints: Optional[Dict[str, str]] = None
    ) -> Optional[str]:
        """
        Apply user-provided extraction hints (CSS selectors, XPath).

        This is a placeholder for future implementation.
        Concrete extractors should implement this based on their parsing library.

        Args:
            raw_content: Raw HTML content
            user_hints: Dictionary of user hints (e.g., {'content_selector': 'article.main'})

        Returns:
            Extracted content using hints, or None if hints don't match
        """
        # Placeholder - concrete implementations will use BeautifulSoup, lxml, etc.
        return None

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text.

        Args:
            text: Raw extracted text

        Returns:
            Cleaned and normalized text
        """
        if not text:
            return ""

        # Basic cleaning: normalize whitespace
        lines = [line.strip() for line in text.split('\n')]
        lines = [line for line in lines if line]  # Remove empty lines
        return '\n'.join(lines)
