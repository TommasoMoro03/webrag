import logging

from webrag.utils.exceptions import ConfigurationError
from webrag.crawler.base import BaseCrawler
from webrag.fetcher.base import BaseFetcher
from webrag.extractors.base import BaseExtractor
from webrag.chunking.base import BaseChunker
from webrag.output.base import BaseExporter

from typing import Optional
from webrag.config import AIConfig

logger = logging.getLogger(__name__)

def _get_default_fetcher() -> BaseFetcher:
    """Get default fetcher implementation."""
    # Try to import and use a concrete implementation
    try:
        from webrag.fetcher.static_fetcher import StaticFetcher
        return StaticFetcher()
    except ImportError:
        logger.warning(
            "No concrete fetcher implementation found. "
            "Please implement StaticFetcher or provide a custom fetcher."
        )
        raise ConfigurationError(
            "No fetcher available. Please provide a fetcher instance or "
            "implement webrag.fetcher.static_fetcher.StaticFetcher"
        )


def _get_default_extractor() -> BaseExtractor:
    """Get default extractor implementation."""
    try:
        from webrag.extractors.trafilatura_extractor import TrafilaturaExtractor
        return TrafilaturaExtractor()
    except ImportError:
        logger.warning(
            "No concrete extractor implementation found. "
            "Please implement TrafilaturaExtractor or provide a custom extractor."
        )
        raise ConfigurationError(
            "No extractor available. Please provide an extractor instance or "
            "implement webrag.extractors.trafilatura_extractor.TrafilaturaExtractor"
        )


def _get_default_chunker() -> BaseChunker:
    """Get default chunker implementation."""
    try:
        from webrag.chunking.semantic_chunker import SemanticChunker
        return SemanticChunker()
    except ImportError:
        logger.warning(
            "No concrete chunker implementation found. "
            "Please implement SemanticChunker or provide a custom chunker."
        )
        raise ConfigurationError(
            "No chunker available. Please provide a chunker instance or "
            "implement webrag.chunking.semantic_chunker.SemanticChunker"
        )


def _get_default_exporter() -> BaseExporter:
    """Get default exporter implementation."""
    try:
        from webrag.output.json_exporter import JSONExporter
        return JSONExporter()
    except ImportError:
        logger.warning(
            "No concrete exporter implementation found. "
            "Please implement JSONExporter or provide a custom exporter."
        )
        raise ConfigurationError(
            "No exporter available. Please provide an exporter instance or "
            "implement webrag.output.json_exporter.JSONExporter"
        )


def _get_default_crawler(ai_config: Optional[AIConfig]) -> BaseCrawler:
    """
    Get default crawler implementation with smart selection.

    Uses AI crawler if:
    1. AI is enabled in config
    2. AICrawler implementation is available

    Otherwise falls back to SimpleCrawler.
    """
    # Try AI crawler first if AI is enabled
    if ai_config.enabled:
        try:
            from webrag.crawler.ai_crawler import AICrawler

            # Get LLM function from config
            llm_function = ai_config.get_llm_function()

            logger.info("Using AI-powered crawler")
            return AICrawler(
                llm_function=llm_function,
                max_links_per_page=200,
                min_confidence=0.6,
                enable_smart_grouping=True,
            )
        except ImportError as e:
            logger.warning(f"AI crawler not available: {e}, falling back to SimpleCrawler")
        except Exception as e:
            logger.warning(f"Could not initialize AI crawler: {e}, falling back to SimpleCrawler")

    # Fallback to SimpleCrawler
    try:
        from webrag.crawler.simple_crawler import SimpleCrawler
        logger.info("Using heuristic-based SimpleCrawler")
        return SimpleCrawler()
    except ImportError:
        logger.warning(
            "No concrete crawler implementation found. "
            "Please implement SimpleCrawler or provide a custom crawler."
        )
        raise ConfigurationError(
            "No crawler available. Please provide a crawler instance or "
            "implement webrag.crawler.simple_crawler.SimpleCrawler"
        )