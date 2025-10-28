"""
Main orchestrator for the WebRAG pipeline.

This module coordinates the entire RAG ingestion pipeline:
1. Load and validate sources
2. Fetch content from URLs
3. Extract meaningful content
4. Chunk content for RAG systems
5. Export to various formats
"""

from typing import Union, List, Optional, Dict, Any
from pathlib import Path
from datetime import datetime
import logging

from webrag.sources.sources_loader import SourceLoader
from webrag.schemas.source_profile import SourceProfile
from webrag.schemas.document import DocumentChunk, ExtractionResult, DocumentGroup
from webrag.schemas.response import PipelineResult
from webrag.fetcher import BaseFetcher
from webrag.extractors import BaseExtractor
from webrag.chunking import BaseChunker
from webrag.output import BaseExporter
from webrag.crawler import BaseCrawler
from webrag.utils.exceptions import (
    PipelineError,
    ConfigurationError,
    FetchError,
    ExtractionError,
    ChunkingError,
    ExportError,
)

logger = logging.getLogger(__name__)


class WebRAG:
    """
    Main orchestrator class for the WebRAG pipeline.

    This class provides a simple, chainable API for ingesting websites
    into RAG-ready document chunks.

    Example:
        >>> rag = WebRAG("sources.json")
        >>> rag.build()
        >>> rag.export(format="json")
        >>> rag.save("output/results.json")
    """

    def __init__(
        self,
        sources: Union[str, Path, List[dict], List[SourceProfile]],
        fetcher: Optional[BaseFetcher] = None,
        extractor: Optional[BaseExtractor] = None,
        chunker: Optional[BaseChunker] = None,
        exporter: Optional[BaseExporter] = None,
        crawler: Optional[BaseCrawler] = None,
        enable_crawling: bool = True,
        fail_fast: bool = False,
        verbose: bool = False,
    ):
        """
        Initialize the WebRAG orchestrator.

        Args:
            sources: Can be:
                - Path to JSON file (str or Path)
                - List of dictionaries representing sources
                - List of SourceProfile objects
            fetcher: Custom fetcher instance (if None, uses default)
            extractor: Custom extractor instance (if None, uses default)
            chunker: Custom chunker instance (if None, uses default)
            exporter: Custom exporter instance (if None, uses default)
            crawler: Custom crawler instance (if None, uses default when enable_crawling=True)
            enable_crawling: Enable link discovery and multi-page crawling (default: True)
            fail_fast: If True, stop on first error; if False, collect errors and continue
            verbose: Enable verbose logging

        Raises:
            InvalidSourceError: If sources cannot be loaded or validated
            ConfigurationError: If configuration is invalid
        """
        # Configure logging
        if verbose:
            logging.basicConfig(level=logging.INFO)
        else:
            logging.basicConfig(level=logging.WARNING)

        # Load and validate sources
        if isinstance(sources, list) and all(isinstance(s, SourceProfile) for s in sources):
            self.sources = sources
        else:
            self.sources = SourceLoader.load(sources)

        logger.info(f"Loaded {len(self.sources)} source(s)")

        # Store configuration
        self.fail_fast = fail_fast
        self.verbose = verbose
        self.enable_crawling = enable_crawling

        # Initialize pipeline components
        self.fetcher = fetcher or self._get_default_fetcher()
        self.extractor = extractor or self._get_default_extractor()
        self.chunker = chunker or self._get_default_chunker()
        self.exporter = exporter or self._get_default_exporter()
        self.crawler = crawler or (self._get_default_crawler() if enable_crawling else None)

        # Pipeline state
        self.result: Optional[PipelineResult] = None
        self.chunks: List[DocumentChunk] = []
        self._extraction_results: List[ExtractionResult] = []
        self._document_groups: List[DocumentGroup] = []
        self._errors: List[str] = []
        self._warnings: List[str] = []
        self._started_at: Optional[datetime] = None
        self._completed_at: Optional[datetime] = None

    def _get_default_fetcher(self) -> BaseFetcher:
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

    def _get_default_extractor(self) -> BaseExtractor:
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

    def _get_default_chunker(self) -> BaseChunker:
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

    def _get_default_exporter(self) -> BaseExporter:
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

    def _get_default_crawler(self) -> BaseCrawler:
        """Get default crawler implementation."""
        try:
            from webrag.crawler.simple_crawler import SimpleCrawler
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

    def build(self) -> "WebRAG":
        """
        Execute the main pipeline with optional crawling.

        Pipeline flow:
        1. Fetch source pages
        2. [If crawling enabled] Discover links and fetch discovered pages
        3. [If crawling enabled] Group related pages
        4. Extract content from all pages
        5. Chunk content with group associations

        Returns:
            self (for method chaining)

        Raises:
            PipelineError: If fail_fast=True and any step fails
        """
        logger.info("Starting WebRAG pipeline...")
        self._started_at = datetime.utcnow()

        # Reset state
        self.chunks = []
        self._extraction_results = []
        self._document_groups = []
        self._errors = []
        self._warnings = []

        sources_processed = 0

        for i, source in enumerate(self.sources):
            logger.info(f"Processing source {i+1}/{len(self.sources)}: {source.url}")

            try:
                if self.enable_crawling and self.crawler and source.crawl_settings.max_depth > 0:
                    # CRAWLING MODE: Multi-page processing
                    chunks = self._process_source_with_crawling(source)
                else:
                    # SIMPLE MODE: Single page processing
                    chunks = self._process_source_simple(source)

                self.chunks.extend(chunks)
                sources_processed += 1

                logger.info(
                    f"Successfully processed {source.url}: "
                    f"created {len(chunks)} chunk(s)"
                )

            except Exception as e:
                error_msg = f"Error processing {source.url}: {str(e)}"
                logger.error(error_msg)
                self._errors.append(error_msg)

                if self.fail_fast:
                    raise PipelineError(error_msg) from e

        self._completed_at = datetime.utcnow()

        # Create pipeline result
        duration = (self._completed_at - self._started_at).total_seconds()
        self.result = PipelineResult(
            documents=self.chunks,
            total_sources_processed=sources_processed,
            total_chunks_created=len(self.chunks),
            errors=self._errors if self._errors else None,
            warnings=self._warnings if self._warnings else None,
            started_at=self._started_at,
            completed_at=self._completed_at,
            duration_seconds=duration,
            metadata={
                "total_sources": len(self.sources),
                "success_rate": sources_processed / len(self.sources) if self.sources else 0,
                "crawling_enabled": self.enable_crawling,
                "document_groups_created": len(self._document_groups),
            }
        )

        logger.info(
            f"Pipeline complete: {sources_processed}/{len(self.sources)} sources processed, "
            f"{len(self.chunks)} chunks created in {duration:.2f}s"
        )

        return self

    def _fetch_source(self, source: SourceProfile) -> Dict[str, Any]:
        """
        Fetch content from a source.

        Args:
            source: SourceProfile to fetch

        Returns:
            Fetch result dictionary

        Raises:
            FetchError: If fetching fails
        """
        try:
            return self.fetcher.fetch(source)
        except Exception as e:
            raise FetchError(f"Failed to fetch {source.url}: {str(e)}") from e

    def _extract_content(
        self,
        fetch_result: Dict[str, Any],
        source: SourceProfile
    ) -> ExtractionResult:
        """
        Extract meaningful content from fetched data.

        Args:
            fetch_result: Result from fetcher
            source: Original source profile

        Returns:
            ExtractionResult object

        Raises:
            ExtractionError: If extraction fails
        """
        try:
            raw_content = fetch_result.get('content', '')
            metadata = {
                'status_code': fetch_result.get('status_code'),
                'fetch_time_ms': fetch_result.get('fetch_time_ms'),
                'headers': fetch_result.get('headers'),
            }

            return self.extractor.extract(raw_content, source, metadata)
        except Exception as e:
            raise ExtractionError(
                f"Failed to extract content from {source.url}: {str(e)}"
            ) from e

    def _chunk_content(self, extraction_result: ExtractionResult) -> List[DocumentChunk]:
        """
        Chunk extracted content into RAG-ready pieces.

        Args:
            extraction_result: Result from extractor

        Returns:
            List of DocumentChunk objects

        Raises:
            ChunkingError: If chunking fails
        """
        try:
            return self.chunker.chunk(extraction_result)
        except Exception as e:
            raise ChunkingError(
                f"Failed to chunk content from {extraction_result.url}: {str(e)}"
            ) from e

    def _process_source_simple(self, source: SourceProfile) -> List[DocumentChunk]:
        """
        Process a single source without crawling (simple mode).

        Args:
            source: SourceProfile to process

        Returns:
            List of DocumentChunk objects
        """
        # Step 1: Fetch
        fetch_result = self._fetch_source(source)

        # Step 2: Extract
        extraction_result = self._extract_content(fetch_result, source)
        self._extraction_results.append(extraction_result)

        # Step 3: Chunk
        chunks = self._chunk_content(extraction_result)

        return chunks

    def _process_source_with_crawling(self, source: SourceProfile) -> List[DocumentChunk]:
        """
        Process a source with link discovery and multi-page crawling.

        Pipeline:
        1. Fetch source page
        2. Discover links using crawler
        3. Fetch discovered pages
        4. Group related pages
        5. Extract content from all pages
        6. Chunk with group associations

        Args:
            source: SourceProfile to process

        Returns:
            List of DocumentChunk objects
        """
        all_chunks = []

        # Step 1: Fetch source page
        logger.info(f"  [1/6] Fetching source page: {source.url}")
        source_fetch_result = self._fetch_source(source)
        self.crawler.mark_visited(str(source.url))

        # Step 2: Discover links
        logger.info(f"  [2/6] Discovering links from source page...")
        discovered_links = self.crawler.discover_links(
            source_fetch_result['content'],
            str(source.url),
            source
        )
        logger.info(f"  Found {len(discovered_links)} link(s) to crawl")

        # Step 3: Fetch discovered pages
        logger.info(f"  [3/6] Fetching discovered pages...")
        discovered_fetch_results = {}
        for link in discovered_links[:source.crawl_settings.max_pages or len(discovered_links)]:
            try:
                # Create temporary source profile for discovered page
                temp_source = SourceProfile(
                    url=link,
                    type=source.type,
                    crawl_settings=source.crawl_settings,
                    enabled=source.enabled
                )
                fetch_result = self._fetch_source(temp_source)
                discovered_fetch_results[link] = fetch_result
                self.crawler.mark_visited(link)
            except Exception as e:
                logger.warning(f"  Failed to fetch {link}: {e}")
                continue

        # Step 4: Group related pages
        logger.info(f"  [4/6] Grouping related pages...")
        all_urls = [str(source.url)] + list(discovered_fetch_results.keys())
        document_groups = self.crawler.group_related_pages(all_urls)
        self._document_groups.extend(document_groups)
        logger.info(f"  Created {len(document_groups)} document group(s)")

        # Step 5: Extract content from all pages
        logger.info(f"  [5/6] Extracting content from all pages...")
        extractions_by_url = {}

        # Extract source page
        try:
            source_extraction = self._extract_content(source_fetch_result, source)
            extractions_by_url[str(source.url)] = source_extraction
            self._extraction_results.append(source_extraction)
        except Exception as e:
            logger.warning(f"  Failed to extract from {source.url}: {e}")

        # Extract discovered pages
        for url, fetch_result in discovered_fetch_results.items():
            try:
                temp_source = SourceProfile(url=url, type=source.type, enabled=source.enabled)
                extraction = self._extract_content(fetch_result, temp_source)
                extractions_by_url[url] = extraction
                self._extraction_results.append(extraction)
            except Exception as e:
                logger.warning(f"  Failed to extract from {url}: {e}")
                continue

        # Step 6: Chunk with group associations
        logger.info(f"  [6/6] Chunking content with group associations...")

        if document_groups:
            # Process grouped pages
            for group in document_groups:
                group_chunks = self._chunk_document_group(group, extractions_by_url)
                all_chunks.extend(group_chunks)

            # Process ungrouped pages (those not in any group)
            grouped_urls = set()
            for group in document_groups:
                grouped_urls.update(str(url) for url in group.page_urls)

            for url, extraction in extractions_by_url.items():
                if url not in grouped_urls:
                    try:
                        chunks = self._chunk_content(extraction)
                        all_chunks.extend(chunks)
                    except Exception as e:
                        logger.warning(f"  Failed to chunk {url}: {e}")
        else:
            # No groups created, process all individually
            for extraction in extractions_by_url.values():
                try:
                    chunks = self._chunk_content(extraction)
                    all_chunks.extend(chunks)
                except Exception as e:
                    logger.warning(f"  Failed to chunk content: {e}")

        return all_chunks

    def _chunk_document_group(
        self,
        group: DocumentGroup,
        extractions_by_url: Dict[str, ExtractionResult]
    ) -> List[DocumentChunk]:
        """
        Chunk a document group, maintaining group associations.

        Args:
            group: DocumentGroup to chunk
            extractions_by_url: Map of URL to ExtractionResult

        Returns:
            List of DocumentChunk objects with group information
        """
        group_chunks = []

        for page_index, page_url in enumerate(group.page_urls):
            url_str = str(page_url)

            if url_str not in extractions_by_url:
                continue

            extraction = extractions_by_url[url_str]

            try:
                # Chunk this page
                page_chunks = self._chunk_content(extraction)

                # Add group information to each chunk
                for chunk in page_chunks:
                    chunk.document_group_id = group.group_id
                    chunk.page_url = page_url
                    chunk.page_index = page_index
                    chunk.total_pages = len(group.page_urls)

                group_chunks.extend(page_chunks)

            except Exception as e:
                logger.warning(f"  Failed to chunk page {url_str} in group {group.group_id}: {e}")
                continue

        return group_chunks

    def export(self, format: str = "json", **kwargs) -> List[Dict[str, Any]]:
        """
        Export chunks to specified format and return as list of dicts.

        Args:
            format: Export format (currently only "json" supported)
            **kwargs: Additional export parameters

        Returns:
            List of chunk dictionaries

        Raises:
            PipelineError: If build() hasn't been called yet
            ExportError: If export fails
        """
        if not self.chunks:
            raise PipelineError(
                "No chunks available. Please call build() first."
            )

        # For now, just serialize to dicts
        # In the future, this will use different exporters based on format
        try:
            return [chunk.model_dump(mode='python') for chunk in self.chunks]
        except Exception as e:
            raise ExportError(f"Failed to export chunks: {str(e)}") from e

    def save(self, path: Union[str, Path], format: str = "json", **kwargs) -> str:
        """
        Save chunks to disk.

        Args:
            path: Output file path
            format: Export format
            **kwargs: Additional export parameters

        Returns:
            Path to saved file

        Raises:
            PipelineError: If build() hasn't been called yet
            ExportError: If saving fails
        """
        if not self.chunks:
            raise PipelineError(
                "No chunks available. Please call build() first."
            )

        try:
            # Use the exporter to save
            if self.result:
                output_path = self.exporter.export_pipeline_result(
                    self.result,
                    str(path),
                    **kwargs
                )
            else:
                output_path = self.exporter.export(
                    self.chunks,
                    str(path),
                    **kwargs
                )

            logger.info(f"Saved {len(self.chunks)} chunks to {output_path}")
            return output_path

        except Exception as e:
            raise ExportError(f"Failed to save to {path}: {str(e)}") from e

    def get_result(self) -> Optional[PipelineResult]:
        """
        Get the complete pipeline result.

        Returns:
            PipelineResult if build() has been called, None otherwise
        """
        return self.result

    def get_chunks(self) -> List[DocumentChunk]:
        """
        Get all generated chunks.

        Returns:
            List of DocumentChunk objects
        """
        return self.chunks

    def get_errors(self) -> List[str]:
        """
        Get all errors encountered during pipeline execution.

        Returns:
            List of error messages
        """
        return self._errors

    def get_warnings(self) -> List[str]:
        """
        Get all warnings from pipeline execution.

        Returns:
            List of warning messages
        """
        return self._warnings

    def filter_chunks(
        self,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        tags: Optional[List[str]] = None,
        **kwargs
    ) -> List[DocumentChunk]:
        """
        Filter chunks based on criteria.

        Args:
            min_length: Minimum content length
            max_length: Maximum content length
            tags: Filter by tags (chunks must have at least one matching tag)
            **kwargs: Additional filter criteria

        Returns:
            Filtered list of chunks
        """
        filtered = self.chunks

        if min_length is not None:
            filtered = [c for c in filtered if len(c.content) >= min_length]

        if max_length is not None:
            filtered = [c for c in filtered if len(c.content) <= max_length]

        if tags:
            filtered = [
                c for c in filtered
                if c.tags and any(tag in c.tags for tag in tags)
            ]

        return filtered

    def schedule_updates(self):
        """
        Schedule periodic updates for sources.

        This is a placeholder for future implementation.
        """
        raise NotImplementedError(
            "Scheduling is not yet implemented. "
            "This feature is planned for a future release."
        )

    def __repr__(self) -> str:
        """String representation of WebRAG instance."""
        return (
            f"WebRAG(sources={len(self.sources)}, "
            f"chunks={len(self.chunks)}, "
            f"errors={len(self._errors)})"
        )

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        if hasattr(self.fetcher, 'close'):
            self.fetcher.close()
