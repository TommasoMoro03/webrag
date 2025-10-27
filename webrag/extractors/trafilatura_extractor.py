"""Content extractor using the trafilatura library."""

from typing import Dict, Any, Optional, List
from datetime import datetime
import trafilatura
from trafilatura.settings import use_config

from webrag.extractors.base import BaseExtractor
from webrag.schemas.document import ExtractionResult
from webrag.schemas.source_profile import SourceProfile
from webrag.utils.exceptions import ExtractionError, ContentNotFoundError, ParsingError


class TrafilaturaExtractor(BaseExtractor):
    """
    Content extractor using trafilatura library.

    Trafilatura is a Python library for extracting text from web pages.
    It's optimized for extracting main content while removing boilerplate.
    """

    def __init__(self, **kwargs):
        """
        Initialize the trafilatura extractor.

        Args:
            **kwargs: Configuration options
                - include_images: Extract images (default: True)
                - include_links: Extract links (default: True)
                - include_tables: Extract tables (default: True)
                - favor_precision: Favor precision over recall (default: False)
        """
        super().__init__(**kwargs)
        self.include_images = kwargs.get('include_images', True)
        self.include_links = kwargs.get('include_links', True)
        self.include_tables = kwargs.get('include_tables', True)
        self.favor_precision = kwargs.get('favor_precision', False)

        # Configure trafilatura
        self.config = use_config()
        if self.favor_precision:
            self.config.set("DEFAULT", "MIN_EXTRACTED_SIZE", "500")

    def extract(
        self,
        raw_content: str,
        source: SourceProfile,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ExtractionResult:
        """
        Extract meaningful content from raw HTML.

        Args:
            raw_content: Raw HTML content
            source: SourceProfile with hints and configuration
            metadata: Optional metadata from fetch stage

        Returns:
            ExtractionResult containing extracted content and metadata

        Raises:
            ExtractionError: If extraction fails
            ContentNotFoundError: If no content can be extracted
            ParsingError: If HTML parsing fails
        """
        if not raw_content or not isinstance(raw_content, str):
            raise ExtractionError("Invalid raw content provided")

        try:
            # Extract main content
            extracted_text = trafilatura.extract(
                raw_content,
                include_tables=self.include_tables,
                include_images=self.include_images,
                include_links=self.include_links,
                config=self.config,
                favor_precision=self.favor_precision,
                url=str(source.url),
            )

            if not extracted_text:
                raise ContentNotFoundError(
                    f"No content could be extracted from {source.url}"
                )

            # Validate content
            if not self.validate_content(extracted_text):
                raise ContentNotFoundError(
                    f"Extracted content from {source.url} is too short or invalid"
                )

            # Extract metadata
            extracted_metadata = self.extract_metadata(raw_content)

            # Extract links if needed
            links = None
            if self.include_links:
                links = self.extract_links(raw_content, str(source.url))

            # Calculate confidence score
            confidence = self.calculate_confidence_score(raw_content, extracted_text)

            # Merge metadata from different sources
            final_metadata = extracted_metadata.copy()
            if metadata:
                final_metadata.update({
                    k: v for k, v in metadata.items()
                    if v is not None and k not in final_metadata
                })

            # Add source metadata if available
            if source.metadata:
                final_metadata.update(source.metadata)

            # Create extraction result
            result = ExtractionResult(
                url=source.url,
                title=extracted_metadata.get('title'),
                content=self.clean_text(extracted_text),
                raw_html=raw_content if len(raw_content) < 100000 else None,  # Limit raw HTML storage
                metadata=final_metadata,
                links=links,
                language=extracted_metadata.get('language'),
                extraction_method='trafilatura',
                extracted_at=datetime.utcnow(),
                confidence_score=confidence,
                status_code=metadata.get('status_code') if metadata else None,
                fetch_time_ms=metadata.get('fetch_time_ms') if metadata else None,
            )

            return result

        except ContentNotFoundError:
            raise
        except Exception as e:
            raise ParsingError(
                f"Failed to parse HTML from {source.url}: {str(e)}"
            ) from e

    def extract_metadata(self, raw_content: str) -> Dict[str, Any]:
        """
        Extract metadata from raw HTML content.

        Args:
            raw_content: Raw HTML content

        Returns:
            Dictionary of extracted metadata
        """
        metadata = {}

        try:
            # Use trafilatura's metadata extraction
            meta = trafilatura.extract_metadata(raw_content)

            if meta:
                metadata['title'] = meta.title
                metadata['author'] = meta.author
                metadata['description'] = meta.description
                metadata['sitename'] = meta.sitename
                metadata['date'] = meta.date
                metadata['categories'] = meta.categories
                metadata['tags'] = meta.tags
                metadata['language'] = meta.language
                metadata['canonical_url'] = meta.url

                # Remove None values
                metadata = {k: v for k, v in metadata.items() if v is not None}

        except Exception:
            # If metadata extraction fails, return empty dict
            pass

        return metadata

    def extract_links(self, raw_content: str, base_url: str) -> List[str]:
        """
        Extract links from HTML content.

        Args:
            raw_content: Raw HTML content
            base_url: Base URL for resolving relative links

        Returns:
            List of absolute URLs
        """
        links = []

        try:
            from urllib.parse import urljoin
            from lxml import html as lxml_html

            # Parse HTML
            tree = lxml_html.fromstring(raw_content)

            # Extract all href attributes
            for element in tree.xpath('//a[@href]'):
                href = element.get('href')
                if href:
                    # Convert to absolute URL
                    absolute_url = urljoin(base_url, href)
                    if absolute_url.startswith('http'):
                        links.append(absolute_url)

            # Remove duplicates while preserving order
            links = list(dict.fromkeys(links))

        except Exception:
            # If link extraction fails, return empty list
            pass

        return links
