"""Abstract base class for URL relationship analysis."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from webrag.schemas.document import DocumentGroup


class BaseURLRelationshipAnalyzer(ABC):
    """
    Abstract base class for analyzing relationships between URLs.

    This component sits between crawler and extraction, analyzing the
    URLs discovered during crawling to identify relationships such as:
    - Multi-page articles (same content split across pages)
    - Article series (related topics)
    - Documentation sections (related docs)
    - Any other logical groupings

    Pipeline position: Fetcher → Crawler → **URL Relationships** → Scraper → Content Enricher → Chunker
    """

    def __init__(self, **kwargs):
        """
        Initialize the URL relationship analyzer.

        Args:
            **kwargs: Analyzer-specific configuration
        """
        self.config = kwargs

    @abstractmethod
    def analyze_relationships(
        self,
        urls: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[DocumentGroup]:
        """
        Analyze URLs to identify relationships and create document groups.

        Args:
            urls: List of URLs to analyze
            metadata: Optional metadata about the URLs (e.g., titles, extracted info)

        Returns:
            List of DocumentGroup objects representing related URLs.
            URLs not in any group should not be returned.
        """
        raise NotImplementedError

    def get_grouped_urls(self, groups: List[DocumentGroup]) -> set:
        """
        Get all URLs that are part of any group.

        Args:
            groups: List of DocumentGroup objects

        Returns:
            Set of URLs that are part of groups
        """
        grouped_urls = set()
        for group in groups:
            grouped_urls.update(str(url) for url in group.page_urls)
        return grouped_urls

    def get_ungrouped_urls(
        self,
        all_urls: List[str],
        groups: List[DocumentGroup]
    ) -> List[str]:
        """
        Get URLs that are not part of any group.

        Args:
            all_urls: All URLs being analyzed
            groups: List of DocumentGroup objects

        Returns:
            List of URLs not in any group
        """
        grouped_urls = self.get_grouped_urls(groups)
        return [url for url in all_urls if url not in grouped_urls]
