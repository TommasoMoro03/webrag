"""Abstract base class for web crawlers."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Set
from urllib.parse import urljoin, urlparse
import re
from pydantic import BaseModel

from webrag.schemas.source_profile import SourceProfile


class CrawlResult(BaseModel):
    """Result of a crawl operation."""
    discovered_urls: List[str]
    metadata: Dict[str, Any]


class BaseCrawler(ABC):
    """
    Abstract base class for crawling and link discovery.

    The crawler is responsible for:
    1. Analyzing fetched content to discover links
    2. Deciding which links are relevant to follow
    3. Respecting crawl settings (depth, patterns, robots.txt)

    The crawler works AFTER fetch but BEFORE URL relationship analysis:
    Fetch → Crawl/Discover → Fetch discovered pages → URL Relationships → Extract all → Chunk
    """

    def __init__(self, **kwargs):
        """
        Initialize the crawler.

        Args:
            **kwargs: Crawler-specific configuration
        """
        self.config = kwargs
        self.visited_urls: Set[str] = set()

    @abstractmethod
    def discover_links(
        self,
        raw_content: str,
        source_url: str,
        source: SourceProfile
    ) -> List[str]:
        """
        Discover links from raw HTML content.

        Args:
            raw_content: Raw HTML content
            source_url: URL of the page being analyzed
            source: SourceProfile with crawl settings

        Returns:
            List of discovered URLs (absolute URLs)
        """
        raise NotImplementedError

    @abstractmethod
    def should_follow_link(
        self,
        link_url: str,
        source_url: str,
        source: SourceProfile
    ) -> bool:
        """
        Determine if a link should be followed based on crawl settings.

        Args:
            link_url: The link to evaluate
            source_url: The page containing the link
            source: SourceProfile with crawl settings

        Returns:
            True if link should be followed
        """
        raise NotImplementedError

    def normalize_url(self, url: str, base_url: str) -> str:
        """
        Normalize and convert relative URL to absolute.

        Args:
            url: URL to normalize
            base_url: Base URL for resolving relatives

        Returns:
            Normalized absolute URL
        """
        # Convert to absolute URL
        absolute = urljoin(base_url, url)

        # Remove fragments
        if '#' in absolute:
            absolute = absolute.split('#')[0]

        # Remove trailing slash for consistency
        if absolute.endswith('/') and absolute.count('/') > 3:
            absolute = absolute.rstrip('/')

        return absolute

    def is_same_domain(self, url: str, base_url: str) -> bool:
        """
        Check if URL is from the same domain as base URL.

        Args:
            url: URL to check
            base_url: Base URL to compare against

        Returns:
            True if same domain
        """
        url_domain = urlparse(url).netloc
        base_domain = urlparse(base_url).netloc
        return url_domain == base_domain

    def matches_pattern(self, url: str, patterns: List[str]) -> bool:
        """
        Check if URL matches any of the given regex patterns.

        Args:
            url: URL to check
            patterns: List of regex patterns

        Returns:
            True if URL matches any pattern
        """
        for pattern in patterns:
            if re.search(pattern, url):
                return True
        return False

    def filter_links(
        self,
        links: List[str],
        source: SourceProfile
    ) -> List[str]:
        """
        Filter links based on crawl settings.

        Args:
            links: List of links to filter
            source: SourceProfile with crawl settings

        Returns:
            Filtered list of links
        """
        filtered = []
        source_url = str(source.url)

        for link in links:
            # Skip if already visited
            if link in self.visited_urls:
                continue

            # Check domain restrictions
            if not source.crawl_settings.follow_external_links:
                if not self.is_same_domain(link, source_url):
                    continue

            # Check allowed domains
            if source.crawl_settings.allowed_domains:
                link_domain = urlparse(link).netloc
                if link_domain not in source.crawl_settings.allowed_domains:
                    continue

            # Check exclude patterns
            if source.crawl_settings.exclude_patterns:
                if self.matches_pattern(link, source.crawl_settings.exclude_patterns):
                    continue

            # Check if we should follow this link
            if self.should_follow_link(link, source_url, source):
                filtered.append(link)

        return filtered

    def mark_visited(self, url: str) -> None:
        """Mark a URL as visited to avoid re-crawling."""
        self.visited_urls.add(url)

    def reset(self) -> None:
        """Reset crawler state (clear visited URLs)."""
        self.visited_urls.clear()


# Import after class definition to avoid circular imports
from pydantic import BaseModel
