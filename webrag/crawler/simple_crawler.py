"""Simple crawler implementation for discovering and following links."""

from typing import List
from urllib.parse import urlparse
from lxml import html as lxml_html

from webrag.crawler.base import BaseCrawler
from webrag.schemas.source_profile import SourceProfile


class SimpleCrawler(BaseCrawler):
    """
    Simple crawler that discovers links using basic heuristics.

    This implementation:
    - Extracts all <a> tags from HTML
    - Filters links based on crawl settings
    - Uses pattern matching to skip non-content URLs
    """

    def __init__(self, **kwargs):
        """
        Initialize the simple crawler.

        Args:
            **kwargs: Configuration options
                - max_links_per_page: Maximum links to extract per page (default: 100)
                - link_selector: CSS selector for links (default: 'a[href]')
        """
        super().__init__(**kwargs)
        self.max_links_per_page = kwargs.get('max_links_per_page', 100)
        self.link_selector = kwargs.get('link_selector', 'a[href]')

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
        if not raw_content:
            return []

        try:
            # Parse HTML
            tree = lxml_html.fromstring(raw_content)

            # Extract links
            links = []
            for element in tree.xpath('//a[@href]'):
                href = element.get('href')
                if href:
                    # Normalize to absolute URL
                    absolute_url = self.normalize_url(href, source_url)

                    # Basic validation
                    if self._is_valid_url(absolute_url):
                        links.append(absolute_url)

                    # Limit number of links
                    if len(links) >= self.max_links_per_page:
                        break

            # Remove duplicates while preserving order
            links = list(dict.fromkeys(links))

            # Filter based on crawl settings
            filtered_links = self.filter_links(links, source)

            return filtered_links

        except Exception as e:
            # If link extraction fails, return empty list
            return []

    def should_follow_link(
        self,
        link_url: str,
        source_url: str,
        source: SourceProfile
    ) -> bool:
        """
        Determine if a link should be followed.

        Basic heuristics:
        - Skip non-content links (login, search, etc.)
        - Prefer links that look like articles/content
        - Respect max_depth setting

        Args:
            link_url: The link to evaluate
            source_url: The page containing the link
            source: SourceProfile with crawl settings

        Returns:
            True if link should be followed
        """
        # Skip obviously non-content URLs
        skip_patterns = [
            r'/login', r'/logout', r'/signup', r'/register',
            r'/search', r'/tag/', r'/category/',
            r'/feed', r'/rss', r'/api/',
            r'\.(pdf|jpg|jpeg|png|gif|zip|tar|gz)$'
        ]

        for pattern in skip_patterns:
            if self.matches_pattern(link_url, [pattern]):
                return False

        # If max_depth is 0, don't follow any links
        if source.crawl_settings.max_depth == 0:
            return False

        # Check if link is within reasonable depth
        # (count path segments as depth indicator)
        source_depth = len(urlparse(source_url).path.strip('/').split('/'))
        link_depth = len(urlparse(link_url).path.strip('/').split('/'))

        if link_depth > source_depth + source.crawl_settings.max_depth:
            return False

        return True

    def _is_valid_url(self, url: str) -> bool:
        """
        Check if URL is valid and should be considered.

        Args:
            url: URL to validate

        Returns:
            True if valid
        """
        if not url or not isinstance(url, str):
            return False

        # Must be HTTP(S)
        if not (url.startswith('http://') or url.startswith('https://')):
            return False

        # Shouldn't be just a hash/anchor
        if url.startswith('#'):
            return False

        return True
