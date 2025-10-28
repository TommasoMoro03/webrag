"""Simple crawler implementation for discovering and following links."""

from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
from lxml import html as lxml_html
import hashlib

from webrag.crawler.base import BaseCrawler
from webrag.schemas.source_profile import SourceProfile
from webrag.schemas.document import DocumentGroup


class SimpleCrawler(BaseCrawler):
    """
    Simple crawler that discovers links and groups related pages.

    This implementation:
    - Extracts all <a> tags from HTML
    - Filters links based on crawl settings
    - Groups pages by URL patterns (e.g., /article/title and /article/title/page-2)
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

    def group_related_pages(
        self,
        urls: List[str],
        metadata: Optional[Dict[str, List[Dict[str, Any]]]] = None
    ) -> List[DocumentGroup]:
        """
        Group related URLs into DocumentGroup objects.

        This implementation uses URL pattern matching to detect:
        - Multi-page articles (e.g., /article/page-1, /article/page-2)
        - Article series with similar paths

        Args:
            urls: List of URLs to analyze
            metadata: Optional extracted metadata for each URL

        Returns:
            List of DocumentGroup objects
        """
        groups = []
        processed_urls = set()

        for url in urls:
            if url in processed_urls:
                continue

            # Find related URLs
            related = self._find_related_urls(url, urls)

            if related:
                # Create a document group
                group = self._create_document_group(url, related, metadata)
                groups.append(group)

                # Mark as processed
                processed_urls.add(url)
                processed_urls.update(related)

        return groups

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

    def _find_related_urls(self, base_url: str, all_urls: List[str]) -> List[str]:
        """
        Find URLs that are related to base_url.

        Looks for patterns like:
        - /article/title and /article/title/page-2
        - /blog/post-1 and /blog/post-2

        Args:
            base_url: URL to find relations for
            all_urls: All available URLs

        Returns:
            List of related URLs
        """
        related = []
        base_path = urlparse(base_url).path

        for url in all_urls:
            if url == base_url:
                continue

            url_path = urlparse(url).path

            # Check if paths are similar
            if self._are_paths_related(base_path, url_path):
                related.append(url)

        return related

    def _are_paths_related(self, path1: str, path2: str) -> bool:
        """
        Check if two URL paths are related.

        Args:
            path1: First path
            path2: Second path

        Returns:
            True if related
        """
        # Remove trailing slashes
        path1 = path1.rstrip('/')
        path2 = path2.rstrip('/')

        # Check if one is a subpath of the other
        if path2.startswith(path1 + '/') or path1.startswith(path2 + '/'):
            return True

        # Check for pagination patterns (/page-1, /page-2, etc.)
        import re
        page_pattern = r'/page[-_]?\d+$'

        path1_base = re.sub(page_pattern, '', path1)
        path2_base = re.sub(page_pattern, '', path2)

        if path1_base == path2_base and path1_base:
            return True

        return False

    def _create_document_group(
        self,
        primary_url: str,
        related_urls: List[str],
        metadata: Optional[Dict[str, List[Dict[str, Any]]]] = None
    ) -> DocumentGroup:
        """
        Create a DocumentGroup from a primary URL and its related URLs.

        Args:
            primary_url: Main URL for the group
            related_urls: Related URLs
            metadata: Optional metadata

        Returns:
            DocumentGroup object
        """
        all_urls = [primary_url] + related_urls

        # Generate group ID from primary URL
        group_id = self._generate_group_id(primary_url)

        # Extract title from metadata if available
        title = None
        if metadata and primary_url in metadata:
            url_metadata = metadata[primary_url]
            if url_metadata and isinstance(url_metadata, list) and url_metadata:
                title = url_metadata[0].get('title')

        return DocumentGroup(
            group_id=group_id,
            source_url=primary_url,
            page_urls=all_urls,
            title=title,
            relationship_type="multi_page_article",
            metadata={}
        )

    def _generate_group_id(self, url: str) -> str:
        """
        Generate a unique group ID from URL.

        Args:
            url: URL to generate ID from

        Returns:
            Group ID string
        """
        # Use hash of URL path as ID
        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
        return f"group_{url_hash}"
