"""Heuristic-based URL relationship analyzer using pattern matching."""

from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
import hashlib
import re
import logging

from webrag.url_relationships.base import BaseURLRelationshipAnalyzer
from webrag.schemas.document import DocumentGroup

logger = logging.getLogger(__name__)


class HeuristicURLRelationshipAnalyzer(BaseURLRelationshipAnalyzer):
    """
    Heuristic-based URL relationship analyzer.

    Uses URL pattern matching and structural analysis to identify relationships.
    This is a lightweight, fast alternative to AI-based analysis.

    Detection strategies:
    - Multi-page articles: /article/title/page-1, /article/title/page-2
    - Pagination patterns: /blog/post?page=1, /blog/post?page=2
    - Path similarity: /docs/intro, /docs/getting-started
    - Numbered sequences: /tutorial-1, /tutorial-2
    """

    def __init__(self, **kwargs):
        """
        Initialize the heuristic URL relationship analyzer.

        Args:
            **kwargs: Configuration options
                - min_group_size: Minimum URLs for a group (default: 2)
                - enable_grouping: Enable/disable grouping entirely (default: True)
                - similarity_threshold: How similar paths must be (default: 0.7)
        """
        super().__init__(**kwargs)
        self.min_group_size = kwargs.get('min_group_size', 2)
        self.enable_grouping = kwargs.get('enable_grouping', True)
        self.similarity_threshold = kwargs.get('similarity_threshold', 0.7)

    def analyze_relationships(
        self,
        urls: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[DocumentGroup]:
        """
        Analyze URL relationships using heuristic pattern matching.

        Args:
            urls: List of URLs to analyze
            metadata: Optional metadata about URLs (e.g., titles)

        Returns:
            List of DocumentGroup objects representing related URLs
        """
        if not self.enable_grouping:
            logger.info("Heuristic grouping disabled in configuration")
            return []

        if len(urls) < self.min_group_size:
            logger.info(f"Not enough URLs ({len(urls)}) for grouping (min: {self.min_group_size})")
            return []

        try:
            logger.info(f"Analyzing relationships between {len(urls)} URLs using heuristics")
            groups = self._analyze_patterns(urls, metadata)
            logger.info(f"Heuristic analysis identified {len(groups)} document group(s)")
            return groups
        except Exception as e:
            logger.warning(f"Heuristic relationship analysis failed: {e}")
            return []

    def _analyze_patterns(
        self,
        urls: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[DocumentGroup]:
        """
        Analyze URL patterns to identify groups.

        Args:
            urls: URLs to analyze
            metadata: Optional metadata

        Returns:
            List of DocumentGroup objects
        """
        groups = []
        processed_urls = set()

        for url in urls:
            if url in processed_urls:
                continue

            # Find related URLs using multiple strategies
            related = self._find_related_urls(url, urls, processed_urls)

            if related and len(related) + 1 >= self.min_group_size:
                # Create a document group
                group = self._create_document_group(url, related, metadata)
                groups.append(group)

                # Mark as processed
                processed_urls.add(url)
                processed_urls.update(related)

        return groups

    def _find_related_urls(
        self,
        base_url: str,
        all_urls: List[str],
        processed_urls: set
    ) -> List[str]:
        """
        Find URLs that are related to base_url.

        Looks for patterns like:
        - /article/title and /article/title/page-2
        - /blog/post-1 and /blog/post-2
        - /docs/section-a and /docs/section-b

        Args:
            base_url: URL to find relations for
            all_urls: All available URLs
            processed_urls: URLs already in groups

        Returns:
            List of related URLs
        """
        related = []
        base_path = urlparse(base_url).path

        for url in all_urls:
            if url == base_url or url in processed_urls:
                continue

            url_path = urlparse(url).path

            # Check if paths are related using various strategies
            if self._are_paths_related(base_path, url_path):
                related.append(url)

        return related

    def _are_paths_related(self, path1: str, path2: str) -> bool:
        """
        Check if two URL paths are related using multiple heuristics.

        Args:
            path1: First path
            path2: Second path

        Returns:
            True if related
        """
        # Remove trailing slashes
        path1 = path1.rstrip('/')
        path2 = path2.rstrip('/')

        # Strategy 1: Check if one is a subpath of the other
        # e.g., /article/title and /article/title/page-2
        if path2.startswith(path1 + '/') or path1.startswith(path2 + '/'):
            return True

        # Strategy 2: Check for pagination patterns
        # e.g., /blog/post/page-1, /blog/post/page-2
        page_patterns = [
            r'/page[-_]?\d+$',      # /page-1, /page_2
            r'[?&]page=\d+',         # ?page=1
            r'/p\d+$',               # /p1, /p2
        ]

        for pattern in page_patterns:
            path1_base = re.sub(pattern, '', path1)
            path2_base = re.sub(pattern, '', path2)

            if path1_base == path2_base and path1_base:
                return True

        # Strategy 3: Check for numbered sequences
        # e.g., /tutorial-1, /tutorial-2
        number_pattern = r'[-_]?\d+$'
        path1_base = re.sub(number_pattern, '', path1)
        path2_base = re.sub(number_pattern, '', path2)

        if path1_base == path2_base and path1_base and len(path1_base) > 5:
            # Extract numbers
            match1 = re.search(r'\d+$', path1)
            match2 = re.search(r'\d+$', path2)
            if match1 and match2:
                num1 = int(match1.group())
                num2 = int(match2.group())
                # Numbers should be close (within 10 of each other)
                if abs(num1 - num2) <= 10:
                    return True

        # Strategy 4: Path similarity (shared prefix)
        # e.g., /docs/getting-started and /docs/advanced-usage
        parts1 = [p for p in path1.split('/') if p]
        parts2 = [p for p in path2.split('/') if p]

        if parts1 and parts2 and len(parts1) == len(parts2):
            # Compare parent directory
            if len(parts1) > 1 and parts1[:-1] == parts2[:-1]:
                # Same parent directory, different final component
                # Only group if parent is specific enough
                if len(parts1) >= 2:
                    return True

        return False

    def _create_document_group(
        self,
        primary_url: str,
        related_urls: List[str],
        metadata: Optional[Dict[str, Any]] = None
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

        # Detect relationship type from URL patterns
        relationship_type = self._detect_relationship_type(primary_url, related_urls)

        # Extract title from metadata if available
        title = None
        if metadata and primary_url in metadata:
            url_metadata = metadata[primary_url]
            if isinstance(url_metadata, dict):
                title = url_metadata.get('title')
            elif isinstance(url_metadata, list) and url_metadata:
                title = url_metadata[0].get('title')

        logger.info(
            f"Created group '{group_id}': {len(all_urls)} URLs, "
            f"type: {relationship_type}, title: {title}"
        )

        return DocumentGroup(
            group_id=group_id,
            source_url=primary_url,
            page_urls=all_urls,
            title=title,
            relationship_type=relationship_type,
            metadata={}
        )

    def _detect_relationship_type(
        self,
        primary_url: str,
        related_urls: List[str]
    ) -> str:
        """
        Detect the type of relationship between URLs.

        Args:
            primary_url: Primary URL
            related_urls: Related URLs

        Returns:
            Relationship type string
        """
        primary_path = urlparse(primary_url).path

        # Check for pagination
        page_patterns = [r'/page[-_]?\d+$', r'[?&]page=\d+', r'/p\d+$']
        for pattern in page_patterns:
            if re.search(pattern, primary_path):
                return "multi_page"

        # Check for numbered sequences
        if re.search(r'[-_]?\d+$', primary_path):
            return "series"

        # Default to related
        return "related"

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
        return f"heuristic_group_{url_hash}"
