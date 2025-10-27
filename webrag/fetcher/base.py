"""Abstract base class for content fetchers."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from webrag.schemas.source_profile import SourceProfile


class BaseFetcher(ABC):
    """
    Abstract base class for fetching web content.

    Fetchers are responsible for retrieving raw content from URLs,
    handling HTTP requests, retries, rate limiting, and returning
    the raw response data.

    Future implementations may include:
    - StaticFetcher (requests-based)
    - DynamicFetcher (Selenium/Playwright for JS-heavy sites)
    - APIFetcher (for REST/GraphQL APIs)
    - CachedFetcher (with local caching)
    """

    def __init__(self, timeout: int = 30, max_retries: int = 3, **kwargs):
        """
        Initialize the fetcher.

        Args:
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            **kwargs: Additional fetcher-specific configuration
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.config = kwargs

    @abstractmethod
    def fetch(self, source: SourceProfile) -> Dict[str, Any]:
        """
        Fetch content from a source URL.

        Args:
            source: SourceProfile containing URL and fetch configuration

        Returns:
            Dict containing:
                - 'content': Raw HTML/text content (str)
                - 'status_code': HTTP status code (int)
                - 'headers': Response headers (dict)
                - 'url': Final URL after redirects (str)
                - 'fetch_time_ms': Time taken to fetch (int)
                - 'encoding': Content encoding (str, optional)

        Raises:
            URLFetchError: If fetching fails
            TimeoutError: If request times out
            RateLimitError: If rate limited
        """
        raise NotImplementedError

    @abstractmethod
    def fetch_multiple(self, sources: list[SourceProfile]) -> list[Dict[str, Any]]:
        """
        Fetch content from multiple sources.

        May implement parallel/concurrent fetching for efficiency.

        Args:
            sources: List of SourceProfile objects

        Returns:
            List of fetch results (same format as fetch())

        Raises:
            Same exceptions as fetch()
        """
        raise NotImplementedError

    @abstractmethod
    def validate_url(self, url: str) -> bool:
        """
        Validate if a URL can be fetched by this fetcher.

        Args:
            url: URL to validate

        Returns:
            True if URL is valid and fetchable, False otherwise
        """
        raise NotImplementedError

    def set_rate_limit(self, delay: float) -> None:
        """
        Set rate limiting delay between requests.

        Args:
            delay: Delay in seconds between requests
        """
        self.rate_limit_delay = delay

    def set_headers(self, headers: Dict[str, str]) -> None:
        """
        Set custom HTTP headers for requests.

        Args:
            headers: Dictionary of HTTP headers
        """
        self.custom_headers = headers

    def close(self) -> None:
        """
        Clean up resources (close sessions, connections, etc.).

        Should be called when fetcher is no longer needed.
        """
        pass  # Default implementation does nothing

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
