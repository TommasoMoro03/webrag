"""Static content fetcher using requests library."""

import time
from typing import Dict, Any, List
import requests
from webrag.fetcher.base import BaseFetcher
from webrag.schemas.source_profile import SourceProfile
from webrag.utils.exceptions import URLFetchError, TimeoutError as WebRAGTimeoutError


class StaticFetcher(BaseFetcher):
    """
    Fetcher for static HTML content using the requests library.

    This is suitable for websites that don't require JavaScript execution.
    For JavaScript-heavy sites, use a dynamic fetcher (Selenium/Playwright).
    """

    def __init__(self, timeout: int = 30, max_retries: int = 3, **kwargs):
        """
        Initialize the static fetcher.

        Args:
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            **kwargs: Additional configuration
        """
        super().__init__(timeout, max_retries, **kwargs)
        self.session = requests.Session()
        self.rate_limit_delay = kwargs.get('rate_limit_delay', 1.0)
        self.custom_headers = kwargs.get('headers', {})
        self._last_request_time = 0

    def fetch(self, source: SourceProfile) -> Dict[str, Any]:
        """
        Fetch content from a source URL.

        Args:
            source: SourceProfile containing URL and fetch configuration

        Returns:
            Dict containing:
                - content: Raw HTML content
                - status_code: HTTP status code
                - headers: Response headers
                - url: Final URL after redirects
                - fetch_time_ms: Time taken to fetch
                - encoding: Content encoding

        Raises:
            URLFetchError: If fetching fails
            WebRAGTimeoutError: If request times out
        """
        url = str(source.url)

        # Apply rate limiting
        self._apply_rate_limit(source.crawl_settings.rate_limit_delay)

        # Prepare headers
        headers = self._prepare_headers(source)

        # Attempt fetch with retries
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()

                response = self.session.get(
                    url,
                    headers=headers,
                    timeout=source.crawl_settings.timeout or self.timeout,
                    allow_redirects=True,
                )

                fetch_time_ms = int((time.time() - start_time) * 1000)

                # Check for HTTP errors
                response.raise_for_status()

                return {
                    'content': response.text,
                    'status_code': response.status_code,
                    'headers': dict(response.headers),
                    'url': response.url,
                    'fetch_time_ms': fetch_time_ms,
                    'encoding': response.encoding or 'utf-8',
                }

            except requests.exceptions.Timeout as e:
                if attempt == self.max_retries - 1:
                    raise WebRAGTimeoutError(
                        f"Request to {url} timed out after {self.max_retries} attempts"
                    ) from e

            except requests.exceptions.HTTPError as e:
                raise URLFetchError(
                    url=url,
                    reason=str(e),
                    status_code=response.status_code if 'response' in locals() else None
                ) from e

            except requests.exceptions.RequestException as e:
                if attempt == self.max_retries - 1:
                    raise URLFetchError(
                        url=url,
                        reason=str(e)
                    ) from e

            # Wait before retry
            time.sleep(1.0 * (attempt + 1))

        # Should not reach here, but just in case
        raise URLFetchError(url=url, reason="Max retries exceeded")

    def fetch_multiple(self, sources: List[SourceProfile]) -> List[Dict[str, Any]]:
        """
        Fetch content from multiple sources sequentially.

        Args:
            sources: List of SourceProfile objects

        Returns:
            List of fetch results

        Raises:
            URLFetchError: If any fetch fails
        """
        results = []
        for source in sources:
            result = self.fetch(source)
            results.append(result)
        return results

    def validate_url(self, url: str) -> bool:
        """
        Validate if a URL can be fetched.

        Args:
            url: URL to validate

        Returns:
            True if URL is valid and fetchable
        """
        if not url or not isinstance(url, str):
            return False

        # Basic URL validation
        if not (url.startswith('http://') or url.startswith('https://')):
            return False

        return True

    def _prepare_headers(self, source: SourceProfile) -> Dict[str, str]:
        """
        Prepare HTTP headers for request.

        Args:
            source: SourceProfile with potential custom headers

        Returns:
            Dictionary of HTTP headers
        """
        headers = {
            'User-Agent': (
                source.crawl_settings.user_agent or
                'Mozilla/5.0 (compatible; WebRAG/1.0; +https://github.com/yourusername/webrag)'
            ),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }

        # Merge with custom headers
        if self.custom_headers:
            headers.update(self.custom_headers)

        return headers

    def _apply_rate_limit(self, delay: float = None) -> None:
        """
        Apply rate limiting between requests.

        Args:
            delay: Delay in seconds (uses default if None)
        """
        delay = delay if delay is not None else self.rate_limit_delay

        if delay > 0:
            time_since_last = time.time() - self._last_request_time
            if time_since_last < delay:
                time.sleep(delay - time_since_last)

        self._last_request_time = time.time()

    def close(self) -> None:
        """Close the requests session."""
        if self.session:
            self.session.close()
