"""AI-powered crawler using LLM prompts for intelligent link discovery and content navigation."""

from typing import List, Dict, Any, Optional, Callable
from urllib.parse import urlparse
from lxml import html as lxml_html
import hashlib
import json
import re
import logging

from webrag.crawler.base import BaseCrawler
from webrag.schemas.source_profile import SourceProfile

logger = logging.getLogger(__name__)


class AICrawler(BaseCrawler):
    """
    AI-powered crawler that uses LLM prompts instead of heuristics.

    This implementation:
    - Uses AI to analyze HTML and determine which links are relevant
    - Intelligently decides which links to follow based on content understanding
    - Adapts to different website structures without hardcoded rules

    The crawler requires an LLM function to be provided during initialization.
    """

    def __init__(
        self,
        llm_function: Callable[[str], str],
        **kwargs
    ):
        """
        Initialize the AI crawler.

        Args:
            llm_function: A callable that takes a prompt (str) and returns an LLM response (str)
                         Should be a simple function like: lambda prompt: client.chat(prompt)
            **kwargs: Configuration options
                - max_links_per_page: Maximum links to analyze per page (default: 200)
                - min_confidence: Minimum confidence score for following links (default: 0.6)
                - temperature: LLM temperature for creativity (default: 0.3)
                - extraction_selector: CSS selector for content area (default: 'body')
        """
        super().__init__(**kwargs)
        self.llm_function = llm_function
        self.max_links_per_page = kwargs.get('max_links_per_page', 200)
        self.min_confidence = kwargs.get('min_confidence', 0.6)
        self.temperature = kwargs.get('temperature', 0.3)
        self.extraction_selector = kwargs.get('extraction_selector', 'body')

        # Cache for AI responses to avoid redundant calls
        self._response_cache: Dict[str, Any] = {}

    def discover_links(
        self,
        raw_content: str,
        source_url: str,
        source: SourceProfile
    ) -> List[str]:
        """
        Discover links using AI analysis of HTML content.

        Applies ai_validation_strategy from source.crawl_settings:
        - 'always': Validates each URL individually (expensive)
        - 'never': Only uses initial batch filtering
        - 'threshold': Uses individual validation only if URLs <= threshold
        - 'batch': Uses batch filtering (default, most efficient)

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

            # Extract all links with context
            link_contexts = self._extract_links_with_context(tree, source_url)

            if not link_contexts:
                return []

            # Limit to avoid overwhelming the LLM
            link_contexts = link_contexts[:self.max_links_per_page]

            # Apply AI filtering based on strategy
            strategy = source.crawl_settings.ai_validation_strategy
            logger.info(f"Using AI validation strategy: {strategy}")

            if strategy == "never":
                # No AI filtering, just return all links
                relevant_links = [link['url'] for link in link_contexts]
                logger.info(f"Strategy 'never': Skipping AI validation, accepting all {len(relevant_links)} links")

            elif strategy == "batch":
                # Use batch filtering (default, most efficient)
                relevant_links = self._ai_filter_links_batch(
                    link_contexts,
                    source_url,
                    source
                )
                logger.info(f"Strategy 'batch': Filtered to {len(relevant_links)} links")

            elif strategy == "threshold":
                # Use individual validation only if under threshold
                threshold = source.crawl_settings.ai_validation_threshold
                if len(link_contexts) <= threshold:
                    logger.info(f"Strategy 'threshold': {len(link_contexts)} <= {threshold}, using individual validation")
                    relevant_links = self._ai_filter_links_individual(
                        link_contexts,
                        source_url,
                        source
                    )
                else:
                    logger.info(f"Strategy 'threshold': {len(link_contexts)} > {threshold}, using batch filtering")
                    relevant_links = self._ai_filter_links_batch(
                        link_contexts,
                        source_url,
                        source
                    )

            elif strategy == "always":
                # Validate each URL individually (expensive)
                logger.warning(f"Strategy 'always': Validating {len(link_contexts)} links individually (expensive!)")
                relevant_links = self._ai_filter_links_individual(
                    link_contexts,
                    source_url,
                    source
                )

            else:
                # Fallback to batch
                logger.warning(f"Unknown strategy '{strategy}', using 'batch'")
                relevant_links = self._ai_filter_links_batch(
                    link_contexts,
                    source_url,
                    source
                )

            # Apply basic filters from crawl settings
            filtered_links = self.filter_links(relevant_links, source)

            return filtered_links

        except Exception as e:
            logger.error(f"Error in discover_links: {e}")
            # Fallback to empty list if extraction fails
            return []

    def should_follow_link(
        self,
        link_url: str,
        source_url: str,
        source: SourceProfile
    ) -> bool:
        """
        Determine if a link should be followed.

        Note: This method is now mainly for basic validation.
        AI validation happens based on ai_validation_strategy in discover_links().

        Args:
            link_url: The link to evaluate
            source_url: The page containing the link
            source: SourceProfile with crawl settings

        Returns:
            True if link should be followed (basic check only)
        """
        # Check max_depth first (hard constraint)
        if source.crawl_settings.max_depth == 0:
            return False

        # Basic depth check using path segments
        source_depth = len(urlparse(source_url).path.strip('/').split('/'))
        link_depth = len(urlparse(link_url).path.strip('/').split('/'))

        if link_depth > source_depth + source.crawl_settings.max_depth:
            return False

        # For AICrawler, most validation happens in discover_links()
        # This is just a basic sanity check
        return True

    # ========== Private Helper Methods ==========

    def _extract_links_with_context(
        self,
        tree: lxml_html.HtmlElement,
        base_url: str
    ) -> List[Dict[str, str]]:
        """
        Extract links along with their contextual information.

        Args:
            tree: Parsed HTML tree
            base_url: Base URL for resolving relative links

        Returns:
            List of dicts with 'url', 'text', 'context', 'parent_tag'
        """
        link_contexts = []

        for element in tree.xpath('//a[@href]'):
            href = element.get('href')
            if not href:
                continue

            # Normalize to absolute URL
            absolute_url = self.normalize_url(href, base_url)

            if not self._is_valid_url(absolute_url):
                continue

            # Extract link text
            link_text = element.text_content().strip()

            # Extract parent context (e.g., nav, article, aside)
            parent = element.getparent()
            parent_tag = parent.tag if parent is not None else 'unknown'

            # Get some surrounding context
            context = self._get_element_context(element)

            link_contexts.append({
                'url': absolute_url,
                'text': link_text[:200],  # Limit length
                'context': context[:300],  # Limit length
                'parent_tag': parent_tag
            })

        # Remove duplicates while preserving order
        seen_urls = set()
        unique_links = []
        for link in link_contexts:
            if link['url'] not in seen_urls:
                seen_urls.add(link['url'])
                unique_links.append(link)

        return unique_links

    def _get_element_context(self, element: lxml_html.HtmlElement) -> str:
        """
        Get contextual text around an element.

        Args:
            element: HTML element

        Returns:
            Context string
        """
        try:
            # Try to get parent's text content
            parent = element.getparent()
            if parent is not None:
                return parent.text_content().strip()[:300]
            return ""
        except Exception:
            return ""

    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid for crawling."""
        if not url or not isinstance(url, str):
            return False

        if not (url.startswith('http://') or url.startswith('https://')):
            return False

        if url.startswith('#'):
            return False

        # Skip obvious non-content file types
        skip_extensions = ['.pdf', '.jpg', '.jpeg', '.png', '.gif', '.zip', '.tar', '.gz', '.mp4', '.mp3']
        if any(url.lower().endswith(ext) for ext in skip_extensions):
            return False

        return True

    def _ai_filter_links_batch(
        self,
        link_contexts: List[Dict[str, str]],
        source_url: str,
        source: SourceProfile
    ) -> List[str]:
        """
        Use AI to filter and select relevant links in a single batch call.

        This is the most efficient strategy - one AI call for all links.

        Args:
            link_contexts: List of link dictionaries with context
            source_url: Source page URL
            source: SourceProfile

        Returns:
            List of relevant URLs
        """
        # Build prompt for LLM
        prompt = self._build_link_filtering_prompt(
            link_contexts,
            source_url,
            source
        )

        # Get cache key
        cache_key = hashlib.md5(prompt.encode()).hexdigest()

        # Check cache
        if cache_key in self._response_cache:
            response = self._response_cache[cache_key]
        else:
            # Call LLM once for all links
            logger.info(f"AI batch filtering {len(link_contexts)} links in one call")
            response = self.llm_function(prompt)
            self._response_cache[cache_key] = response

        # Parse response to extract URLs
        relevant_urls = self._parse_link_selection_response(response, link_contexts)

        return relevant_urls

    def _ai_filter_links_individual(
        self,
        link_contexts: List[Dict[str, str]],
        source_url: str,
        source: SourceProfile
    ) -> List[str]:
        """
        Use AI to validate each link individually.

        WARNING: This makes one AI call PER LINK - expensive!
        Only use with small numbers of links.

        Args:
            link_contexts: List of link dictionaries with context
            source_url: Source page URL
            source: SourceProfile

        Returns:
            List of relevant URLs
        """
        relevant_urls = []

        for i, link_ctx in enumerate(link_contexts, 1):
            logger.info(f"AI validating link {i}/{len(link_contexts)}: {link_ctx['url']}")

            try:
                confidence = self._ai_evaluate_link_relevance(
                    link_ctx['url'],
                    source_url,
                    source
                )

                if confidence >= self.min_confidence:
                    relevant_urls.append(link_ctx['url'])
                    logger.info(f"  ✓ Accepted (confidence: {confidence:.2f})")
                else:
                    logger.info(f"  ✗ Rejected (confidence: {confidence:.2f})")

            except Exception as e:
                logger.warning(f"  ✗ Error validating link: {e}")
                continue

        return relevant_urls

    def _build_link_filtering_prompt(
        self,
        link_contexts: List[Dict[str, str]],
        source_url: str,
        source: SourceProfile
    ) -> str:
        """Build prompt for link filtering."""

        # Format links for the prompt
        links_text = ""
        for i, link in enumerate(link_contexts):
            links_text += f"{i}. [{link['text'][:100]}] {link['url']}\n"
            links_text += f"   Context: {link['context'][:150]}\n"
            links_text += f"   Parent tag: {link['parent_tag']}\n\n"

        print("LINKS TEXT:", links_text)

        prompt = f"""You are analyzing a webpage to identify relevant content links to crawl.

Source URL: {source_url}
Purpose: Extract relevant content for a knowledge base / RAG system

Links found on the page:
{links_text}

Task: Identify which links are most relevant for content extraction. Focus on:
- Main content pages (articles, documentation, blog posts, guides)
- Avoid navigation links (home, about, contact, login)
- Avoid utility pages (search, RSS feeds, social media)
- Avoid links to media files or downloads
- Avoid links that suggest pages with categories or tags, not specific readable content
- Avoid pagination or duplicate content
- In general, avoid links to external domains unless clearly relevant
- Prioritize links that seem to contain substantial textual content, useful for RAG systems

Respond with ONLY a JSON array of link indices (0-based) that should be followed.
Example: [0, 2, 5, 7]

Response:"""

        return prompt

    def _parse_link_selection_response(
        self,
        response: str,
        link_contexts: List[Dict[str, str]]
    ) -> List[str]:
        """
        Parse LLM response to extract selected URLs.

        Args:
            response: LLM response
            link_contexts: Original link contexts

        Returns:
            List of selected URLs
        """
        try:
            # Extract JSON array from response
            json_match = re.search(r'\[[\d,\s]+\]', response)
            if not json_match:
                return []

            indices = json.loads(json_match.group())

            # Extract URLs at those indices
            selected_urls = []
            for idx in indices:
                if 0 <= idx < len(link_contexts):
                    selected_urls.append(link_contexts[idx]['url'])

            return selected_urls
        except Exception:
            return []

    def _ai_evaluate_link_relevance(
        self,
        link_url: str,
        source_url: str,
        source: SourceProfile
    ) -> float:
        """
        Use AI to evaluate link relevance with a confidence score.

        Args:
            link_url: Link to evaluate
            source_url: Source page
            source: SourceProfile

        Returns:
            Confidence score (0.0 to 1.0)
        """
        prompt = f"""Evaluate if this link should be followed for content extraction.

Source URL: {source_url}
Target Link: {link_url}

Consider:
- Is this likely a content page (article, documentation, guide)?
- Does it fit the same domain/topic as the source?
- Is it likely to contain valuable textual content?

Respond with ONLY a confidence score from 0.0 to 1.0.
- 1.0 = Definitely follow (high-value content)
- 0.5 = Uncertain (might be content)
- 0.0 = Don't follow (navigation/utility page)

Score:"""

        # Get cache key
        cache_key = hashlib.md5(prompt.encode()).hexdigest()

        # Check cache
        if cache_key in self._response_cache:
            response = self._response_cache[cache_key]
        else:
            response = self.llm_function(prompt)
            self._response_cache[cache_key] = response

        # Parse score
        try:
            # Extract first number from response
            match = re.search(r'([0-1]\.?\d*)', response)
            if match:
                score = float(match.group(1))
                return min(max(score, 0.0), 1.0)  # Clamp to [0, 1]
            return 0.5  # Default to uncertain
        except Exception:
            return 0.5
