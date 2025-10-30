"""AI-based URL relationship analyzer using LLM prompts."""

from typing import List, Dict, Any, Optional, Callable
import hashlib
import json
import re
import logging

from webrag.url_relationships.base import BaseURLRelationshipAnalyzer
from webrag.schemas.document import DocumentGroup

logger = logging.getLogger(__name__)


class AIURLRelationshipAnalyzer(BaseURLRelationshipAnalyzer):
    """
    AI-powered URL relationship analyzer.

    Uses LLM prompts to intelligently identify relationships between URLs
    based on semantic similarity, content analysis, and URL structure.

    This implementation can detect:
    - Multi-page articles (same content split across pages)
    - Article series (related topics)
    - Documentation sections (related docs)
    - Any other semantic relationships the AI can identify
    """

    def __init__(
        self,
        llm_function: Callable[[str], str],
        **kwargs
    ):
        """
        Initialize the AI URL relationship analyzer.

        Args:
            llm_function: A callable that takes a prompt (str) and returns an LLM response (str)
                         Should be a simple function like: lambda prompt: client.chat(prompt)
            **kwargs: Configuration options
                - min_group_size: Minimum URLs for a group (default: 2)
                - enable_grouping: Enable/disable grouping entirely (default: True)
                - temperature: LLM temperature for creativity (default: 0.3)
        """
        super().__init__(**kwargs)
        self.llm_function = llm_function
        self.min_group_size = kwargs.get('min_group_size', 2)
        self.enable_grouping = kwargs.get('enable_grouping', True)
        self.temperature = kwargs.get('temperature', 0.3)

        # Cache for AI responses to avoid redundant calls
        self._response_cache: Dict[str, Any] = {}

    def analyze_relationships(
        self,
        urls: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[DocumentGroup]:
        """
        Use AI to analyze URL relationships and create document groups.

        Args:
            urls: List of URLs to analyze
            metadata: Optional metadata about URLs (e.g., titles, descriptions)

        Returns:
            List of DocumentGroup objects representing related URLs
        """
        if not self.enable_grouping:
            logger.info("AI grouping disabled in configuration")
            return []

        if len(urls) < self.min_group_size:
            logger.info(f"Not enough URLs ({len(urls)}) for grouping (min: {self.min_group_size})")
            return []

        try:
            logger.info(f"Analyzing relationships between {len(urls)} URLs using AI")
            groups = self._ai_analyze_groups(urls, metadata)
            logger.info(f"AI identified {len(groups)} document group(s)")
            return groups
        except Exception as e:
            logger.warning(f"AI relationship analysis failed: {e}")
            # Return empty list on failure (no groups)
            return []

    def _ai_analyze_groups(
        self,
        urls: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[DocumentGroup]:
        """
        Use AI to intelligently group related pages.

        Args:
            urls: URLs to group
            metadata: Optional metadata

        Returns:
            List of DocumentGroup objects
        """
        if len(urls) < 2:
            return []

        # Build prompt
        prompt = self._build_grouping_prompt(urls, metadata)

        # Get cache key
        cache_key = hashlib.md5(prompt.encode()).hexdigest()

        # Check cache
        if cache_key in self._response_cache:
            logger.info("Using cached AI grouping response")
            response = self._response_cache[cache_key]
        else:
            logger.info("Calling AI for URL grouping analysis")
            response = self.llm_function(prompt)
            self._response_cache[cache_key] = response

        # Parse grouping response
        groups = self._parse_grouping_response(response, urls, metadata)

        return groups

    def _build_grouping_prompt(
        self,
        urls: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Build prompt for AI-based page grouping.

        Args:
            urls: URLs to analyze
            metadata: Optional metadata

        Returns:
            Prompt string
        """
        urls_text = ""
        for i, url in enumerate(urls):
            urls_text += f"{i}. {url}\n"

            # Add metadata if available
            if metadata and url in metadata:
                meta = metadata[url]
                if isinstance(meta, dict):
                    title = meta.get('title', 'N/A')
                    urls_text += f"   Title: {title}\n"
                elif isinstance(meta, list) and meta:
                    # Handle list of metadata dicts
                    title = meta[0].get('title', 'N/A')
                    urls_text += f"   Title: {title}\n"

        prompt = f"""Analyze these URLs to identify groups of related pages.

URLs:
{urls_text}

Task: Identify groups of related pages such as:
- Multi-page articles (same content split across pages)
- Article series (related topics)
- Documentation sections (related docs)
- Tutorial sequences
- Any other logical groupings

Respond with ONLY a JSON array of groups. Each group is an object with:
- "indices": array of URL indices (0-based)
- "relationship": string ("multi_page" or "series" or "related" or "tutorial" or "docs")
- "title": string (optional group title describing what the group is about)

Example:
[
  {{"indices": [0, 1, 2], "relationship": "multi_page", "title": "Complete Guide"}},
  {{"indices": [5, 8], "relationship": "series", "title": "Tutorial Series"}}
]

If no clear groups, respond with empty array: []

Response:"""

        return prompt

    def _parse_grouping_response(
        self,
        response: str,
        urls: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[DocumentGroup]:
        """
        Parse AI grouping response into DocumentGroup objects.

        Args:
            response: LLM response
            urls: Original URLs
            metadata: Optional metadata

        Returns:
            List of DocumentGroup objects
        """
        try:
            # Extract JSON array
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if not json_match:
                logger.warning("No JSON array found in AI response")
                return []

            groups_data = json.loads(json_match.group())

            if not isinstance(groups_data, list):
                logger.warning("AI response is not a JSON array")
                return []

            document_groups = []
            for group_data in groups_data:
                indices = group_data.get('indices', [])
                if len(indices) < self.min_group_size:
                    logger.info(f"Skipping group with {len(indices)} URLs (min: {self.min_group_size})")
                    continue

                # Get URLs for this group
                group_urls = [urls[i] for i in indices if 0 <= i < len(urls)]
                if not group_urls:
                    continue

                # Create DocumentGroup
                primary_url = group_urls[0]
                group_id = self._generate_group_id(primary_url)

                relationship_type = group_data.get('relationship', 'related')
                title = group_data.get('title')

                document_groups.append(DocumentGroup(
                    group_id=group_id,
                    source_url=primary_url,
                    page_urls=group_urls,
                    title=title,
                    relationship_type=relationship_type,
                    metadata={}
                ))

                logger.info(
                    f"Created group '{group_id}': {len(group_urls)} URLs, "
                    f"type: {relationship_type}, title: {title}"
                )

            return document_groups

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse AI response as JSON: {e}")
            return []
        except Exception as e:
            logger.warning(f"Error parsing AI grouping response: {e}")
            return []

    def _generate_group_id(self, url: str) -> str:
        """
        Generate unique group ID from URL.

        Args:
            url: URL to generate ID from

        Returns:
            Group ID string
        """
        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
        return f"ai_group_{url_hash}"
