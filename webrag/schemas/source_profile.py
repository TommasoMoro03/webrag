"""Source profile and crawl settings schemas."""

from pydantic import BaseModel, HttpUrl, Field, field_validator
from typing import Optional, Literal, Dict, List
from datetime import datetime


class CrawlSettings(BaseModel):
    """Configuration for crawling behavior (future extensibility)."""

    max_depth: int = Field(
        default=0,
        ge=0,
        description="Maximum depth for recursive crawling (0 = single page only)"
    )
    max_pages: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum number of pages to crawl"
    )
    follow_external_links: bool = Field(
        default=False,
        description="Whether to follow links outside the original domain"
    )
    respect_robots_txt: bool = Field(
        default=True,
        description="Whether to respect robots.txt rules"
    )
    rate_limit_delay: float = Field(
        default=1.0,
        ge=0.0,
        description="Delay in seconds between requests"
    )
    timeout: int = Field(
        default=30,
        ge=1,
        description="Request timeout in seconds"
    )
    user_agent: Optional[str] = Field(
        default=None,
        description="Custom user agent string"
    )
    allowed_domains: Optional[List[str]] = Field(
        default=None,
        description="List of allowed domains for crawling"
    )
    exclude_patterns: Optional[List[str]] = Field(
        default=None,
        description="URL patterns to exclude (regex)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "max_depth": 2,
                "max_pages": 100,
                "follow_external_links": False,
                "rate_limit_delay": 1.0
            }
        }


class SourceProfile(BaseModel):
    """Profile for a single website source to be ingested."""

    url: HttpUrl = Field(
        ...,
        description="The URL of the website to ingest"
    )
    name: Optional[str] = Field(
        default=None,
        description="Human-readable name for this source"
    )
    type: Literal["static", "dynamic", "api", "unknown"] = Field(
        default="unknown",
        description="Type of website content"
    )
    update_frequency: Optional[Literal["hourly", "daily", "weekly", "monthly"]] = Field(
        default=None,
        description="Expected update frequency for scheduled re-ingestion"
    )
    priority: Literal["low", "medium", "high"] = Field(
        default="medium",
        description="Processing priority"
    )
    crawl_settings: CrawlSettings = Field(
        default_factory=CrawlSettings,
        description="Crawling configuration"
    )
    user_hints: Optional[Dict[str, str]] = Field(
        default=None,
        description="User-provided extraction hints (e.g., CSS selectors, XPath)"
    )
    tags: Optional[List[str]] = Field(
        default=None,
        description="Tags for categorization and filtering"
    )
    metadata: Optional[Dict[str, str]] = Field(
        default=None,
        description="Additional custom metadata"
    )

    # Future extensibility fields (placeholders)
    version: Optional[str] = Field(
        default=None,
        description="Version identifier for this source configuration"
    )
    last_ingested_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp of last successful ingestion"
    )
    enabled: bool = Field(
        default=True,
        description="Whether this source is enabled for processing"
    )

    @field_validator('name', mode='before')
    @classmethod
    def generate_name_from_url(cls, v, info):
        """Auto-generate name from URL if not provided."""
        if v is None and 'url' in info.data:
            url = str(info.data['url'])
            # Extract domain as default name
            return url.split('/')[2] if len(url.split('/')) > 2 else url
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "url": "https://example.com/docs",
                "name": "Example Documentation",
                "type": "static",
                "update_frequency": "daily",
                "priority": "high",
                "crawl_settings": {
                    "max_depth": 2,
                    "max_pages": 50
                },
                "user_hints": {
                    "content_selector": "article.main-content",
                    "title_selector": "h1.page-title"
                },
                "tags": ["documentation", "technical"]
            }
        }
