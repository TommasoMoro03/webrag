"""Document and extraction result schemas."""

from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, Dict, List, Any
from datetime import datetime
from enum import Enum


class ContentType(str, Enum):
    """Types of extracted content."""
    TEXT = "text"
    CODE = "code"
    TABLE = "table"
    LIST = "list"
    HEADING = "heading"
    METADATA = "metadata"
    UNKNOWN = "unknown"


class ExtractionResult(BaseModel):
    """Result of content extraction from a webpage."""

    url: HttpUrl = Field(
        ...,
        description="Source URL of the extracted content"
    )
    title: Optional[str] = Field(
        default=None,
        description="Page title"
    )
    content: str = Field(
        ...,
        description="Main extracted content (cleaned text)"
    )
    raw_html: Optional[str] = Field(
        default=None,
        description="Original HTML content (for debugging/reprocessing)"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extracted metadata (author, description, keywords, etc.)"
    )
    links: Optional[List[str]] = Field(
        default=None,
        description="Extracted links for potential crawling"
    )
    images: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="Extracted images with URLs and alt text"
    )
    language: Optional[str] = Field(
        default=None,
        description="Detected language code (e.g., 'en', 'es')"
    )
    extraction_method: str = Field(
        default="unknown",
        description="Method used for extraction (e.g., 'trafilatura', 'beautifulsoup')"
    )
    extracted_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when extraction occurred"
    )

    # Future extensibility fields
    confidence_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence score of extraction quality (0-1)"
    )
    status_code: Optional[int] = Field(
        default=None,
        description="HTTP status code from fetch"
    )
    fetch_time_ms: Optional[int] = Field(
        default=None,
        ge=0,
        description="Time taken to fetch the page in milliseconds"
    )
    content_hash: Optional[str] = Field(
        default=None,
        description="Hash of content for change detection"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "url": "https://example.com/article",
                "title": "Example Article",
                "content": "This is the main content...",
                "metadata": {
                    "author": "John Doe",
                    "published_date": "2024-01-15",
                    "description": "An example article"
                },
                "language": "en",
                "extraction_method": "trafilatura",
                "confidence_score": 0.95
            }
        }


class DocumentChunk(BaseModel):
    """A chunk of a document ready for RAG ingestion."""

    chunk_id: str = Field(
        ...,
        description="Unique identifier for this chunk"
    )
    source_url: HttpUrl = Field(
        ...,
        description="Original source URL"
    )
    content: str = Field(
        ...,
        description="The chunked content text"
    )
    content_type: ContentType = Field(
        default=ContentType.TEXT,
        description="Type of content in this chunk"
    )

    # Position and context
    chunk_index: int = Field(
        ...,
        ge=0,
        description="Index of this chunk in the sequence (0-based)"
    )
    total_chunks: int = Field(
        ...,
        ge=1,
        description="Total number of chunks from this document"
    )
    start_char: Optional[int] = Field(
        default=None,
        ge=0,
        description="Starting character position in original content"
    )
    end_char: Optional[int] = Field(
        default=None,
        ge=0,
        description="Ending character position in original content"
    )

    # Metadata and enrichment
    title: Optional[str] = Field(
        default=None,
        description="Document title"
    )
    section_title: Optional[str] = Field(
        default=None,
        description="Section or heading this chunk belongs to"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata inherited from source and extraction"
    )
    tags: Optional[List[str]] = Field(
        default=None,
        description="Tags for categorization"
    )

    # Timestamps
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this chunk was created"
    )
    source_updated_at: Optional[datetime] = Field(
        default=None,
        description="When the source was last updated (if known)"
    )

    # Future extensibility fields (placeholders)
    confidence_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence score for chunk quality/relevance"
    )
    embedding: Optional[List[float]] = Field(
        default=None,
        description="Vector embedding (for future vector DB integration)"
    )
    version: Optional[str] = Field(
        default=None,
        description="Version identifier for change tracking"
    )
    parent_chunk_id: Optional[str] = Field(
        default=None,
        description="ID of parent chunk for hierarchical chunking strategies"
    )
    child_chunk_ids: Optional[List[str]] = Field(
        default=None,
        description="IDs of child chunks for hierarchical chunking strategies"
    )
    chunking_strategy: Optional[str] = Field(
        default=None,
        description="Strategy used to create this chunk (e.g., 'fixed_size', 'semantic')"
    )
    token_count: Optional[int] = Field(
        default=None,
        ge=0,
        description="Approximate token count for LLM context planning"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "chunk_id": "doc123_chunk_0",
                "source_url": "https://example.com/article",
                "content": "First paragraph of content...",
                "content_type": "text",
                "chunk_index": 0,
                "total_chunks": 5,
                "title": "Example Article",
                "section_title": "Introduction",
                "metadata": {
                    "author": "John Doe",
                    "category": "documentation"
                },
                "tags": ["tutorial", "beginner"],
                "confidence_score": 0.95,
                "chunking_strategy": "semantic",
                "token_count": 150
            }
        }
