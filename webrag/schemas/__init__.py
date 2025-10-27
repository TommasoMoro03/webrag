"""Schema definitions for web-rag."""

from webrag.schemas.source_profile import SourceProfile, CrawlSettings
from webrag.schemas.document import (
    ExtractionResult,
    DocumentChunk,
    ContentType
)
from webrag.schemas.response import PipelineResult

__all__ = [
    "SourceProfile",
    "CrawlSettings",
    "ExtractionResult",
    "DocumentChunk",
    "ContentType",
    "PipelineResult",
]
