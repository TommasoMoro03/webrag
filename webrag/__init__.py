"""
web-rag: Production-grade library for ingesting websites into RAG-ready documents.

This library provides a modular, extensible architecture for:
- Fetching content from websites
- Extracting meaningful content and metadata
- Chunking content for RAG systems
- Exporting to various formats
"""

from webrag.orchestrator import WebRAG

# Core schemas
from webrag.schemas import (
    SourceProfile,
    CrawlSettings,
    ExtractionResult,
    DocumentChunk,
    ContentType,
    PipelineResult,
)

# Source loading
from webrag.sources import SourceLoader

# Abstract base classes
from webrag.fetcher import BaseFetcher
from webrag.extractors import BaseExtractor
from webrag.chunking import BaseChunker
from webrag.output import BaseExporter

# Version
__version__ = "0.0.1"

__all__ = [
    # Main orchestrator
    "WebRAG",

    # Schemas
    "SourceProfile",
    "CrawlSettings",
    "ExtractionResult",
    "DocumentChunk",
    "ContentType",
    "PipelineResult",

    # Loaders
    "SourceLoader",

    # Base classes
    "BaseFetcher",
    "BaseExtractor",
    "BaseChunker",
    "BaseExporter",

    # Version
    "__version__",
]
