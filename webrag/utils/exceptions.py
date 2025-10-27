"""Custom exceptions for web-rag library."""


class WebRAGError(Exception):
    """Base exception for all web-rag errors."""
    pass


# Configuration and validation errors
class InvalidSourceError(WebRAGError):
    """Raised when source configuration is invalid or cannot be validated."""
    pass


class ConfigurationError(WebRAGError):
    """Raised when there's an error in library configuration."""
    pass


# Fetching errors
class FetchError(WebRAGError):
    """Base exception for fetching-related errors."""
    pass


class URLFetchError(FetchError):
    """Raised when a URL cannot be fetched."""

    def __init__(self, url: str, reason: str = None, status_code: int = None):
        self.url = url
        self.status_code = status_code
        message = f"Failed to fetch URL: {url}"
        if status_code:
            message += f" (status code: {status_code})"
        if reason:
            message += f" - {reason}"
        super().__init__(message)


class TimeoutError(FetchError):
    """Raised when a fetch operation times out."""
    pass


class RateLimitError(FetchError):
    """Raised when rate limit is exceeded."""
    pass


# Extraction errors
class ExtractionError(WebRAGError):
    """Base exception for content extraction errors."""
    pass


class ContentNotFoundError(ExtractionError):
    """Raised when expected content cannot be found or extracted."""
    pass


class ParsingError(ExtractionError):
    """Raised when HTML/content parsing fails."""
    pass


# Chunking errors
class ChunkingError(WebRAGError):
    """Base exception for chunking-related errors."""
    pass


class InvalidChunkSizeError(ChunkingError):
    """Raised when chunk size configuration is invalid."""
    pass


# Export errors
class ExportError(WebRAGError):
    """Base exception for export-related errors."""
    pass


class UnsupportedFormatError(ExportError):
    """Raised when an unsupported export format is requested."""
    pass


class FileWriteError(ExportError):
    """Raised when writing output file fails."""
    pass


# Pipeline errors
class PipelineError(WebRAGError):
    """Raised when the pipeline encounters a fatal error."""
    pass


class DependencyError(WebRAGError):
    """Raised when a required dependency is missing or incompatible."""
    pass


# Future extensibility errors (placeholders)
class CrawlerError(WebRAGError):
    """Base exception for crawling-related errors (future)."""
    pass


class RobotsTxtError(CrawlerError):
    """Raised when robots.txt cannot be parsed or blocks access."""
    pass


class SchedulerError(WebRAGError):
    """Base exception for scheduling-related errors (future)."""
    pass


class StorageError(WebRAGError):
    """Base exception for storage/persistence errors (future)."""
    pass


class ValidationError(WebRAGError):
    """Raised when data validation fails (complementary to Pydantic)."""
    pass
