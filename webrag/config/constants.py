"""Constants and default configuration values for web-rag."""

from typing import Final

# Library metadata
LIBRARY_NAME: Final[str] = "web-rag"
VERSION: Final[str] = "0.0.1"
USER_AGENT: Final[str] = f"{LIBRARY_NAME}/{VERSION}"

# Default fetch settings
DEFAULT_TIMEOUT: Final[int] = 30  # seconds
DEFAULT_RATE_LIMIT_DELAY: Final[float] = 1.0  # seconds
DEFAULT_MAX_RETRIES: Final[int] = 3
DEFAULT_RETRY_DELAY: Final[float] = 2.0  # seconds

# Default crawl settings
DEFAULT_MAX_DEPTH: Final[int] = 0  # 0 = single page only
DEFAULT_MAX_PAGES: Final[int] = 100
DEFAULT_RESPECT_ROBOTS_TXT: Final[bool] = True

# Default chunking settings
DEFAULT_CHUNK_SIZE: Final[int] = 512  # characters
DEFAULT_CHUNK_OVERLAP: Final[int] = 50  # characters
DEFAULT_MIN_CHUNK_SIZE: Final[int] = 100  # minimum viable chunk
DEFAULT_MAX_CHUNK_SIZE: Final[int] = 2000  # maximum chunk size

# Token estimation (rough approximation: 1 token ~= 4 characters)
CHARS_PER_TOKEN: Final[float] = 4.0

# Content extraction settings
DEFAULT_EXTRACTION_METHOD: Final[str] = "trafilatura"
EXTRACTION_METHODS: Final[tuple] = ("trafilatura", "beautifulsoup", "readability", "custom")

# Supported content types
SUPPORTED_CONTENT_TYPES: Final[tuple] = (
    "text/html",
    "application/xhtml+xml",
    "text/plain"
)

# Export formats
SUPPORTED_EXPORT_FORMATS: Final[tuple] = ("json", "jsonl", "csv", "parquet")
DEFAULT_EXPORT_FORMAT: Final[str] = "json"

# File handling
DEFAULT_OUTPUT_DIR: Final[str] = "./output"
DEFAULT_CACHE_DIR: Final[str] = "./.webrag_cache"
MAX_FILE_SIZE_MB: Final[int] = 50

# Logging
DEFAULT_LOG_LEVEL: Final[str] = "INFO"
LOG_FORMAT: Final[str] = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# HTTP headers
DEFAULT_HEADERS: Final[dict] = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1"
}

# Regex patterns (placeholders for future use)
URL_PATTERN: Final[str] = r"https?://[^\s]+"
EMAIL_PATTERN: Final[str] = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"

# Language detection
DEFAULT_LANGUAGE: Final[str] = "en"
SUPPORTED_LANGUAGES: Final[tuple] = ("en", "es", "fr", "de", "it", "pt", "zh", "ja", "ko", "ru")

# Content quality thresholds (for future scoring)
MIN_CONTENT_LENGTH: Final[int] = 50  # minimum characters for valid content
MIN_WORD_COUNT: Final[int] = 10  # minimum words for valid content
MIN_CONFIDENCE_SCORE: Final[float] = 0.5  # minimum extraction confidence

# Parallel processing settings (future)
DEFAULT_MAX_WORKERS: Final[int] = 5
DEFAULT_CONCURRENT_REQUESTS: Final[int] = 3

# Cache settings (future)
CACHE_ENABLED: Final[bool] = True
CACHE_EXPIRY_HOURS: Final[int] = 24

# Update frequency mappings (in seconds)
UPDATE_FREQUENCY_MAP: Final[dict] = {
    "hourly": 3600,
    "daily": 86400,
    "weekly": 604800,
    "monthly": 2592000
}

# Priority weights (for future scheduling)
PRIORITY_WEIGHTS: Final[dict] = {
    "low": 1,
    "medium": 5,
    "high": 10
}

# Error handling
MAX_ERROR_MESSAGES: Final[int] = 100  # max errors to store in pipeline result
CONTINUE_ON_ERROR: Final[bool] = True  # whether to continue pipeline on individual source errors

# HTML parsing
REMOVE_TAGS: Final[tuple] = ("script", "style", "nav", "footer", "header", "aside", "iframe", "noscript")
PRESERVE_TAGS: Final[tuple] = ("article", "main", "section", "div", "p", "h1", "h2", "h3", "h4", "h5", "h6", "ul", "ol", "li", "pre", "code", "table")

# Metadata extraction
METADATA_FIELDS: Final[tuple] = (
    "title",
    "description",
    "keywords",
    "author",
    "published_date",
    "modified_date",
    "canonical_url",
    "language"
)
