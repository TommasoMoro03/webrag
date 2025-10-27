"""Abstract base class for content chunkers."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from webrag.schemas.document import ExtractionResult, DocumentChunk


class BaseChunker(ABC):
    """
    Abstract base class for chunking extracted content into RAG-ready pieces.

    Chunkers are responsible for splitting content into appropriately-sized
    chunks suitable for embedding and retrieval, while preserving context
    and semantic coherence.

    Future implementations may include:
    - FixedSizeChunker (character/token-based splitting)
    - SemanticChunker (sentence/paragraph-based)
    - StructureAwareChunker (respects headings, sections)
    - RecursiveChunker (hierarchical chunking)
    - LLMChunker (using LLM to intelligently split content)
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        **kwargs
    ):
        """
        Initialize the chunker.

        Args:
            chunk_size: Target size for each chunk (in characters or tokens)
            chunk_overlap: Number of characters/tokens to overlap between chunks
            **kwargs: Chunker-specific configuration
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.config = kwargs

    @abstractmethod
    def chunk(
        self,
        extraction_result: ExtractionResult,
        **kwargs
    ) -> List[DocumentChunk]:
        """
        Chunk extracted content into DocumentChunk objects.

        Args:
            extraction_result: ExtractionResult from the extraction stage
            **kwargs: Additional chunking parameters

        Returns:
            List of DocumentChunk objects ready for RAG ingestion

        Raises:
            ChunkingError: If chunking fails
            InvalidChunkSizeError: If chunk configuration is invalid
        """
        raise NotImplementedError

    @abstractmethod
    def estimate_chunks(self, content: str) -> int:
        """
        Estimate the number of chunks that will be created from content.

        Useful for pre-allocation and progress tracking.

        Args:
            content: Content to estimate chunks for

        Returns:
            Estimated number of chunks
        """
        raise NotImplementedError

    def validate_chunk_size(self) -> bool:
        """
        Validate that chunk size configuration is valid.

        Returns:
            True if configuration is valid, False otherwise
        """
        if self.chunk_size <= 0:
            return False
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            return False
        return True

    def generate_chunk_id(
        self,
        source_url: str,
        chunk_index: int,
        strategy: str = "default"
    ) -> str:
        """
        Generate a unique ID for a chunk.

        Args:
            source_url: Source URL
            chunk_index: Index of the chunk
            strategy: ID generation strategy

        Returns:
            Unique chunk ID string
        """
        # Simple implementation: hash + index
        # More sophisticated implementations could include timestamps, versions, etc.
        url_hash = str(hash(source_url))[-8:]  # Last 8 chars of hash
        return f"{url_hash}_chunk_{chunk_index}"

    def calculate_token_count(self, text: str) -> int:
        """
        Estimate token count for a piece of text.

        Default implementation uses simple heuristic (4 chars per token).
        Can be overridden to use actual tokenizer.

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated token count
        """
        # Simple approximation: ~4 characters per token
        return len(text) // 4

    def merge_metadata(
        self,
        extraction_metadata: Dict[str, Any],
        chunk_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge metadata from extraction result with chunk-specific metadata.

        Args:
            extraction_metadata: Metadata from ExtractionResult
            chunk_metadata: Chunk-specific metadata

        Returns:
            Merged metadata dictionary
        """
        merged = extraction_metadata.copy()
        merged.update(chunk_metadata)
        return merged

    def preserve_context(
        self,
        chunks: List[str],
        context_size: int = 100
    ) -> List[str]:
        """
        Add contextual overlap between chunks for better coherence.

        This is a helper method for implementations that need to
        preserve context across chunk boundaries.

        Args:
            chunks: List of chunk texts
            context_size: Number of characters to use as context

        Returns:
            List of chunks with overlapping context
        """
        if not chunks or context_size <= 0:
            return chunks

        contextualized_chunks = []
        for i, chunk in enumerate(chunks):
            if i > 0:
                # Add end of previous chunk as prefix
                prev_context = chunks[i-1][-context_size:]
                chunk = prev_context + " " + chunk
            contextualized_chunks.append(chunk)

        return contextualized_chunks

    def extract_section_title(
        self,
        content: str,
        chunk_start: int
    ) -> Optional[str]:
        """
        Extract the section/heading title that a chunk belongs to.

        This is a placeholder for future implementation.
        Concrete chunkers can implement heading detection based on
        their parsing strategy.

        Args:
            content: Full document content
            chunk_start: Starting position of the chunk

        Returns:
            Section title if found, None otherwise
        """
        # Placeholder - concrete implementations will parse headings
        return None

    def split_on_separators(
        self,
        text: str,
        separators: List[str] = None
    ) -> List[str]:
        """
        Split text on multiple separators in order of preference.

        Args:
            text: Text to split
            separators: List of separators in preference order
                       (default: paragraph, sentence, word boundaries)

        Returns:
            List of text segments
        """
        if separators is None:
            separators = ['\n\n', '\n', '. ', ' ']

        segments = [text]
        for separator in separators:
            new_segments = []
            for segment in segments:
                new_segments.extend(segment.split(separator))
            segments = new_segments

        return [s.strip() for s in segments if s.strip()]
