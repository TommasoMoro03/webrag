"""Semantic chunker for splitting content into meaningful chunks."""

from typing import List, Optional
import re

from webrag.chunking.base import BaseChunker
from webrag.schemas.document import ExtractionResult, DocumentChunk, ContentType
from webrag.utils.exceptions import ChunkingError, InvalidChunkSizeError


class SemanticChunker(BaseChunker):
    """
    Semantic chunker that splits content based on natural language boundaries.

    This chunker attempts to preserve semantic coherence by splitting on:
    1. Paragraph boundaries
    2. Sentence boundaries
    3. Word boundaries (as fallback)

    It avoids splitting in the middle of sentences when possible.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        **kwargs
    ):
        """
        Initialize the semantic chunker.

        Args:
            chunk_size: Target size for each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            **kwargs: Additional configuration
        """
        super().__init__(chunk_size, chunk_overlap, **kwargs)

        if not self.validate_chunk_size():
            raise InvalidChunkSizeError(
                f"Invalid chunk configuration: size={chunk_size}, overlap={chunk_overlap}"
            )

    def chunk(
        self,
        extraction_result: ExtractionResult,
        **kwargs
    ) -> List[DocumentChunk]:
        """
        Chunk extracted content into DocumentChunk objects.

        Args:
            extraction_result: ExtractionResult from extraction stage
            **kwargs: Additional chunking parameters

        Returns:
            List of DocumentChunk objects

        Raises:
            ChunkingError: If chunking fails
        """
        try:
            content = extraction_result.content

            if not content or len(content.strip()) == 0:
                raise ChunkingError("Cannot chunk empty content")

            # Split content into chunks
            chunk_texts = self._split_content(content)

            # Create DocumentChunk objects
            chunks = []
            for i, chunk_text in enumerate(chunk_texts):
                chunk = self._create_chunk(
                    chunk_text=chunk_text,
                    chunk_index=i,
                    total_chunks=len(chunk_texts),
                    extraction_result=extraction_result,
                )
                chunks.append(chunk)

            return chunks

        except ChunkingError:
            raise
        except Exception as e:
            raise ChunkingError(f"Chunking failed: {str(e)}") from e

    def _split_content(self, content: str) -> List[str]:
        """
        Split content into chunks preserving semantic boundaries.

        Args:
            content: Content text to split

        Returns:
            List of chunk texts
        """
        # First, try to split by paragraphs
        paragraphs = re.split(r'\n\n+', content)

        chunks = []
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # If paragraph fits in remaining space, add it
            if len(current_chunk) + len(para) + 2 <= self.chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para

            # If paragraph is larger than chunk size, split it
            elif len(para) > self.chunk_size:
                # Save current chunk if exists
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""

                # Split large paragraph by sentences
                sentences = self._split_by_sentences(para)
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue

                    if len(current_chunk) + len(sentence) + 1 <= self.chunk_size:
                        if current_chunk:
                            current_chunk += " " + sentence
                        else:
                            current_chunk = sentence
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)

                        # If single sentence is too long, split it
                        if len(sentence) > self.chunk_size:
                            word_chunks = self._split_by_words(sentence)
                            chunks.extend(word_chunks[:-1])
                            current_chunk = word_chunks[-1] if word_chunks else ""
                        else:
                            current_chunk = sentence

            # Paragraph doesn't fit, save current and start new
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = para

        # Don't forget the last chunk
        if current_chunk:
            chunks.append(current_chunk)

        # Apply overlap if configured
        if self.chunk_overlap > 0 and len(chunks) > 1:
            chunks = self._apply_overlap(chunks)

        return chunks

    def _split_by_sentences(self, text: str) -> List[str]:
        """
        Split text by sentence boundaries.

        Args:
            text: Text to split

        Returns:
            List of sentences
        """
        # Simple sentence splitting regex
        # Handles common sentence endings: . ! ?
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_pattern, text)
        return [s.strip() for s in sentences if s.strip()]

    def _split_by_words(self, text: str) -> List[str]:
        """
        Split text by word boundaries when nothing else works.

        Args:
            text: Text to split

        Returns:
            List of chunks
        """
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0

        for word in words:
            word_length = len(word) + 1  # +1 for space

            if current_length + word_length <= self.chunk_size:
                current_chunk.append(word)
                current_length += word_length
            else:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = len(word)

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        """
        Apply overlap between chunks.

        Args:
            chunks: List of chunk texts

        Returns:
            List of chunks with overlap applied
        """
        overlapped_chunks = [chunks[0]]  # First chunk stays the same

        for i in range(1, len(chunks)):
            prev_chunk = chunks[i - 1]
            current_chunk = chunks[i]

            # Take last N characters from previous chunk as overlap
            overlap_text = prev_chunk[-self.chunk_overlap:].lstrip()

            # Add overlap to current chunk
            overlapped_chunks.append(overlap_text + " " + current_chunk)

        return overlapped_chunks

    def _create_chunk(
        self,
        chunk_text: str,
        chunk_index: int,
        total_chunks: int,
        extraction_result: ExtractionResult,
    ) -> DocumentChunk:
        """
        Create a DocumentChunk object from chunk text.

        Args:
            chunk_text: The chunk content
            chunk_index: Index of this chunk
            total_chunks: Total number of chunks
            extraction_result: Original extraction result

        Returns:
            DocumentChunk object
        """
        # Generate chunk ID
        chunk_id = self.generate_chunk_id(
            str(extraction_result.url),
            chunk_index
        )

        # Calculate token count
        token_count = self.calculate_token_count(chunk_text)

        # Create chunk
        chunk = DocumentChunk(
            chunk_id=chunk_id,
            source_url=extraction_result.url,
            content=chunk_text,
            content_type=ContentType.TEXT,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            title=extraction_result.title,
            metadata=extraction_result.metadata.copy(),
            tags=extraction_result.metadata.get('tags'),
            confidence_score=extraction_result.confidence_score,
            chunking_strategy='semantic',
            token_count=token_count,
        )

        return chunk

    def estimate_chunks(self, content: str) -> int:
        """
        Estimate the number of chunks that will be created.

        Args:
            content: Content to estimate

        Returns:
            Estimated number of chunks
        """
        if not content:
            return 0

        # Simple estimation based on content length and chunk size
        effective_chunk_size = self.chunk_size - self.chunk_overlap
        estimated = (len(content) + effective_chunk_size - 1) // effective_chunk_size

        return max(1, estimated)
