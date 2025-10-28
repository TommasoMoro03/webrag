"""Base converter interface for converting DocumentChunks to various RAG library formats."""

from abc import ABC, abstractmethod
from typing import List, Any, Dict, Optional
from webrag.schemas.document import DocumentChunk


class BaseConverter(ABC):
    """
    Abstract base class for converting DocumentChunk objects to specific RAG library formats.

    This allows users to get output chunks in formats compatible with:
    - LangChain
    - LlamaIndex
    - Haystack
    - Other RAG frameworks
    """

    def __init__(self, **kwargs):
        """
        Initialize the converter.

        Args:
            **kwargs: Converter-specific configuration options
        """
        self.config = kwargs

    @abstractmethod
    def convert(self, chunk: DocumentChunk) -> Any:
        """
        Convert a single DocumentChunk to the target format.

        Args:
            chunk: DocumentChunk object to convert

        Returns:
            Object in the target library's format (e.g., LangChain Document, LlamaIndex Node)
        """
        raise NotImplementedError

    def convert_batch(self, chunks: List[DocumentChunk]) -> List[Any]:
        """
        Convert multiple DocumentChunks to the target format.

        Args:
            chunks: List of DocumentChunk objects

        Returns:
            List of converted objects
        """
        return [self.convert(chunk) for chunk in chunks]

    @abstractmethod
    def get_format_name(self) -> str:
        """
        Get the name of the target format.

        Returns:
            Format name (e.g., 'langchain', 'llamaindex', 'haystack')
        """
        raise NotImplementedError

    def prepare_metadata(self, chunk: DocumentChunk) -> Dict[str, Any]:
        """
        Prepare metadata dictionary from DocumentChunk.

        Converts all chunk metadata into a flat dictionary suitable for most RAG frameworks.

        Args:
            chunk: DocumentChunk object

        Returns:
            Dictionary of metadata
        """
        metadata = {
            'chunk_id': chunk.chunk_id,
            'source_url': str(chunk.source_url),
            'chunk_index': chunk.chunk_index,
            'total_chunks': chunk.total_chunks,
            'content_type': chunk.content_type.value if chunk.content_type else 'text',
        }

        # Add optional fields if present
        if chunk.title:
            metadata['title'] = chunk.title

        if chunk.section_title:
            metadata['section_title'] = chunk.section_title

        if chunk.start_char is not None:
            metadata['start_char'] = chunk.start_char

        if chunk.end_char is not None:
            metadata['end_char'] = chunk.end_char

        if chunk.tags:
            metadata['tags'] = chunk.tags

        if chunk.confidence_score is not None:
            metadata['confidence_score'] = chunk.confidence_score

        if chunk.token_count is not None:
            metadata['token_count'] = chunk.token_count

        if chunk.chunking_strategy:
            metadata['chunking_strategy'] = chunk.chunking_strategy

        if chunk.document_group_id:
            metadata['document_group_id'] = chunk.document_group_id

        if chunk.page_url:
            metadata['page_url'] = str(chunk.page_url)

        if chunk.page_index is not None:
            metadata['page_index'] = chunk.page_index

        if chunk.total_pages is not None:
            metadata['total_pages'] = chunk.total_pages

        if chunk.created_at:
            metadata['created_at'] = chunk.created_at.isoformat()

        if chunk.source_updated_at:
            metadata['source_updated_at'] = chunk.source_updated_at.isoformat()

        # Merge in custom metadata from chunk
        if chunk.metadata:
            # Avoid overwriting reserved keys
            for key, value in chunk.metadata.items():
                if key not in metadata:
                    metadata[key] = value

        return metadata

    def validate_dependencies(self) -> bool:
        """
        Check if required dependencies for this converter are installed.

        Returns:
            True if dependencies are available, False otherwise
        """
        try:
            self._check_dependencies()
            return True
        except ImportError:
            return False

    @abstractmethod
    def _check_dependencies(self) -> None:
        """
        Check if required dependencies are installed.

        Raises:
            ImportError: If required packages are not installed
        """
        raise NotImplementedError

    def get_installation_command(self) -> str:
        """
        Get the pip install command for required dependencies.

        Returns:
            Installation command string
        """
        return f"pip install {self.get_format_name()}"
