"""Converter for Haystack Document format."""

from typing import Any, Optional, Dict
from webrag.schemas.document import DocumentChunk
from webrag.output.converters.base_converter import BaseConverter


class HaystackConverter(BaseConverter):
    """
    Convert DocumentChunk objects to Haystack Document format.

    Haystack Documents have:
    - content: The main text content
    - meta: Dictionary of metadata
    - id: Unique identifier
    - embedding: Optional embedding vector
    """

    def __init__(self, **kwargs):
        """
        Initialize Haystack converter.

        Args:
            **kwargs: Configuration options
                - include_embedding: Include embedding vector (default: False)
                - content_type: Haystack content type (default: 'text')
        """
        super().__init__(**kwargs)
        self.include_embedding = kwargs.get('include_embedding', False)
        self.content_type = kwargs.get('content_type', 'text')

    def convert(self, chunk: DocumentChunk) -> Any:
        """
        Convert a DocumentChunk to a Haystack Document.

        Args:
            chunk: DocumentChunk object to convert

        Returns:
            Haystack Document object

        Raises:
            ImportError: If haystack is not installed
        """
        self._check_dependencies()

        from haystack import Document

        # Prepare metadata
        meta = self.prepare_metadata(chunk)

        # Create Haystack Document
        doc = Document(
            id=chunk.chunk_id,
            content=chunk.content,
            meta=meta,
        )

        # Add embedding if available and requested
        if self.include_embedding and chunk.embedding:
            doc.embedding = chunk.embedding

        return doc

    def get_format_name(self) -> str:
        """Get the format name."""
        return 'haystack'

    def _check_dependencies(self) -> None:
        """
        Check if Haystack is installed.

        Raises:
            ImportError: If haystack is not installed
        """
        try:
            from haystack import Document
        except ImportError as e:
            raise ImportError(
                "Haystack is not installed. Install it with: pip install haystack-ai"
            ) from e

    def get_installation_command(self) -> str:
        """Get installation command for Haystack."""
        return "pip install haystack-ai"


class Haystack1Converter(BaseConverter):
    """
    Convert DocumentChunk objects to Haystack 1.x Document format.

    For legacy Haystack 1.x compatibility.
    """

    def __init__(self, **kwargs):
        """
        Initialize Haystack 1.x converter.

        Args:
            **kwargs: Configuration options
                - include_embedding: Include embedding vector (default: False)
        """
        super().__init__(**kwargs)
        self.include_embedding = kwargs.get('include_embedding', False)

    def convert(self, chunk: DocumentChunk) -> Any:
        """
        Convert a DocumentChunk to a Haystack 1.x Document.

        Args:
            chunk: DocumentChunk object to convert

        Returns:
            Haystack 1.x Document object

        Raises:
            ImportError: If haystack 1.x is not installed
        """
        self._check_dependencies()

        from haystack.schema import Document

        # Prepare metadata
        meta = self.prepare_metadata(chunk)

        # Create document arguments
        doc_kwargs = {
            'id': chunk.chunk_id,
            'content': chunk.content,
            'meta': meta,
            'content_type': 'text',
        }

        # Add embedding if available and requested
        if self.include_embedding and chunk.embedding:
            doc_kwargs['embedding'] = chunk.embedding

        return Document(**doc_kwargs)

    def get_format_name(self) -> str:
        """Get the format name."""
        return 'haystack-1'

    def _check_dependencies(self) -> None:
        """
        Check if Haystack 1.x is installed.

        Raises:
            ImportError: If haystack 1.x is not installed
        """
        try:
            from haystack.schema import Document
        except ImportError as e:
            raise ImportError(
                "Haystack 1.x is not installed. Install it with: pip install farm-haystack"
            ) from e

    def get_installation_command(self) -> str:
        """Get installation command for Haystack 1.x."""
        return "pip install farm-haystack"
