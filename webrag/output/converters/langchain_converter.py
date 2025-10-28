"""Converter for LangChain Document format."""

from typing import Any, Optional
from webrag.schemas.document import DocumentChunk
from webrag.output.converters.base_converter import BaseConverter


class LangChainConverter(BaseConverter):
    """
    Convert DocumentChunk objects to LangChain Document format.

    LangChain Documents have:
    - page_content: The main text content
    - metadata: Dictionary of metadata
    """

    def __init__(self, **kwargs):
        """
        Initialize LangChain converter.

        Args:
            **kwargs: Configuration options
                - include_embedding: Include embedding in metadata (default: False)
        """
        super().__init__(**kwargs)
        self.include_embedding = kwargs.get('include_embedding', False)

    def convert(self, chunk: DocumentChunk) -> Any:
        """
        Convert a DocumentChunk to a LangChain Document.

        Args:
            chunk: DocumentChunk object to convert

        Returns:
            LangChain Document object

        Raises:
            ImportError: If langchain is not installed
        """
        self._check_dependencies()

        from langchain.schema import Document

        # Prepare metadata
        metadata = self.prepare_metadata(chunk)

        # Optionally include embedding
        if self.include_embedding and chunk.embedding:
            metadata['embedding'] = chunk.embedding

        # Create LangChain Document
        return Document(
            page_content=chunk.content,
            metadata=metadata
        )

    def get_format_name(self) -> str:
        """Get the format name."""
        return 'langchain'

    def _check_dependencies(self) -> None:
        """
        Check if LangChain is installed.

        Raises:
            ImportError: If langchain is not installed
        """
        try:
            import langchain
            from langchain.schema import Document
        except ImportError as e:
            raise ImportError(
                "LangChain is not installed. Install it with: pip install langchain"
            ) from e

    def get_installation_command(self) -> str:
        """Get installation command for LangChain."""
        return "pip install langchain"


class LangChainCoreConverter(BaseConverter):
    """
    Convert DocumentChunk objects to LangChain-Core Document format.

    LangChain-Core is the lightweight core of LangChain with fewer dependencies.
    """

    def __init__(self, **kwargs):
        """
        Initialize LangChain-Core converter.

        Args:
            **kwargs: Configuration options
                - include_embedding: Include embedding in metadata (default: False)
        """
        super().__init__(**kwargs)
        self.include_embedding = kwargs.get('include_embedding', False)

    def convert(self, chunk: DocumentChunk) -> Any:
        """
        Convert a DocumentChunk to a LangChain-Core Document.

        Args:
            chunk: DocumentChunk object to convert

        Returns:
            LangChain-Core Document object

        Raises:
            ImportError: If langchain-core is not installed
        """
        self._check_dependencies()

        from langchain_core.documents import Document

        # Prepare metadata
        metadata = self.prepare_metadata(chunk)

        # Optionally include embedding
        if self.include_embedding and chunk.embedding:
            metadata['embedding'] = chunk.embedding

        # Create Document
        return Document(
            page_content=chunk.content,
            metadata=metadata
        )

    def get_format_name(self) -> str:
        """Get the format name."""
        return 'langchain-core'

    def _check_dependencies(self) -> None:
        """
        Check if LangChain-Core is installed.

        Raises:
            ImportError: If langchain-core is not installed
        """
        try:
            from langchain_core.documents import Document
        except ImportError as e:
            raise ImportError(
                "LangChain-Core is not installed. Install it with: pip install langchain-core"
            ) from e

    def get_installation_command(self) -> str:
        """Get installation command for LangChain-Core."""
        return "pip install langchain-core"
