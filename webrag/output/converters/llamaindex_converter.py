"""Converter for LlamaIndex Node formats."""

from typing import Any, Optional, Dict
from webrag.schemas.document import DocumentChunk
from webrag.output.converters.base_converter import BaseConverter


class LlamaIndexConverter(BaseConverter):
    """
    Convert DocumentChunk objects to LlamaIndex Node format.

    LlamaIndex has several node types:
    - TextNode: For text content (most common)
    - Document: Higher-level wrapper
    - ImageNode, IndexNode, etc.

    This converter creates TextNode objects by default.
    """

    def __init__(self, use_document: bool = False, **kwargs):
        """
        Initialize LlamaIndex converter.

        Args:
            use_document: If True, create Document objects instead of TextNode (default: False)
            **kwargs: Configuration options
                - include_embedding: Include embedding vector (default: False)
        """
        super().__init__(**kwargs)
        self.use_document = use_document
        self.include_embedding = kwargs.get('include_embedding', False)

    def convert(self, chunk: DocumentChunk) -> Any:
        """
        Convert a DocumentChunk to a LlamaIndex TextNode or Document.

        Args:
            chunk: DocumentChunk object to convert

        Returns:
            LlamaIndex TextNode or Document object

        Raises:
            ImportError: If llama-index is not installed
        """
        self._check_dependencies()

        if self.use_document:
            return self._convert_to_document(chunk)
        else:
            return self._convert_to_text_node(chunk)

    def _convert_to_text_node(self, chunk: DocumentChunk) -> Any:
        """
        Convert to LlamaIndex TextNode.

        TextNode is the most common node type for text chunks.
        """
        from llama_index.core.schema import TextNode

        # Prepare metadata
        metadata = self.prepare_metadata(chunk)

        # Create TextNode
        node = TextNode(
            id_=chunk.chunk_id,
            text=chunk.content,
            metadata=metadata,
        )

        # Add embedding if available and requested
        if self.include_embedding and chunk.embedding:
            node.embedding = chunk.embedding

        # Add relationships if this is part of a sequence
        if chunk.parent_chunk_id:
            from llama_index.core.schema import NodeRelationship, RelatedNodeInfo
            node.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(
                node_id=chunk.parent_chunk_id
            )

        if chunk.child_chunk_ids:
            from llama_index.core.schema import NodeRelationship, RelatedNodeInfo
            for child_id in chunk.child_chunk_ids:
                node.relationships[NodeRelationship.CHILD] = RelatedNodeInfo(
                    node_id=child_id
                )

        return node

    def _convert_to_document(self, chunk: DocumentChunk) -> Any:
        """
        Convert to LlamaIndex Document.

        Document is a higher-level wrapper, often used at ingestion time.
        """
        from llama_index.core import Document

        # Prepare metadata
        metadata = self.prepare_metadata(chunk)

        # Create Document
        doc = Document(
            id_=chunk.chunk_id,
            text=chunk.content,
            metadata=metadata,
        )

        # Add embedding if available and requested
        if self.include_embedding and chunk.embedding:
            doc.embedding = chunk.embedding

        return doc

    def get_format_name(self) -> str:
        """Get the format name."""
        return 'llamaindex'

    def _check_dependencies(self) -> None:
        """
        Check if LlamaIndex is installed.

        Raises:
            ImportError: If llama-index is not installed
        """
        try:
            from llama_index.core.schema import TextNode
            from llama_index.core import Document
        except ImportError as e:
            raise ImportError(
                "LlamaIndex is not installed. Install it with: pip install llama-index"
            ) from e

    def get_installation_command(self) -> str:
        """Get installation command for LlamaIndex."""
        return "pip install llama-index"


class LlamaIndexCoreConverter(BaseConverter):
    """
    Convert DocumentChunk objects to LlamaIndex-Core formats.

    llama-index-core is the lightweight core package with fewer dependencies.
    """

    def __init__(self, use_document: bool = False, **kwargs):
        """
        Initialize LlamaIndex-Core converter.

        Args:
            use_document: If True, create Document objects instead of TextNode (default: False)
            **kwargs: Configuration options
        """
        super().__init__(**kwargs)
        self.use_document = use_document
        self.include_embedding = kwargs.get('include_embedding', False)

    def convert(self, chunk: DocumentChunk) -> Any:
        """
        Convert a DocumentChunk to a LlamaIndex-Core TextNode or Document.

        Args:
            chunk: DocumentChunk object to convert

        Returns:
            LlamaIndex-Core TextNode or Document object

        Raises:
            ImportError: If llama-index-core is not installed
        """
        self._check_dependencies()

        if self.use_document:
            return self._convert_to_document(chunk)
        else:
            return self._convert_to_text_node(chunk)

    def _convert_to_text_node(self, chunk: DocumentChunk) -> Any:
        """Convert to LlamaIndex-Core TextNode."""
        from llama_index.core.schema import TextNode

        metadata = self.prepare_metadata(chunk)

        node = TextNode(
            id_=chunk.chunk_id,
            text=chunk.content,
            metadata=metadata,
        )

        if self.include_embedding and chunk.embedding:
            node.embedding = chunk.embedding

        return node

    def _convert_to_document(self, chunk: DocumentChunk) -> Any:
        """Convert to LlamaIndex-Core Document."""
        from llama_index.core import Document

        metadata = self.prepare_metadata(chunk)

        doc = Document(
            id_=chunk.chunk_id,
            text=chunk.content,
            metadata=metadata,
        )

        if self.include_embedding and chunk.embedding:
            doc.embedding = chunk.embedding

        return doc

    def get_format_name(self) -> str:
        """Get the format name."""
        return 'llamaindex-core'

    def _check_dependencies(self) -> None:
        """
        Check if LlamaIndex-Core is installed.

        Raises:
            ImportError: If llama-index-core is not installed
        """
        try:
            from llama_index.core.schema import TextNode
            from llama_index.core import Document
        except ImportError as e:
            raise ImportError(
                "LlamaIndex-Core is not installed. Install it with: pip install llama-index-core"
            ) from e

    def get_installation_command(self) -> str:
        """Get installation command."""
        return "pip install llama-index-core"
