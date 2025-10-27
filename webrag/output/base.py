"""Abstract base class for output exporters."""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from pathlib import Path
from webrag.schemas.document import DocumentChunk
from webrag.schemas.response import PipelineResult


class BaseExporter(ABC):
    """
    Abstract base class for exporting processed documents to various formats.

    Exporters are responsible for serializing DocumentChunk objects
    into specific output formats suitable for downstream RAG systems,
    vector databases, or storage.

    Future implementations may include:
    - JSONExporter (single JSON file or JSONL)
    - CSVExporter (tabular format)
    - ParquetExporter (efficient columnar storage)
    - VectorDBExporter (direct upload to Pinecone, Weaviate, etc.)
    - ElasticsearchExporter (direct indexing)
    - MarkdownExporter (human-readable format)
    """

    def __init__(self, output_dir: Optional[str] = None, **kwargs):
        """
        Initialize the exporter.

        Args:
            output_dir: Directory to write output files
            **kwargs: Exporter-specific configuration
        """
        self.output_dir = Path(output_dir) if output_dir else Path("./output")
        self.config = kwargs

    @abstractmethod
    def export(
        self,
        chunks: List[DocumentChunk],
        output_path: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Export document chunks to the specified format.

        Args:
            chunks: List of DocumentChunk objects to export
            output_path: Optional specific output file path
            **kwargs: Additional export parameters

        Returns:
            Path to the exported file/directory

        Raises:
            ExportError: If export fails
            FileWriteError: If file writing fails
            UnsupportedFormatError: If format is not supported
        """
        raise NotImplementedError

    @abstractmethod
    def export_pipeline_result(
        self,
        result: PipelineResult,
        output_path: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Export complete pipeline result including metadata.

        Args:
            result: PipelineResult object containing chunks and metadata
            output_path: Optional specific output file path
            **kwargs: Additional export parameters

        Returns:
            Path to the exported file/directory

        Raises:
            Same exceptions as export()
        """
        raise NotImplementedError

    @abstractmethod
    def get_file_extension(self) -> str:
        """
        Get the file extension for this export format.

        Returns:
            File extension (e.g., 'json', 'csv', 'parquet')
        """
        raise NotImplementedError

    def validate_output_path(self, output_path: str) -> bool:
        """
        Validate that the output path is writable.

        Args:
            output_path: Path to validate

        Returns:
            True if path is valid and writable, False otherwise
        """
        try:
            path = Path(output_path)
            # Check if parent directory exists or can be created
            path.parent.mkdir(parents=True, exist_ok=True)
            return True
        except (OSError, PermissionError):
            return False

    def ensure_output_dir(self) -> None:
        """
        Ensure output directory exists, creating it if necessary.

        Raises:
            FileWriteError: If directory cannot be created
        """
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError) as e:
            from webrag.utils.exceptions import FileWriteError
            raise FileWriteError(f"Cannot create output directory: {e}")

    def generate_output_filename(
        self,
        base_name: str = "output",
        include_timestamp: bool = True
    ) -> str:
        """
        Generate a unique output filename.

        Args:
            base_name: Base name for the file
            include_timestamp: Whether to include timestamp in filename

        Returns:
            Generated filename with extension
        """
        from datetime import datetime

        extension = self.get_file_extension()

        if include_timestamp:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            return f"{base_name}_{timestamp}.{extension}"
        else:
            return f"{base_name}.{extension}"

    def chunk_to_dict(self, chunk: DocumentChunk) -> Dict[str, Any]:
        """
        Convert a DocumentChunk to a dictionary for serialization.

        Args:
            chunk: DocumentChunk object

        Returns:
            Dictionary representation
        """
        return chunk.model_dump(mode='python')

    def serialize_chunks(
        self,
        chunks: List[DocumentChunk],
        include_embeddings: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Serialize a list of chunks to dictionaries.

        Args:
            chunks: List of DocumentChunk objects
            include_embeddings: Whether to include embedding vectors

        Returns:
            List of dictionaries
        """
        serialized = []
        for chunk in chunks:
            chunk_dict = self.chunk_to_dict(chunk)

            # Optionally exclude embeddings to reduce file size
            if not include_embeddings and 'embedding' in chunk_dict:
                del chunk_dict['embedding']

            serialized.append(chunk_dict)

        return serialized

    def calculate_export_size(self, chunks: List[DocumentChunk]) -> int:
        """
        Estimate the size of the export in bytes.

        This is a rough estimate for progress tracking.

        Args:
            chunks: List of chunks to export

        Returns:
            Estimated size in bytes
        """
        # Rough estimate: sum of content lengths + metadata overhead
        total_size = sum(len(chunk.content) for chunk in chunks)
        # Add ~50% overhead for metadata and formatting
        return int(total_size * 1.5)

    def validate_chunks(self, chunks: List[DocumentChunk]) -> bool:
        """
        Validate that chunks are ready for export.

        Args:
            chunks: List of chunks to validate

        Returns:
            True if all chunks are valid, False otherwise
        """
        if not chunks:
            return False

        for chunk in chunks:
            if not chunk.content or not chunk.chunk_id:
                return False

        return True

    def batch_export(
        self,
        chunks: List[DocumentChunk],
        batch_size: int = 1000,
        **kwargs
    ) -> List[str]:
        """
        Export chunks in batches (for large datasets).

        Args:
            chunks: List of all chunks
            batch_size: Number of chunks per batch
            **kwargs: Additional export parameters

        Returns:
            List of paths to exported batch files

        Raises:
            Same exceptions as export()
        """
        batch_paths = []

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_num = i // batch_size + 1

            # Generate batch-specific output path
            base_name = f"batch_{batch_num:04d}"
            output_path = self.output_dir / self.generate_output_filename(
                base_name=base_name,
                include_timestamp=False
            )

            # Export this batch
            path = self.export(batch, str(output_path), **kwargs)
            batch_paths.append(path)

        return batch_paths
