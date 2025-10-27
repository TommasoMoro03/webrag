"""JSON exporter for document chunks."""

import json
from typing import List, Optional, Dict, Any
from pathlib import Path
from datetime import datetime

from webrag.output.base import BaseExporter
from webrag.schemas.document import DocumentChunk
from webrag.schemas.response import PipelineResult
from webrag.utils.exceptions import ExportError, FileWriteError


class JSONExporter(BaseExporter):
    """
    Exporter for JSON and JSONL formats.

    Supports:
    - Single JSON file with all chunks
    - JSONL (JSON Lines) format with one chunk per line
    - Pretty-printed or compact JSON
    """

    def __init__(
        self,
        output_dir: Optional[str] = None,
        pretty: bool = True,
        jsonl: bool = False,
        **kwargs
    ):
        """
        Initialize the JSON exporter.

        Args:
            output_dir: Directory to write output files
            pretty: If True, use pretty-printed JSON
            jsonl: If True, use JSONL format (one JSON object per line)
            **kwargs: Additional configuration
        """
        super().__init__(output_dir, **kwargs)
        self.pretty = pretty
        self.jsonl = jsonl
        self.indent = 2 if pretty else None

    def export(
        self,
        chunks: List[DocumentChunk],
        output_path: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Export document chunks to JSON format.

        Args:
            chunks: List of DocumentChunk objects to export
            output_path: Optional specific output file path
            **kwargs: Additional export parameters
                - include_embeddings: Include embedding vectors (default: False)
                - pretty: Override instance pretty setting
                - jsonl: Override instance jsonl setting

        Returns:
            Path to the exported file

        Raises:
            ExportError: If export fails
            FileWriteError: If file writing fails
        """
        if not self.validate_chunks(chunks):
            raise ExportError("Invalid chunks: cannot export empty or invalid chunks")

        # Get export parameters
        include_embeddings = kwargs.get('include_embeddings', False)
        pretty = kwargs.get('pretty', self.pretty)
        jsonl = kwargs.get('jsonl', self.jsonl)

        # Determine output path
        if not output_path:
            self.ensure_output_dir()
            filename = self.generate_output_filename(
                base_name="chunks",
                include_timestamp=True
            )
            output_path = str(self.output_dir / filename)
        else:
            # Ensure parent directory exists
            if not self.validate_output_path(output_path):
                raise FileWriteError(f"Cannot write to path: {output_path}")

        # Serialize chunks
        serialized = self.serialize_chunks(chunks, include_embeddings=include_embeddings)

        try:
            # Write to file
            with open(output_path, 'w', encoding='utf-8') as f:
                if jsonl:
                    # JSONL format: one JSON object per line
                    for chunk_dict in serialized:
                        json_line = json.dumps(chunk_dict, ensure_ascii=False)
                        f.write(json_line + '\n')
                else:
                    # Standard JSON format
                    json.dump(
                        serialized,
                        f,
                        indent=2 if pretty else None,
                        ensure_ascii=False
                    )

            return output_path

        except IOError as e:
            raise FileWriteError(f"Failed to write to {output_path}: {str(e)}") from e
        except Exception as e:
            raise ExportError(f"Export failed: {str(e)}") from e

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
            Path to the exported file

        Raises:
            ExportError: If export fails
        """
        include_embeddings = kwargs.get('include_embeddings', False)
        pretty = kwargs.get('pretty', self.pretty)

        # Determine output path
        if not output_path:
            self.ensure_output_dir()
            filename = self.generate_output_filename(
                base_name="pipeline_result",
                include_timestamp=True
            )
            output_path = str(self.output_dir / filename)
        else:
            if not self.validate_output_path(output_path):
                raise FileWriteError(f"Cannot write to path: {output_path}")

        try:
            # Convert PipelineResult to dict
            result_dict = result.model_dump(mode='python')

            # Optionally remove embeddings from documents
            if not include_embeddings and 'documents' in result_dict:
                for doc in result_dict['documents']:
                    if 'embedding' in doc:
                        del doc['embedding']

            # Convert datetime objects to ISO format strings
            result_dict = self._serialize_datetimes(result_dict)

            # Write to file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(
                    result_dict,
                    f,
                    indent=2 if pretty else None,
                    ensure_ascii=False,
                    default=str  # Handle any remaining non-serializable objects
                )

            return output_path

        except Exception as e:
            raise ExportError(f"Failed to export pipeline result: {str(e)}") from e

    def get_file_extension(self) -> str:
        """
        Get the file extension for JSON export.

        Returns:
            File extension
        """
        return 'jsonl' if self.jsonl else 'json'

    def export_summary(
        self,
        chunks: List[DocumentChunk],
        output_path: Optional[str] = None
    ) -> str:
        """
        Export a summary of chunks (metadata only, no content).

        Useful for getting an overview without the full content.

        Args:
            chunks: List of DocumentChunk objects
            output_path: Optional output file path

        Returns:
            Path to exported summary file
        """
        if not output_path:
            self.ensure_output_dir()
            filename = self.generate_output_filename(
                base_name="summary",
                include_timestamp=True
            )
            output_path = str(self.output_dir / filename)

        try:
            summary = {
                'total_chunks': len(chunks),
                'total_tokens': sum(c.token_count or 0 for c in chunks),
                'sources': list(set(str(c.source_url) for c in chunks)),
                'chunks_per_source': self._count_chunks_per_source(chunks),
                'average_chunk_size': sum(len(c.content) for c in chunks) // len(chunks) if chunks else 0,
                'generated_at': datetime.utcnow().isoformat(),
            }

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            return output_path

        except Exception as e:
            raise ExportError(f"Failed to export summary: {str(e)}") from e

    def _serialize_datetimes(self, obj: Any) -> Any:
        """
        Recursively convert datetime objects to ISO format strings.

        Args:
            obj: Object to process

        Returns:
            Object with datetimes converted to strings
        """
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: self._serialize_datetimes(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_datetimes(item) for item in obj]
        else:
            return obj

    def _count_chunks_per_source(self, chunks: List[DocumentChunk]) -> Dict[str, int]:
        """
        Count chunks per source URL.

        Args:
            chunks: List of chunks

        Returns:
            Dictionary mapping source URLs to chunk counts
        """
        counts: Dict[str, int] = {}
        for chunk in chunks:
            url = str(chunk.source_url)
            counts[url] = counts.get(url, 0) + 1
        return counts
