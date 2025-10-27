"""Pipeline result and response schemas."""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from webrag.schemas.document import DocumentChunk


class PipelineResult(BaseModel):
    """Result of the complete RAG pipeline execution."""

    documents: List[DocumentChunk] = Field(
        default_factory=list,
        description="List of processed document chunks ready for RAG"
    )
    total_sources_processed: int = Field(
        default=0,
        ge=0,
        description="Number of sources successfully processed"
    )
    total_chunks_created: int = Field(
        default=0,
        ge=0,
        description="Total number of chunks created"
    )
    errors: Optional[List[str]] = Field(
        default=None,
        description="List of errors encountered during processing"
    )
    warnings: Optional[List[str]] = Field(
        default=None,
        description="List of warnings during processing"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional pipeline execution metadata"
    )
    started_at: Optional[datetime] = Field(
        default=None,
        description="Pipeline start timestamp"
    )
    completed_at: Optional[datetime] = Field(
        default=None,
        description="Pipeline completion timestamp"
    )
    duration_seconds: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Total execution time in seconds"
    )

    # Future extensibility
    skipped_sources: Optional[List[str]] = Field(
        default=None,
        description="URLs of sources that were skipped"
    )
    retry_count: Optional[int] = Field(
        default=0,
        ge=0,
        description="Number of retries performed"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "total_sources_processed": 5,
                "total_chunks_created": 42,
                "documents": [],
                "errors": [],
                "warnings": ["Rate limit approached for example.com"],
                "metadata": {
                    "pipeline_version": "1.0.0",
                    "config_hash": "abc123"
                },
                "duration_seconds": 12.5
            }
        }
