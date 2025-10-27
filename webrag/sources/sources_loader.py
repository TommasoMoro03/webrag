"""SourceLoader for loading and validating source configurations."""

import json
from pathlib import Path
from typing import List, Union
from pydantic import ValidationError
from webrag.schemas.source_profile import SourceProfile
from webrag.utils.exceptions import InvalidSourceError, ConfigurationError


class SourceLoader:
    """
    Loads user website sources from JSON or dicts and validates them.

    This class is responsible for:
    - Loading source configurations from JSON files or dict lists
    - Validating each source using Pydantic models
    - Providing helpful error messages for configuration issues
    - Supporting multiple input formats for flexibility
    """

    @staticmethod
    def load(sources_input: Union[str, Path, List[dict]]) -> List[SourceProfile]:
        """
        Load and validate source profiles from various input formats.

        Args:
            sources_input: Can be:
                - Path to JSON file (str or Path)
                - List of dictionaries representing sources

        Returns:
            List[SourceProfile]: Validated and normalized source objects

        Raises:
            InvalidSourceError: If file not found, JSON is invalid, or validation fails
            ConfigurationError: If input format is not supported

        Examples:
            # From JSON file
            >>> sources = SourceLoader.load("sources.json")

            # From dict list
            >>> sources = SourceLoader.load([
            ...     {"url": "https://example.com", "type": "static"}
            ... ])
        """
        sources_data = None

        # If it's a file path
        if isinstance(sources_input, (str, Path)):
            sources_data = SourceLoader._load_from_file(sources_input)

        # If it's already a list of dicts
        elif isinstance(sources_input, list):
            sources_data = sources_input

        else:
            raise ConfigurationError(
                f"sources_input must be a file path or a list of dicts, "
                f"got {type(sources_input).__name__}"
            )

        # Validate and convert to SourceProfile objects
        return SourceLoader._validate_sources(sources_data)

    @staticmethod
    def _load_from_file(file_path: Union[str, Path]) -> List[dict]:
        """
        Load sources data from a JSON file.

        Args:
            file_path: Path to the JSON file

        Returns:
            List of source dictionaries

        Raises:
            InvalidSourceError: If file not found or JSON is invalid
        """
        path = Path(file_path)

        if not path.exists():
            raise InvalidSourceError(f"Source file not found: {file_path}")

        if not path.is_file():
            raise InvalidSourceError(f"Path is not a file: {file_path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise InvalidSourceError(
                f"Invalid JSON in {file_path}: {e.msg} at line {e.lineno}, column {e.colno}"
            )
        except Exception as e:
            raise InvalidSourceError(f"Error reading file {file_path}: {e}")

        # Ensure data is a list
        if isinstance(data, dict):
            # If it's a dict with a "sources" key, extract that
            if "sources" in data:
                data = data["sources"]
            else:
                # Single source object, wrap in list
                data = [data]

        if not isinstance(data, list):
            raise InvalidSourceError(
                f"JSON file must contain a list or object with 'sources' key, "
                f"got {type(data).__name__}"
            )

        return data

    @staticmethod
    def _validate_sources(sources_data: List[dict]) -> List[SourceProfile]:
        """
        Validate a list of source dictionaries using Pydantic.

        Args:
            sources_data: List of source dictionaries

        Returns:
            List of validated SourceProfile objects

        Raises:
            InvalidSourceError: If any source fails validation
        """
        if not sources_data:
            raise InvalidSourceError("No sources provided")

        validated_sources = []
        errors = []

        for i, src in enumerate(sources_data):
            if not isinstance(src, dict):
                errors.append(f"Source at index {i}: expected dict, got {type(src).__name__}")
                continue

            try:
                validated_source = SourceProfile(**src)
                validated_sources.append(validated_source)
            except ValidationError as e:
                # Format Pydantic validation errors nicely
                error_details = SourceLoader._format_validation_error(e)
                errors.append(f"Source at index {i}: {error_details}")

        # If any errors occurred, raise with all error messages
        if errors:
            error_msg = "Validation errors:\n" + "\n".join(f"  - {err}" for err in errors)
            raise InvalidSourceError(error_msg)

        return validated_sources

    @staticmethod
    def _format_validation_error(error: ValidationError) -> str:
        """
        Format a Pydantic ValidationError into a readable message.

        Args:
            error: Pydantic ValidationError

        Returns:
            Formatted error message string
        """
        error_messages = []
        for err in error.errors():
            field = ".".join(str(loc) for loc in err['loc'])
            msg = err['msg']
            error_messages.append(f"{field}: {msg}")

        return "; ".join(error_messages)

    @staticmethod
    def load_and_filter(
        sources_input: Union[str, Path, List[dict]],
        enabled_only: bool = True,
        tags: List[str] = None,
        priority: str = None
    ) -> List[SourceProfile]:
        """
        Load sources and apply filters.

        Args:
            sources_input: Source input (file path or dict list)
            enabled_only: Only return enabled sources
            tags: Filter by tags (sources must have at least one matching tag)
            priority: Filter by priority level

        Returns:
            Filtered list of SourceProfile objects

        Examples:
            # Load only high-priority enabled sources
            >>> sources = SourceLoader.load_and_filter(
            ...     "sources.json",
            ...     enabled_only=True,
            ...     priority="high"
            ... )
        """
        sources = SourceLoader.load(sources_input)

        # Apply filters
        filtered = sources

        if enabled_only:
            filtered = [s for s in filtered if s.enabled]

        if tags:
            filtered = [
                s for s in filtered
                if s.tags and any(tag in s.tags for tag in tags)
            ]

        if priority:
            filtered = [s for s in filtered if s.priority == priority]

        return filtered

    @staticmethod
    def validate_file(file_path: Union[str, Path]) -> tuple[bool, str]:
        """
        Validate a source configuration file without loading it fully.

        Useful for pre-flight checks.

        Args:
            file_path: Path to the JSON file

        Returns:
            Tuple of (is_valid, error_message)
            If valid: (True, "")
            If invalid: (False, "error description")

        Examples:
            >>> is_valid, error = SourceLoader.validate_file("sources.json")
            >>> if not is_valid:
            ...     print(f"Invalid config: {error}")
        """
        try:
            SourceLoader.load(file_path)
            return True, ""
        except (InvalidSourceError, ConfigurationError) as e:
            return False, str(e)
        except Exception as e:
            return False, f"Unexpected error: {e}"
