"""Converters for transforming DocumentChunks to various RAG library formats."""

from webrag.output.converters.base_converter import BaseConverter
from webrag.output.converters.langchain_converter import LangChainConverter, LangChainCoreConverter
from webrag.output.converters.llamaindex_converter import LlamaIndexConverter, LlamaIndexCoreConverter
from webrag.output.converters.haystack_converter import HaystackConverter, Haystack1Converter

__all__ = [
    "BaseConverter",
    "LangChainConverter",
    "LangChainCoreConverter",
    "LlamaIndexConverter",
    "LlamaIndexCoreConverter",
    "HaystackConverter",
    "Haystack1Converter",
    "get_converter",
    "list_available_converters",
]


def get_converter(format_name: str, **kwargs) -> BaseConverter:
    """
    Get a converter instance by format name.

    Args:
        format_name: Name of the format ('langchain', 'llamaindex', 'haystack', etc.)
        **kwargs: Configuration options to pass to the converter

    Returns:
        BaseConverter instance

    Raises:
        ValueError: If format_name is not recognized

    Examples:
        >>> converter = get_converter('langchain')
        >>> converter = get_converter('llamaindex', use_document=True)
        >>> converter = get_converter('haystack', include_embedding=True)
    """
    format_map = {
        'langchain': LangChainConverter,
        'langchain-core': LangChainCoreConverter,
        'llamaindex': LlamaIndexConverter,
        'llamaindex-core': LlamaIndexCoreConverter,
        'haystack': HaystackConverter,
        'haystack-1': Haystack1Converter,
    }

    format_name_lower = format_name.lower()

    if format_name_lower not in format_map:
        available = ', '.join(format_map.keys())
        raise ValueError(
            f"Unknown format '{format_name}'. Available formats: {available}"
        )

    return format_map[format_name_lower](**kwargs)


def list_available_converters() -> dict:
    """
    List all available converters with their descriptions.

    Returns:
        Dictionary mapping format names to descriptions

    Examples:
        >>> converters = list_available_converters()
        >>> for name, desc in converters.items():
        ...     print(f"{name}: {desc}")
    """
    return {
        'langchain': 'LangChain Document format (full package)',
        'langchain-core': 'LangChain-Core Document format (lightweight)',
        'llamaindex': 'LlamaIndex TextNode/Document format (full package)',
        'llamaindex-core': 'LlamaIndex-Core TextNode/Document format (lightweight)',
        'haystack': 'Haystack 2.x Document format',
        'haystack-1': 'Haystack 1.x Document format (legacy)',
    }
