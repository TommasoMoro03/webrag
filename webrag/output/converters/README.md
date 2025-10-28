# Output Converters

Convert WebRAG's `DocumentChunk` objects to formats compatible with popular RAG libraries and frameworks.

## Overview

WebRAG output converters allow you to seamlessly integrate processed chunks into your existing RAG pipeline, regardless of which framework you're using. Instead of manually transforming data structures, simply use the appropriate converter to get chunks in your preferred format.

## Supported Formats

| Format | Library | Package | Description |
|--------|---------|---------|-------------|
| `langchain` | LangChain | `langchain` | Full LangChain package with all components |
| `langchain-core` | LangChain Core | `langchain-core` | Lightweight core package |
| `llamaindex` | LlamaIndex | `llama-index` | Full LlamaIndex package |
| `llamaindex-core` | LlamaIndex Core | `llama-index-core` | Lightweight core package |
| `haystack` | Haystack 2.x | `haystack-ai` | Modern Haystack framework |
| `haystack-1` | Haystack 1.x | `farm-haystack` | Legacy Haystack (v1.x) |

## Quick Start

### Basic Usage

```python
from webrag.output import get_converter

# Get chunks from your WebRAG pipeline
chunks = pipeline.process("https://example.com")

# Convert to LangChain Documents
converter = get_converter('langchain')
langchain_docs = converter.convert_batch(chunks)

# Use directly with LangChain
from langchain.vectorstores import Chroma
vectorstore = Chroma.from_documents(langchain_docs, embeddings)
```

### List Available Converters

```python
from webrag.output import list_available_converters

converters = list_available_converters()
for name, description in converters.items():
    print(f"{name}: {description}")
```

## Converter Examples

### LangChain

```python
from webrag.output import LangChainConverter

# Create converter
converter = LangChainConverter(include_embedding=True)

# Convert single chunk
langchain_doc = converter.convert(chunk)

# Convert multiple chunks
langchain_docs = converter.convert_batch(chunks)

# Use with LangChain components
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

# Add to vector store
vectorstore = FAISS.from_documents(langchain_docs, embeddings)

# Use with retrievers
retriever = vectorstore.as_retriever()
```

**LangChain Document Structure:**
```python
Document(
    page_content="Your content text here...",
    metadata={
        'chunk_id': 'chunk_001',
        'source_url': 'https://example.com',
        'title': 'Page Title',
        'chunk_index': 0,
        'total_chunks': 5,
        # ... all other metadata fields
    }
)
```

### LlamaIndex

```python
from webrag.output import LlamaIndexConverter

# Create TextNode objects (default)
converter = LlamaIndexConverter(use_document=False)
nodes = converter.convert_batch(chunks)

# Or create Document objects
doc_converter = LlamaIndexConverter(use_document=True)
documents = doc_converter.convert_batch(chunks)

# Use with LlamaIndex
from llama_index.core import VectorStoreIndex

# Create index from nodes
index = VectorStoreIndex(nodes)

# Query
query_engine = index.as_query_engine()
response = query_engine.query("What is...?")
```

**LlamaIndex TextNode Structure:**
```python
TextNode(
    id_='chunk_001',
    text='Your content text here...',
    metadata={
        'source_url': 'https://example.com',
        'title': 'Page Title',
        'chunk_index': 0,
        # ... all other metadata
    },
    embedding=[...],  # if include_embedding=True
)
```

### Haystack

```python
from webrag.output import HaystackConverter

# Create converter
converter = HaystackConverter(include_embedding=True)

# Convert to Haystack Documents
haystack_docs = converter.convert_batch(chunks)

# Use with Haystack 2.x
from haystack import Pipeline
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore

# Add to document store
document_store = InMemoryDocumentStore()
document_store.write_documents(haystack_docs)
```

**Haystack Document Structure:**
```python
Document(
    id='chunk_001',
    content='Your content text here...',
    meta={
        'source_url': 'https://example.com',
        'title': 'Page Title',
        'chunk_index': 0,
        # ... all other metadata
    },
    embedding=[...],  # if include_embedding=True
)
```

## Configuration Options

### Include Embeddings

If your chunks already have embeddings, you can include them in the converted output:

```python
converter = get_converter('langchain', include_embedding=True)
docs = converter.convert_batch(chunks)
```

### LlamaIndex: TextNode vs Document

```python
# Use TextNode (recommended for most cases)
converter = LlamaIndexConverter(use_document=False)
nodes = converter.convert_batch(chunks)

# Use Document (higher-level wrapper)
converter = LlamaIndexConverter(use_document=True)
documents = converter.convert_batch(chunks)
```

## Metadata Preservation

All converters preserve metadata from `DocumentChunk`:

**Standard Fields:**
- `chunk_id` - Unique identifier
- `source_url` - Original URL
- `chunk_index` - Position in sequence
- `total_chunks` - Total chunks from source
- `content_type` - Type of content (text, code, etc.)
- `title` - Document title
- `section_title` - Section/heading
- `tags` - Categorization tags
- `confidence_score` - Quality score
- `token_count` - Approximate token count
- `created_at` - Creation timestamp

**Custom Metadata:**
Any custom fields in `chunk.metadata` are also preserved.

## Dependency Management

### Check Dependencies

```python
converter = get_converter('langchain')

# Check if required package is installed
if converter.validate_dependencies():
    print("Ready to use!")
else:
    print(f"Install with: {converter.get_installation_command()}")
```

### Installation Commands

```bash
# LangChain
pip install langchain
# or lightweight version
pip install langchain-core

# LlamaIndex
pip install llama-index
# or lightweight version
pip install llama-index-core

# Haystack
pip install haystack-ai
# or legacy version
pip install farm-haystack
```

## Complete Pipeline Example

```python
from webrag import WebRAGPipeline
from webrag.output import get_converter

# Initialize pipeline
pipeline = WebRAGPipeline(
    fetcher="requests",
    extractor="trafilatura",
    chunker="semantic",
)

# Process URLs
chunks = pipeline.process_urls([
    "https://docs.example.com/intro",
    "https://docs.example.com/advanced",
])

# Convert to your preferred format
converter = get_converter('langchain')  # or 'llamaindex', 'haystack'
framework_docs = converter.convert_batch(chunks)

# Use in your RAG pipeline
# (framework-specific code here)
```

## Direct Converter Import

You can also import converters directly:

```python
from webrag.output.converters import (
    LangChainConverter,
    LlamaIndexConverter,
    HaystackConverter,
)

# Use specific converter
converter = LangChainConverter(include_embedding=False)
docs = converter.convert_batch(chunks)
```

## Error Handling

```python
try:
    converter = get_converter('langchain')
    docs = converter.convert_batch(chunks)
except ImportError as e:
    print(f"Required package not installed: {e}")
    print(f"Install with: {converter.get_installation_command()}")
except ValueError as e:
    print(f"Unknown format: {e}")
```

## Custom Converters

You can create custom converters by extending `BaseConverter`:

```python
from webrag.output.converters import BaseConverter
from typing import Any

class MyCustomConverter(BaseConverter):
    def convert(self, chunk):
        # Your conversion logic
        return {
            'text': chunk.content,
            'metadata': self.prepare_metadata(chunk),
            # ... custom fields
        }

    def get_format_name(self):
        return 'my-custom-format'

    def _check_dependencies(self):
        # Check for required packages
        import my_rag_library  # raises ImportError if not installed
```

## Performance Tips

1. **Batch Conversion**: Always use `convert_batch()` for multiple chunks instead of calling `convert()` in a loop
2. **Dependency Checking**: Check dependencies once at initialization, not for every conversion
3. **Embedding Inclusion**: Only include embeddings if you need them to reduce memory usage

## Troubleshooting

### ImportError: Package not installed

Install the required package for your target format:
```bash
pip install langchain  # or llamaindex, haystack-ai, etc.
```

### ValueError: Unknown format

Check available formats:
```python
from webrag.output import list_available_converters
print(list_available_converters())
```

### Metadata Missing

Ensure your chunks have the metadata populated before conversion. Converters preserve all existing metadata but don't generate new fields.

## See Also

- [Examples](../../../examples/converter_examples.py) - Complete working examples
- [Base Converter](base_converter.py) - Base class for custom converters
- [WebRAG Documentation](../../../README.md) - Main library documentation
