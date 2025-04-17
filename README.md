# Structuring the Unstructured: Advanced Document Parsing for AI Workflows

## Overview
This project provides tools and examples for converting unstructured documents (e.g., PDFs, images, presentations) into structured formats like JSON or Markdown. It leverages advanced AI models for document layout analysis, table structure recognition, and retrieval-augmented generation (RAG) pipelines.

## Features
- **Document Conversion**: Convert PDFs, images, and other formats into structured data.
- **Layout Analysis**: Detect and analyze document layouts using AI models.
- **Table Recognition**: Extract and structure table data from documents.
- **RAG Pipelines**: Implement retrieval-augmented generation for querying document content.
- **Streamlit App**: Interactive Q&A application for document exploration.

## Setup

### Prerequisites
- Python 3.8 or higher
- Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Environment Variables
Create a `.env` file in the project root with the following variables:

```
HF_TOKEN=your_huggingface_token
EMBED_MODEL_ID=sentence-transformers/all-MiniLM-L6-v2
GEN_MODEL_ID=mistralai/Mixtral-8x7B-Instruct-v0.1
TOP_K=3
```

## Usage

### 1. Document Conversion
Use the `DocumentConverter` class from the `docling` library to convert documents:

```python
from docling.document_converter import DocumentConverter

converter = DocumentConverter()
result = converter.convert("path/to/document.pdf")
print(result.document.export_to_markdown())
```

### 2. Streamlit App
Run the Streamlit app for interactive Q&A:

```bash
streamlit run 6-streamlit-rag-app.py
```

### 3. Examples
Explore the Jupyter notebooks in the repository for detailed examples:
- `0-basic-example.ipynb`: Basic HTML parsing and token counting.
- `1-experimentation.ipynb`: PDF and HTML extraction using Docling.
- `2-multi-format-conversion.ipynb`: Multi-format document conversion.
- `5-rag_langchain_docling.ipynb`: RAG pipeline with LangChain and Docling.

## License
This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## Contributing
Contributions are welcome! Please submit issues or pull requests to improve the project.