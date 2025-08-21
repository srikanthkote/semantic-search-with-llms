# Semantic Search Pipeline for PDF Documents

A powerful semantic search system that enables natural language querying of PDF documents using LangChain, HuggingFace embeddings, and ChromaDB vector storage.

## Features

- **PDF Document Processing**: Automatically loads and processes PDF files from a specified directory
- **Intelligent Text Chunking**: Splits documents into manageable chunks with configurable overlap for better context preservation
- **Semantic Embeddings**: Uses HuggingFace's sentence-transformers for high-quality text embeddings
- **Vector Storage**: Leverages ChromaDB for efficient similarity search and retrieval
- **RAG (Retrieval-Augmented Generation)**: Combines document retrieval with language model generation for accurate answers
- **Advanced Reranking**: Utilizes a `CrossEncoderReranker` to refine search results, ensuring the most relevant documents are prioritized for answer generation.
- **Multiple Search Methods**: Supports both similarity search and Maximum Marginal Relevance (MMR) retrieval
- **Flexible Model Support**: Compatible with various HuggingFace models for embeddings and text generation

## Installation

1. **Clone the repository** (if applicable) or download the files
2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Setup

### 1. HuggingFace API Token
You'll need a HuggingFace API token to use the embedding models:
- Sign up at [HuggingFace](https://huggingface.co/)
- Generate an API token from your account settings
- The script will prompt you for the token on first run, or you can set it as an environment variable:
  ```bash
  export HUGGINGFACEHUB_API_TOKEN="your_token_here"
  ```

### 2. PDF Documents
- Place your PDF documents in a directory (default: `~/Downloads/search-sources/`)
- Update the directory path in the `main()` function if needed

## Usage

### Basic Usage
```bash
python semantic_search.py
```

### Customizing the Pipeline

#### Change PDF Directory
```python
directory = "/path/to/your/pdf/documents/"
pipeline = RAGPipeline(Path(directory))
```

#### Adjust Chunking Parameters
```python
chunker = DocumentChunker(chunk_size=512, chunk_overlap=100)
```

#### Use Different Embedding Model
```python
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
```

#### Configure Retrieval Parameters
```python
retriever = Retriever(vectorstore, k=5)  # Retrieve top 5 documents
```

## Architecture

The system consists of several key components:

### Core Classes

- **`DocumentLoader`**: Loads PDF files from a directory using PyPDFLoader
- **`DocumentChunker`**: Splits documents into smaller, overlapping chunks
- **`HuggingFaceEmbeddings`**: Creates vector embeddings using HuggingFace models
- **`VectorStore`**: Manages ChromaDB vector storage and similarity search
- **`SimilaritySearch`**: Performs similarity search with scoring
- **`Retriever`**: Implements MMR-based document retrieval
- **`PromptManager`**: Creates structured prompts for the language model
- **`ResponseGenerator`**: Integrates the reranker and language model to generate answers from the most relevant documents.
- **`RAGPipeline`**: Orchestrates the entire semantic search workflow

### Pipeline Flow

1. **Document Loading**: PDF files are loaded and parsed
2. **Text Chunking**: Documents are split into manageable pieces
3. **Embedding Generation**: Text chunks are converted to vector embeddings
4. **Vector Storage**: Embeddings are stored in ChromaDB for efficient retrieval
5. **Query Processing**: User queries are embedded and matched against stored vectors
6. **Document Retrieval**: Relevant document chunks are retrieved using similarity search
7. **Reranking**: A `CrossEncoderReranker` re-sorts the retrieved documents to prioritize the most relevant ones.
8. **Answer Generation**: A language model generates responses based on the refined context

## Prompting

The `PromptManager` class is responsible for creating the prompt that is sent to the language model. The default prompt is designed to make the model respond as a "story teller":

```
You are a story teller, answering questions in an excited, insightful, and empathetic way. Answer the question based only on the provided context:

<context>
{context}
</context>

Question: {question}
```

This ensures that the answers are not only accurate but also engaging and easy to understand. You can customize the prompt by modifying the `prompt_template` attribute in the `PromptManager` class.

## Configuration Options

### Embedding Models
- `sentence-transformers/all-MiniLM-L6-v2` (default, lightweight)
- `sentence-transformers/all-mpnet-base-v2` (higher quality)
- `sentence-transformers/all-distilroberta-v1` (balanced performance)

### Language Models
- `microsoft/DialoGPT-medium` (default)
- Custom HuggingFace models supported

### Search Parameters
- **Chunk Size**: 256 tokens (default)
- **Chunk Overlap**: 50 tokens (default)
- **Retrieval Count**: 3-4 documents (configurable)
- **Search Type**: MMR (Maximum Marginal Relevance)

## Example Output

The system provides detailed output for each stage of the pipeline, using the `tabulate` library to format the results in an easy-to-read grid. This includes:
- Document processing progress
- Similarity search results with scores
- Retrieved document content and metadata, clearly showing the source of the information
- The final generated answer alongside the source documents that were used to create it

## Requirements

- Python 3.8+
- HuggingFace API token
- PDF documents to search
- Sufficient memory for embedding models (2GB+ recommended)

## GPU Support

For faster processing with CUDA-enabled GPUs:
1. Uncomment GPU-related packages in `requirements.txt`
2. Install PyTorch with CUDA support
3. The system will automatically detect and use GPU acceleration

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce chunk size or use smaller embedding models
2. **API Token Issues**: Ensure your HuggingFace token is valid and has appropriate permissions
3. **PDF Loading Errors**: Check PDF file integrity and permissions
4. **Model Download Issues**: Ensure stable internet connection for initial model downloads

### Performance Tips

- Use GPU acceleration for large document collections
- Adjust chunk size based on document complexity
- Consider using smaller embedding models for faster processing
- Implement caching for frequently accessed embeddings

## Testing

This project includes a comprehensive test suite to ensure the reliability of each component. The tests use Python's `unittest` framework and `unittest.mock` to isolate components and test them independently.

To run the tests, navigate to the root directory of the project and run the following command:

```bash
python -m unittest discover tests
```

This will automatically discover and run all the tests in the `tests/` directory.

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve the semantic search pipeline.

## License

This project is open source and available under the MIT License.
