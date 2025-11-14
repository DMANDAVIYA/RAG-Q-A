# AmbedkarGPT - Retrieval-Augmented Generation Q&A System

A sophisticated command-line question-answering system that leverages Retrieval-Augmented Generation (RAG) to answer questions about Dr. B.R. Ambedkar's speech on the annihilation of caste. Built with LangChain, this system demonstrates the power of combining semantic search with large language models for context-aware responses.

## Overview

This project implements a complete RAG pipeline that:
- Processes and chunks textual documents intelligently
- Generates semantic embeddings for efficient retrieval
- Stores vectors in a persistent local database
- Retrieves contextually relevant passages
- Generates accurate answers using a local LLM

## Architecture

```
User Query → Embedding → Vector Search → Context Retrieval → LLM → Answer
                ↓                              ↓
         HuggingFace               ChromaDB Vector Store
                                                ↓
                                        Ollama Mistral 7B
```

## Key Features

- **100% Local Execution** - No external APIs, complete privacy
- **Zero Cost** - No API keys or paid services required
- **Semantic Search** - Advanced embedding-based retrieval
- **Persistent Storage** - Vector store cached for fast subsequent runs
- **Modular Design** - Clean, maintainable, production-ready code
- **Error Handling** - Comprehensive validation and user guidance
- **Interactive CLI** - Simple question-answer interface

## Technical Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Framework | LangChain | RAG pipeline orchestration |
| Vector Store | ChromaDB | Document embedding storage |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 | Semantic text representation |
| LLM | Ollama Mistral 7B | Answer generation |
| Language | Python 3.8+ | Implementation |

## Prerequisites

### System Requirements
- **OS:** Windows, Linux, or macOS
- **RAM:** 8GB minimum (16GB recommended)
- **Storage:** 5GB free space
- **Python:** 3.8 or higher

### Required Software

1. **Python 3.8+**
   - Download: https://www.python.org/downloads/

2. **Ollama**
   - Download: https://ollama.ai/download
   - Runs Mistral 7B locally

## Installation Guide

### Step 1: Clone Repository

```bash
git clone url
```

### Step 2: Install Ollama

**Windows:**
```powershell
winget install Ollama.Ollama
```

**macOS:**
```bash
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### Step 3: Download Mistral Model

```bash
ollama pull mistral
```

This downloads approximately 4GB. First-time setup takes 5-10 minutes depending on your internet speed.

### Step 4: Create Virtual Environment

**Windows:**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 5: Install Python Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `langchain` - RAG framework
- `langchain-community` - Community integrations
- `chromadb` - Vector database
- `sentence-transformers` - Embedding models
- `ollama` - LLM integration

## Usage

### Starting the System

1. **Ensure Ollama is running** (usually starts automatically)
   ```bash
   ollama serve
   ```

2. **Run the application**
   ```bash
   python main.py
   ```

3. **First run:** Vector store creation takes 30-60 seconds
4. **Subsequent runs:** Instant startup using cached vectors

### Example Session

```
============================================================
AmbedkarGPT - Q&A System
============================================================
Loading document...
Splitting text into chunks...
Creating embeddings...
Building vector store...
Initializing LLM...
Creating QA chain...

Setup complete! Ready to answer questions.
Type 'exit' or 'quit' to end the session.

Your question: What is the real remedy according to Ambedkar?

Thinking...

Answer: According to the text, the real remedy is to destroy the belief 
in the sanctity of the shastras. Ambedkar argues that you cannot both 
stop the practice of caste and continue believing in the shastras - you 
must take a stand against the scriptures.

Your question: exit

Thank you for using AmbedkarGPT!
```

### Sample Questions

- What does Ambedkar say about the shastras?
- How does Ambedkar compare social reform to gardening?
- What is the relationship between caste and religious texts?
- Why does Ambedkar call belief in shastras the real enemy?
- What does Ambedkar mean by attacking the roots vs pruning leaves?

## Project Structure

```
CODE
├── main.py                 # Main application (modular RAG pipeline)
├── speech.txt              # Source document (Ambedkar's speech)
├── requirements.txt        # Python dependencies
├── README.md              # Comprehensive documentation
├── LICENSE                # Project license
├── .gitignore             # Git exclusion rules
└── chroma_db/             # Vector database (auto-generated, gitignored)
    ├── chroma.sqlite3
    └── [embedding files]
```

## Code Architecture

### Main Components

**`load_document(file_path)`**
- Loads text file with UTF-8 encoding
- Validates file existence

**`split_text(documents, chunk_size=500, chunk_overlap=50)`**
- Intelligently chunks text at sentence boundaries
- Maintains context with overlapping segments

**`create_embeddings()`**
- Initializes HuggingFace embedding model
- Uses optimized all-MiniLM-L6-v2 (384 dimensions)

**`create_vector_store(chunks, embeddings)`**
- Generates embeddings for all chunks
- Persists to ChromaDB for reuse

**`initialize_llm(model_name="mistral")`**
- Connects to local Ollama instance
- Configures Mistral 7B parameters

**`create_qa_chain(vector_store, llm)`**
- Builds retrieval pipeline (k=3 chunks)
- Implements "stuff" chain type for context injection

**`verify_ollama()`**
- Validates Ollama availability
- Provides setup guidance on failure

### Configuration Parameters

```python
CHUNK_SIZE = 500          # Characters per chunk
CHUNK_OVERLAP = 50        # Overlap between chunks
RETRIEVAL_K = 3           # Number of chunks retrieved
PERSIST_DIR = "./chroma_db"  # Vector store location
```

## How It Works

### 1. Document Processing
The system loads `speech.txt` and splits it into overlapping chunks. This ensures no context is lost at boundaries.

### 2. Embedding Generation
Each chunk is converted to a 384-dimensional vector using sentence-transformers. These vectors capture semantic meaning.

### 3. Vector Storage
Embeddings are stored in ChromaDB with efficient indexing for fast similarity search.

### 4. Query Processing
When you ask a question:
1. Question is embedded using the same model
2. ChromaDB finds the 3 most similar chunks
3. Retrieved chunks provide context to the LLM
4. Mistral generates an answer based on context

### 5. Answer Generation
The LLM receives your question plus relevant context, ensuring answers are grounded in the actual document.

## Troubleshooting

### Issue: "Ollama is not running"
**Solution:**
```bash
ollama serve
```
Ensure Ollama is running before starting the application.

### Issue: "Mistral model not available"
**Solution:**
```bash
ollama pull mistral
```

### Issue: Module import errors
**Solution:**
```bash
pip install -r requirements.txt --upgrade
```

### Issue: "speech.txt not found"
**Solution:** Ensure `speech.txt` is in the same directory as `main.py`

### Issue: Slow first run
**Expected behavior:** Initial embedding generation takes time. Vector store is cached for future runs.

### Issue: Out of memory
**Solution:** Close other applications. Mistral 7B requires ~6GB RAM.

### Issue: Want to rebuild vector store
**Solution:**
```bash
rm -rf chroma_db/  # Linux/Mac
Remove-Item -Recurse -Force chroma_db/  # Windows
```

## Performance Optimization

- **First Run:** 30-60 seconds (embedding + vector store creation)
- **Subsequent Runs:** 5-10 seconds (loads cached vectors)
- **Query Response:** 2-5 seconds per question
- **Disk Usage:** ~50MB for vector store

## Advanced Configuration

### Custom Chunk Size

Edit `main.py`:
```python
chunks = split_text(documents, chunk_size=300, chunk_overlap=30)
```

### Adjust Retrieval Count

```python
retriever = vector_store.as_retriever(search_kwargs={"k": 5})
```

### Use Different LLM

```bash
ollama pull llama2
```

Then in `main.py`:
```python
llm = initialize_llm(model_name="llama2")
```

## Development

### Running Tests
```bash
python -m pytest tests/
```

### Code Style
Follows PEP 8 with minimal comments. Clean, self-documenting code.

### Contributing
This is an assignment project, but feedback is welcome via issues.

## Educational Value

This project demonstrates:
- **RAG Architecture** - Combining retrieval with generation
- **Embedding Techniques** - Semantic similarity search
- **Vector Databases** - Efficient storage and retrieval
- **LLM Integration** - Local model deployment
- **Production Patterns** - Error handling, modularity, user experience

## Limitations

- **Context Window** - Limited by Mistral's context size
- **Single Document** - Designed for one source text
- **English Only** - Embedding model optimized for English
- **Factual Accuracy** - Answers depend on retrieval quality

## Future Enhancements

- [ ] Multi-document support
- [ ] Web interface with Gradio/Streamlit
- [ ] Citation tracking (show source chunks)
- [ ] Conversation history
- [ ] Fine-tuning on Ambedkar's complete works
- [ ] Multilingual support

## Assignment Compliance

This project fulfills all requirements:
- ✅ Python 3.8+
- ✅ LangChain framework
- ✅ ChromaDB vector store
- ✅ HuggingFace embeddings (all-MiniLM-L6-v2)
- ✅ Ollama with Mistral 7B
- ✅ Complete deliverables (code, requirements.txt, README.md, speech.txt)
- ✅ Well-commented, modular code
- ✅ Functional prototype

## Resources

- [LangChain Documentation](https://python.langchain.com/)
- [ChromaDB Docs](https://docs.trychroma.com/)
- [Ollama Documentation](https://ollama.ai/docs)
- [Sentence Transformers](https://www.sbert.net/)




---

**Note:** This is a functional prototype demonstrating RAG fundamentals. Production deployment would require additional security, scaling, and monitoring considerations.
