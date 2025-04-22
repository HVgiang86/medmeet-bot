# MedBot - AI Health Assistant

MedBot is a RAG-based conversational AI for health and medical information, built using LangChain and powered by Ollama.

## Features

- 🔍 Retrieval Augmented Generation (RAG) for accurate medical information
- 💾 Persistent vector database for efficient document retrieval
- 🌐 Support for multiple languages (with Vietnamese responses)
- 💬 Session-based conversation history
- 🖥️ Command-line and Streamlit web interfaces

## Requirements

- Python 3.12+
- Ollama running locally with the following models:
  - llama3.2 (for LLM)
  - nomic-embed-text (for embeddings)

## Installation

1. Clone this repository:

   ```
   git clone <repository-url>
   cd medicalagent
   ```

2. Install dependencies using Poetry:

   ```
   poetry install
   ```

3. Make sure Ollama is running with the required models:
   ```
   ollama pull llama3.2
   ollama pull nomic-embed-text
   ```

## Usage

### Command Line Interface

Run the CLI version with:

```
poetry run python main.py
```

### Web Interface

Run the Streamlit web app with:

```
poetry run streamlit run app.py
```

Then open your browser at http://localhost:8501

## Configuration

- Place PDF documents in the `data/` directory
- The vector database is automatically saved to the `chroma_db/` directory

## Project Structure

- `main.py` - Core RAG implementation and CLI interface
- `app.py` - Streamlit web interface
- `data/` - Directory for PDF documents
- `chroma_db/` - Directory where vector database is stored
