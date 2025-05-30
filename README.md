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
- The vector database is automatically saved to the `faiss_db/` directory

## Project Structure

- `main.py` - Core RAG implementation and CLI interface
- `app.py` - Streamlit web interface
- `data/` - Directory for PDF documents
- `faiss_db/` - Directory where vector database is stored

# AI Chat Backend API

This is a FastAPI application that provides a backend for AI chat conversations using MongoDB.

## Features

- Store AI chat conversations in MongoDB
- Manage conversations and messages per user
- RESTful API for creating, reading, updating, and deleting conversations and messages

## API Endpoints

### Conversations

- `GET /api/conversations` - Get all conversations for a user
- `GET /api/conversations/{conversation_id}` - Get a specific conversation
- `POST /api/conversations` - Create a new conversation
- `PUT /api/conversations/{conversation_id}` - Update a conversation's title
- `DELETE /api/conversations/{conversation_id}` - Delete a conversation and all its messages

### Messages

- `GET /api/conversations/{conversation_id}/messages` - Get messages for a specific conversation
- `POST /api/conversations/{conversation_id}/messages` - Add a new message to a conversation

## Setup

1. Install dependencies:

   ```
   poetry add pymongo motor fastapi uvicorn pydantic pydantic-settings
   ```

2. Configure MongoDB:
   Create a `.env` file with:

   ```
   MONGODB_URL=mongodb://localhost:27017
   MONGODB_DB_NAME=ai_chat
   ```

3. Run the application:
   ```
   python run_api.py
   ```

## MongoDB Schema

The application uses two collections:

1. `conversations` - Stores conversation metadata

   ```json
   {
     "_id": ObjectId,
     "user_id": ObjectId,
     "title": String,
     "created_at": DateTime,
     "updated_at": DateTime
   }
   ```

2. `messages` - Stores individual messages
   ```json
   {
     "_id": ObjectId,
     "conversation_id": ObjectId,
     "content": String,
     "is_user": Boolean,
     "created_at": DateTime
   }
   ```

## API Usage Examples

### Create a new conversation

```bash
curl -X POST "http://localhost:8000/api/conversations" \
  -H "Content-Type: application/json" \
  -d '{"title": "New Chat", "initial_message": "Hello AI!", "user_id": "60d21b4667d0d8992e610c85"}'
```

### Get messages from a conversation

```bash
curl -X GET "http://localhost:8000/api/conversations/60d21b4667d0d8992e610c86/messages?user_id=60d21b4667d0d8992e610c85"
```

### Add a message to a conversation

```bash
curl -X POST "http://localhost:8000/api/conversations/60d21b4667d0d8992e610c86/messages" \
  -H "Content-Type: application/json" \
  -d '{"content": "How does this work?", "is_user": true, "user_id": "60d21b4667d0d8992e610c85"}'
```
