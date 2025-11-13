# LangChain RAG API with Supabase and Gemini

This project is a production-grade, retrieval-augmented generation (RAG) API it provides a conversational interface to query a knowledge base of PDF documents, leveraging a sophisticated, database-backed memory system.

The application is built with Python, FastAPI, LangChain, Supabase (`pgvector`), and Google's Gemini models.

## Core Features

- **Modular FastAPI Backend**: A clean, scalable API built with a professional, modular structure.
- **Advanced Conversational Memory**: Implements **embedding-based recall** to provide the LLM with only the most relevant parts of the conversation history, ensuring efficiency and context-awareness.
- **Persistent, Database-Backed History**: All conversations are stored and retrieved from a Supabase PostgreSQL database, ensuring state is maintained across sessions.
- **Local Embeddings**: Uses the `all-MiniLM-L6-v2` sentence-transformer model to generate embeddings locally, requiring no API keys or cost for the embedding process.
- **Vector Search**: Leverages Supabase with the `pgvector` extension for efficient document and history retrieval.
- **Built-in Evaluation**: Includes a `/eval` endpoint to compute `precision@k` for the retrieval component, allowing for quantitative performance assessment.
- **Comprehensive Unit Tests**: Core components for chunking, retrieval, and memory are verified with a suite of passing unit tests.

## Project Structure

```
/rag-project
|-- app/                 # Main FastAPI application source code
|   |-- main.py          # API endpoints
|   |-- core/
|   |   |-- config.py    # Environment loading and client initializations
|   |-- models/
|   |   |-- api_models.py# Pydantic request/response models
|   |-- services/
|       |-- rag_service.py # Core RAG logic, chains, and history management
|-- documents/           # Source PDF files for ingestion
|-- scripts/             # Standalone scripts
|   |-- ingest.py
|-- tests/               # Unit tests
|   |-- test_chunker.py
|   |-- test_memory.py
|   |-- test_retriever.py
|-- .env                 # Environment variables (API keys, etc.)
|-- requirements.txt     # Python dependencies
|-- README.md            # This file
```

## Setup and Installation

### 1. Prerequisites

- Python 3.8+
- A Supabase account (free tier is sufficient)
- A Google AI Studio API key

### 2. Initial Setup

- Clone this repository.
- Create a new project on [Supabase](https://supabase.com/).
- In the Supabase dashboard, navigate to the **SQL Editor** and run the contents of the `database_setup.sql` file (or the script below) to enable `pgvector` and create all necessary tables and functions.

### 3. Database Setup Script

Run this full script in the Supabase SQL Editor:

```sql
-- Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create the table for document chunks
CREATE TABLE documents (
  id UUID NOT NULL PRIMARY KEY DEFAULT gen_random_uuid(),
  content TEXT,
  metadata JSONB,
  embedding VECTOR(384) -- Dimension for all-MiniLM-L6-v2
);

-- Create the table for chat history embeddings
CREATE TABLE chat_history (
  id UUID NOT NULL PRIMARY KEY DEFAULT gen_random_uuid(),
  conversation_id UUID NOT NULL,
  message_type TEXT NOT NULL, -- 'human' or 'ai'
  content TEXT,
  embedding VECTOR(384),
  created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX ON chat_history (conversation_id);

-- Create the function to search for documents
CREATE OR REPLACE FUNCTION match_documents (
  query_embedding VECTOR(384),
  match_count INT,
  filter JSONB DEFAULT '{}'
) RETURNS TABLE (
  id UUID,
  content TEXT,
  metadata JSONB,
  similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT
    id,
    content,
    metadata,
    1 - (documents.embedding <=> query_embedding) AS similarity
  FROM documents
  WHERE metadata @> filter
  ORDER BY documents.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;

-- Create the function to search for relevant chat history
CREATE OR REPLACE FUNCTION match_chat_history (
  query_embedding VECTOR(384),
  p_conversation_id UUID,
  match_count INT
) RETURNS TABLE (
  id UUID,
  content TEXT,
  message_type TEXT,
  similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT
    chat_history.id,
    chat_history.content,
    chat_history.message_type,
    1 - (chat_history.embedding <=> query_embedding) AS similarity
  FROM chat_history
  WHERE chat_history.conversation_id = p_conversation_id
  ORDER BY chat_history.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;
```

### 4. Environment Configuration

- Create a file named `.env` in the project root.
- Add your credentials to the `.env` file:
  ```dotenv
  # Supabase Credentials (from your project's API settings)
  SUPABASE_URL="YOUR_SUPABASE_URL"
  SUPABASE_ANON_KEY="YOUR_SUPABASE_ANON_KEY"

  # Google AI Credentials (from Google AI Studio)
  GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
  ```

### 5. Install Dependencies

- Create and activate a Python virtual environment:
  ```bash
  python -m venv venv
  # On Windows: .\venv\Scripts\activate
  # On macOS/Linux: source venv/bin/activate
  ```
- Install the required packages:
  ```bash
  pip install -r requirements.txt
  ```

## Usage

### 1. Ingest Documents

- Place your PDF files into the `/documents` directory.
- Run the ingestion script from the project root:
  ```bash
  python scripts/ingest.py
  ```

  *(Note: The first run will download the embedding model, which may take a few minutes.)*

### 2. Run the API Server

- Start the FastAPI server with Uvicorn:
  ```bash
  uvicorn app.main:app --reload
  ```
- The API will be available at `http://127.0.0.1:8000`.
- Interactive documentation is available at `http://127.0.0.1:8000/docs`.

### 3. Run Unit Tests

- To verify all components, run the test suite from the project root:
  ```bash
  python -m unittest discover tests
  ```

## API Endpoints

### `POST /query`

Handles conversational RAG queries.

- **Request Body:**
  ```json
  {
    "question": "What is Supabase?",
    "conversation_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6" // Optional
  }
  ```
- **Response Body:**
  ```json
  {
    "answer": "Supabase is an open-source backend-as-a-service...",
    "sources": ["document1.pdf", "document2.pdf"],
    "conversation_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6"
  }
  ```

### `GET /eval`

Runs a predefined set of queries to evaluate the retrieval performance.

- **Request:** `GET http://127.0.0.1:8000/eval`
- **Response:** A JSON report detailing `precision@k` for each query and an overall average.

## Design Tradeoffs Explained

### 1. Embedding Model Selection

- **Choice:** `all-MiniLM-L6-v2` (a local Sentence Transformer model).
- **Reasoning:** To ensure the project is fully functional without requiring paid cloud services, a high-quality, open-source local model was chosen over a proprietary API like OpenAI or Google's embedding service.
- **Tradeoffs:**
  - **Pros:** No cost, no API keys, no network latency for embedding.
  - **Cons:** Requires a one-time model download. Performance might be slightly lower than the largest proprietary models, but is excellent for this scale.

### 2. Vector Index Parameters

- **Choice:** Supabase `pgvector` with custom RPC functions (`match_documents`, `match_chat_history`).
- **Reasoning:** The dimension `384` was chosen to match the output of our selected embedding model. Using custom RPC functions provides a clean interface for similarity search and gives us precise control over the retrieval logic (e.g., `LIMIT` clause).
- **Tradeoffs:**
  - **Pros:** `pgvector` is extremely convenient as it co-exists with our primary data in PostgreSQL, simplifying the architecture.
  - **Cons:** For applications with billions of vectors, a dedicated vector database might offer more advanced indexing strategies (e.g., HNSW) and better performance. For this project's scale, `pgvector` is ideal.

### 3. Memory Strategy

- **Choice:** **Embedding-Based Recall**.
- **Reasoning:** The project requires "compressed memory." Rather than simple summarization, we implemented the more advanced embedding-based recall technique. When a new question is asked, we perform a vector search on the past conversation to find the most *semantically relevant* exchanges. This is a more intelligent and context-aware form of memory compression.
- **Tradeoffs:**
  - **Pros:** Highly efficient, as only the most relevant parts of the history are loaded into the prompt. This leads to better answers for follow-up questions and scales well, as the context passed to the LLM does not grow linearly with the conversation length.
  - **Cons:** More complex to implement, requiring an additional vector table (`chat_history`) and a dedicated search function. It incurs a small overhead per query to perform the history search, but this is negligible compared to the token savings and performance gains in the LLM call.
