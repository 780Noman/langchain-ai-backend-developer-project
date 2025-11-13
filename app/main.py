import os
from uuid import UUID, uuid4

from fastapi import FastAPI
from app.models.api_models import QueryRequest, QueryResponse
from app.services.rag_service import (
    rag_chain,
    get_relevant_history,
    add_message_to_history,
    get_retrieved_documents,
)

# --- FastAPI Application ---
app = FastAPI(
    title="LangChain RAG API with Supabase and Gemini",
    description="A Retrieval-Augmented Generation (RAG) API using LangChain, Supabase, and Google Gemini.",
    version="4.0.0",
)

# --- API Endpoints ---

@app.get("/")
async def read_root():
    """A simple endpoint to confirm the API is running."""
    return {"message": "Welcome to the LangChain RAG API!"}

@app.post("/query", response_model=QueryResponse)
async def query_rag_endpoint(request: QueryRequest):
    """
    Handles conversational RAG queries. It uses embedding-based recall
    to fetch relevant parts of the conversation history.
    """
    conversation_id = request.conversation_id or uuid4()
    session_id_str = str(conversation_id)
    
    # 1. Recall relevant chat history based on the current question
    recalled_history = get_relevant_history(session_id_str, request.question)

    # 2. Invoke the RAG chain with the recalled history
    answer = rag_chain.invoke({
        "input": request.question,
        "chat_history": recalled_history.messages
    })

    # 3. Add the new user question and AI answer to the history table for future recall
    add_message_to_history(session_id_str, "human", request.question)
    add_message_to_history(session_id_str, "ai", answer)

    # 4. Retrieve sources for the final response
    match_response = get_retrieved_documents(request.question)
    sources = list(set([doc["metadata"].get("source", "Unknown") for doc in match_response.data])) if match_response.data else []

    return QueryResponse(
        answer=answer,
        sources=sources,
        conversation_id=conversation_id,
    )

@app.get("/eval")
async def evaluate_rag():
    """
    Runs a predefined set of queries to evaluate the retrieval performance
    of the RAG system and returns a JSON report.
    """
    eval_results = []
    k = 3
    eval_queries = [
        {"question": "How do I initialize the Supabase client in Python?", "expected_sources": ["Client_Initialization_and_Setup.pdf", "Supabase_Python_Introduction.pdf"]},
        {"question": "What are the authentication methods supported by Supabase?", "expected_sources": ["Authentication_Methods.pdf"]},
        {"question": "How can I perform CRUD operations on the database?", "expected_sources": ["Database_Operations.pdf"]},
        {"question": "What are Supabase Edge Functions?", "expected_sources": ["Edge_Functions_and_API_Integration.pdf"]},
        {"question": "How does Supabase handle real-time subscriptions?", "expected_sources": ["Real-time_Subscriptions.pdf"]},
        {"question": "What are security best practices for Supabase?", "expected_sources": ["Error_Handling_and_Security.pdf"]},
        {"question": "How do I upload files to Supabase storage?", "expected_sources": ["Storage_Management.pdf"]},
        {"question": "What is the purpose of the .env file?", "expected_sources": ["Client_Initialization_and_Setup.pdf"]},
        {"question": "Can Supabase handle database relationships?", "expected_sources": ["Database_Operations.pdf"]},
        {"question": "What is the embedding dimension for all-MiniLM-L6-v2?", "expected_sources": []}
    ]

    for i, query_data in enumerate(eval_queries):
        question = query_data["question"]
        expected_sources = query_data["expected_sources"]
        
        match_response = get_retrieved_documents(question)

        retrieved_sources = []
        if match_response.data:
            retrieved_sources = [os.path.basename(doc["metadata"].get("source", "")) for doc in match_response.data]
        
        num_relevant_retrieved = sum(1 for source in retrieved_sources if source in expected_sources)
        
        precision_at_k = num_relevant_retrieved / k if k > 0 else 0

        eval_results.append({
            "query_id": i + 1,
            "question": question,
            "expected_sources": expected_sources,
            "retrieved_sources": retrieved_sources,
            "precision_at_k": precision_at_k
        })
    
    overall_precision = sum([res["precision_at_k"] for res in eval_results]) / len(eval_results) if eval_results else 0

    return {
        "evaluation_summary": {"total_queries": len(eval_queries), "k_value": k, "overall_average_precision_at_k": overall_precision},
        "query_results": eval_results
    }
