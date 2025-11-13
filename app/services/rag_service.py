import os
import json
from uuid import UUID

from app.core.config import supabase_client, embeddings_model, llm_client

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# --- Prompts for the Chains ---
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

qa_system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, just say "
    "that you don't know. Use three sentences maximum and keep "
    "the answer concise."
    "\n\n"
    "{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

# --- Core Logic Functions ---

def get_retrieved_documents(question: str):
    """Embeds a question and retrieves documents from Supabase."""
    question_embedding = embeddings_model.embed_query(question)
    return supabase_client.rpc(
        "match_documents",
        {"query_embedding": question_embedding, "match_count": 5},
    ).execute()

# --- New Embedding-Based Recall Functions ---

def get_relevant_history(session_id: str, question: str) -> BaseChatMessageHistory:
    """
    Recalls relevant chat history using embedding-based search.
    """
    question_embedding = embeddings_model.embed_query(question)
    
    # Find relevant messages from history
    match_response = supabase_client.rpc(
        "match_chat_history",
        {
            "query_embedding": question_embedding,
            "p_conversation_id": session_id,
            "match_count": 4, # Recall top 4 relevant messages
        },
    ).execute()

    recalled_messages = []
    if match_response.data:
        for doc in match_response.data:
            if doc["message_type"] == "human":
                recalled_messages.append(HumanMessage(content=doc["content"]))
            elif doc["message_type"] == "ai":
                recalled_messages.append(AIMessage(content=doc["content"]))
    
    return ChatMessageHistory(messages=recalled_messages)

def add_message_to_history(session_id: str, message_type: str, content: str):
    """
    Adds a new message and its embedding to the chat_history table.
    """
    content_embedding = embeddings_model.embed_query(content)
    
    supabase_client.table("chat_history").insert({
        "conversation_id": session_id,
        "message_type": message_type,
        "content": content,
        "embedding": content_embedding
    }).execute()

# --- History Aware RAG Chain ---
history_aware_rephraser = contextualize_q_prompt | llm_client | StrOutputParser()

rag_chain = (
    RunnablePassthrough.assign(
        context=history_aware_rephraser
        | (lambda query: get_retrieved_documents(query))
        | (lambda response: "\n\n".join([doc["content"] for doc in response.data if "content" in doc]))
    )
    | qa_prompt
    | llm_client
    | StrOutputParser()
)