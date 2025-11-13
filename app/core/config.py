import os
from dotenv import load_dotenv
from supabase.client import Client, create_client
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from the project's .env file
load_dotenv()

# --- Environment Variables ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LLM_MODEL_NAME = os.getenv("LLM_MODEL", "gemini-1.5-flash")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# --- Validate Environment Variables ---
if not SUPABASE_URL or not SUPABASE_KEY or not GOOGLE_API_KEY:
    raise ValueError("Supabase URL/Key and Google API Key must be set in the .env file")

# --- Initialize Clients (Singleton pattern) ---
# These clients are initialized once and reused throughout the application.
supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

llm_client = ChatGoogleGenerativeAI(
    model=LLM_MODEL_NAME,
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3,
)
