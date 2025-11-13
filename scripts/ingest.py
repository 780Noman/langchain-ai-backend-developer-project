import os
import sys
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import Client, create_client

# Add the project root to the Python path to allow for module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load environment variables from .env file
load_dotenv()

# Get Supabase credentials from environment variables
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_ANON_KEY")

# Check if Supabase credentials are set
if not supabase_url or not supabase_key:
    raise ValueError("Supabase URL and Key must be set in the .env file")

# Initialize Supabase client
print("Initializing Supabase client...")
supabase: Client = create_client(supabase_url, supabase_key)

def ingest_documents():
    """
    Loads documents from the 'documents' directory, splits them into chunks,
    generates embeddings using a local model, and stores them in Supabase.
    """
    try:
        # 1. Load PDF documents
        documents_path = os.path.join(os.path.dirname(__file__), '..', 'documents')
        print(f"Loading documents from '{documents_path}' directory...")
        loader = PyPDFDirectoryLoader(documents_path)
        documents = loader.load()
        if not documents:
            print("No documents found in the 'documents' directory. Exiting.")
            return
        print(f"Loaded {len(documents)} document pages.")

        # 2. Split documents into smaller chunks
        print("Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        print(f"Split into {len(docs)} chunks.")

        # 3. Initialize the local embedding model
        # We are using a free, open-source model from Hugging Face.
        model_name = "all-MiniLM-L6-v2"
        print(f"Initializing local embedding model: {model_name}...")
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        print("Local embedding model initialized.")

        # 4. Ingest documents and embeddings into Supabase
        print("Ingesting documents into Supabase vector store... This may take a few minutes.")
        SupabaseVectorStore.from_documents(
            documents=docs,
            embedding=embeddings,
            client=supabase,
            table_name="documents",
            query_name="match_documents",
            chunk_size=500
        )
        print("Ingestion complete!")

    except Exception as e:
        print(f"An error occurred during ingestion: {e}")

if __name__ == "__main__":
    ingest_documents()