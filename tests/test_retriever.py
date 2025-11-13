import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the function we want to test from its new location
from app.services.rag_service import get_retrieved_documents

class TestRetriever(unittest.TestCase):

    @patch('app.services.rag_service.supabase_client') # Patch the client where it is used
    @patch('app.services.rag_service.embeddings_model') # Patch the embeddings model to avoid network calls
    def test_retrieval_logic(self, mock_embeddings, mock_supabase):
        """
        Tests the get_retrieved_documents function by mocking the database call.
        """
        # 1. Define a sample question
        question = "What is Supabase?"
        
        # 2. Define the mock response from the 'match_documents' RPC call
        mock_rpc_response_data = [
            {"content": "Supabase is an open-source Firebase alternative.", "metadata": {"source": "doc1.pdf"}},
            {"content": "It provides a suite of tools including a Postgres database.", "metadata": {"source": "doc2.pdf"}},
        ]
        
        # 3. Configure the mocks
        mock_embeddings.embed_query.return_value = [0.1] * 384 # Return a dummy embedding
        mock_supabase.rpc.return_value.execute.return_value = MagicMock(data=mock_rpc_response_data)

        # 4. Call the function we are testing
        response = get_retrieved_documents(question)

        # 5. Assertions
        mock_embeddings.embed_query.assert_called_once_with(question)
        mock_supabase.rpc.assert_called_once()
        rpc_args = mock_supabase.rpc.call_args[0]
        self.assertEqual(rpc_args[0], "match_documents")
        self.assertIn("query_embedding", rpc_args[1])
        self.assertEqual(rpc_args[1]["match_count"], 5)
        
        # Check that the response from our function matches the mock data
        self.assertEqual(response.data, mock_rpc_response_data)

if __name__ == '__main__':
    unittest.main()
