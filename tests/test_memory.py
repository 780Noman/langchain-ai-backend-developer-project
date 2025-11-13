import unittest
from unittest.mock import patch, MagicMock
import sys
import os
from uuid import uuid4

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the functions we want to test from their new location
from app.services.rag_service import get_relevant_history, add_message_to_history
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

class TestEmbeddingBasedMemory(unittest.TestCase):

    @patch('app.services.rag_service.supabase_client')
    @patch('app.services.rag_service.embeddings_model')
    def test_add_message_to_history(self, mock_embeddings, mock_supabase):
        """
        Tests that add_message_to_history correctly embeds and inserts a message.
        """
        session_id = str(uuid4())
        message_type = "human"
        content = "This is a test message."
        dummy_embedding = [0.1, 0.2, 0.3]

        # Configure mocks
        mock_embeddings.embed_query.return_value = dummy_embedding
        mock_supabase.table.return_value.insert.return_value.execute.return_value = MagicMock()

        # Call the function
        add_message_to_history(session_id, message_type, content)

        # Assertions
        mock_embeddings.embed_query.assert_called_once_with(content)
        mock_supabase.table.assert_called_once_with("chat_history")
        
        # Check the data passed to insert
        insert_args = mock_supabase.table.return_value.insert.call_args[0][0]
        self.assertEqual(insert_args["conversation_id"], session_id)
        self.assertEqual(insert_args["message_type"], message_type)
        self.assertEqual(insert_args["content"], content)
        self.assertEqual(insert_args["embedding"], dummy_embedding)

    @patch('app.services.rag_service.supabase_client')
    @patch('app.services.rag_service.embeddings_model')
    def test_get_relevant_history(self, mock_embeddings, mock_supabase):
        """
        Tests that get_relevant_history recalls and constructs history correctly.
        """
        session_id = str(uuid4())
        question = "What was my last question?"
        dummy_embedding = [0.3, 0.2, 0.1]

        # Mock the response from the match_chat_history RPC
        mock_rpc_response_data = [
            {"content": "This was my previous question.", "message_type": "human"},
            {"content": "This was the AI's answer.", "message_type": "ai"},
        ]
        mock_supabase.rpc.return_value.execute.return_value = MagicMock(data=mock_rpc_response_data)
        mock_embeddings.embed_query.return_value = dummy_embedding

        # Call the function
        history = get_relevant_history(session_id, question)

        # Assertions
        mock_embeddings.embed_query.assert_called_once_with(question)
        mock_supabase.rpc.assert_called_once_with(
            "match_chat_history",
            {
                "query_embedding": dummy_embedding,
                "p_conversation_id": session_id,
                "match_count": 4,
            },
        )
        
        self.assertIsInstance(history, ChatMessageHistory)
        self.assertEqual(len(history.messages), 2)
        self.assertIsInstance(history.messages[0], HumanMessage)
        self.assertEqual(history.messages[0].content, "This was my previous question.")
        self.assertIsInstance(history.messages[1], AIMessage)
        self.assertEqual(history.messages[1].content, "This was the AI's answer.")

if __name__ == '__main__':
    unittest.main()