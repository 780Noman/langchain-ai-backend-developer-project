import unittest
import sys
import os

# Add the project root to the Python path to allow for module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_text_splitters import RecursiveCharacterTextSplitter

class TestChunker(unittest.TestCase):

    def test_recursive_character_splitter(self):
        """
        Tests that the RecursiveCharacterTextSplitter splits text into expected chunks.
        """
        # Sample text that is longer than the chunk size
        long_text = "This is a long sentence for testing the text splitter. " * 100
        
        # Initialize the splitter with the same parameters as in our ingestion script
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Create documents from the text
        documents = text_splitter.create_documents([long_text])
        
        # Assertions
        self.assertGreater(len(documents), 1, "The text should be split into more than one chunk.")
        
        # Check that each chunk is within the size limit (approximately)
        for doc in documents:
            self.assertLessEqual(len(doc.page_content), 1000, "Each chunk should be at most 1000 characters long.")
            
        # Check that the overlap is working by seeing if the end of one chunk
        # matches the beginning of the next.
        if len(documents) > 1:
            end_of_first_chunk = documents[0].page_content[-100:] # Last 100 chars
            start_of_second_chunk = documents[1].page_content[:100] # First 100 chars
            # This is a probabilistic check, but overlap should ensure some commonality
            self.assertGreater(len(set(end_of_first_chunk.split()) & set(start_of_second_chunk.split())), 0, "There should be some overlap between consecutive chunks.")

if __name__ == '__main__':
    unittest.main()
