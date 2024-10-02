import llama_cpp
from qdrant_client import QdrantClient

# Define the class to handle LLM and Qdrant client
class Chatbot:
    def __init__(self, model_path, embedding_model_path, qdrant_path, collection_name):
        # Initialize the Llama model (used for generating responses)
        self.llm = llama_cpp.Llama(
            model_path=model_path,
            verbose=False,
            n_ctx=2048
        )
        
        # Initialize the Llama model for embedding generation
        self.embedding_llm = llama_cpp.Llama(
            model_path=embedding_model_path,
            embedding=True,
            verbose=False
        )
        
        # Initialize the Qdrant client
        self.qdrant_client = QdrantClient(path=qdrant_path)
        self.collection_name = collection_name

    def create_embeddings(self, text):
        """Create embeddings for a given text using the embedding model."""
        return self.embedding_llm.create_embedding(text)['data'][0]['embedding']

    def search_similar_documents(self, query_embedding, limit=5):
        """Search for similar documents in the Qdrant collection."""
        return self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit
        )
    
    def generate_response(self, context, question):
        """Generate a response based on the provided context and question."""

        template = """You are a helpful assistant who answers questions using only the provided context.
        If you don't know the answer, simply state that you don't know.

        {context}

        Question: {question}"""

        # Create a chat completion using the Llama model
        response = self.llm.create_chat_completion(
            messages=[
                {"role": "user", "content": template.format(context=context, question=question)}
            ],
            stream=False
        )
      
        return response['choices'][0]['message'].get('content', '')

    def ask_question(self, question, limit=5):
        """Handles the process of embedding the question, searching for relevant documents, and generating a response."""
        # Step 1: Create an embedding for the user's question
        query_embedding = self.create_embeddings(question)

        # Step 2: Search for relevant documents using the query embedding
        search_results = self.search_similar_documents(query_embedding, limit=limit)

        # Step 3: Combine the context from the search results
        context = "\n\n".join([row.payload['text'] for row in search_results])

        # Step 4: Generate a response using the retrieved context
        return self.generate_response(context, question)
