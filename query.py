import llama_cpp
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct 
import uuid

# Define the class to handle LLM and Qdrant client
class Chatbot:
    def __init__(self, model_path, embedding_model_path, qdrant_path, collection_name, tmp_interaction_collection_name):
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
        self.interaction_collection_name = tmp_interaction_collection_name

        # interaction collection is temporary and will be deleted after the session
        self.qdrant_client.delete_collection(collection_name=self.interaction_collection_name)

        # Collection for previous question-answer interactions
        self.qdrant_client.create_collection(
            collection_name=self.interaction_collection_name,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
        )


    def create_embeddings(self, text):
        """Create embeddings for a given text using the embedding model."""
        return self.embedding_llm.create_embedding(text)['data'][0]['embedding']

    def store_interaction(self, question, response):
        """Store the question-answer pair in Qdrant as embeddings."""
        # Create embeddings for both the question and the response combined
        combined_text = f"Q: {question} A: {response}"
        embedding = self.create_embeddings(combined_text)

        # Store the interaction as a point in Qdrant
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                "question": question,
                "response": response,
            }
        )

        self.qdrant_client.upsert(
            collection_name=self.interaction_collection_name,
            wait=True,
            points=[point]
        )

    def search_similar_documents(self, query_embedding, limit=5):
        """Search for relevant documents from both collections: interactions and documents."""

        document_results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit
        )

        interaction_results = self.qdrant_client.search(
            collection_name=self.interaction_collection_name,
            query_vector=query_embedding,
            limit=limit
        )

        # print(f'Document results: {document_results}')
        # print(f'Interaction results: {interaction_results}')
        # print(f'Combined results: {document_results + interaction_results}')

        return document_results + interaction_results

    
    def generate_response(self, context, question):
        """Generate a response based on the provided context and question."""

        template = """Context information is given below:

        {context}

        Given the context information above, now I want you to think step by step to answer the query in a careful manner.
        In case you don't know the answer, state 'I don't know' or 'I am not sure'.

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

        # Step 3: Combine the context from the search results (documents and past interactions)
        context_parts = []
        for row in search_results:
            if 'text' in row.payload:
                context_parts.append(row.payload['text'])
            else:
                context_parts.append(f"Q: {row.payload['question']} A: {row.payload['response']}")

        # Step 4: Generate the full context by joining relevant parts
        context = "\n\n".join(context_parts)

        # Step 5: Generate a response 
        return self.generate_response(context, question)
