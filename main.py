from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from query import Chatbot
import time


# Initialize FastAPI app
app = FastAPI()

# Define the input model for queries
class QueryRequest(BaseModel):
    question: str

# Initialize the chatbot instance at startup
chatbot = Chatbot(
    model_path="./models/Llama-3-Instruct-8B-SPPO-Iter3-Q4_K_M.gguf",
    embedding_model_path="./models/mxbai-embed-large-v1-f16.gguf",
    qdrant_path="embeddings",
    collection_name="rag_chunk_500",
    tmp_interaction_collection_name="rag_interactions_500"
)

@app.post("/ask")
async def ask_question(request: QueryRequest):
    try:
        # Start measuring time
        start_time = time.time()

        # Get the chatbot response based on user input
        response = chatbot.ask_question(request.question, limit=10)

        if not response:
            raise HTTPException(status_code=404, detail="No relevant context found.")

        # End measuring time
        time_taken = time.time() - start_time

        print(f'response: {response}')

        # Store the question and its response into Qdrant
        chatbot.store_interaction(request.question, response)

         # Return response along with time taken
        return {
            "response": response,
            "time_taken": time_taken
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

