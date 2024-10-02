# A Simple Chatbot to Practice Retrieval Augmented Generation

### Description 

Develop a chatbot that can answer queries related to a document that has not been included in the training data.

### Installation

To reproduce the results, first set up an environment with the required packages.

Create a virtual environment using Python 3.10 and activate it:

```bash
conda create -n rag python=3.10
conda activate rag
```

Install the necessary packages:
```bash
pip install -r requirements.txt
```

### Building and running the chatbot

#### Step 0 (Optional): Convert Llama 2 paper from pdf to text and save into a txt file

This step can be skipped if you already have a txt version of the document.

```
python convert.py
```

#### Step 1: Download checkpoints for embedding model and LLama model

Create a new folder in the current name **_models_**. All the checkpoints that we are going to download will be saved there.

For embedding model: Download the model 
 **_xbai-embed-large-v1_** [here](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1/blob/7130e2d16051fdf3e0157e841f8b5a8d0d5e63ef/gguf/mxbai-embed-large-v1-f16.gguf).

 For Llama model: Download the model **_Llama-3-Instruct-8B-SPPO-Iter3_** [here](https://huggingface.co/UCLA-AGI/Llama-3-Instruct-8B-SPPO-Iter3).

#### Step 2: Generate embeddings for the information of the paper

```
python create_embeddings.py <path to the txt document> <name of the collection>
```

E.g.

```
python create_embeddings.py docs/llama.txt  rag_chunk_500
```

Once it's done, all the embeddings will be saved in the database under the folder named **_embeddings_**.

#### Step 3: Run the chatbot 

```
uvicorn main:app --reload
```

#### Step 4: Testing - Asking questions to the chatbot

```
curl -X 'POST'   'http://127.0.0.1:8000/ask'   -H 'Content-Type: application/json'   -d '{"question": "What is novel in the paper Llama 2?"}'
```

If successfull, you will get a response that looks like:

```
{
    "response":"What's novel about Llama 2 is...",
    "time_taken": 50.0
}
```