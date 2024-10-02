from langchain_text_splitters import RecursiveCharacterTextSplitter 
import llama_cpp
from itertools import islice
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct 
import uuid
import argparse

# import pdb

def chunk(arr_range, chunk_size): 
    arr_range = iter(arr_range) 
    return iter(lambda: list(islice(arr_range, chunk_size)), []) 


def create_embedding(file, collection_name):
# file = 'docs/llama2.txt'
  with open(file, 'r') as f:
      text = f.read()

  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=500,
      chunk_overlap=50,
      length_function=len,
      is_separator_regex=False,
  )

  documents = text_splitter.create_documents([text])
  print(len(documents))   

  # Initialize the Llama model for embedding generation
  llm = llama_cpp.Llama(
    model_path="./models/mxbai-embed-large-v1-f16.gguf", 
    embedding=True, 
    verbose=False
  )

  batch_size = 100
  documents_embeddings = []
  batches = list(chunk(documents, batch_size))

  for batch in batches:
    embeddings = llm.create_embedding([item.page_content for item in batch])
    documents_embeddings.extend(
      [ 
        (document, embeddings['embedding'])  
        for document, embeddings in zip(batch, embeddings['data'])
      ]
    )

  print(f'processed {len(documents)} documents')

  # Create collection in Qdrant
  client = QdrantClient(path="embeddings")
  client.create_collection(
      collection_name=collection_name,
      vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
  )

  # Store documemts in Qdrant
  points = [
    PointStruct(
      id = str(uuid.uuid4()),
      vector = embeddings,
      payload = {
        "text": doc.page_content
      }
    )
    for doc, embeddings in documents_embeddings
  ]

  operation_info = client.upsert(
      collection_name="test",
      wait=True,
      points=points
  )

  print(f'Done creating embeddings for {file}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="input txt file")
    parser.add_argument("collection_name", help="collection name")
    args = parser.parse_args()

    create_embedding(args.file, args.collection_name)
