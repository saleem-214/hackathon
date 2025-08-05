import faiss
import numpy as np
import json
import os

# Note: You need to install faiss-cpu or faiss-gpu first.
# `pip install faiss-cpu`

def build_and_save_faiss_index(embeddings: list, texts: list, index_path: str, texts_path: str):
    """Builds a FAISS index and saves it along with the corresponding texts."""
    print("Building FAISS index...")
    
    # Convert embeddings to a numpy array of the correct type
    embedding_array = np.array(embeddings).astype('float32')
    dimension = embedding_array.shape[1]
    
    # Create an index. IndexFlatL2 is a simple, exact search index.
    index = faiss.IndexFlatL2(dimension)
    index.add(embedding_array)
    
    # Save the FAISS index
    faiss.write_index(index, index_path)
    
    # Save the corresponding texts
    with open(texts_path, 'w', encoding='utf-8') as f:
        json.dump(texts, f, indent=2)

    print(f"FAISS index and texts saved to {index_path} and {texts_path}")

def load_faiss_index(index_path: str, texts_path: str):
    """Loads a FAISS index and the corresponding texts."""
    if not os.path.exists(index_path) or not os.path.exists(texts_path):
        raise FileNotFoundError(f"Index or texts file not found. Run the ingestion script first.")
    
    print("Loading FAISS index...")
    index = faiss.read_index(index_path)
    
    with open(texts_path, 'r', encoding='utf-8') as f:
        texts = json.load(f)
    
    return index, texts

def faiss_semantic_search(index, query_embedding, top_k=10):
    """Performs a semantic search on the FAISS index."""
    query_vector = np.array(query_embedding).reshape(1, -1).astype('float32')
    
    # D is distances, I is indices
    distances, indices = index.search(query_vector, top_k)
    
    return indices[0], distances[0]