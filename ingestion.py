# ingestion.py
from app.modules.document_processor import extract_text
from app.modules.text_chunker import chunk_text
from app.modules.embedder import get_embeddings_in_parallel
from app.modules.vector_search import build_and_save_faiss_index

import json
import os
from dotenv import load_dotenv

load_dotenv()

def run_ingestion(source: str, output_dir="data"):
    """
    Runs the full ingestion pipeline: extracts text, chunks, embeds in parallel,
    and builds a FAISS index.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Extract text from the source (URL or local path)
    print("📄 Extracting text from document...")
    text = extract_text(source)
    
    # Step 2: Chunk the text
    chunks = chunk_text(text)
    print(f"🔹 Total Chunks: {len(chunks)}")
    
    # Step 3: Get embeddings in parallel
    embeddings = get_embeddings_in_parallel(chunks)
    
    # Step 4: Build and save the FAISS index along with the original texts
    base_name = "policy_index"
    index_path = os.path.join(output_dir, f"{base_name}.faiss")
    texts_path = os.path.join(output_dir, f"{base_name}_texts.json")
    
    build_and_save_faiss_index(embeddings, chunks, index_path, texts_path)
    
    print("✅ Ingestion complete.")
    return index_path, texts_path

if __name__ == "__main__":
    # Example usage with the sample URL from your prompt
    # Note: The URL might be expired, so you might need to use a local file for testing
    #sample_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
    sample_data="data/doc2.pdf"
    run_ingestion(sample_data, output_dir="data")