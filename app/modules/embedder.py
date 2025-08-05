import os
from dotenv import load_dotenv
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY not found in .env file")

genai.configure(api_key=GOOGLE_API_KEY)

def get_embedding(text: str) -> list:
    """Get vector embedding from Google embedding API."""
    response = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="retrieval_query"
    )
    return response['embedding']

def get_embeddings_in_parallel(chunks: list, max_workers=os.cpu_count()):
    """Get embeddings for a list of text chunks in parallel."""
    print(f"Starting parallel embedding with {max_workers} workers...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        embeddings = list(executor.map(get_embedding, chunks))
    print("Parallel embedding completed.")
    return embeddings