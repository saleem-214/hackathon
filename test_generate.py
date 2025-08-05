# rag_pipeline.py
from app.modules.vector_search import load_faiss_index, faiss_semantic_search
from app.modules.embedder import get_embedding
from app.modules.llm import build_prompt, generate_answer

import json
import os

def run_query(query: str, index, texts: list):
    """Runs a single query through the RAG pipeline."""
    # Step 1: Get embedding for query
    print("1. Embedding query...")
    query_embedding = get_embedding(query)
    
    # Step 2: Semantic search on FAISS index
    print("2. Performing semantic search...")
    top_indices, scores = faiss_semantic_search(index, query_embedding, top_k=3)
    
    # Step 3: Retrieve top chunks
    top_chunks_with_indices = {i: texts[i] for i in top_indices}
    # Remove potential duplicates and preserve order
    unique_chunks = list(dict.fromkeys(top_chunks_with_indices.values()))
    
    print(f"3. Retrieved {len(unique_chunks)} relevant chunks.")
    
    # Step 4: Build prompt and get answer
    print("4. Generating answer with LLM...")
    prompt = build_prompt(query, unique_chunks)
    answer = generate_answer(prompt)
    
    # Step 5: Format output as structured JSON
    structured_response = {
        "question": query,
        "answer": answer,
        "explainable_reasoning": "Answer derived from retrieved policy clauses."
    }
    
    return structured_response

def main():
    # Ensure the ingestion script has been run
    try:
        index, texts = load_faiss_index("data/policy_index.faiss", "data/policy_index_texts.json")
    except FileNotFoundError as e:
        print(e)
        print("Please run `python ingestion.py` first to process the document.")
        return

    # List of questions from your sample request
    # questions = [
    #     "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
    #     "What is the waiting period for pre-existing diseases (PED) to be covered?",
    #     "Does this policy cover maternity expenses, and what are the conditions?",
    #     "What is the waiting period for cataract surgery?",
    #     "Are the medical expenses for an organ donor covered under this policy?",
    #     "What is the No Claim Discount (NCD) offered in this policy?",
    #     "Is there a benefit for preventive health check-ups?",
    #     "How does the policy define a 'Hospital'?",
    #     "What is the extent of coverage for AYUSH treatments?",
    #     "Are there any sub-limits on room rent and ICU charges for Plan A?"
    # ]

    # all_answers = []
    # for q in questions:
    #     print(f"\n--- Processing Query: {q} ---")
    #     response = run_query(q, index, texts)
    #     all_answers.append(response)
        
    # # Print the final structured output
    # print("\n\n--- Final Structured Output ---")
    # print(json.dumps(all_answers, indent=2))
    
    while True:
        query = input("\n🔍 Enter your query (type 'stop' to end):\n> ")
        if query.lower() == 'stop':
            print("Exiting RAG pipeline.")
            break

        print(f"\n--- Processing Query: {query} ---")
        response = run_query(query, index, texts)

        # Print the structured output for the current query
        print("\n\n--- Gemini Answer ---")
        print(json.dumps(response, indent=2))

if __name__ == "__main__":
    main()