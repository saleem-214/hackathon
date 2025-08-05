import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

generation_model = genai.GenerativeModel("gemini-1.5-flash")

def build_prompt(query: str, contexts: list) -> str:
    """Build a prompt for the LLM with explicit instructions for citations."""
    context_block = "\n\n".join([f"Clause {i+1}: {c}" for i, c in enumerate(contexts)])
    prompt = (
        "You are a policy analyst. Use the following clauses to answer the question accurately. "
        "Only answer based on the provided context. If not found, say 'Not mentioned in the policy.' "
        "Example: 'The policy covers maternity expenses.' "
        "Do not invent new clauses.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question:\n{query}\n\n"
        "Answer:"
    )
    return prompt

def generate_answer(prompt: str) -> str:
    response = generation_model.generate_content(prompt)
    return response.text.strip()