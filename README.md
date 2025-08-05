# hackathon
# 🛡️ HackRx - Policy Document Q&A API

A production-ready document-based Question-Answering (RAG) API built for the HackRx API Challenge. It accepts a link to a policy PDF and answers specific insurance-related questions by extracting, chunking, embedding, and analyzing the content using LLMs.

---

## 🚀 Features

- 📄 Accepts any PDF-based policy document (via URL)
- ❓ Accepts multiple natural language questions
- 🧠 Uses Google Gemini or OpenAI (e.g. GPT-4-turbo) for answer generation
- 🧩 RAG pipeline: document parsing → embedding → FAISS vector search → LLM
- 🔒 Supports Bearer token authorization
- 🌐 Fully public HTTPS-compatible endpoint
- ⚡ Responds under 30 seconds

---

## 📦 Tech Stack

- **FastAPI** – API backend
- **FAISS** – Vector search engine
- **Google Gemini 1.5 Flash / OpenAI GPT-4** – LLM
- **PyMuPDF / pdfminer** – PDF text extraction
- **Uvicorn** – ASGI server
- **Pydantic** – Data validation
- **Render / Railway / Heroku** – Deployment (HTTPS supported)

---

## 📌 API Specification

### ✅ Endpoint

```http
POST /hackrx/run

---
Authentication
Authorization: Bearer <api_key>
---

📤 Request Format:
{
  "documents": "https://example.com/policy.pdf",
  "questions": [
    "What is the grace period for premium payment?",
    "Are maternity benefits covered?",
    "What is the cataract surgery waiting period?"
  ]
}
---

📥 Response Format:
{
  "answers": [
    "A grace period of thirty days is provided after the due date.",
    "Yes, maternity is covered after 24 months of continuous coverage.",
    "There is a waiting period of two years for cataract surgery."
  ]
}
----

🧪 Example curl Request:
curl -X POST https://your-domain.com/hackrx/run \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer test_key" \
  -d '{"documents": "https://example.com/policy.pdf", "questions": ["What is the waiting period for cataract surgery?", "Are organ donor expenses covered?"]}'

---
🛠️ Setup (Local)
1. Clone & Install
git clone https://github.com/yourusername/hackrx-policy-api.git
cd hackrx-policy-api
pip install -r requirements.txt
2. Run Locally
uvicorn main:app --reload --port 8000
3. Test with curl or Postman
🔐 Environment Variables
Set up your .env file like this:

API_KEY=test_key
GOOGLE_API_KEY=your_gemini_api_key
# or
OPENAI_API_KEY=your_openai_key
🔄 Updating
This project supports:

Adding new LLMs (e.g., Claude, Mistral)

Adding UI via Streamlit or React

Switching embedding models

Storing history in DB (PostgreSQL, Supabase, etc.)

📤 Deployment (Render / Railway)
Connect GitHub repo

Set build command:

pip install -r requirements.txt
Set start command:

uvicorn main:app --host 0.0.0.0 --port 10000
Add environment variables in dashboard

Deploy and get your public HTTPS URL

