# RAG Document Q&A Assistant

An AI-powered document question-answering system built with LangChain, FAISS, FastAPI, and Google Gemini.

## What it does
Upload any PDF document and ask natural language questions about its contents. The system retrieves the most relevant sections and generates accurate answers using Gemini AI.

## Tech Stack
- **LangChain** — RAG pipeline orchestration
- **FAISS** — Vector similarity search
- **Google Gemini** — Embeddings and answer generation
- **FastAPI** — REST API backend
- **Streamlit** — Interactive frontend UI
- **Python 3.13**

## Architecture
PDF Upload → Text Chunking → Gemini Embeddings → FAISS Index

Question → Embedding → Vector Search → Context Retrieval → Gemini LLM → Answer

## Setup

1. Clone the repository
2. Create a virtual environment and install dependencies:
```bash
pip install -r requirements.txt
```
3. Create a `.env` file with your Gemini API key:
4. Run the FastAPI backend:
```bash
python -m uvicorn app.main:app --reload
```
5. Run the Streamlit frontend:
```bash
streamlit run frontend/streamlit_app.py
```
6. Open `http://localhost:8501` in your browser

## Author
Akash Arjun Suresh Babu — MSc Applied AI, Aston University
