import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.rag_pipeline import build_vectorstore, load_vectorstore, answer_question
from app.utils import save_uploaded_file

app = FastAPI(
    title="RAG Document Q&A API",
    description="Upload a PDF and ask questions about it using Gemini AI",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

vectorstore = None


class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def root():
    return {"message": "RAG Document Q&A API is running"}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    global vectorstore
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    try:
        file_bytes = await file.read()
        file_path = save_uploaded_file(file_bytes, file.filename)
        vectorstore = build_vectorstore(file_path)
        return {"message": f"Document '{file.filename}' uploaded and indexed successfully", "status": "ready"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask")
def ask_question(request: QuestionRequest):
    global vectorstore
    if vectorstore is None:
        try:
            vectorstore = load_vectorstore()
        except Exception:
            raise HTTPException(status_code=400, detail="No document indexed yet. Please upload a PDF first.")
    try:
        answer = answer_question(request.question, vectorstore)
        return {"question": request.question, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    return {"status": "healthy", "index_loaded": vectorstore is not None}