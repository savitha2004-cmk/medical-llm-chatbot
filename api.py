from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from src.retrieval import get_answer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    query: str

@app.get("/")
def root():
    return {"message": "API working"}

@app.post("/chat")
def chat(q: Query):
    result = get_answer(q.query)

    return {
        "answer": result.get("answer"),
        "confidence": result.get("confidence"),
        "sources": result.get("sources", [])
    }