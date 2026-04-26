import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from src.llm import ask_llm

model = SentenceTransformer("all-MiniLM-L6-v2")

index = faiss.read_index("embeddings/index.faiss")

with open("embeddings/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)


def similarity_to_confidence(distance):
    return float(1 / (1 + distance))


def keyword_match(query, chunks):
    query_lower = query.lower()
    for chunk in chunks:
        if any(word in chunk.lower() for word in query_lower.split()):
            return chunk
    return None


def get_answer(query):
    try:
        query_vec = model.encode([query])
        query_vec = np.array(query_vec)

        distances, indices = index.search(query_vec, 3)

        results = [chunks[i] for i in indices[0]]
        scores = [similarity_to_confidence(d) for d in distances[0]]

        avg_conf = sum(scores) / len(scores)

        print("Query:", query)
        print("Distances:", distances)
        print("Results:", results)

        # 🔥 STEP 1: Try semantic match
        if avg_conf >= 0.4:
            context = "\n".join(results)

            prompt = f"""
Answer ONLY from context.

Context:
{context}

Question:
{query}
"""
            answer = ask_llm(prompt)

            return {
                "answer": answer,
                "confidence": round(avg_conf, 2),
                "sources": results
            }

        # 🔥 STEP 2: fallback keyword match
        keyword_result = keyword_match(query, chunks)

        if keyword_result:
            return {
                "answer": keyword_result,
                "confidence": 0.9,
                "sources": [keyword_result]
            }

        # 🔥 STEP 3: strict fallback
        return {
            "answer": "NOT FOUND in documents",
            "confidence": round(avg_conf, 2),
            "sources": []
        }

    except Exception as e:
        return {
            "answer": f"Error: {str(e)}",
            "confidence": 0.0,
            "sources": []
        }