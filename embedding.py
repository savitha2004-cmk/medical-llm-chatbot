import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

with open("embeddings/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

embeddings = model.encode(chunks)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

faiss.write_index(index, "embeddings/index.faiss")

print("✅ Embeddings + Index created")