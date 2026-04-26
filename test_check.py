import pickle

with open("embeddings/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

print("TOTAL CHUNKS:", len(chunks))

for c in chunks:
    if "diagnosis" in c.lower():
        print("FOUND:", c)