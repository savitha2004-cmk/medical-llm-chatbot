import pickle
from src.ingestion import extract_text_from_pdf, get_all_pdfs

def create_chunks():
    all_chunks = []

    pdfs = get_all_pdfs()

    for pdf in pdfs:
        print(f"Processing: {pdf}")

        text = extract_text_from_pdf(pdf)
        lines = text.split("\n")

        for line in lines:
            line = line.strip()

            # 🔥 FIX: allow shorter medical lines
            if len(line) > 5:
                all_chunks.append(line)

    return all_chunks


if __name__ == "__main__":
    chunks = create_chunks()

    with open("embeddings/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print(f"✅ {len(chunks)} chunks saved")