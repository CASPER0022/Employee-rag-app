from pathlib import Path

from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np

base_dir = Path(__file__).resolve().parent
db_path = base_dir / "db"
index_path = db_path / "faiss_index.bin"
texts_path = db_path / "texts.pkl"

# Load model, FAISS index, and texts
model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index(str(index_path))
with open(texts_path, "rb") as f:
    texts = pickle.load(f)

def search(query, top_k=3):
    # Convert query to embedding
    query_embedding = model.encode(query).astype('float32').reshape(1, -1)
    
    # Search in FAISS
    distances, indices = index.search(query_embedding, top_k)
    
    # Return matching documents
    return [texts[i] for i in indices[0]]

# Interactive query loop
while True:
    query = input("\nAsk about employees (or 'quit'): ")
    if query.lower() == 'quit':
        break
    
    results = search(query)
    print("\n--- Top Matches ---")
    for i, doc in enumerate(results, 1):
        print(f"\n[{i}]")
        print(doc)