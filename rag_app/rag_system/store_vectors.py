from pathlib import Path

import pickle
import faiss
import numpy as np

base_dir = Path(__file__).resolve().parent
output_dir = base_dir / "embeddings"
embeddings_path = output_dir / "embeddings.pkl"
db_path = base_dir / "db"
index_path = db_path / "faiss_index.bin"
texts_path = db_path / "texts.pkl"

# Load embeddings
with open(embeddings_path, "rb") as f:
    data = pickle.load(f)

# Create FAISS index
print(f"Creating FAISS index with {len(data['embeddings'])} vectors...")
embeddings = np.array(data["embeddings"]).astype('float32')
dimension = embeddings.shape[1]

# Use IndexFlatL2 for exact search
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save index and texts
db_path.mkdir(parents=True, exist_ok=True)
faiss.write_index(index, str(index_path))
with open(texts_path, "wb") as f:
    pickle.dump(data["texts"], f)

print(f"Done! Saved {index.ntotal} vectors to {index_path}")