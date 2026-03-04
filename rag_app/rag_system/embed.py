from pathlib import Path

import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer

base_dir = Path(__file__).resolve().parent
bench_dir = base_dir.parent
data_path = bench_dir / "rag_data" / "Employees.xlsx"
output_dir = base_dir / "embeddings"
output_path = output_dir / "embeddings.pkl"

# Load Excel
df = pd.read_excel(data_path)
print(f"Loaded {len(df)} employees")

# Convert to text
texts = []
for _, row in df.iterrows():
    text = "\n".join([f"{col}: {val}" for col, val in row.items()])
    texts.append(text)

# Generate embeddings
print("Generating embeddings...")
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(texts, show_progress_bar=True)

# Save
output_dir.mkdir(parents=True, exist_ok=True)
with open(output_path, "wb") as f:
    pickle.dump({"texts": texts, "embeddings": embeddings}, f)

print("Done!")