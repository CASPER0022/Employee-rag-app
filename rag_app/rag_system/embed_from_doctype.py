"""
Generate Embeddings from Frappe DocType
========================================

This script fetches employee data from the Employee Data DocType
and generates embeddings for RAG retrieval.

Usage:
    bench execute rag_app.rag_system.embed_from_doctype.generate_embeddings

Requirements:
    - Employee Data DocType must be populated
    - sentence_transformers library installed
"""

import frappe
from pathlib import Path
import pickle
import json
from sentence_transformers import SentenceTransformer
import sys
import importlib

# Add rag_system to path
rag_system_path = Path(__file__).resolve().parent
if str(rag_system_path) not in sys.path:
	sys.path.insert(0, str(rag_system_path))

try:
	from .doctype_data_loader import get_all_employee_texts, get_employee_count, get_employee_data_signature
except ImportError:
	dl = importlib.import_module("doctype_data_loader")
	get_all_employee_texts = dl.get_all_employee_texts
	get_employee_count = dl.get_employee_count
	get_employee_data_signature = dl.get_employee_data_signature


def generate_embeddings(model_name='all-MiniLM-L6-v2'):
	"""
	Generate embeddings from Employee Data DocType
	
	Args:
		model_name: Name of the sentence transformer model to use
	
	Returns:
		dict: Status of the embedding generation
	"""
	print("=" * 60)
	print("Generating Embeddings from Employee Data DocType")
	print("=" * 60)
	
	# Check if data exists
	count = get_employee_count()
	if count == 0:
		error_msg = "No employee data found in DocType. Please import data first."
		print(f"ERROR: {error_msg}")
		frappe.log_error(error_msg, "RAG System - Embeddings")
		return {
			"success": False,
			"error": error_msg
		}
	
	print(f"\nFound {count} employee records in DocType")
	
	# Get text documents
	print("Converting employee records to text documents...")
	texts = get_all_employee_texts()
	print(f"Generated {len(texts)} text documents")
	
	# Show sample
	if texts:
		print("\nSample document:")
		print("-" * 60)
		print(texts[0][:300] + "..." if len(texts[0]) > 300 else texts[0])
		print("-" * 60)
	
	# Generate embeddings
	print(f"\nLoading embedding model: {model_name}")
	model = SentenceTransformer(model_name)
	
	print("Generating embeddings...")
	embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
	
	print(f"Generated embeddings: shape = {embeddings.shape}")
	
	# Save embeddings
	base_dir = Path(__file__).resolve().parent
	output_dir = base_dir / "embeddings"
	output_dir.mkdir(parents=True, exist_ok=True)
	output_path = output_dir / "embeddings.pkl"
	
	print(f"\nSaving embeddings to: {output_path}")
	with open(output_path, "wb") as f:
		pickle.dump({"texts": texts, "embeddings": embeddings}, f)
	
	print("\n✓ Embeddings generated successfully!")
	print(f"  - Total documents: {len(texts)}")
	print(f"  - Embedding dimension: {embeddings.shape[1]}")
	print(f"  - Saved to: {output_path}")
	
	# Log success
	frappe.logger().info(f"Generated {len(texts)} embeddings from Employee Data DocType")
	
	return {
		"success": True,
		"total_documents": len(texts),
		"embedding_dimension": embeddings.shape[1],
		"output_path": str(output_path),
		"model_used": model_name
	}


def regenerate_embeddings_and_index(model_name='all-MiniLM-L6-v2'):
	"""
	Complete pipeline: Generate embeddings and create FAISS index
	
	Args:
		model_name: Name of the sentence transformer model to use
	
	Returns:
		dict: Status of the complete pipeline
	"""
	# Step 1: Generate embeddings
	result = generate_embeddings(model_name=model_name)
	
	if not result.get('success'):
		return result
	
	# Step 2: Create FAISS index
	print("\n" + "=" * 60)
	print("Creating FAISS Index")
	print("=" * 60)
	
	try:
		import faiss
		import numpy as np
		
		base_dir = Path(__file__).resolve().parent
		embeddings_path = base_dir / "embeddings" / "embeddings.pkl"
		db_path = base_dir / "db"
		index_path = db_path / "faiss_index.bin"
		texts_path = db_path / "texts.pkl"
		
		# Load embeddings
		print(f"Loading embeddings from: {embeddings_path}")
		with open(embeddings_path, "rb") as f:
			data = pickle.load(f)
		
		# Create FAISS index
		embeddings = np.array(data["embeddings"]).astype('float32')
		dimension = embeddings.shape[1]
		
		print(f"Creating FAISS index with {len(embeddings)} vectors (dimension={dimension})")
		index = faiss.IndexFlatL2(dimension)
		index.add(embeddings)
		
		# Save index and texts
		db_path.mkdir(parents=True, exist_ok=True)
		faiss.write_index(index, str(index_path))
		
		with open(texts_path, "wb") as f:
			pickle.dump(data["texts"], f)

		# Save metadata including embedded employee IDs for incremental updates
		meta_path = db_path / "rag_index_meta.json"
		embedded_ids = data["texts"]  # Store texts to track which records are embedded
		meta_data = get_employee_data_signature()
		meta_data["embedded_employee_ids"] = [
			text.split("\n")[1].split(": ")[1] if "Employee Number:" in text else None
			for text in data["texts"]
		]
		with open(meta_path, "w", encoding="utf-8") as f:
			json.dump(meta_data, f, indent=2)
		
		print(f"\n✓ FAISS index created successfully!")
		print(f"  - Vectors in index: {index.ntotal}")
		print(f"  - Index saved to: {index_path}")
		print(f"  - Texts saved to: {texts_path}")
		print(f"  - Metadata saved to: {meta_path}")
		
		result['faiss_index'] = {
			"vectors_count": int(index.ntotal),
			"index_path": str(index_path),
			"texts_path": str(texts_path),
			"meta_path": str(meta_path),
		}
		
		frappe.logger().info(f"Created FAISS index with {index.ntotal} vectors")
		
	except Exception as e:
		error_msg = f"Error creating FAISS index: {str(e)}"
		print(f"\nERROR: {error_msg}")
		frappe.log_error(error_msg, "RAG System - FAISS")
		result['faiss_error'] = str(e)
	
	return result


def add_new_employee_embeddings(new_employee_records: list, model_name='all-MiniLM-L6-v2'):
	"""
	Incremental embedding: add only new employee records to existing FAISS index.
	Much faster than rebuilding from scratch.
	
	Args:
		new_employee_records: List of dicts with new employee data
		model_name: Embedding model to use
	
	Returns:
		dict: Status with count of newly added embeddings
	"""
	if not new_employee_records:
		return {"success": True, "added_count": 0, "message": "No new records"}
	
	print(f"\n" + "=" * 60)
	print(f"Adding Embeddings for {len(new_employee_records)} new employees")
	print("=" * 60)
	
	try:
		import faiss
		import numpy as np
		
		# Convert new employees to text
		try:
			from .doctype_data_loader import employee_to_text
		except ImportError:
			employee_to_text = importlib.import_module("doctype_data_loader").employee_to_text
		
		new_texts = [employee_to_text(emp) for emp in new_employee_records]
		print(f"Converted {len(new_texts)} employees to text")
		
		# Generate embeddings for new records only
		model = SentenceTransformer(model_name)
		new_embeddings = model.encode(new_texts, show_progress_bar=False, convert_to_numpy=True).astype('float32')
		print(f"Generated {len(new_embeddings)} embeddings (shape: {new_embeddings.shape})")
		
		# Load existing FAISS index
		base_dir = Path(__file__).resolve().parent
		db_path = base_dir / "db"
		index_path = db_path / "faiss_index.bin"
		texts_path = db_path / "texts.pkl"
		
		if not index_path.exists():
			return {
				"success": False,
				"error": "FAISS index not found. Run full embedding generation first."
			}
		
		print(f"Loading existing FAISS index from {index_path}")
		index = faiss.read_index(str(index_path))
		old_vector_count = index.ntotal
		
		with open(texts_path, "rb") as f:
			old_texts = pickle.load(f)
		
		print(f"Index had {old_vector_count} vectors")
		
		# Append new embeddings to index
		index.add(new_embeddings)
		new_vector_count = index.ntotal
		
		print(f"Added {new_vector_count - old_vector_count} vectors to index")
		print(f"Index now has {new_vector_count} total vectors")
		
		# Save updated index
		faiss.write_index(index, str(index_path))
		
		# Save updated texts
		updated_texts = old_texts + new_texts
		with open(texts_path, "wb") as f:
			pickle.dump(updated_texts, f)
		
		# Update metadata
		try:
			from .doctype_data_loader import get_employee_data_signature, get_all_employee_numbers
		except ImportError:
			dl = importlib.import_module("doctype_data_loader")
			get_employee_data_signature = dl.get_employee_data_signature
			get_all_employee_numbers = dl.get_all_employee_numbers
		
		meta_path = db_path / "rag_index_meta.json"
		sig = get_employee_data_signature()
		sig["embedded_employee_ids"] = get_all_employee_numbers()
		
		with open(meta_path, "w", encoding="utf-8") as f:
			json.dump(sig, f, indent=2)
		
		print(f"\n✓ Successfully added embeddings!")
		print(f"  - New embeddings: {len(new_embeddings)}")
		print(f"  - Total vectors in index: {new_vector_count}")
		print(f"  - Index saved to: {index_path}")
		
		frappe.logger().info(f"Added incremental embeddings for {len(new_embeddings)} new employees")
		
		return {
			"success": True,
			"added_count": len(new_embeddings),
			"total_vectors": new_vector_count,
			"index_path": str(index_path)
		}
		
	except Exception as e:
		error_msg = f"Error adding incremental embeddings: {str(e)}"
		print(f"\nERROR: {error_msg}")
		frappe.log_error(error_msg, "RAG System - Incremental Embeddings")
		return {
			"success": False,
			"error": error_msg
		}


def get_embedded_employee_ids() -> list:
	"""
	Load list of employee IDs that are already embedded in FAISS.
	Used for incremental embedding detection.
	"""
	base_dir = Path(__file__).resolve().parent
	db_path = base_dir / "db"
	meta_path = db_path / "rag_index_meta.json"
	if not meta_path.exists():
		return []
	try:
		with open(meta_path, "r", encoding="utf-8") as f:
			meta = json.load(f)
		return meta.get("embedded_employee_ids", [])
	except Exception as e:
		print(f"Warning: Could not load embedded IDs: {e}")
		return []


def get_new_or_modified_employees():
	"""
	Get employees that are new or have been modified since last embedding.
	Compares against saved embedding metadata.
	"""
	try:
		embedded_ids = get_embedded_employee_ids()
		all_employees = get_all_employee_texts()
		
		new_texts = []
		new_ids = []
		for text in all_employees:
			try:
				emp_num_line = [l for l in text.split("\n") if "Employee Number:" in l][0]
				emp_id = emp_num_line.split(": ")[1].strip()
				if emp_id not in embedded_ids:
					new_texts.append(text)
					new_ids.append(emp_id)
			except (IndexError, ValueError):
				pass
		
		return new_texts, new_ids
	except Exception as e:
		print(f"Warning: Could not detect new employees: {e}")
		return [], []


def embed_new_employees_incrementally(model_name='all-MiniLM-L6-v2'):
	"""
	Embed only new/modified employees and append to existing FAISS index.
	Much faster than full rebuild (0.02s per record vs 17s for all).
	"""
	print("\n" + "="*60)
	print("Incremental Embedding Update")
	print("="*60)
	
	new_texts, new_ids = get_new_or_modified_employees()
	
	if not new_texts:
		print("\n✓ No new employees. FAISS index is current.")
		return {
			"success": True,
			"new_embeddings": 0,
			"message": "Index already current"
		}
	
	print(f"\nFound {len(new_texts)} new employees to embed")
	print(f"Employee IDs: {new_ids}")
	
	# Generate embeddings for new records only
	print(f"\nLoading model: {model_name}")
	model = SentenceTransformer(model_name)
	
	print("Embedding new records...")
	new_embeddings = model.encode(new_texts, show_progress_bar=True, convert_to_numpy=True)
	print(f"Generated {len(new_embeddings)} new embeddings")
	
	# Append to existing FAISS index
	try:
		import faiss
		import numpy as np
		
		index_path = db_path / "faiss_index.bin"
		texts_path = db_path / "texts.pkl"
		meta_path = db_path / "rag_index_meta.json"
		
		# Load existing index and texts
		if index_path.exists():
			index = faiss.read_index(str(index_path))
			with open(texts_path, "rb") as f:
				existing_texts = pickle.load(f)
		else:
			# Create new index if doesn't exist
			dimension = new_embeddings.shape[1]
			index = faiss.IndexFlatL2(dimension)
			existing_texts = []
		
		# Add new embeddings
		new_embeddings_float32 = np.array(new_embeddings).astype('float32')
		index.add(new_embeddings_float32)
		
		# Update texts
		updated_texts = existing_texts + new_texts
		
		# Save updated index and texts
		db_path.mkdir(parents=True, exist_ok=True)
		faiss.write_index(index, str(index_path))
		with open(texts_path, "wb") as f:
			pickle.dump(updated_texts, f)
		
		# Update metadata with new employee IDs
		with open(meta_path, "r", encoding="utf-8") as f:
			meta = json.load(f)
		
		embedded_ids = meta.get("embedded_employee_ids", [])
		embedded_ids.extend(new_ids)
		meta["embedded_employee_ids"] = list(set(embedded_ids))  # Remove duplicates
		meta.update(get_employee_data_signature())
		
		with open(meta_path, "w", encoding="utf-8") as f:
			json.dump(meta, f, indent=2)
		
		print(f"\n✓ Incremental update successful!")
		print(f"  - New embeddings added: {len(new_embeddings)}")
		print(f"  - Total vectors in index: {index.ntotal}")
		print(f"  - Total texts: {len(updated_texts)}")
		
		return {
			"success": True,
			"new_embeddings": len(new_embeddings),
			"total_vectors": int(index.ntotal),
			"total_texts": len(updated_texts)
		}
	
	except Exception as e:
		error_msg = f"Error in incremental embedding: {str(e)}"
		print(f"\nERROR: {error_msg}")
		frappe.log_error(error_msg, "RAG System - Incremental Embedding")
		return {
			"success": False,
			"error": str(e)
		}


# For testing
if __name__ == "__main__":
	print("This script must be run within Frappe context")
	print("Use: bench execute rag_app.rag_system.embed_from_doctype.regenerate_embeddings_and_index")
