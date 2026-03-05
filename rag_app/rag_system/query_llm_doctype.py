"""
RAG Query System - Frappe DocType Version
==========================================

This module handles employee queries using either:
1. SQL for analytical queries (counts, averages, rankings)
2. RAG for specific employee lookups

Data is loaded from Frappe Employee Data DocType instead of Excel files.

Usage:
    from rag_app.rag_system.query_llm_doctype import ask
    result = ask("What is John's salary?")

Or via Frappe API:
    frappe.call('rag_app.rag_system.frappe_integration.query_employee', 
                args={'question': 'What is John salary?'})
"""

import os
import sqlite3
import json
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np
import pandas as pd
from ollama import Client
import sys
import importlib

# Add rag_system to path for imports
rag_system_path = Path(__file__).resolve().parent
if str(rag_system_path) not in sys.path:
	sys.path.insert(0, str(rag_system_path))

load_dotenv()

base_dir = Path(__file__).resolve().parent
db_path = base_dir / "db"

# Use incremental embedding by default (only embed new records, don't rebuild)
USE_INCREMENTAL_EMBEDDING = True

# Setup models and connections
ollama = Client(host="https://ollama.com", headers={'Authorization': 'Bearer ' + os.getenv('OLLAMA_API_KEY')})
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load FAISS for RAG
try:
	index = faiss.read_index(str(db_path / "faiss_index.bin"))
	with open(db_path / "texts.pkl", "rb") as f:
		texts = pickle.load(f)
	print(f"Loaded FAISS index with {index.ntotal} vectors")
except Exception as e:
	print(f"Warning: Could not load FAISS index: {e}")
	index = None
	texts = []

# In-memory SQLite for analytical queries
sql_conn = None
columns = []
_initialized = False  # Track if we've already initialized

def initialize_sql_from_frappe():
	"""
	Initialize in-memory SQLite database from Frappe DocType data
	This function should be called when running within Frappe context
	"""
	global sql_conn, columns
	
	try:
		import frappe
		try:
			from .doctype_data_loader import get_employees_dataframe
		except ImportError:
			get_employees_dataframe = importlib.import_module("doctype_data_loader").get_employees_dataframe
		
		print("Loading employee data from Frappe DocType...")
		df = get_employees_dataframe()
		
		if df.empty:
			print("Warning: No employee data found in DocType")
			return False
		
		# Create in-memory SQLite database
		sql_conn = sqlite3.connect(":memory:", check_same_thread=False)
		df.to_sql("employees", sql_conn, index=False, if_exists="replace")
		columns = list(df.columns)
		
		print(f"Loaded {len(df)} employees into in-memory database")
		print(f"Columns: {columns}")
		return True
		
	except ImportError:
		print("Warning: Not running in Frappe context, cannot load DocType data")
		return False
	except Exception as e:
		print(f"Error loading data from DocType: {e}")
		return False


def initialize_sql_from_dataframe(df: pd.DataFrame):
	"""
	Initialize in-memory SQLite from a provided DataFrame
	Useful for testing or standalone usage
	
	Args:
		df: Pandas DataFrame with employee data
	"""
	global sql_conn, columns
	
	sql_conn = sqlite3.connect(":memory:", check_same_thread=False)
	df.to_sql("employees", sql_conn, index=False, if_exists="replace")
	columns = list(df.columns)
	
	print(f"Loaded {len(df)} employees into in-memory database")


# Get schema for LLM
def get_schema():
	"""Get database schema for SQL generation"""
	if columns:
		return f"Table: employees\nColumns: {', '.join(columns)}"
	return "Table: employees\nColumns: employee_number, first_name, last_name, full_name, gender, start_date, years_of_service, department, country, center, monthly_salary, annual_salary, job_rate, sick_leaves, unpaid_leaves, overtime_hours"

SCHEMA = get_schema()


def is_analytical(question):
	"""Check if question needs SQL (aggregates, rankings, comparisons)"""
	keywords = ['average', 'avg', 'sum', 'total', 'count', 'how many', 'top', 'highest', 
				'lowest', 'maximum', 'minimum', 'max', 'min', 'most', 'least', 'rank',
				'all employees', 'list all', 'everyone', 'compare', 'filter', 'greater than',
				'less than', 'between', 'department', 'group by', 'salary', 'salary above', 'salary below',
				'compensation', 'pay']
	return any(kw in question.lower() for kw in keywords)


def _reload_faiss_index() -> bool:
	"""Reload FAISS index and associated texts from disk."""
	global index, texts
	try:
		index = faiss.read_index(str(db_path / "faiss_index.bin"))
		with open(db_path / "texts.pkl", "rb") as f:
			texts = pickle.load(f)
		return True
	except Exception as e:
		print(f"Warning: failed to reload FAISS index: {e}")
		index = None
		texts = []
		return False


def _load_saved_index_signature():
	"""Load previously saved DocType signature for the current FAISS index."""
	meta_path = db_path / "rag_index_meta.json"
	if not meta_path.exists():
		return None
	try:
		with open(meta_path, "r", encoding="utf-8") as f:
			return json.load(f)
	except Exception as e:
		print(f"Warning: failed to read index metadata: {e}")
		return None


def _get_newly_added_records(last_embedded_ids):
	"""Get employee records that were added since last embedding."""
	try:
		try:
			from .doctype_data_loader import get_new_employees_since
		except ImportError:
			get_new_employees_since = importlib.import_module("doctype_data_loader").get_new_employees_since
		
		return get_new_employees_since(last_embedded_ids or [])
	except Exception as e:
		print(f"Warning: failed to get new records: {e}")
		return []


def ensure_rag_index_current(auto_rebuild=True):
	"""
	Ensure FAISS vector count matches current Employee Data.
	Smart incremental updates: only re-embed NEW records instead of rebuilding everything.
	Only does full rebuild if field schema changes.
	"""
	try:
		try:
			from .doctype_data_loader import get_employee_count, get_employee_data_signature
		except ImportError:
			dl = importlib.import_module("doctype_data_loader")
			get_employee_count = dl.get_employee_count
			get_employee_data_signature = dl.get_employee_data_signature

		current_count = get_employee_count()
		index_count = int(index.ntotal) if index is not None else 0
		current_sig = get_employee_data_signature()
		saved_sig = _load_saved_index_signature()
		
		# Check if field schema changed (requires full rebuild)
		fields_changed = bool(saved_sig) and (saved_sig.get("fields") != current_sig.get("fields"))
		
		if fields_changed:
			print(f"Field schema changed! Full rebuild required.")
			if not auto_rebuild:
				return False
			
			try:
				from .embed_from_doctype import regenerate_embeddings_and_index
			except ImportError:
				regenerate_embeddings_and_index = importlib.import_module("embed_from_doctype").regenerate_embeddings_and_index
			
			result = regenerate_embeddings_and_index()
			if not result.get("success"):
				return False
			return _reload_faiss_index()
		
		# Check if there are new records (use incremental update)
		last_embedded_ids = saved_sig.get("embedded_employee_ids", []) if saved_sig else []
		new_records = _get_newly_added_records(last_embedded_ids)
		
		if new_records:
			print(f"Found {len(new_records)} new employee records. Adding embeddings incrementally...")
			
			try:
				from .embed_from_doctype import add_new_employee_embeddings
			except ImportError:
				add_new_employee_embeddings = importlib.import_module("embed_from_doctype").add_new_employee_embeddings
			
			result = add_new_employee_embeddings(new_records)
			if not result.get("success"):
				print(f"Incremental embedding failed: {result.get('error')}")
				return False
			
			return _reload_faiss_index()
		
		# Index is fresh (same fields, no new records)
		if current_count == index_count and index_count > 0 and texts:
			return True
		
		# Count mismatch but no new_records found (data was deleted or something weird)
		if current_count < index_count:
			print(f"Data count decreased ({index_count} → {current_count}). Full rebuild required.")
			if not auto_rebuild:
				return False
			
			try:
				from .embed_from_doctype import regenerate_embeddings_and_index
			except ImportError:
				regenerate_embeddings_and_index = importlib.import_module("embed_from_doctype").regenerate_embeddings_and_index
			
			result = regenerate_embeddings_and_index()
			if not result.get("success"):
				return False
			return _reload_faiss_index()
		
		# Index is empty or not loaded
		if index is None or index_count == 0:
			print("FAISS index not initialized. Generating embeddings...")
			if not auto_rebuild:
				return False
			
			try:
				from .embed_from_doctype import regenerate_embeddings_and_index
			except ImportError:
				regenerate_embeddings_and_index = importlib.import_module("embed_from_doctype").regenerate_embeddings_and_index
			
			result = regenerate_embeddings_and_index()
			if not result.get("success"):
				return False
			return _reload_faiss_index()
		
		return True
		
	except Exception as e:
		print(f"Warning: failed to validate RAG index freshness: {e}")
		return False


def ask_sql(question):
	"""Generate and execute SQL for analytical queries"""
	if not sql_conn:
		return {
			"answer": "SQL database not initialized. Please load data first.",
			"query_type": "SQL",
			"error": "Database not initialized"
		}
	
	schema = get_schema()
	
	response = ollama.chat(model="gpt-oss:120b", messages=[
		{"role": "system", "content": f"""You are a SQL expert. Generate SQLite queries for employee data.
{schema}
Rules:
- Return ONLY the SQL query, no explanation
- Use proper SQLite syntax
- For "top" queries, use ORDER BY with LIMIT
- Column names use underscores (e.g., First_Name, not First Name)
- Column names are case-sensitive
- Always use SELECT statements"""},
		{"role": "user", "content": question}
	])
	
	sql = response['message']['content'].strip()
	# Clean up if wrapped in code blocks
	sql = sql.replace("```sql", "").replace("```", "").strip()
	
	try:
		result_df = pd.read_sql_query(sql, sql_conn)
		raw_result = result_df.to_string(index=False)
		
		# Get natural language answer from LLM
		answer_response = ollama.chat(model="gpt-oss:120b", messages=[
			{"role": "system", "content": "Convert the SQL query results into a clear, natural language answer. Be concise and direct."},
			{"role": "user", "content": f"Question: {question}\n\nSQL Result:\n{raw_result}"}
		])
		
		return {
			"answer": answer_response['message']['content'],
			"query_type": "SQL",
			"sql": sql,
			"raw_data": result_df.to_dict('records')
		}
	except Exception as e:
		return {
			"answer": f"SQL Error: {e}",
			"query_type": "SQL",
			"sql": sql,
			"error": str(e)
		}


def ask_rag(question):
	"""Use RAG for specific employee lookups"""
	if index is None or not texts:
		return {
			"answer": "RAG system not initialized. Please generate embeddings first.",
			"query_type": "RAG",
			"error": "FAISS index not loaded"
		}
	
	try:
		# Search
		query_embedding = model.encode(question).astype('float32').reshape(1, -1)
		distances, indices = index.search(query_embedding, 5)
		matched_docs = [texts[i] for i in indices[0]]
		context = "\n\n".join(matched_docs)
		
		# Ask LLM
		response = ollama.chat(model="gpt-oss:120b", messages=[
			{"role": "system", "content": "Answer based only on the provided employee records. Be concise."},
			{"role": "user", "content": f"Records:\n{context}\n\nQuestion: {question}"}
		])
		
		return {
			"answer": response['message']['content'],
			"query_type": "RAG",
			"matched_docs": matched_docs,
			"distances": distances[0].tolist()
		}
	except Exception as e:
		return {
			"answer": f"RAG Error: {str(e)}",
			"query_type": "RAG",
			"error": str(e)
		}


def ask(question, force_init=True):
	"""
	Main query function - routes to SQL or RAG based on question type
	
	Args:
		question: Natural language question about employees
		force_init: If True, initialize SQL from Frappe only on first query
	
	Returns:
		dict with answer and metadata
	"""
	global _initialized
	
	# Only initialize on first query (subsequent queries reuse cached data)
	if force_init and not _initialized:
		initialize_sql_from_frappe()
		ensure_rag_index_current(auto_rebuild=True)
		_initialized = True
	
	if is_analytical(question):
		return ask_sql(question)
	else:
		return ask_rag(question)


# Standalone CLI mode
if __name__ == "__main__":
	print("Employee RAG System - DocType Version")
	print("=" * 50)
	print("Data Source: Frappe Employee Data DocType")
	print("=" * 50)
	
	# Try to initialize from Frappe
	if not initialize_sql_from_frappe():
		print("\nCannot run in standalone mode without Frappe context")
		print("Use: bench execute rag_app.rag_system.query_llm_doctype.test_queries")
		sys.exit(1)
	
	print("\nAsk analytical questions (counts, averages, top N) or specific employee queries")
	print()
	
	while True:
		q = input("\nAsk (or 'quit'): ")
		if q.lower() == 'quit':
			break
		
		result = ask(q, force_init=False)
		print(f"\n[{result['query_type']}] {result['answer']}")
		
		if result['query_type'] == 'SQL' and 'sql' in result:
			print(f"\nGenerated SQL:\n{result['sql']}")
		elif result['query_type'] == 'RAG':
			print(f"\n(Retrieved {len(result.get('matched_docs', []))} records)")


def test_queries():
	"""Test function to run sample queries"""
	print("Testing RAG System with DocType Data")
	print("=" * 50)
	
	# Initialize
	if not initialize_sql_from_frappe():
		print("Failed to initialize from Frappe")
		return
	
	test_questions = [
		"How many employees are there?",
		"What is the average salary?",
		"Who are the top 5 highest paid employees?",
		"Tell me about employee number 1",
	]
	
	for q in test_questions:
		print(f"\nQ: {q}")
		result = ask(q, force_init=False)
		print(f"A [{result['query_type']}]: {result['answer']}")
		print("-" * 50)
