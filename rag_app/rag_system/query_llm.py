import os
import sqlite3
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np
import pandas as pd
from ollama import Client

load_dotenv() 

base_dir = Path(__file__).resolve().parent
db_path = base_dir / "db"
data_path = base_dir.parent / "rag_data"

# Setup models and connections
ollama = Client(host="https://ollama.com", headers={'Authorization': 'Bearer ' + os.getenv('OLLAMA_API_KEY')})
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load FAISS for RAG
index = faiss.read_index(str(db_path / "faiss_index.bin"))
with open(db_path / "texts.pkl", "rb") as f:
    texts = pickle.load(f)

# Load Excel into SQLite for analytical queries
sql_conn = sqlite3.connect(":memory:", check_same_thread=False)
df = pd.read_excel(str(data_path / "Employees.xlsx"))
# Normalize column names (remove spaces)
df.columns = [col.replace(" ", "_") for col in df.columns]
df.to_sql("employees", sql_conn, index=False)
columns = list(df.columns)

# Get schema for LLM
SCHEMA = f"Table: employees\nColumns: {', '.join(columns)}"

def is_analytical(question):
    """Check if question needs SQL (aggregates, rankings, comparisons)"""
    keywords = ['average', 'avg', 'sum', 'total', 'count', 'how many', 'top', 'highest', 
                'lowest', 'maximum', 'minimum', 'max', 'min', 'most', 'least', 'rank',
                'all employees', 'list all', 'everyone', 'compare', 'filter', 'greater than',
                'less than', 'between', 'department', 'group by', 'salary above', 'salary below']
    return any(kw in question.lower() for kw in keywords)

def ask_sql(question):
    """Generate and execute SQL for analytical queries"""
    response = ollama.chat(model="gpt-oss:120b", messages=[
        {"role": "system", "content": f"""You are a SQL expert. Generate SQLite queries for employee data.
{SCHEMA}
Rules:
- Return ONLY the SQL query, no explanation
- Use proper SQLite syntax
- For "top" queries, use ORDER BY with LIMIT
- Column names use underscores (e.g., Full_Name, not Full Name)
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

def ask(question):
    """Main query function - routes to SQL or RAG based on question type"""
    if is_analytical(question):
        return ask_sql(question)
    else:
        return ask_rag(question)

# For Frappe integration - can be called as: from rag_system.query_llm import ask
# result = ask("What is John's salary?")

# Standalone CLI mode
if __name__ == "__main__":
    print("Employee RAG System (SQL + RAG)")
    print("================================")
    print("Ask analytical questions (counts, averages, top N) or specific employee queries")
    print()
    
    while True:
        q = input("\nAsk (or 'quit'): ")
        if q.lower() == 'quit':
            break
        
        result = ask(q)
        print(f"\n[{result['query_type']}] {result['answer']}")
        
        if result['query_type'] == 'SQL' and 'sql' in result:
            print(f"\nGenerated SQL:\n{result['sql']}")
        elif result['query_type'] == 'RAG':
            print(f"\n(Retrieved {len(result['matched_docs'])} records)")