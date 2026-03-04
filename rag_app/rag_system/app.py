import os
import sqlite3
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np
from pathlib import Path
from ollama import Client

load_dotenv()

@st.cache_resource
def load_models():
    base_dir = Path(__file__).resolve().parent
    db_path = base_dir / "db"
    ollama = Client(host="https://ollama.com", headers={'Authorization': 'Bearer ' + os.getenv('OLLAMA_API_KEY')})
    model = SentenceTransformer('all-MiniLM-L6-v2')
    index = faiss.read_index(str(db_path / "faiss_index.bin"))
    with open(db_path / "texts.pkl", "rb") as f:
        texts = pickle.load(f)
    return ollama, model, index, texts

@st.cache_resource
def load_sql_db():
    df = pd.read_excel("data/Employees.xlsx")
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    # Normalize column names (remove spaces)
    df.columns = [col.replace(" ", "_") for col in df.columns]
    df.to_sql("employees", conn, index=False)
    return conn, list(df.columns)

ollama, model, faiss_index, faiss_texts = load_models()
sql_conn, columns = load_sql_db()

# Get schema for LLM
SCHEMA = f"Table: employees\nColumns: {', '.join(columns)}"

def is_analytical(question):
    """Check if question needs SQL (aggregates, rankings, comparisons)"""
    keywords = ['average', 'avg', 'sum', 'total', 'count', 'how many', 'top', 'highest', 
                'lowest', 'maximum', 'minimum', 'max', 'min', 'most', 'least', 'rank',
                'all employees', 'list all', 'everyone', 'compare']
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
- Column names are case-sensitive"""},
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
        return answer_response['message']['content'], sql
    except Exception as e:
        return f"SQL Error: {e}", sql

def ask_rag(question):
    """Use RAG for specific employee lookups"""
    query_embedding = model.encode(question).astype('float32').reshape(1, -1)
    distances, indices = faiss_index.search(query_embedding, 5)
    matched_docs = [faiss_texts[i] for i in indices[0]]
    context = "\n\n".join(matched_docs)
    
    response = ollama.chat(model="gpt-oss:120b", messages=[
        {"role": "system", "content": "Answer based only on the provided employee records. Be concise."},
        {"role": "user", "content": f"Records:\n{context}\n\nQuestion: {question}"}
    ])
    return response['message']['content'], matched_docs

# UI
st.title("Employee RAG System")
st.write("Ask questions about employees")

question = st.text_input("Your question:")

if st.button("Ask") and question:
    with st.spinner("Searching..."):
        if is_analytical(question):
            answer, sql_or_docs = ask_sql(question)
            query_type = "SQL"
        else:
            answer, sql_or_docs = ask_rag(question)
            query_type = "RAG"
    
    st.subheader("Answer")
    st.write(answer)
    
    if query_type == "SQL":
        with st.expander("Generated SQL"):
            st.code(sql_or_docs, language="sql")
    else:
        with st.expander("Retrieved Records"):
            for i, doc in enumerate(sql_or_docs, 1):
                st.text(f"[{i}]\n{doc}\n")