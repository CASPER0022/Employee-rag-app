# Employee RAG System with SQL Analytics

A hybrid query system that intelligently routes questions to either SQL (for analytics) or RAG (for specific lookups).

## Features

- **Intelligent Query Routing**: Automatically detects if a question needs SQL or RAG
- **SQL Analytics**: Handles aggregations, rankings, filtering, comparisons
- **RAG for Lookups**: Fast semantic search for specific employee information
- **Frappe Integration**: Ready-to-use whitelisted methods for Frappe apps
- **Standalone CLI**: Can be used without Frappe

## Architecture

```
Question → is_analytical() → SQL or RAG
                ↓                ↓
            ask_sql()        ask_rag()
                ↓                ↓
            Result Dict      Result Dict
```

## Installation

1. Install required packages:
```bash
pip install pandas openpyxl sentence-transformers faiss-cpu ollama python-dotenv
```

2. Set up environment variables in `.env`:
```
OLLAMA_API_KEY=your_api_key_here
```

3. Add local data and generate local artifacts:
    - Put your dataset at `rag_data/Employees.xlsx` (not committed)
    - Generate index artifacts locally:

```bash
python create_db.py
python embed.py
```

## Usage

### 1. Standalone CLI

```bash
python query_llm.py
```

Example queries:
- "How many employees are there?" (SQL)
- "What is the average salary?" (SQL)
- "Who are the top 5 highest paid?" (SQL)
- "What is John's email?" (RAG)

### 2. Python Import

```python
from query_llm import ask

# Query returns a structured dict
result = ask("How many employees in engineering?")

print(result['answer'])          # Natural language answer
print(result['query_type'])      # "SQL" or "RAG"

# SQL-specific fields
if result['query_type'] == 'SQL':
    print(result['sql'])         # Generated SQL query
    print(result['raw_data'])    # DataFrame as list of dicts

# RAG-specific fields
if result['query_type'] == 'RAG':
    print(result['matched_docs']) # Retrieved documents
    print(result['distances'])    # Similarity scores
```

### 3. Frappe Integration

#### Backend (Python):
```python
import frappe

result = frappe.call(
    'rag_system.frappe_integration.query_employee',
    question="What is the average salary?"
)

print(result['answer'])
```

#### Frontend (JavaScript):
```javascript
frappe.call({
    method: 'rag_system.frappe_integration.query_employee',
    args: {
        question: 'How many employees in sales?'
    },
    callback: function(r) {
        if (r.message) {
            console.log(r.message.answer);
            if (r.message.query_type === 'SQL') {
                console.log('SQL:', r.message.sql);
            }
        }
    }
});
```

#### Frappe Page/Form:
```python
# In a Frappe DocType controller
def get_employee_insights(self):
    from rag_system.query_llm import ask
    
    result = ask(f"How many employees in {self.department}?")
    self.employee_count = result['raw_data'][0]['count']
```

## Query Types

### SQL Queries (Analytical)
Triggered by keywords: average, avg, sum, total, count, how many, top, highest, lowest, rank, all employees, compare, etc.

Examples:
- "How many employees are there?"
- "What is the average salary?"
- "Top 10 highest paid employees"
- "Count employees by department"
- "All employees with salary above 80000"

### RAG Queries (Specific Lookups)
For specific employee information or context-based questions.

Examples:
- "What is John Smith's email?"
- "Tell me about Sarah's role"
- "What department is Michael in?"
- "Find employees named David"

## Response Format

```python
{
    "answer": "Natural language answer",
    "query_type": "SQL" or "RAG",
    
    # SQL-specific
    "sql": "SELECT ...",
    "raw_data": [{...}, {...}],
    
    # RAG-specific
    "matched_docs": ["doc1", "doc2", ...],
    "distances": [0.23, 0.45, ...]
}
```

## Configuration

### Modify Query Routing

Edit `is_analytical()` function to add/remove trigger keywords:

```python
def is_analytical(question):
    keywords = ['your', 'custom', 'keywords']
    return any(kw in question.lower() for kw in keywords)
```

### Change LLM Model

```python
# In query_llm.py
response = ollama.chat(
    model="your-model:tag",  # Change this
    messages=[...]
)
```

### Adjust Retrieval Count

```python
# In ask_rag()
distances, indices = index.search(query_embedding, 5)  # Change 5 to desired count
```

## Testing

Run the test suite:

```bash
python test_queries.py
```

This will test both SQL and RAG routing with various question types.

## Troubleshooting

### "No module named 'query_llm'"
Make sure the rag_system folder is in your Python path or use absolute imports.

### "Table employees not found"
Ensure `Employees.xlsx` exists in `rag_data/` directory.

### "FAISS index not found"
Generate artifacts first:
```bash
python create_db.py
python embed.py
```

### SQL Generation Errors
- Check that column names in your Excel match the schema
- Verify SQLite syntax compatibility
- Review generated SQL in the error message

## Files

- `query_llm.py` - Main query engine
- `frappe_integration.py` - Frappe whitelisted methods
- `test_queries.py` - Test suite
- `db/faiss_index.bin` - Vector index (generated locally)
- `db/texts.pkl` - Text records (generated locally)
- `rag_data/Employees.xlsx` - Employee data (local/private)

## Performance

- **SQL queries**: ~2-3 seconds (includes LLM SQL generation + execution + natural language conversion)
- **RAG queries**: ~1-2 seconds (includes embedding + FAISS search + LLM response)
- **In-memory SQLite**: Fast query execution
- **FAISS**: Efficient vector similarity search

## Future Enhancements

- [ ] Cache frequent queries
- [ ] Support multiple tables/joins
- [ ] Add query history
- [ ] Implement query validation
- [ ] Add user feedback loop
- [ ] Support real-time data updates
