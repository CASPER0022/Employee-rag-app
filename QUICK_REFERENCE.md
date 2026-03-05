# RAG System Quick Reference

## One-Command Setup

```bash
# Complete migration (from Excel to DocType)
bench execute rag_app.rag_system.migrate_to_doctype.run_complete_migration

# Check system status
bench execute rag_app.rag_system.migrate_to_doctype.check_migration_status
```

---

## Individual Steps

### 1. Import Data
```bash
# Import from Excel to DocType
bench execute rag_app.rag_system.import_excel_to_doctype.import_employees

# With overwrite
bench execute "rag_app.rag_system.import_excel_to_doctype.import_employees(overwrite=True)"

# Check count
bench execute rag_app.rag_system.import_excel_to_doctype.get_employee_count
```

### 2. Generate Embeddings
```bash
# Generate embeddings + FAISS index
bench execute rag_app.rag_system.embed_from_doctype.regenerate_embeddings_and_index

# Just embeddings (no FAISS)
bench execute rag_app.rag_system.embed_from_doctype.generate_embeddings
```

### 3. Test System
```bash
# Test data loader
bench execute rag_app.rag_system.doctype_data_loader.test_data_loader

# Test queries
bench execute rag_app.rag_system.query_llm_doctype.test_queries
```

---

## API Endpoints

### Query Employee
```python
# Python
result = frappe.call(
    'rag_app.rag_system.frappe_integration_doctype.query_employee',
    question="What is the average salary?"
)
```

```javascript
// JavaScript
frappe.call({
    method: 'rag_app.rag_system.frappe_integration_doctype.query_employee',
    args: { question: 'What is the average salary?' },
    callback: function(r) {
        console.log(r.message.answer);
    }
});
```

### Get Statistics
```python
stats = frappe.call('rag_app.rag_system.frappe_integration_doctype.get_query_stats')
# Returns: {"total_employees": 1000, "status": "connected"}
```

### Rebuild Embeddings
```python
result = frappe.call('rag_app.rag_system.frappe_integration_doctype.rebuild_embeddings')
```

---

## Query Types

### Analytical (SQL)
- "How many employees?"
- "What is the average salary?"
- "Top 10 highest paid employees"
- "Employees in Engineering department"
- "Salary distribution by country"

### Semantic (RAG)
- "Tell me about John Doe"
- "What does employee 12345 do?"
- "Who works in the Seattle office?"

---

## File Structure

```
rag_app/
├── rag_system/
│   ├── doctype_data_loader.py          # Fetch data from DocType
│   ├── embed_from_doctype.py           # Generate embeddings
│   ├── query_llm_doctype.py            # Query system (SQL + RAG)
│   ├── frappe_integration_doctype.py   # API endpoints
│   ├── migrate_to_doctype.py           # Migration automation
│   ├── import_excel_to_doctype.py      # Import Excel → DocType
│   ├── embeddings/                     # Generated embeddings
│   └── db/                             # FAISS index
├── rag_app/doctype/employee_data/      # DocType definition
└── rag_data/                           # Excel source data
```

---

## Common Issues

### No employee data
```bash
bench execute rag_app.rag_system.import_excel_to_doctype.import_employees
```

### FAISS index not found
```bash
bench execute rag_app.rag_system.embed_from_doctype.regenerate_embeddings_and_index
```

### Check logs
```bash
bench tail
```

---

## Code Examples

### Fetch from DocType
```python
from rag_app.rag_system.doctype_data_loader import (
    get_all_employees,
    get_employees_dataframe,
    get_all_employee_texts
)

# Get as list of dicts
employees = get_all_employees()

# Get as DataFrame
df = get_employees_dataframe()

# Get as text documents
texts = get_all_employee_texts()
```

### Query System
```python
from rag_app.rag_system.query_llm_doctype import ask

# Ask question
result = ask("What is the average salary?")

# Result contains:
# - answer: Natural language answer
# - query_type: "SQL" or "RAG"
# - sql: Generated SQL (if SQL query)
# - raw_data: Query results (if SQL query)
# - matched_docs: Retrieved docs (if RAG query)
```

---

## Maintenance

### When employee data changes
```bash
# Regenerate embeddings
bench execute rag_app.rag_system.embed_from_doctype.regenerate_embeddings_and_index

# Or via API
frappe.call('rag_app.rag_system.frappe_integration_doctype.rebuild_embeddings')
```

### Verify system
```bash
bench execute rag_app.rag_system.migrate_to_doctype.check_migration_status
```

---

## Full Documentation

See [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) for complete details.
