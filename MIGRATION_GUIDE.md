# Migration Guide: Excel to Frappe DocType RAG System

## Overview

This guide walks you through migrating the RAG system from Excel-based data to Frappe DocType-based data.

### Architecture Changes

**Old Pipeline:**
```
Excel File → Embeddings → FAISS → LLM
Excel File → SQLite → SQL Queries
```

**New Pipeline:**
```
Frappe DocType → Retrieve Records → Embeddings → FAISS → LLM
Frappe DocType → In-Memory SQLite → SQL Queries
```

---

## Prerequisites

1. Frappe bench setup with `rag_app` installed
2. Employee Data DocType created (already exists)
3. Python packages installed:
   - sentence-transformers
   - faiss-cpu (or faiss-gpu)
   - pandas
   - ollama
   - python-dotenv

---

## Migration Steps

### Step 1: Import Data from Excel to DocType

The Employee Data DocType is already created with these fields:
- employee_number (unique identifier)
- first_name, last_name, full_name
- gender, start_date, years_of_service
- department, country, center
- monthly_salary, annual_salary, job_rate
- sick_leaves, unpaid_leaves, overtime_hours

**Import the data:**

```bash
# Navigate to your bench directory
cd /workspaces/frappe_codespace/frappe-bench

# Import employees from Excel
bench execute rag_app.rag_system.import_excel_to_doctype.import_employees
```

**Options:**
- To overwrite existing data:
  ```bash
  bench execute "rag_app.rag_system.import_excel_to_doctype.import_employees(overwrite=True)"
  ```

- To check employee count:
  ```bash
  bench execute rag_app.rag_system.import_excel_to_doctype.get_employee_count
  ```

**Expected Output:**
```
Reading data from: /path/to/rag_data/Employees.xlsx
Found 1000 records in Excel file
Imported 100 records...
Imported 200 records...
...
=== Import Complete ===
Successfully imported: 1000 records
Errors: 0 records
Total in Excel: 1000 records
```

---

### Step 2: Test Data Loader

Verify that data can be loaded from the DocType:

```bash
bench execute rag_app.rag_system.doctype_data_loader.test_data_loader
```

**Expected Output:**
```
Testing Employee Data Loader
==================================================

Total Employees: 1000

DataFrame shape: (1000, 16)
Columns: ['No', 'First_Name', 'Last_Name', ...]

First 3 rows:
   No  First_Name  Last_Name  ...

Generated 1000 text documents

Sample text document:
--------------------------------------------------
Employee Number: 1
Name: John Doe
Gender: Male
Start Date: 2020-01-15
...
--------------------------------------------------
```

---

### Step 3: Generate Embeddings from DocType

Generate embeddings from the DocType data (replaces the old embed.py):

```bash
bench execute rag_app.rag_system.embed_from_doctype.regenerate_embeddings_and_index
```

This command:
1. Fetches all employee records from the DocType
2. Converts them to text documents
3. Generates embeddings using sentence-transformers
4. Creates a FAISS index for similarity search
5. Saves everything to disk

**Expected Output:**
```
============================================================
Generating Embeddings from Employee Data DocType
============================================================

Found 1000 employee records in DocType
Converting employee records to text documents...
Generated 1000 text documents

Sample document:
------------------------------------------------------------
Employee Number: 1
Name: John Doe
...
------------------------------------------------------------

Loading embedding model: all-MiniLM-L6-v2
Generating embeddings...
Batches: 100%|████████████████| 32/32 [00:05<00:00,  6.12it/s]
Generated embeddings: shape = (1000, 384)

Saving embeddings to: /path/to/embeddings/embeddings.pkl

✓ Embeddings generated successfully!
  - Total documents: 1000
  - Embedding dimension: 384
  - Saved to: /path/to/embeddings/embeddings.pkl

============================================================
Creating FAISS Index
============================================================

Loading embeddings from: /path/to/embeddings/embeddings.pkl
Creating FAISS index with 1000 vectors (dimension=384)

✓ FAISS index created successfully!
  - Vectors in index: 1000
  - Index saved to: /path/to/db/faiss_index.bin
  - Texts saved to: /path/to/db/texts.pkl
```

---

### Step 4: Test the Query System

Test queries using the new DocType-based system:

```bash
bench execute rag_app.rag_system.query_llm_doctype.test_queries
```

**Sample Test Queries:**
```
Q: How many employees are there?
A [SQL]: There are 1000 employees.
--------------------------------------------------

Q: What is the average salary?
A [SQL]: The average salary is $75,000.
--------------------------------------------------

Q: Who are the top 5 highest paid employees?
A [SQL]: The top 5 highest paid employees are:
1. Jane Smith - $120,000
2. Bob Johnson - $115,000
...
--------------------------------------------------

Q: Tell me about employee number 1
A [RAG]: Employee 1 is John Doe, working in Engineering department...
--------------------------------------------------
```

---

### Step 5: Update API Integration

The new Frappe API methods are available at:

**Main Query Endpoint:**
```python
# From Python
result = frappe.call(
    'rag_app.rag_system.frappe_integration_doctype.query_employee',
    question="What is the average salary in Engineering?"
)
```

```javascript
// From JavaScript
frappe.call({
    method: 'rag_app.rag_system.frappe_integration_doctype.query_employee',
    args: { 
        question: 'What is the average salary in Engineering?' 
    },
    callback: function(r) {
        console.log(r.message.answer);
    }
});
```

**Statistics Endpoint:**
```python
stats = frappe.call('rag_app.rag_system.frappe_integration_doctype.get_query_stats')
# Returns: {"total_employees": 1000, "status": "connected", "data_source": "Frappe DocType"}
```

**Rebuild Embeddings (Admin):**
```python
result = frappe.call('rag_app.rag_system.frappe_integration_doctype.rebuild_embeddings')
# Regenerates embeddings when DocType data changes
```

---

## File Structure

### New Files Created:
- `doctype_data_loader.py` - Utility to fetch and convert DocType data
- `embed_from_doctype.py` - Generate embeddings from DocType
- `query_llm_doctype.py` - Query system using DocType data
- `frappe_integration_doctype.py` - Frappe API endpoints

### Existing Files (unchanged):
- `import_excel_to_doctype.py` - Import Excel to DocType (already existed)
- `rag_app/doctype/employee_data/` - DocType definition (already existed)

### Old Files (can be kept for reference):
- `embed.py` - Old Excel-based embedding generation
- `query_llm.py` - Old Excel-based query system
- `frappe_integration.py` - Old API endpoints

---

## Code Examples

### Fetching Data from DocType

```python
from rag_app.rag_system.doctype_data_loader import (
    get_all_employees,
    get_employees_dataframe,
    get_all_employee_texts
)

# Get raw employee records
employees = get_all_employees()
# Returns: List of dicts with employee data

# Get as DataFrame for SQL
df = get_employees_dataframe()
# Returns: Pandas DataFrame

# Get as text documents for embeddings
texts = get_all_employee_texts()
# Returns: List of formatted text strings
```

### Querying the System

```python
from rag_app.rag_system.query_llm_doctype import ask

# Analytical query (uses SQL)
result = ask("What is the average salary?")
print(result['answer'])  # Natural language answer
print(result['sql'])     # Generated SQL query
print(result['raw_data']) # Raw query results

# Semantic query (uses RAG)
result = ask("Tell me about John Doe")
print(result['answer'])  # Natural language answer
print(result['matched_docs'])  # Retrieved employee records
```

### Programmatic Control

```python
from rag_app.rag_system.query_llm_doctype import (
    initialize_sql_from_frappe,
    is_analytical
)

# Initialize SQL database manually
initialize_sql_from_frappe()

# Check query type
is_analytical("How many employees?")  # True
is_analytical("What is John's job?")  # False
```

---

## Automatic Updates

### When Employee Data Changes

After adding/updating/deleting employees in the DocType:

1. **Regenerate embeddings:**
   ```bash
   bench execute rag_app.rag_system.embed_from_doctype.regenerate_embeddings_and_index
   ```

2. **Or use the API:**
   ```python
   frappe.call('rag_app.rag_system.frappe_integration_doctype.rebuild_embeddings')
   ```

### Automating with Hooks

Add to `hooks.py` to auto-rebuild on data changes:

```python
# rag_app/hooks.py

doc_events = {
    "Employee Data": {
        "after_insert": "rag_app.rag_system.hooks.rebuild_embeddings_async",
        "on_update": "rag_app.rag_system.hooks.rebuild_embeddings_async",
        "on_trash": "rag_app.rag_system.hooks.rebuild_embeddings_async",
    }
}
```

Create `rag_app/rag_system/hooks.py`:
```python
import frappe

def rebuild_embeddings_async(doc, method):
    """Queue embedding rebuild as background job"""
    frappe.enqueue(
        'rag_app.rag_system.embed_from_doctype.regenerate_embeddings_and_index',
        queue='long',
        timeout=3600
    )
```

---

## Benefits of DocType-Based Approach

1. **No File Dependencies** - Data stored in Frappe database
2. **Real-Time Updates** - Query the latest data without file imports
3. **Permissions** - Leverage Frappe's permission system
4. **Scalability** - Database queries scale better than file parsing
5. **Integration** - Direct access from any Frappe controller/API
6. **Validation** - DocType validations ensure data quality
7. **Audit Trail** - Track changes with Frappe's version control

---

## SQL Query Support

Both analytical and semantic queries are supported:

### Analytical Queries (SQL)
- Counts: "How many employees?"
- Aggregates: "What is the average/total salary?"
- Rankings: "Top 10 highest paid employees"
- Filters: "Employees with salary above 80000"
- Comparisons: "Compare departments by headcount"
- Group by: "Average salary by department"

### Semantic Queries (RAG)
- Specific lookup: "Tell me about John Doe"
- Employee details: "What is employee 12345's department?"
- Natural language: "Who manages the engineering team?"

---

## Troubleshooting

### Issue: "No employee data found in DocType"
**Solution:** Run the import script:
```bash
bench execute rag_app.rag_system.import_excel_to_doctype.import_employees
```

### Issue: "FAISS index not loaded"
**Solution:** Generate embeddings:
```bash
bench execute rag_app.rag_system.embed_from_doctype.regenerate_embeddings_and_index
```

### Issue: "SQL database not initialized"
**Solution:** The system auto-initializes on first query, but you can force it:
```python
from rag_app.rag_system.query_llm_doctype import initialize_sql_from_frappe
initialize_sql_from_frappe()
```

### Issue: Import errors
**Solution:** Check that Excel file exists at:
```
/workspaces/frappe_codespace/frappe-bench/apps/rag_app/rag_app/rag_data/Employees.xlsx
```

---

## Performance Considerations

- **Embeddings:** Generated once, can be cached
- **SQL Database:** In-memory for fast analytical queries
- **FAISS Search:** Efficient for 1000s of records
- **When to Rebuild:** Only when employee data changes significantly

---

## Testing Commands Summary

```bash
# 1. Import data
bench execute rag_app.rag_system.import_excel_to_doctype.import_employees

# 2. Test data loading
bench execute rag_app.rag_system.doctype_data_loader.test_data_loader

# 3. Generate embeddings
bench execute rag_app.rag_system.embed_from_doctype.regenerate_embeddings_and_index

# 4. Test queries
bench execute rag_app.rag_system.query_llm_doctype.test_queries

# 5. Check stats
bench execute "import frappe; print(frappe.call('rag_app.rag_system.frappe_integration_doctype.get_query_stats'))"
```

---

## Next Steps

1. ✅ Import employee data to DocType
2. ✅ Generate embeddings from DocType
3. ✅ Test queries
4. Update your application to use the new API endpoints
5. (Optional) Remove old Excel-based files
6. (Optional) Set up automatic embedding rebuilds

---

## Support

For issues or questions:
- Check Frappe logs: `bench tail`
- Review error logs in desk: Error Log DocType
- Debug with: `bench execute rag_app.rag_system.query_llm_doctype.test_queries`
