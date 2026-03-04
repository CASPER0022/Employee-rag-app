# RAG Data Folder

Place your local employee source file(s) here, for example `Employees.xlsx`.

This folder is intentionally ignored by Git so private employee data is not pushed.

## Expected local file

- `Employees.xlsx`

## Next steps

After adding your local data, regenerate vector artifacts:

```bash
python rag_app/rag_system/create_db.py
python rag_app/rag_system/embed.py
```
