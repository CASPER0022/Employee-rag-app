import sqlite3

conn = sqlite3.connect("db/employees.db")
cursor = conn.cursor()

cursor.execute("SELECT * FROM employees")
rows = cursor.fetchall()

# Get column names
columns = [description[0] for description in cursor.description]

# Convert each row to text
texts = []
for row in rows:
    text = "\n".join([f"{col}: {val}" for col, val in zip(columns, row)])
    texts.append(text)

# Print first example to see the format
print(texts[0])
print("---")
print(f"Total records converted: {len(texts)}")

conn.close()