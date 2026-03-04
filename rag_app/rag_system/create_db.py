import sqlite3
import pandas as pd


df = pd.read_csv("data/employee_records.csv")

conn = sqlite3.connect("db/employees.db")
df.to_sql("employees", conn, if_exists="replace", index=False)
conn.close()

print(f"Successfully stored {len(df)} employee records in db/employees.db")
