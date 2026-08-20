import pandas as pd
import sqlite3

# Load CSV and write to SQLite table
df = pd.read_csv("boiler_inspection_data.csv")

conn = sqlite3.connect("boiler_data.db")
df.to_sql("boiler_inspections", conn, if_exists="replace", index=False)
conn.close()

print("✅ Created 'boiler_data.db' with table 'boiler_inspections'.")