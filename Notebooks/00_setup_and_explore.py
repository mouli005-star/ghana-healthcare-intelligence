# Databricks notebook source
# MAGIC %pip install tabulate

# COMMAND ----------

# DBTITLE 1,Find your CSV file
import os

# Check FileStore location (as per instructions)
print("Checking /FileStore/tables/ ...")
try:
    files = dbutils.fs.ls("dbfs:/FileStore/tables/")
    print(f"   Found {len(files)} files:")
    for f in files:
        print(f"   - {f.name}")
except Exception as e:
    print(f"   FileStore not accessible: {str(e)[:100]}...")

# Check workspace directory (actual location)
print("\nChecking workspace directory...")
workspace_dir = "/Workspace/Users/burugula_b220794ce@nitc.ac.in/Accenture Hack"
if os.path.exists(workspace_dir):
    files = [f for f in os.listdir(workspace_dir) if f.endswith('.csv')]
    if files:
        print(f"   Found {len(files)} CSV file(s):")
        for f in files:
            file_path = os.path.join(workspace_dir, f)
            size_kb = os.path.getsize(file_path) / 1024
            print(f"   - {f} ({size_kb:.2f} KB)")
        print(f"\nFull path to use in your code:")
        print(f'   file_path = "{os.path.join(workspace_dir, files[0])}"')
    else:
        print("   No CSV files found")
else:
    print("   Directory doesn't exist")

# COMMAND ----------

# DBTITLE 1,Load the raw CSV and take a first look
import pandas as pd

# Load the raw CSV — dtype=str means we read EVERYTHING as text first
# This is intentional — we don't want Python to misinterpret numbers or nulls yet
raw_df = pd.read_csv(
    "/Workspace/Users/burugula_b220794ce@nitc.ac.in/Accenture Hack/Virtue Foundation Ghana v0.3 - Sheet1.csv",
    dtype=str,
    keep_default_na=False  # Don't auto-convert empty strings to NaN yet
)

print(f"Total rows: {len(raw_df)}")
print(f"Total columns: {len(raw_df.columns)}")
print(f"\nColumn names:")
for col in raw_df.columns:
    print(f"  - {col}")

# COMMAND ----------

# DBTITLE 1,See a sample row
# Look at the first 5 rows to understand the shape of the data
# Using display() for better table visibility in Databricks
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 80)

print(f"Showing first 5 rows of {len(raw_df)} total rows:\n")
display(raw_df.head(5))

# COMMAND ----------

# DBTITLE 1,Cell 5
# MAGIC %pip install torch

# COMMAND ----------

import torch
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")