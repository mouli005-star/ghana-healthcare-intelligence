# Databricks notebook source
import pandas as pd
import json

raw_df = pd.read_csv(
    "/Workspace/Users/burugula_b220794ce@nitc.ac.in/Accenture Hack/Virtue Foundation Ghana v0.3 - Sheet1.csv",
    dtype=str,
    keep_default_na=False
)

# How many unique facilities do we actually have?
total_rows = len(raw_df)
unique_by_id = raw_df['pk_unique_id'].replace('', None).dropna().nunique()
duplicate_rows = total_rows - unique_by_id

print(f"Total rows in file:        {total_rows}")
print(f"Unique facility IDs:       {unique_by_id}")
print(f"Duplicate/extra rows:      {duplicate_rows}")
print(f"Rows with no ID at all:    {(raw_df['pk_unique_id'] == '').sum()}")

# COMMAND ----------

print("Organization types breakdown:")
print(raw_df['organization_type'].value_counts().to_string())

print("\nFacility types breakdown:")
print(raw_df['facilityTypeId'].value_counts().to_string())

# COMMAND ----------

def is_effectively_empty(val):
    """Check if a value is meaningfully empty regardless of how it's stored"""
    if not val:
        return True
    val = val.strip()
    return val in ('', 'null', 'NULL', '[]', '[""]', 'None')

fields_to_check = [
    'procedure', 'equipment', 'capability',
    'specialties', 'description', 'numberDoctors',
    'capacity', 'operatorTypeId', 'yearEstablished',
    'address_city', 'officialWebsite', 'email'
]

print(f"{'Field':<25} {'Empty':>8} {'Filled':>8} {'% Empty':>10}")
print("-" * 55)
for field in fields_to_check:
    empty = raw_df[field].apply(is_effectively_empty).sum()
    filled = total_rows - empty
    pct = (empty / total_rows) * 100
    print(f"{field:<25} {empty:>8} {filled:>8} {pct:>9.1f}%")

# COMMAND ----------

print("Top 15 cities by facility count:")
city_counts = raw_df[raw_df['address_city'] != '']['address_city'].value_counts().head(15)
print(city_counts.to_string())

print("\n\nRegion distribution:")
region_counts = raw_df[raw_df['address_stateOrRegion'] != '']['address_stateOrRegion'].value_counts()
print(region_counts.to_string())

# COMMAND ----------

from collections import Counter
import json

specialty_counter = Counter()

for val in raw_df['specialties']:
    if is_effectively_empty(val):
        continue
    try:
        specs = json.loads(val)
        if isinstance(specs, list):
            specialty_counter.update(specs)
    except:
        pass

print("Top 20 medical specialties across all facilities:")
print(f"\n{'Specialty':<40} {'Count':>8}")
print("-" * 50)
for spec, count in specialty_counter.most_common(20):
    print(f"{spec:<40} {count:>8}")

# COMMAND ----------

# Save a summary to workspace folder where your CSV is located
summary = {
    "total_rows": int(total_rows),
    "unique_facilities": int(unique_by_id),
    "duplicate_rows": int(duplicate_rows),
    "farmacy_typo_count": int((raw_df['facilityTypeId'] == 'farmacy').sum()),
    "empty_procedure_pct": round(raw_df['procedure'].apply(is_effectively_empty).mean() * 100, 1),
    "empty_equipment_pct": round(raw_df['equipment'].apply(is_effectively_empty).mean() * 100, 1),
    "top_city": raw_df[raw_df['address_city'] != '']['address_city'].value_counts().index[0],
}

import json
summary_path = '/Workspace/Users/burugula_b220794ce@nitc.ac.in/Accenture Hack/profiling_summary.json'

with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"✅ Profiling report saved to:")
print(f"   {summary_path}\n")
print(json.dumps(summary, indent=2))

# COMMAND ----------

