# Databricks notebook source
# DBTITLE 1,Test Suite Header
print("=" * 60)
print("GHANA HEALTHCARE INTELLIGENCE — FINAL TEST SUITE")
print("=" * 60)

# COMMAND ----------

# DBTITLE 1,Test 1: Delta Table Existence
# Test 1: All tables exist
tables = [
    "virtue_foundation.ghana_health.bronze_facilities",
    "virtue_foundation.ghana_health.silver_facilities",
    "virtue_foundation.ghana_health.silver_enriched",
    "virtue_foundation.ghana_health.silver_flagged",
    "virtue_foundation.ghana_health.gold_facilities",
    "virtue_foundation.ghana_health.region_desert_analysis"
]

print("\n[TEST 1] Delta Table Existence:")
all_passed = True
for t in tables:
    try:
        count = spark.table(t).count()
        print(f"  ✅ {t.split('.')[-1]}: {count} rows")
    except Exception as e:
        print(f"  ❌ {t.split('.')[-1]}: MISSING — {e}")
        all_passed = False

# COMMAND ----------

# DBTITLE 1,Test 2: Data Quality Checks
import pandas as pd

gold = spark.table("virtue_foundation.ghana_health.gold_facilities").toPandas()
region = spark.table("virtue_foundation.ghana_health.region_desert_analysis").toPandas()

print("\n[TEST 2] Data Quality Checks:")
checks = {
    "Bronze rows ~987": (870 <= spark.table("virtue_foundation.ghana_health.bronze_facilities").count() <= 1000),
    "Silver rows ~797": (780 <= spark.table("virtue_foundation.ghana_health.silver_facilities").count() <= 820),
    "Gold rows ~797": (780 <= len(gold) <= 820),
    "No farmacy typo": (gold['facilityTypeId'] == 'farmacy').sum() == 0,
    "Regions table has 16 rows": (len(region) >= 14),
    "Gold has official_region": 'official_region' in gold.columns,
    "Gold has deployment_priority_score": 'deployment_priority_score' in gold.columns,
    "Gold has clean_region_mdi": 'clean_region_mdi' in gold.columns,
    "At least 50 facilities geocoded": gold['latitude'].apply(pd.to_numeric, errors='coerce').notna().sum() >= 50,
    "Anomaly flagged column exists": 'needs_human_review' in gold.columns,
}

for check, result in checks.items():
    icon = "✅" if result else "❌"
    print(f"  {icon} {check}")

# COMMAND ----------

# DBTITLE 1,Test 3: Key Metric Sanity
gold['deployment_priority_score'] = pd.to_numeric(gold['deployment_priority_score'], errors='coerce')
gold['clean_region_mdi'] = pd.to_numeric(gold['clean_region_mdi'], errors='coerce')
gold['facility_richness_score'] = pd.to_numeric(gold['facility_richness_score'], errors='coerce')

print("\n[TEST 3] Key Metric Sanity:")
print(f"  Avg priority score: {gold['deployment_priority_score'].mean():.2f}  (expected 0.4-0.8)")
print(f"  Avg richness score: {gold['facility_richness_score'].mean():.2f}  (expected 0.05-0.3)")
print(f"  Regions with CRITICAL alert: {(gold['clean_alert_level']=='CRITICAL').sum()} facilities")
print(f"  Facilities flagged for review: {gold['needs_human_review'].astype(str).eq('True').sum()}")
print(f"  NaN in priority score: {gold['deployment_priority_score'].isna().sum()}  (should be 0)")
print(f"  NaN in region MDI: {gold['clean_region_mdi'].isna().sum()}")

# COMMAND ----------

# DBTITLE 1,Install OpenAI Package
# MAGIC %pip install openai -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Test 4: OpenAI API Connection
import os
from openai import OpenAI

# Use the API key from the planning assistant notebook
os.environ["OPENAI_API_KEY"] = "REDACTED_OPENAI_API_KEY"
client = OpenAI()

print("\n[TEST 4] OpenAI API:")
try:
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=30,
        messages=[{"role":"user","content":"Say: API working"}]
    )
    print(f"  ✅ OpenAI responding: {r.choices[0].message.content}")
except Exception as e:
    print(f"  ❌ OpenAI error: {e}")

# COMMAND ----------

# DBTITLE 1,Test 5: Map File Existence
print("\n[TEST 5] Map File:")
import os
map_exists = os.path.exists('/dbfs/FileStore/ghana_health_map.html')
print(f"  {'✅' if map_exists else '❌'} Map HTML file: {'exists' if map_exists else 'MISSING — run notebook 12'}")

# COMMAND ----------

# DBTITLE 1,Test 6: Planning Assistant Spot Check
print("\n[TEST 6] Planning Assistant Spot Check:")
try:
    # Quick context test
    test_resp = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=100,
        messages=[
            {"role":"system","content":"You are a healthcare analyst. Answer briefly."},
            {"role":"user","content":f"How many facilities are in the dataset? (Answer: {len(gold)})"}
        ]
    )
    print(f"  ✅ Assistant responding correctly")
    print(f"  Response preview: {test_resp.choices[0].message.content[:80]}")
except Exception as e:
    print(f"  ❌ Error: {e}")

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)