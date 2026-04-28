# Databricks notebook source
# DBTITLE 1,Load both sides
import pandas as pd, json

silver_df = spark.table("virtue_foundation.ghana_health.silver_facilities").toPandas()

# Load the checkpoint results
CHECKPOINT_PATH = "/Workspace/Users/burugula_b220794ce@nitc.ac.in/phase2_checkpoint.json"
with open(CHECKPOINT_PATH, 'r') as f:
    processed_results = json.load(f)

results_df = pd.DataFrame(list(processed_results.values()))

print(f"Silver rows:  {len(silver_df)}")
print(f"Results rows: {len(results_df)}")

# COMMAND ----------

# DBTITLE 1,Merge IDP results onto silver records
# Merge on pk_unique_id
enriched_df = silver_df.merge(
    results_df[[
        'pk_unique_id',
        'extracted_procedure', 'extracted_equipment',
        'extracted_capability', 'extracted_specialties',
        'anomaly_flags', 'confidence_score',
        'needs_human_review', '_processing_status',
        '_freeform_status', '_specialty_status'
    ]],
    on='pk_unique_id',
    how='left'
)

print(f"Enriched table shape: {enriched_df.shape}")

# Use extracted values as the new authoritative values
# Logic: if extraction produced data, use it; otherwise keep the original

def merge_json_lists(original, extracted):
    """Combine original and extracted lists, deduplicated"""
    try:
        orig_list = json.loads(original) if original and original not in ('', '[]') else []
    except:
        orig_list = []
    try:
        ext_list = json.loads(extracted) if extracted and extracted not in ('', '[]') else []
    except:
        ext_list = []
    
    # Combine and deduplicate
    combined = orig_list.copy()
    seen_lower = {item.lower() for item in orig_list}
    for item in ext_list:
        if item and item.lower() not in seen_lower:
            combined.append(item)
            seen_lower.add(item.lower())
    
    return json.dumps(combined)

# Apply merge for each fact field
enriched_df['final_procedure'] = enriched_df.apply(
    lambda r: merge_json_lists(r.get('procedure', '[]'), r.get('extracted_procedure', '[]')), axis=1
)
enriched_df['final_equipment'] = enriched_df.apply(
    lambda r: merge_json_lists(r.get('equipment', '[]'), r.get('extracted_equipment', '[]')), axis=1
)
enriched_df['final_capability'] = enriched_df.apply(
    lambda r: merge_json_lists(r.get('capability', '[]'), r.get('extracted_capability', '[]')), axis=1
)

# For specialties: use extracted if reclassified, else keep original
def pick_specialties(row):
    if row.get('_specialty_status') == 'success':
        return row.get('extracted_specialties', '[]')
    return row.get('specialties', '[]')

enriched_df['final_specialties'] = enriched_df.apply(pick_specialties, axis=1)

# Add timestamp
from datetime import datetime
enriched_df['_enriched_at'] = datetime.utcnow().isoformat()
enriched_df['_idp_version'] = 'v1.0'

print("Merge complete ✅")

# COMMAND ----------

# DBTITLE 1,Quick quality check before saving
# Compare before and after
before_procedures = silver_df['procedure'].apply(
    lambda x: len(json.loads(x)) > 0 if x and x not in ('', '[]') else False
).sum()

after_procedures = enriched_df['final_procedure'].apply(
    lambda x: len(json.loads(x)) > 0 if x and x not in ('', '[]') else False
).sum()

before_equipment = silver_df['equipment'].apply(
    lambda x: len(json.loads(x)) > 0 if x and x not in ('', '[]') else False
).sum()

after_equipment = enriched_df['final_equipment'].apply(
    lambda x: len(json.loads(x)) > 0 if x and x not in ('', '[]') else False
).sum()

print("PHASE 2 IMPACT SUMMARY")
print("=" * 45)
print(f"{'Metric':<35} {'Before':>6} {'After':>6}")
print("-" * 45)
print(f"{'Facilities with procedures':<35} {before_procedures:>6} {after_procedures:>6}")
print(f"{'Facilities with equipment':<35} {before_equipment:>6} {after_equipment:>6}")
print(f"{'Facilities needing review':<35} {'N/A':>6} {enriched_df['needs_human_review'].astype(str).eq('True').sum():>6}")
print(f"{'Avg confidence score':<35} {'N/A':>6} {enriched_df['confidence_score'].astype(float).mean():>6.2f}")

# COMMAND ----------

# DBTITLE 1,Save the enriched silver table
spark_enriched = spark.createDataFrame(enriched_df.astype(str))

(spark_enriched.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("virtue_foundation.ghana_health.silver_enriched"))

count = spark.table("virtue_foundation.ghana_health.silver_enriched").count()
print(f"✅ Enriched silver table saved!")
print(f"   Table: virtue_foundation.ghana_health.silver_enriched")
print(f"   Rows:  {count}")

# COMMAND ----------

# DBTITLE 1,Final verification SQL
display(spark.sql("""
    SELECT
        facilityTypeId,
        COUNT(*) as total,
        SUM(CASE WHEN final_procedure != '[]' THEN 1 ELSE 0 END) as has_procedures,
        SUM(CASE WHEN final_equipment != '[]' THEN 1 ELSE 0 END) as has_equipment,
        SUM(CASE WHEN needs_human_review = 'True' THEN 1 ELSE 0 END) as flagged_for_review,
        ROUND(AVG(CAST(confidence_score AS DOUBLE)), 2) as avg_confidence
    FROM virtue_foundation.ghana_health.silver_enriched
    GROUP BY facilityTypeId
    ORDER BY total DESC
"""))