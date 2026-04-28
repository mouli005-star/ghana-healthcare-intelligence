# Databricks notebook source
# MAGIC %pip install openai tenacity

# COMMAND ----------

# DBTITLE 1,Setup and function definitions
import os, json, time, pandas as pd
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import RateLimitError, APIError
from datetime import datetime

os.environ["OPENAI_API_KEY"] = "REDACTED_OPENAI_API_KEY" # Replace with your API key
client = OpenAI()

# ============================================================================
# PROMPTS
# ============================================================================

FREE_FORM_PROMPT = """You are a specialized medical facility information extractor.

Extract verifiable facts about this specific organization: `{organization}`

Only extract facts that are directly stated in the provided content. Do not infer or assume.

Return ONLY valid JSON with exactly these three keys:
- procedure: array of specific clinical procedures performed (e.g. "Performs cesarean sections")
- equipment: array of physical medical devices and infrastructure (e.g. "Has CT scanner", "Has 12 ICU beds")
- capability: array of medical capabilities and care levels (e.g. "Open 24/7", "Level II trauma center", "Has 50 inpatient beds")

Rules:
- Each item must be a complete declarative sentence
- Include numbers/quantities when mentioned
- Empty array [] if nothing found for that category
- Do not include address or contact information
- Return raw JSON only, no markdown, no explanation"""

SPECIALTY_PROMPT = """You are a medical specialty classifier.

Target Facility: {organization}

Analyze the provided information and identify the medical specialties this facility offers.

You MUST only select from this exact list:
Internal Medicine, Family Medicine, Emergency Medicine, General Surgery, Pediatrics,
Gynecology And Obstetrics, Cardiology, Cardiac Surgery, Orthopedic Surgery, Neurology,
Neurological Surgery, Ophthalmology, Otolaryngology, Dentistry, Dermatology,
Psychiatry, Radiology, Diagnostic Radiology, Pathology, Anesthesia, Urology,
Nephrology, Gastroenterology, Pulmonology, Endocrinology And Diabetes And Metabolism,
Infectious Diseases, Hematology, Medical Oncology, Surgical Oncology, Rheumatology,
Physical Medicine And Rehabilitation, Critical Care Medicine, Neonatology Perinatal Medicine,
Maternal Fetal Medicine Or Perinatology, Reproductive Endocrinology And Infertility,
Plastic Surgery, Vascular Surgery, Colorectal Surgery, Thoracic Surgery,
Hospice And Palliative Internal Medicine, Geriatrics Internal Medicine,
Occupational Medicine, Public Health, Global Health And International Health,
Sports Medicine, Orthodontics, Obstetrics And Maternity Care, Social And Behavioral Sciences

Rules:
- Only select specialties clearly mentioned or strongly implied by the content
- Be conservative — if uncertain, do not include
- Use EXACT spelling with proper spacing from the list above
- If facility name contains "Hospital" or "Medical Center" with no specific specialty → include Internal Medicine
- If facility name contains "Clinic" with no specific specialty → include Family Medicine

Return ONLY valid JSON: {"specialties": ["specialty1", "specialty2"]}
No markdown, no explanation."""

ANOMALY_PROMPT = """You are a medical data quality analyst reviewing healthcare facility records in Ghana.

Review this facility's information and identify any suspicious, contradictory, or implausible claims.

Common anomalies to look for:
- Equipment claimed that is very unlikely for the facility type (e.g. MRI in a small clinic)
- Surgical capabilities without anesthesia or ICU support
- Bed capacity numbers that seem implausible (too high or too low for type)
- Specialties that contradict the facility name or description
- Claims that directly contradict each other
- Very advanced capabilities in extremely rural/remote locations without supporting infrastructure

Return ONLY valid JSON:
{
  "anomalies": [
    {"type": "ANOMALY_TYPE", "description": "what is suspicious", "severity": "high/medium/low"}
  ],
  "confidence_score": 0.85,
  "needs_human_review": false
}

If no anomalies found, return empty anomalies array and confidence_score of 1.0
confidence_score is 0.0 to 1.0 where 1.0 means fully trustworthy data"""

# ============================================================================
# EXTRACTION FUNCTIONS
# ============================================================================

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((RateLimitError, APIError))
)
def extract_free_form_facts(facility_name: str, context: str) -> dict:
    """
    Calls OpenAI to extract procedure/equipment/capability from facility context.
    Returns dict with procedure, equipment, capability lists.
    Has automatic retry on rate limit errors.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            response_format={"type": "json_object"},
            max_tokens=1000,
            messages=[
                {
                    "role": "system",
                    "content": FREE_FORM_PROMPT.replace("{organization}", facility_name)
                },
                {
                    "role": "user",
                    "content": context
                }
            ]
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # Validate structure — ensure all three keys exist
        return {
            "procedure": result.get("procedure", []) or [],
            "equipment": result.get("equipment", []) or [],
            "capability": result.get("capability", []) or [],
            "_extraction_status": "success"
        }
    
    except json.JSONDecodeError:
        return {"procedure": [], "equipment": [], "capability": [], "_extraction_status": "json_error"}
    except Exception as e:
        return {"procedure": [], "equipment": [], "capability": [], "_extraction_status": f"error: {str(e)[:50]}"}


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((RateLimitError, APIError))
)
def classify_specialties(facility_name: str, context: str) -> dict:
    """
    Calls OpenAI to classify medical specialties from facility context.
    Returns dict with specialties list.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            response_format={"type": "json_object"},
            max_tokens=300,
            messages=[
                {
                    "role": "system",
                    "content": SPECIALTY_PROMPT.replace("{organization}", facility_name)
                },
                {
                    "role": "user",
                    "content": context
                }
            ]
        )
        
        result = json.loads(response.choices[0].message.content)
        specialties = result.get("specialties", []) or []
        
        # Filter out any hallucinated specialties not in our approved list
        VALID_SPECIALTIES = {
            "Internal Medicine","Family Medicine","Emergency Medicine","General Surgery",
            "Pediatrics","Gynecology And Obstetrics","Cardiology","Cardiac Surgery",
            "Orthopedic Surgery","Neurology","Neurological Surgery","Ophthalmology",
            "Otolaryngology","Dentistry","Dermatology","Psychiatry","Radiology",
            "Diagnostic Radiology","Pathology","Anesthesia","Urology","Nephrology",
            "Gastroenterology","Pulmonology","Endocrinology And Diabetes And Metabolism",
            "Infectious Diseases","Hematology","Medical Oncology","Surgical Oncology",
            "Rheumatology","Physical Medicine And Rehabilitation","Critical Care Medicine",
            "Neonatology Perinatal Medicine","Maternal Fetal Medicine Or Perinatology",
            "Reproductive Endocrinology And Infertility","Plastic Surgery","Vascular Surgery",
            "Colorectal Surgery","Thoracic Surgery","Hospice And Palliative Internal Medicine",
            "Geriatrics Internal Medicine","Occupational Medicine","Public Health",
            "Global Health And International Health","Sports Medicine","Orthodontics",
            "Obstetrics And Maternity Care","Social And Behavioral Sciences"
        }
        valid_specialties = [s for s in specialties if s in VALID_SPECIALTIES]
        
        return {
            "specialties": valid_specialties,
            "_specialty_status": "success"
        }
    
    except json.JSONDecodeError:
        return {"specialties": [], "_specialty_status": "json_error"}
    except Exception as e:
        return {"specialties": [], "_specialty_status": f"error: {str(e)[:50]}"}


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((RateLimitError, APIError))
)
def detect_anomalies(facility_name: str, context: str, extracted_facts: dict) -> dict:
    """
    Checks for suspicious or contradictory capability claims.
    Only runs for facilities with meaningful data (>3 facts total).
    """
    total_facts = (
        len(extracted_facts.get("procedure", [])) +
        len(extracted_facts.get("equipment", [])) +
        len(extracted_facts.get("capability", []))
    )
    
    # Skip anomaly detection for facilities with very little data — not enough to judge
    if total_facts < 2:
        return {
            "anomalies": [],
            "confidence_score": 0.5,
            "needs_human_review": False,
            "_anomaly_status": "skipped_insufficient_data"
        }
    
    # Build enriched context including the newly extracted facts
    enriched_context = context + "\n\nExtracted facts:\n"
    for fact in extracted_facts.get("procedure", []):
        enriched_context += f"PROCEDURE: {fact}\n"
    for fact in extracted_facts.get("equipment", []):
        enriched_context += f"EQUIPMENT: {fact}\n"
    for fact in extracted_facts.get("capability", []):
        enriched_context += f"CAPABILITY: {fact}\n"
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            response_format={"type": "json_object"},
            max_tokens=500,
            messages=[
                {"role": "system", "content": ANOMALY_PROMPT},
                {"role": "user", "content": f"Facility: {facility_name}\n\n{enriched_context}"}
            ]
        )
        
        result = json.loads(response.choices[0].message.content)
        result["_anomaly_status"] = "success"
        return result
    
    except Exception as e:
        return {
            "anomalies": [],
            "confidence_score": 0.5,
            "needs_human_review": False,
            "_anomaly_status": f"error: {str(e)[:50]}"
        }

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def build_facility_context(row):
    """Combine all text we have about a facility into one input string for the LLM"""
    parts = []
    
    if row.get('name') and row['name'] != '':
        parts.append(f"Facility Name: {row['name']}")
    
    if row.get('facilityTypeId') and row['facilityTypeId'] != '':
        parts.append(f"Facility Type: {row['facilityTypeId']}")
    
    if row.get('operatorTypeId') and row['operatorTypeId'] != '':
        parts.append(f"Operator: {row['operatorTypeId']}")
    
    if row.get('address_city') and row['address_city'] != '':
        region = row.get('address_stateOrRegion', '')
        parts.append(f"Location: {row['address_city']}{', ' + region if region else ''}, Ghana")
    
    if row.get('description') and row['description'] not in ('', 'null'):
        parts.append(f"Description: {row['description']}")
    
    # The raw capability array is the richest source of information
    try:
        cap_list = json.loads(row['capability']) if row.get('capability') and row['capability'] not in ('', '[]') else []
    except:
        cap_list = []
    
    if cap_list:
        parts.append("Available information from source:")
        for item in cap_list:
            if item and str(item).strip() not in ('', '""'):
                parts.append(f"  - {item}")
    
    # Include any existing procedures or equipment already in the record
    try:
        proc_list = json.loads(row['procedure']) if row.get('procedure') and row['procedure'] not in ('', '[]') else []
    except:
        proc_list = []
    
    if proc_list:
        parts.append("Already known procedures:")
        for item in proc_list:
            parts.append(f"  - {item}")
    
    try:
        equip_list = json.loads(row['equipment']) if row.get('equipment') and row['equipment'] not in ('', '[]') else []
    except:
        equip_list = []
    
    if equip_list:
        parts.append("Already known equipment:")
        for item in equip_list:
            parts.append(f"  - {item}")
    
    if row.get('missionStatement') and row['missionStatement'] not in ('', 'null'):
        parts.append(f"Mission: {row['missionStatement']}")
    
    return "\n".join(parts)


def should_reclassify_specialty(row) -> bool:
    """
    Returns True if the specialty needs re-evaluation.
    Cases: only has placeholder internalMedicine, or specialty is empty.
    """
    try:
        specs = json.loads(row['specialties']) if row.get('specialties') and row['specialties'] not in ('', '[]') else []
    except:
        specs = []
    
    if not specs:
        return True
    if specs == ['internalMedicine'] and row.get('facilityTypeId') not in ('hospital', ''):
        return True
    return False


def should_extract_freeform(row) -> bool:
    """
    Returns True if procedure or equipment is empty.
    Even if capability has text, we want structured extraction.
    """
    try:
        proc = json.loads(row['procedure']) if row.get('procedure') and row['procedure'] not in ('', '[]') else []
        equip = json.loads(row['equipment']) if row.get('equipment') and row['equipment'] not in ('', '[]') else []
    except:
        proc, equip = [], []
    
    # Only skip if both are already filled
    return len(proc) == 0 or len(equip) == 0


def process_single_facility(row) -> dict:
    """
    Main function: processes one facility through all three extraction steps.
    Returns enriched record with all extracted fields + metadata.
    """
    facility_name = row.get('name', 'Unknown Facility')
    context = build_facility_context(row)
    
    result = {
        "pk_unique_id": row.get('pk_unique_id', ''),
        "name": facility_name,
        "_processing_status": "processed"
    }
    
    # Step 1: Free-form fact extraction
    if should_extract_freeform(row):
        facts = extract_free_form_facts(facility_name, context)
        result["extracted_procedure"] = json.dumps(facts.get("procedure", []))
        result["extracted_equipment"] = json.dumps(facts.get("equipment", []))
        result["extracted_capability"] = json.dumps(facts.get("capability", []))
        result["_freeform_status"] = facts.get("_extraction_status", "unknown")
    else:
        # Already has data, keep existing
        result["extracted_procedure"] = row.get("procedure", "[]")
        result["extracted_equipment"] = row.get("equipment", "[]")
        result["extracted_capability"] = row.get("capability", "[]")
        result["_freeform_status"] = "skipped_already_filled"
    
    # Small delay between calls to be polite to the API
    time.sleep(0.3)
    
    # Step 2: Specialty reclassification
    if should_reclassify_specialty(row):
        specialty_result = classify_specialties(facility_name, context)
        result["extracted_specialties"] = json.dumps(specialty_result.get("specialties", []))
        result["_specialty_status"] = specialty_result.get("_specialty_status", "unknown")
    else:
        result["extracted_specialties"] = row.get("specialties", "[]")
        result["_specialty_status"] = "skipped_already_classified"
    
    time.sleep(0.3)
    
    # Step 3: Anomaly detection
    extracted_facts = {
        "procedure": json.loads(result.get("extracted_procedure", "[]")),
        "equipment": json.loads(result.get("extracted_equipment", "[]")),
        "capability": json.loads(result.get("extracted_capability", "[]"))
    }
    anomaly_result = detect_anomalies(facility_name, context, extracted_facts)
    result["anomaly_flags"] = json.dumps(anomaly_result.get("anomalies", []))
    result["confidence_score"] = anomaly_result.get("confidence_score", 0.5)
    result["needs_human_review"] = anomaly_result.get("needs_human_review", False)
    result["_anomaly_status"] = anomaly_result.get("_anomaly_status", "unknown")
    
    return result

print("Setup complete ✅")

# COMMAND ----------

# DBTITLE 1,Load silver and set up checkpoint
silver_df = spark.table("virtue_foundation.ghana_health.silver_facilities").toPandas()

# Checkpoint path — we save partial results here as we go
# Use dbutils for AWS compatibility
CHECKPOINT_PATH = "/Workspace/Users/burugula_b220794ce@nitc.ac.in/phase2_checkpoint.json"

# Check if we have a partial checkpoint from a previous run
def load_checkpoint():
    try:
        with open(CHECKPOINT_PATH, 'r') as f:
            data = json.load(f)
        print(f"Resuming from checkpoint: {len(data)} facilities already processed")
        return data
    except:
        print("No checkpoint found — starting fresh")
        return {}

def save_checkpoint(results_dict):
    with open(CHECKPOINT_PATH, 'w') as f:
        json.dump(results_dict, f)

# Load existing progress
processed_results = load_checkpoint()
already_done = set(processed_results.keys())

# Figure out which facilities still need processing
remaining = silver_df[~silver_df['pk_unique_id'].isin(already_done)]
print(f"Total facilities:    {len(silver_df)}")
print(f"Already processed:   {len(already_done)}")
print(f"Remaining to run:    {len(remaining)}")

# COMMAND ----------

# DBTITLE 1,Main batch processing loop
BATCH_SIZE = 50
total = len(remaining)
success_count = 0
error_count = 0
start_time = datetime.now()

print(f"Starting batch processing of {total} facilities...")
print(f"Batch size: {BATCH_SIZE}")
print(f"Started at: {start_time.strftime('%H:%M:%S')}\n")

facilities_list = remaining.to_dict('records')

for i, row in enumerate(facilities_list):
    pk_id = row.get('pk_unique_id', f'unknown_{i}')
    
    try:
        result = process_single_facility(row)
        processed_results[pk_id] = result
        success_count += 1
        
        # Progress update every 25 facilities
        if (i + 1) % 25 == 0:
            elapsed = (datetime.now() - start_time).seconds
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            remaining_count = total - (i + 1)
            eta_seconds = remaining_count / rate if rate > 0 else 0
            eta_minutes = eta_seconds / 60
            
            print(f"Progress: {i+1}/{total} | "
                  f"Success: {success_count} | "
                  f"Errors: {error_count} | "
                  f"ETA: {eta_minutes:.1f} min")
        
        # Save checkpoint every 50 facilities
        if (i + 1) % BATCH_SIZE == 0:
            save_checkpoint(processed_results)
            print(f"  ✅ Checkpoint saved at {i+1} facilities")
        
        # Rate limiting pause between facilities
        # gpt-4o-mini allows ~500 RPM so 3 calls per facility = 0.5s pause is safe
        time.sleep(0.5)
        
    except Exception as e:
        error_count += 1
        processed_results[pk_id] = {
            "pk_unique_id": pk_id,
            "name": row.get('name', 'Unknown'),
            "_processing_status": f"failed: {str(e)[:100]}",
            "extracted_procedure": "[]",
            "extracted_equipment": "[]",
            "extracted_capability": "[]",
            "extracted_specialties": "[]",
            "anomaly_flags": "[]",
            "confidence_score": 0.0,
            "needs_human_review": True
        }
        
        # Still save failed ones in checkpoint so we know about them
        if (i + 1) % BATCH_SIZE == 0:
            save_checkpoint(processed_results)

# Final save
save_checkpoint(processed_results)

end_time = datetime.now()
duration = (end_time - start_time).seconds / 60

print(f"\n{'='*50}")
print(f"BATCH PROCESSING COMPLETE")
print(f"Total processed:  {len(processed_results)}")
print(f"Successful:       {success_count}")
print(f"Errors:           {error_count}")
print(f"Total time:       {duration:.1f} minutes")
print(f"{'='*50}")

# COMMAND ----------

# DBTITLE 1,Convert results to DataFrame and save to gold table
# Turn the dict of results into a clean DataFrame
results_df = pd.DataFrame(list(processed_results.values()))

print(f"Results shape: {results_df.shape}")
print(f"\nProcessing status breakdown:")
print(results_df['_processing_status'].value_counts().to_string())

print(f"\nFreeform extraction breakdown:")
print(results_df['_freeform_status'].value_counts().to_string())

print(f"\nSpecialty classification breakdown:")
print(results_df['_specialty_status'].value_counts().to_string())

# How many facilities now have procedures?
has_procedures = results_df['extracted_procedure'].apply(
    lambda x: len(json.loads(x) if x else []) > 0
).sum()
has_equipment = results_df['extracted_equipment'].apply(
    lambda x: len(json.loads(x) if x else []) > 0
).sum()
needs_review = results_df['needs_human_review'].astype(str).eq('True').sum()

print(f"\nExtraction quality:")
print(f"Facilities with procedures now:  {has_procedures}/{len(results_df)}")
print(f"Facilities with equipment now:   {has_equipment}/{len(results_df)}")
print(f"Flagged for human review:        {needs_review}")

# Convert to Spark DataFrame and save to gold table
print(f"\n{'='*60}")
print(f"Saving to gold table...")
print(f"{'='*60}\n")

spark_df = spark.createDataFrame(results_df)

# Add metadata columns
from pyspark.sql.functions import current_timestamp, lit

spark_df = spark_df.withColumn("_enriched_at", current_timestamp()) \
                   .withColumn("_phase", lit("gold"))

print(f"⏳ Writing to: virtue_foundation.ghana_health.gold_facilities_enriched...")

# Write to gold table (overwrite mode since this is the full enriched dataset)
spark_df.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("virtue_foundation.ghana_health.gold_facilities_enriched")

print(f"✅ Gold table saved successfully!")
print(f"\nTable: virtue_foundation.ghana_health.gold_facilities_enriched")
print(f"Rows: {len(results_df)}")
print(f"Columns: {len(spark_df.columns)}")

# Show sample of enriched data
print(f"\n{'='*80}")
print("SAMPLE ENRICHED RECORD:")
print(f"{'='*80}\n")

sample = results_df[results_df['_processing_status'] == 'processed'].iloc[0]
print(f"Facility: {sample['name']}")
print(f"\nExtracted Procedures: {sample['extracted_procedure']}")
print(f"\nExtracted Equipment: {sample['extracted_equipment']}")
print(f"\nExtracted Specialties: {sample['extracted_specialties']}")
print(f"\nConfidence Score: {sample['confidence_score']}")
print(f"Needs Review: {sample['needs_human_review']}")

print(f"\n{'='*80}")
print("✅ ENRICHMENT COMPLETE - Gold table ready for analysis!")
print(f"{'='*80}")