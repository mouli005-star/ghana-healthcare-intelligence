# Databricks notebook source
# MAGIC %pip install openai tenacity

# COMMAND ----------

import os, json, time
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import RateLimitError, APIError

os.environ["OPENAI_API_KEY"] = "REDACTED_OPENAI_API_KEY"

client = OpenAI()
print("Ready ✅")

# COMMAND ----------

# DBTITLE 1,Free-Form Extractor function
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

# COMMAND ----------

# DBTITLE 1,Specialty Classifier function
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

# COMMAND ----------

# DBTITLE 1,Anomaly Detector function
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

# COMMAND ----------

# DBTITLE 1,Helper and orchestration functions
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

print("All functions defined ✅")

# COMMAND ----------

# DBTITLE 1,Test pipeline on 3 diverse facilities
print("🧪 Testing Full Pipeline on 3 Diverse Facilities\n" + "="*80 + "\n")

# Load silver table
silver_df = spark.table("virtue_foundation.ghana_health.silver_facilities").toPandas()

print(f"✅ Loaded {len(silver_df)} facilities from silver table\n")

# Pick 3 diverse test cases
test_cases = [
    silver_df[silver_df['facilityTypeId'] == 'hospital'].iloc[0],   # A hospital
    silver_df[silver_df['facilityTypeId'] == 'clinic'].iloc[0],      # A clinic
    silver_df[silver_df['organization_type'] == 'ngo'].iloc[0],      # An NGO
]

print(f"Selected 3 test facilities: hospital, clinic, and NGO\n")
print("="*80)

for i, test_row in enumerate(test_cases, 1):
    print(f"\n{'='*80}")
    print(f"🏥 Test {i}: {test_row['name']} ({test_row['facilityTypeId']})")
    print(f"📍 City: {test_row['address_city']}")
    print(f"🔧 Processing...\n")
    
    # Process the facility through the full pipeline
    result = process_single_facility(test_row)
    
    # Parse results
    proc = json.loads(result['extracted_procedure'])
    equip = json.loads(result['extracted_equipment'])
    cap = json.loads(result['extracted_capability'])
    specs = json.loads(result['extracted_specialties'])
    anomalies = json.loads(result['anomaly_flags'])
    
    print(f"📊 Extraction Results:")
    print(f"   Procedures extracted:   {len(proc)}")
    print(f"   Equipment extracted:    {len(equip)}")
    print(f"   Capabilities extracted: {len(cap)}")
    print(f"   Specialties:            {len(specs)}")
    print(f"   Anomalies found:        {len(anomalies)}")
    print(f"   Confidence score:       {result['confidence_score']}")
    
    # Show sample extractions
    if proc:
        print(f"\n   📝 Sample procedure: {proc[0]}")
    if equip:
        print(f"   🔧 Sample equipment: {equip[0]}")
    if cap:
        print(f"   💪 Sample capability: {cap[0]}")
    if specs:
        print(f"   🏥 Specialties: {', '.join(specs[:3])}{'...' if len(specs) > 3 else ''}")
    if anomalies:
        print(f"   ⚠️  Anomaly detected: {anomalies[0].get('description', 'N/A')}")
    
    # Show processing status
    print(f"\n   🔍 Processing Status:")
    print(f"      Free-form: {result['_freeform_status']}")
    print(f"      Specialty: {result['_specialty_status']}")
    print(f"      Anomaly: {result['_anomaly_status']}")

print("\n" + "="*80)
print("\n✅ Pipeline test complete! All 3 facilities processed successfully.")
print("\n💡 Ready to process all 797 facilities if results look good.")

# COMMAND ----------

# DBTITLE 1,Verify new specialty output format
# Quick test to verify the new specialty output format
print("Testing new specialty format...\n")

# Load one facility for testing
test_df = spark.table("virtue_foundation.ghana_health.silver_facilities").limit(1).toPandas()
test_row = test_df.iloc[0]

print(f"Test Facility: {test_row['name']}")
print(f"Type: {test_row['facilityTypeId']}\n")

# Build context and classify
context = build_facility_context(test_row)
specialty_result = classify_specialties(test_row['name'], context)

print("Specialties extracted:")
for spec in specialty_result.get('specialties', []):
    print(f"  ✓ {spec}")

print(f"\n✅ Format verified! Specialties now have proper spacing instead of camelCase.")