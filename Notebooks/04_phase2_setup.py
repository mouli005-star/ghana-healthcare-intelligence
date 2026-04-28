# Databricks notebook source
# MAGIC %pip install openai pydantic tenacity

# COMMAND ----------

# DBTITLE 1,Set up OpenAI API key
import os

# Paste your OpenAI API key here
os.environ["OPENAI_API_KEY"] = "REDACTED_OPENAI_API_KEY"

# Verify it's set
from openai import OpenAI
client = OpenAI()
print("OpenAI client ready ✅")

# COMMAND ----------

# DBTITLE 1,Test OpenAI API connection
from openai import OpenAI
import os

print("🧪 Testing OpenAI API Connection\n" + "="*80 + "\n")

# Initialize client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Test with a simple question
test_question = "What is the capital of Ghana?"

print(f"📝 Test Question: {test_question}\n")

try:
    # Make API call
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": test_question}
        ],
        max_tokens=100,
        temperature=0.7
    )
    
    # Extract and print the answer
    answer = response.choices[0].message.content
    
    print(f"🤖 OpenAI Response:\n")
    print(f"   {answer}\n")
    
    print("="*80)
    print("\n✅ OpenAI API is working correctly!")
    print(f"\n📊 API Call Details:")
    print(f"   Model: {response.model}")
    print(f"   Tokens used: {response.usage.total_tokens}")
    print(f"   Prompt tokens: {response.usage.prompt_tokens}")
    print(f"   Completion tokens: {response.usage.completion_tokens}")
    
except Exception as e:
    print(f"❌ Error: {str(e)}")
    print("\nPlease check:")
    print("   1. Your API key is correct")
    print("   2. You have credits in your OpenAI account")
    print("   3. You restarted the Python kernel after installing packages")

# COMMAND ----------

# DBTITLE 1,Test with one real facility from silver table
import json

print("🏥 Testing OpenAI with Real Facility Data\n" + "="*80 + "\n")

# Load one facility from silver
silver_df = spark.table("virtue_foundation.ghana_health.silver_facilities").toPandas()

print(f"✅ Loaded silver table: {len(silver_df)} facilities\n")

# Pick a facility that has some capability text but empty procedure/equipment
# This is a good test case for enrichment
test_candidates = silver_df[
    (silver_df['capability'] != '[]') & 
    (silver_df['procedure'] == '[]')
]

if len(test_candidates) > 0:
    test_facility = test_candidates.iloc[0]
    
    print(f"📍 Test Facility Selected:\n")
    print(f"   Name: {test_facility['name']}")
    print(f"   Type: {test_facility['facilityTypeId'] if test_facility['facilityTypeId'] else 'Not specified'}")
    print(f"   City: {test_facility['address_city'] if test_facility['address_city'] else 'Not specified'}")
    print(f"   Region: {test_facility['address_stateOrRegion'] if test_facility['address_stateOrRegion'] else 'Not specified'}")
    print(f"   Coordinates: ({test_facility['latitude']:.4f}, {test_facility['longitude']:.4f})\n")
    
    print(f"📋 Existing Data:\n")
    
    capability_text = test_facility['capability']
    if len(capability_text) > 300:
        print(f"   Capability (first 300 chars): {capability_text[:300]}...")
    else:
        print(f"   Capability: {capability_text}")
    
    print(f"\n   Procedure: {test_facility['procedure']}")
    print(f"   Equipment: {test_facility['equipment']}")
    
    print("\n" + "="*80)
    print("\n✅ Test facility loaded successfully!")
    print("\n💡 This facility has capability text but no structured procedure/equipment data.")
    print("   Perfect candidate for OpenAI enrichment!")
    
else:
    # If no facilities with that criteria, just pick the first one
    test_facility = silver_df.iloc[0]
    
    print(f"📍 Test Facility Selected (first in table):\n")
    print(f"   Name: {test_facility['name']}")
    print(f"   Type: {test_facility['facilityTypeId'] if test_facility['facilityTypeId'] else 'Not specified'}")
    print(f"   City: {test_facility['address_city'] if test_facility['address_city'] else 'Not specified'}")
    print(f"   Capability: {test_facility['capability'][:200] if len(test_facility['capability']) > 200 else test_facility['capability']}")
    print(f"   Procedure: {test_facility['procedure']}")
    print(f"   Equipment: {test_facility['equipment']}")

# COMMAND ----------

# DBTITLE 1,Build facility context for LLM input
import json

print("🔨 Building Facility Context for LLM\n" + "="*80 + "\n")

# Build the input text for the LLM from what we already have
def build_facility_context(row):
    """Combine all available text about a facility into one input string"""
    parts = []
    
    if row.get('name'):
        parts.append(f"Facility Name: {row['name']}")
    
    if row.get('facilityTypeId'):
        parts.append(f"Type: {row['facilityTypeId']}")
    
    if row.get('address_city'):
        city = row.get('address_city', '')
        region = row.get('address_stateOrRegion', '')
        parts.append(f"Location: {city}, {region}, Ghana")
    
    if row.get('description') and row['description'] not in ('', 'null'):
        parts.append(f"Description: {row['description']}")
    
    # The capability field has the raw scraped text - this is the gold mine
    cap_list = json.loads(row['capability']) if row.get('capability') and row['capability'] not in ('', '[]') else []
    if cap_list:
        parts.append("Available information:")
        for item in cap_list:
            if item and str(item).strip():
                parts.append(f"  - {item}")
    
    # Existing procedures if any
    proc_list = json.loads(row['procedure']) if row.get('procedure') and row['procedure'] not in ('', '[]') else []
    if proc_list:
        parts.append("Known procedures:")
        for item in proc_list:
            parts.append(f"  - {item}")
    
    return "\n".join(parts)

# Build context for the test facility
context = build_facility_context(test_facility)

print("📝 INPUT TO LLM:\n")
print("="*80)
print(context)
print("="*80)

print(f"\n✅ Context built successfully!")
print(f"\n📊 Context Statistics:")
print(f"   Total characters: {len(context)}")
print(f"   Total lines: {len(context.split(chr(10)))}")
print(f"\n💡 This context will be sent to OpenAI to extract structured data.")

# COMMAND ----------

# DBTITLE 1,Run OpenAI extraction test
import json
from openai import OpenAI
import os

print("🤖 Running OpenAI Extraction Test\n" + "="*80 + "\n")

# Initialize client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# System prompt for structured extraction
FREE_FORM_SYSTEM_PROMPT = """
ROLE
You are a specialized medical facility information extractor. Your task is to analyze website content and images to extract structured facts about healthcare facilities and organizations.

TASK OVERVIEW
Extract verifiable facts about a medical facility/organization from provided content and output them in a structured JSON format.

Do this inference only for the following organization: `{organization}`

CATEGORY DEFINITIONS
- procedure: Clinical procedures, surgical operations, and medical interventions performed at the facility.
- equipment: Physical medical devices, diagnostic machines, infrastructure, and utilities.
- capability: Medical capabilities that define what level and types of clinical care the facility can deliver.

EXTRACTION GUIDELINES
- Only extract facts directly supported by the provided content
- Use clear, declarative statements in plain English
- Include specific quantities when available
- State facts in present tense

CRITICAL: Return ONLY valid JSON with exactly these three keys: procedure, equipment, capability
Each key maps to an array of strings. Arrays can be empty if nothing is found.
Do not include any explanation or markdown. Return raw JSON only.

Example output format:
{"procedure": ["Performs general consultations", "Offers laboratory testing"], "equipment": ["Has X-ray machine"], "capability": ["Operates 24 hours", "Has 50 inpatient beds"]}
"""

print(f"📤 Sending request to OpenAI...\n")

# Make the API call
response = client.chat.completions.create(
    model="gpt-4o-mini",   # cheap and fast, good for structured extraction
    temperature=0,          # 0 = deterministic, no creativity, just facts
    response_format={"type": "json_object"},
    messages=[
        {
            "role": "system",
            "content": FREE_FORM_SYSTEM_PROMPT.replace("{organization}", test_facility['name'])
        },
        {
            "role": "user", 
            "content": context
        }
    ]
)

print(f"📥 Response received!\n")

# Parse the result
result = json.loads(response.choices[0].message.content)

print("="*80)
print("\n🎯 LLM EXTRACTED FACTS\n")
print("="*80)

print(f"\n📊 Summary:")
print(f"   Procedures found:   {len(result.get('procedure', []))}")
print(f"   Equipment found:    {len(result.get('equipment', []))}")
print(f"   Capabilities found: {len(result.get('capability', []))}")

print(f"\n📋 Full Result:\n")
print(json.dumps(result, indent=2))

print("\n" + "="*80)
print("\n✅ OpenAI extraction test completed successfully!")
print(f"\n📊 API Usage:")
print(f"   Model: {response.model}")
print(f"   Total tokens: {response.usage.total_tokens}")
print(f"   Prompt tokens: {response.usage.prompt_tokens}")
print(f"   Completion tokens: {response.usage.completion_tokens}")