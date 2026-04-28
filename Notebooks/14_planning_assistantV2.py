# Databricks notebook source
# DBTITLE 1,Cell 1 — Install and Setup
# MAGIC %pip install openai

# COMMAND ----------

# DBTITLE 1,Load Ghana Health Data and Setup OpenAI
import os
import json
import pandas as pd
from openai import OpenAI

# Set your OpenAI key
os.environ["OPENAI_API_KEY"] = "REDACTED_OPENAI_API_KEY" # Replace with your API key
client = OpenAI()

# Load Gold table
gold_df = spark.table("virtue_foundation.ghana_health.gold_facilities").toPandas()
region_df = spark.table("virtue_foundation.ghana_health.region_desert_analysis").toPandas()

# Fix numeric types
for col in ['facility_richness_score', 'deployment_priority_score', 'clean_region_mdi',
            'confidence_score', 'latitude', 'longitude']:
    if col in gold_df.columns:
        gold_df[col] = pd.to_numeric(gold_df[col], errors='coerce')

# Fix boolean types
bool_caps = ['has_emergency', 'has_surgery', 'has_icu', 'has_maternity',
             'has_lab', 'has_imaging', 'has_blood_bank', 'has_dialysis',
             'has_dental', 'has_eye_care', 'has_mental_health', 'has_pharmacy']
for col in bool_caps:
    if col in gold_df.columns:
        gold_df[col] = gold_df[col].astype(str).str.lower().eq('true')

print(f"✅ Gold table loaded: {len(gold_df)} facilities")
print(f"✅ Region table loaded: {len(region_df)} regions")
print(f"   Hospitals: {(gold_df['facilityTypeId']=='hospital').sum()}")
print(f"   With emergency: {gold_df['has_emergency'].sum()}")
print(f"   With surgery: {gold_df['has_surgery'].sum()}")

# COMMAND ----------

# DBTITLE 1,Build the complete data context
GHANA_16_REGIONS = [
    'Upper East', 'Upper West', 'Savannah', 'North East', 'Northern',
    'Oti', 'Volta', 'Ahafo', 'Bono', 'Bono East', 'Eastern',
    'Western North', 'Western', 'Central', 'Ashanti', 'Greater Accra'
]

def build_full_context():
    lines = []

    # ── Overall stats ──────────────────────────────────────────────
    lines.append("=== GHANA HEALTHCARE SYSTEM OVERVIEW ===")
    lines.append(f"Total facilities in database: {len(gold_df)}")
    lines.append(f"Hospitals: {(gold_df['facilityTypeId']=='hospital').sum()}")
    lines.append(f"Clinics: {(gold_df['facilityTypeId']=='clinic').sum()}")
    lines.append(f"NGOs: {(gold_df['organization_type']=='ngo').sum()}")
    lines.append(f"Health Centers: {(gold_df['facilityTypeId']=='health_center').sum()}")
    lines.append(f"Maternity Homes: {(gold_df['facilityTypeId']=='maternity_home').sum()}")
    lines.append(f"Pharmacies: {(gold_df['facilityTypeId']=='pharmacy').sum()}")
    lines.append(f"Dentists: {(gold_df['facilityTypeId']=='dentist').sum()}")
    lines.append(f"Facilities with emergency care: {gold_df['has_emergency'].sum()}")
    lines.append(f"Facilities with surgery: {gold_df['has_surgery'].sum()}")
    lines.append(f"Facilities with ICU: {gold_df['has_icu'].sum()}")
    lines.append(f"Facilities with maternity services: {gold_df['has_maternity'].sum()}")
    lines.append(f"Facilities with laboratory: {gold_df['has_lab'].sum()}")
    lines.append(f"Facilities with imaging/radiology: {gold_df['has_imaging'].sum()}")
    lines.append(f"Facilities with blood bank: {gold_df['has_blood_bank'].sum()}")
    lines.append(f"Facilities with dialysis: {gold_df['has_dialysis'].sum()}")
    lines.append(f"Facilities with dental: {gold_df['has_dental'].sum()}")
    lines.append(f"Facilities with eye care: {gold_df['has_eye_care'].sum()}")
    lines.append(f"Facilities with mental health: {gold_df['has_mental_health'].sum()}")
    lines.append(f"Facilities flagged for data review: {gold_df['needs_human_review'].astype(str).eq('True').sum()}")
    lines.append(f"Average confidence score: {gold_df['confidence_score'].mean():.2f}")

    # ── Region by region breakdown ────────────────────────────────
    lines.append("\n=== MEDICAL DESERT INDEX BY REGION (0=desert, 1=full coverage) ===")
    for region in GHANA_16_REGIONS:
        g = gold_df[gold_df['official_region'] == region]
        if len(g) == 0:
            lines.append(
                f"{region}: MDI=0.00 [CRITICAL DESERT] "
                f"Facilities=0 | NO DATA RECORDED — assumed desert"
            )
            continue

        mdi = g['clean_region_mdi'].mean()
        alert = g['clean_alert_level'].mode()[0] if len(g) > 0 else 'UNKNOWN'
        hosp = (g['facilityTypeId'] == 'hospital').sum()
        em = g['has_emergency'].sum()
        surg = g['has_surgery'].sum()
        icu = g['has_icu'].sum()
        mat = g['has_maternity'].sum()
        lab = g['has_lab'].sum()
        bb = g['has_blood_bank'].sum()
        img = g['has_imaging'].sum()
        dial = g['has_dialysis'].sum()
        avg_priority = g['deployment_priority_score'].mean()

        missing = []
        for cap, label in [
            ('has_emergency', 'Emergency Care'),
            ('has_surgery', 'Surgery'),
            ('has_icu', 'ICU'),
            ('has_maternity', 'Maternity'),
            ('has_lab', 'Laboratory'),
            ('has_blood_bank', 'Blood Bank'),
            ('has_imaging', 'Medical Imaging'),
            ('has_dialysis', 'Dialysis')
        ]:
            if not g[cap].any():
                missing.append(label)

        lines.append(
            f"{region}: MDI={mdi:.3f} [{alert}] | "
            f"Total={len(g)} | Hospitals={hosp} | "
            f"Emergency={em} | Surgery={surg} | ICU={icu} | "
            f"Maternity={mat} | Lab={lab} | BloodBank={bb} | "
            f"Imaging={img} | Dialysis={dial} | "
            f"AvgPriority={avg_priority:.2f} | "
            f"MISSING: {', '.join(missing) if missing else 'None'}"
        )

    # ── Top 25 priority facilities ────────────────────────────────
    lines.append("\n=== TOP 25 FACILITIES REQUIRING URGENT SUPPORT ===")
    top25 = gold_df.nlargest(25, 'deployment_priority_score')
    for i, (_, row) in enumerate(top25.iterrows(), 1):
        try:
            specs = json.loads(str(row.get('final_specialties') or '[]'))
            specs_text = ', '.join(specs[:3]) if specs else 'not classified'
        except:
            specs_text = 'not classified'

        caps = []
        for cap, label in [
            ('has_emergency', 'Emergency'), ('has_surgery', 'Surgery'),
            ('has_icu', 'ICU'), ('has_maternity', 'Maternity'),
            ('has_lab', 'Lab'), ('has_blood_bank', 'Blood Bank')
        ]:
            if row.get(cap):
                caps.append(label)

        missing_caps = []
        for cap, label in [
            ('has_emergency', 'Emergency'), ('has_surgery', 'Surgery'),
            ('has_icu', 'ICU'), ('has_maternity', 'Maternity')
        ]:
            if not row.get(cap):
                missing_caps.append(label)

        lines.append(
            f"#{i} {row['name']} | "
            f"City: {row.get('address_city', 'Unknown')} | "
            f"Region: {row.get('official_region', 'Unknown')} | "
            f"Type: {row.get('facilityTypeId', 'Unknown')} | "
            f"Priority Score: {row.get('deployment_priority_score', 0):.3f} | "
            f"Region MDI: {row.get('clean_region_mdi', 0):.3f} | "
            f"Alert: {row.get('clean_alert_level', 'UNKNOWN')} | "
            f"Richness: {row.get('facility_richness_score', 0):.3f} | "
            f"Has: {', '.join(caps) if caps else 'No major capabilities'} | "
            f"Missing: {', '.join(missing_caps) if missing_caps else 'None critical'} | "
            f"Specialties: {specs_text} | "
            f"Confidence: {row.get('confidence_score', 0.5):.2f}"
        )

    # ── All hospitals with full details ───────────────────────────
    lines.append("\n=== ALL HOSPITALS — FULL CAPABILITY DETAILS ===")
    hospitals = gold_df[gold_df['facilityTypeId'] == 'hospital'].sort_values(
        'deployment_priority_score', ascending=False
    )
    for _, row in hospitals.iterrows():
        try:
            procs = json.loads(str(row.get('final_procedure') or '[]'))
            equip = json.loads(str(row.get('final_equipment') or '[]'))
            caps_list = json.loads(str(row.get('final_capability') or '[]'))
        except:
            procs, equip, caps_list = [], [], []

        lines.append(
            f"HOSPITAL: {row['name']} | "
            f"{row.get('address_city', '')}, {row.get('official_region', '')} | "
            f"Emergency={'YES' if row.get('has_emergency') else 'NO'} | "
            f"Surgery={'YES' if row.get('has_surgery') else 'NO'} | "
            f"ICU={'YES' if row.get('has_icu') else 'NO'} | "
            f"Maternity={'YES' if row.get('has_maternity') else 'NO'} | "
            f"Lab={'YES' if row.get('has_lab') else 'NO'} | "
            f"Imaging={'YES' if row.get('has_imaging') else 'NO'} | "
            f"BloodBank={'YES' if row.get('has_blood_bank') else 'NO'} | "
            f"Procedures: {'; '.join(procs[:3]) if procs else 'not extracted'} | "
            f"Equipment: {'; '.join(equip[:3]) if equip else 'not extracted'} | "
            f"Priority: {row.get('deployment_priority_score', 0):.3f} | "
            f"Confidence: {row.get('confidence_score', 0.5):.2f}"
        )

    # ── Critical gaps summary ─────────────────────────────────────
    lines.append("\n=== CRITICAL CAPABILITY GAPS — REGIONS WITH ZERO ACCESS ===")
    for cap, label in [
        ('has_emergency', 'Emergency Care'),
        ('has_surgery', 'Surgery'),
        ('has_icu', 'ICU / Intensive Care'),
        ('has_maternity', 'Maternity Services'),
        ('has_blood_bank', 'Blood Bank'),
        ('has_dialysis', 'Kidney Dialysis'),
    ]:
        zero_regions = []
        for region in GHANA_16_REGIONS:
            g = gold_df[gold_df['official_region'] == region]
            if len(g) == 0 or not g[cap].any():
                zero_regions.append(region)
        lines.append(f"Regions with ZERO {label}: {', '.join(zero_regions)}")

    # ── NGO details ───────────────────────────────────────────────
    lines.append("\n=== NGOs OPERATING IN GHANA ===")
    ngos = gold_df[gold_df['organization_type'] == 'ngo']
    for _, row in ngos.iterrows():
        lines.append(
            f"NGO: {row['name']} | "
            f"{row.get('address_city', '')}, {row.get('official_region', '')} | "
            f"Mission: {str(row.get('missionStatement', ''))[:100]}"
        )

    return "\n".join(lines)


DATA_CONTEXT = build_full_context()
print(f"✅ Data context built: {len(DATA_CONTEXT):,} characters")
print(f"   Covers {len(GHANA_16_REGIONS)} regions and {len(gold_df)} facilities")
print("\nPreview of first 300 chars:")
print(DATA_CONTEXT[:300])

# COMMAND ----------

# DBTITLE 1,AI response function with structured output
SYSTEM_PROMPT = """You are a senior healthcare intelligence analyst at the Virtue Foundation, 
specializing in Ghana's healthcare system.

You have been given a comprehensive database of 797 healthcare facilities across Ghana's 
16 administrative regions, including Medical Desert Index (MDI) scores, capability flags, 
deployment priority scores, and facility-level clinical details.

For every question, you MUST respond with EXACTLY this JSON structure and nothing else:

{
  "answer": "A clear, conversational 3-5 sentence answer in plain English. Cite specific facility names, region names, and numbers. Write as if explaining to a non-technical NGO coordinator.",
  
  "findings": [
    {
      "point": "Specific factual finding from the data",
      "citation": "Source: [Facility Name / Region Name] — [specific data field, e.g. MDI=0.05, Priority=0.82]"
    }
  ],
  
  "recommendations": [
    "Concrete actionable recommendation — what the Virtue Foundation should do, specific and immediate"
  ],
  
  "confidence": {
    "level": "HIGH or MEDIUM or LOW",
    "score": 0.85,
    "reason": "One sentence explaining confidence — e.g. Based on complete data for this region vs limited data available"
  }
}

Rules:
- findings: always 3-5 bullet points, each with a citation to actual facility or region data
- recommendations: always 2-4 concrete actions
- confidence HIGH = strong data backing, MEDIUM = partial data, LOW = inferred from limited data
- Never make up facility names — only use names from the provided database
- MDI below 0.30 = medical desert, below 0.15 = critical desert
- Deployment Priority Score: higher = more urgent need for external support
- answer must be readable by a non-technical person
- Return ONLY the JSON, no markdown, no explanation outside the JSON"""


def ask_planning_assistant(question: str, history: list) -> dict:
    """
    Sends question to GPT-4o with full data context.
    Returns structured dict with answer, findings, recommendations, confidence.
    """
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": f"Here is the complete Ghana healthcare database:\n\n{DATA_CONTEXT}"
        },
        {
            "role": "assistant",
            "content": json.dumps({
                "answer": "Ghana healthcare database fully loaded. I have data on 797 facilities across all 16 regions including MDI scores, capability flags, priority scores, and facility details. Ready for your questions.",
                "findings": [],
                "recommendations": [],
                "confidence": {"level": "HIGH", "score": 1.0, "reason": "Data loaded successfully"}
            })
        }
    ]

    # Add conversation history (last 4 exchanges)
    for h in history[-4:]:
        messages.append({"role": "user", "content": h["q"]})
        messages.append({"role": "assistant", "content": json.dumps(h["a"])})

    messages.append({"role": "user", "content": question})

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.15,
            max_tokens=1500,
            response_format={"type": "json_object"},
            messages=messages
        )

        raw = response.choices[0].message.content
        result = json.loads(raw)

        # Validate structure — fill any missing keys
        if "answer" not in result:
            result["answer"] = "Unable to generate answer."
        if "findings" not in result:
            result["findings"] = []
        if "recommendations" not in result:
            result["recommendations"] = []
        if "confidence" not in result:
            result["confidence"] = {"level": "LOW", "score": 0.3, "reason": "Incomplete response"}

        return result

    except json.JSONDecodeError:
        return {
            "answer": "There was a parsing error. Please try your question again.",
            "findings": [],
            "recommendations": ["Try rephrasing your question"],
            "confidence": {"level": "LOW", "score": 0.0, "reason": "JSON parse error"}
        }
    except Exception as e:
        return {
            "answer": f"Error: {str(e)[:100]}. Please check your OpenAI API key and try again.",
            "findings": [],
            "recommendations": ["Check that your OpenAI API key is set correctly in Cell 2"],
            "confidence": {"level": "LOW", "score": 0.0, "reason": "System error"}
        }


# Test with one real question before building UI
print("Testing AI connection...")
test = ask_planning_assistant("How many facilities are in the database?", [])
print(f"✅ Answer: {test['answer'][:100]}...")
print(f"✅ Findings: {len(test['findings'])} points")
print(f"✅ Confidence: {test['confidence']['level']}")

# COMMAND ----------

# DBTITLE 1,Complete professional UI
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output

# ── Conversation state ────────────────────────────────────────────
conversation_history = []
all_messages_html = []

# ── Inject global CSS ─────────────────────────────────────────────
display(HTML("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.vf-wrap * { box-sizing: border-box; font-family: 'Inter', -apple-system, sans-serif; }
.vf-wrap { width: 100%; }

/* Header */
.vf-header {
  background: linear-gradient(135deg, #050d1a, #0d2137, #0a3d6e);
  padding: 18px 24px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  border-bottom: 2px solid #1565c0;
  min-height: 80px;
}
.vf-logo { display: flex; align-items: center; gap: 14px; }
.vf-logo-icon {
  width: 48px; height: 48px;
  background: #1565c0;
  border-radius: 12px;
  display: flex; align-items: center; justify-content: center;
  font-size: 26px;
  flex-shrink: 0;
}
.vf-logo-text { display: flex; flex-direction: column; gap: 4px; }
.vf-title { 
  color: white; 
  font-size: 18px; 
  font-weight: 700; 
  letter-spacing: -0.3px;
  line-height: 1.2;
  margin: 0;
  padding: 0;
}
.vf-subtitle { 
  color: #90caf9; 
  font-size: 12px;
  line-height: 1.2;
  margin: 0;
  padding: 0;
}
.vf-badge {
  background: rgba(255,255,255,0.1);
  border: 1px solid rgba(255,255,255,0.2);
  border-radius: 20px; 
  padding: 6px 16px;
  color: #e3f2fd; 
  font-size: 11px; 
  font-weight: 600;
  white-space: nowrap;
}

/* Stats bar */
.vf-stats {
  background: #070f1c;
  display: flex;
  border-bottom: 1px solid #0d2137;
  padding: 4px 0;
}
.vf-stat {
  flex: 1; 
  padding: 12px 8px; 
  text-align: center;
  border-right: 1px solid #0d2137;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 4px;
}
.vf-stat:last-child { border-right: none; }
.vf-stat-num { 
  font-size: 22px; 
  font-weight: 800; 
  color: white; 
  line-height: 1;
  margin: 0;
  padding: 0;
}
.vf-stat-num.red { color: #ff5252; }
.vf-stat-num.orange { color: #ffa726; }
.vf-stat-num.blue { color: #42a5f5; }
.vf-stat-num.green { color: #66bb6a; }
.vf-stat-num.yellow { color: #ffeb3b; }
.vf-stat-lbl {
  font-size: 9px; 
  color: #78909c; 
  text-transform: uppercase;
  letter-spacing: 1px;
  line-height: 1;
  margin: 0;
  padding: 0;
  font-weight: 600;
}

/* Quick questions */
.vf-chips-label {
  background: #0a1929; 
  padding: 12px 20px 8px;
  font-size: 10px; 
  color: #90caf9;
  text-transform: uppercase; 
  letter-spacing: 1.2px;
  font-weight: 700;
  border-bottom: 1px solid #1565c0;
}
.vf-chips {
  background: #0a1929; 
  padding: 12px 20px 16px;
  display: flex; 
  flex-wrap: wrap; 
  gap: 8px;
  border-bottom: 2px solid #0d2137;
}
.vf-chip {
  background: linear-gradient(135deg, #1565c0, #1976d2);
  border: 1px solid #42a5f5;
  border-radius: 18px; 
  padding: 7px 15px;
  color: #e3f2fd; 
  font-size: 11px; 
  font-weight: 600;
  cursor: pointer; 
  transition: all 0.2s;
  box-shadow: 0 2px 6px rgba(21, 101, 192, 0.4);
}
.vf-chip:hover { 
  background: linear-gradient(135deg, #1976d2, #42a5f5);
  color: white;
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(66, 165, 245, 0.6);
}

/* Chat area */
.vf-chat {
  background: #050d1a; 
  padding: 16px 20px;
  overflow-y: auto; 
  min-height: 400px; 
  max-height: 450px;
}

/* Welcome screen */
.vf-welcome {
  text-align: center; 
  padding: 40px 20px;
}
.vf-welcome-icon { 
  font-size: 48px; 
  margin-bottom: 14px;
}
.vf-welcome-title { 
  color: #64b5f6; 
  font-size: 18px; 
  font-weight: 700; 
  margin-bottom: 8px;
}
.vf-welcome-sub { 
  color: #78909c; 
  font-size: 13px; 
  line-height: 1.7;
}
.vf-welcome-cards {
  display: flex; 
  justify-content: center; 
  gap: 12px;
  margin-top: 20px; 
  flex-wrap: wrap;
}
.vf-welcome-card {
  background: #0d2137; 
  border: 1px solid #1565c0;
  border-radius: 10px; 
  padding: 10px 16px;
  font-size: 12px; 
  color: #64b5f6;
}

/* User message */
.vf-msg-user-wrap {
  display: flex; 
  justify-content: flex-end;
  margin: 12px 0 4px; 
  gap: 10px; 
  align-items: flex-start;
}
.vf-msg-user-label {
  text-align: right; 
  font-size: 10px; 
  color: #546e7a; 
  margin-bottom: 3px;
}
.vf-msg-user {
  background: linear-gradient(135deg, #1565c0, #1976d2);
  color: white; 
  padding: 11px 16px;
  border-radius: 16px 16px 3px 16px;
  font-size: 13px; 
  line-height: 1.6;
  max-width: 75%;
}
.vf-avatar-user {
  width: 32px; 
  height: 32px; 
  border-radius: 50%;
  background: #1565c0; 
  color: white;
  display: flex; 
  align-items: center; 
  justify-content: center;
  font-size: 11px; 
  font-weight: 700; 
  flex-shrink: 0;
}

/* Bot response container */
.vf-response-wrap { margin: 6px 0 16px; }
.vf-response-header {
  display: flex; 
  align-items: center; 
  gap: 10px; 
  margin-bottom: 10px;
}
.vf-avatar-bot {
  width: 32px; 
  height: 32px; 
  border-radius: 50%;
  background: linear-gradient(135deg, #0d2137, #1565c0);
  display: flex; 
  align-items: center; 
  justify-content: center;
  font-size: 16px; 
  flex-shrink: 0;
}
.vf-response-label { 
  font-size: 10px; 
  color: #546e7a;
}

/* CARD 1 - Answer */
.vf-card-answer {
  background: #0d2137;
  border-left: 3px solid #42a5f5;
  border-radius: 10px; 
  padding: 14px 18px;
  margin-bottom: 10px;
}
.vf-card-header {
  font-size: 10px; 
  font-weight: 700; 
  text-transform: uppercase;
  letter-spacing: 1.5px; 
  margin-bottom: 10px;
  display: flex; 
  align-items: center; 
  gap: 6px;
}
.vf-card-answer .vf-card-header { color: #42a5f5; }
.vf-answer-text {
  color: #e3f2fd; 
  font-size: 13px; 
  line-height: 1.75;
}

/* CARD 2 - Findings */
.vf-card-findings {
  background: #040c18;
  border-left: 3px solid #1565c0;
  border-radius: 10px; 
  padding: 14px 18px;
  margin-bottom: 10px;
}
.vf-card-findings .vf-card-header { color: #1565c0; }
.vf-finding-item {
  margin-bottom: 10px; 
  padding-bottom: 10px;
  border-bottom: 1px solid #0d2137;
}
.vf-finding-item:last-child { 
  margin-bottom: 0; 
  padding-bottom: 0; 
  border-bottom: none;
}
.vf-finding-num {
  display: inline-block;
  background: #1565c0; 
  color: white;
  border-radius: 50%; 
  width: 20px; 
  height: 20px;
  text-align: center; 
  line-height: 20px;
  font-size: 11px; 
  font-weight: 700;
  margin-right: 8px;
}
.vf-finding-point { 
  color: #b0bec5; 
  font-size: 12px; 
  line-height: 1.6; 
  margin-bottom: 4px;
}
.vf-citation {
  background: #0d2137; 
  border-radius: 4px; 
  padding: 4px 10px;
  font-size: 10px; 
  color: #546e7a; 
  font-family: 'Courier New', monospace;
  display: inline-block; 
  margin-top: 4px;
}

/* CARD 3 - Recommendations */
.vf-card-recommendations {
  background: #040c18;
  border-left: 3px solid #00c853;
  border-radius: 10px; 
  padding: 14px 18px;
  margin-bottom: 10px;
}
.vf-card-recommendations .vf-card-header { color: #00c853; }
.vf-rec-item {
  display: flex; 
  gap: 10px; 
  margin-bottom: 8px; 
  align-items: flex-start;
  color: #b0bec5; 
  font-size: 12px; 
  line-height: 1.6;
}
.vf-rec-item:last-child { margin-bottom: 0; }
.vf-rec-num {
  background: #00c853; 
  color: #000; 
  border-radius: 50%;
  width: 18px; 
  height: 18px; 
  display: flex; 
  align-items: center;
  justify-content: center; 
  font-size: 9px; 
  font-weight: 800;
  flex-shrink: 0; 
  margin-top: 1px;
}

/* CARD 4 - Confidence */
.vf-card-confidence {
  display: flex; 
  align-items: center; 
  gap: 10px;
  background: #040c18;
  border-radius: 8px; 
  padding: 10px 16px;
}
.vf-conf-badge {
  border-radius: 6px; 
  padding: 3px 10px;
  font-size: 11px; 
  font-weight: 700;
}
.vf-conf-bar-wrap {
  flex: 1; 
  background: #0d2137; 
  border-radius: 4px; 
  height: 6px;
}
.vf-conf-bar { 
  height: 6px; 
  border-radius: 4px;
}
.vf-conf-score { 
  font-size: 13px; 
  font-weight: 700; 
  color: white; 
  min-width: 36px;
}
.vf-conf-reason { 
  font-size: 11px; 
  color: #546e7a;
}

/* Thinking indicator */
.vf-thinking {
  background: #0d2137; 
  border-left: 3px solid #1565c0;
  border-radius: 10px; 
  padding: 14px 18px;
  color: #546e7a; 
  font-size: 13px; 
  font-style: italic;
}

/* Input area */
.vf-input-area {
  background: #070f1c; 
  border-top: 1px solid #0d2137;
  padding: 14px 18px; 
  display: flex; 
  gap: 10px; 
  align-items: center;
}

/* Footer */
.vf-footer {
  background: #040c18; 
  border-top: 1px solid #0d2137;
  padding: 7px 18px;
  display: flex; 
  justify-content: space-between; 
  align-items: center;
}
.vf-footer-text { 
  font-size: 10px; 
  color: #37474f;
}
</style>
"""))

# ── Header widget ─────────────────────────────────────────────────
header = widgets.HTML(f"""
<div class="vf-wrap">
<div class="vf-header">
  <div class="vf-logo">
    <div class="vf-logo-icon">🏥</div>
    <div class="vf-logo-text">
      <div class="vf-title">Ghana Healthcare Planning Intelligence</div>
      <div class="vf-subtitle">Virtue Foundation · Databricks Hackathon 2024</div>
    </div>
  </div>
  <div class="vf-badge">GPT-4o · {len(gold_df)} Facilities · 16 Regions</div>
</div>
</div>
""")

# ── Stats bar ─────────────────────────────────────────────────────
stats_bar = widgets.HTML(f"""
<div class="vf-wrap">
<div class="vf-stats">
  <div class="vf-stat">
    <div class="vf-stat-num">{len(gold_df)}</div>
    <div class="vf-stat-lbl">Facilities</div>
  </div>
  <div class="vf-stat">
    <div class="vf-stat-num red">{(gold_df['clean_alert_level']=='CRITICAL').sum()}</div>
    <div class="vf-stat-lbl">Critical</div>
  </div>
  <div class="vf-stat">
    <div class="vf-stat-num orange">{gold_df['has_emergency'].sum()}</div>
    <div class="vf-stat-lbl">Emergency</div>
  </div>
  <div class="vf-stat">
    <div class="vf-stat-num blue">{gold_df['has_surgery'].sum()}</div>
    <div class="vf-stat-lbl">Surgery</div>
  </div>
  <div class="vf-stat">
    <div class="vf-stat-num green">{gold_df['has_icu'].sum()}</div>
    <div class="vf-stat-lbl">ICU</div>
  </div>
  <div class="vf-stat">
    <div class="vf-stat-num yellow">{gold_df['needs_human_review'].astype(str).eq('True').sum()}</div>
    <div class="vf-stat-lbl">Flagged</div>
  </div>
</div>
</div>
""")

# ── Quick question buttons ────────────────────────────────────────
QUICK_QUESTIONS = [
    "🔴 Which region needs help most urgently?",
    "🔪 Where should we deploy a surgeon?",
    "👶 Which regions have no maternity care?",
    "🏥 What are the top 5 priority facilities?",
    "🚨 Which regions have zero emergency care?",
    "🩸 Where is blood bank service missing?",
    "📍 Full summary of Upper East Region",
    "💉 Which hospital has the most equipment?",
    "🌍 Compare Northern vs Greater Accra",
    "💡 Give a complete resource deployment plan",
]

chip_label = widgets.HTML(
    '<div class="vf-wrap"><div class="vf-chips-label">Quick Questions</div></div>'
)

chip_buttons = []
for q in QUICK_QUESTIONS:
    btn = widgets.Button(
        description=q,
        layout=widgets.Layout(width='auto', height='32px'),
        style={'button_color': '#1565c0', 'font_color': '#e3f2fd', 'font_size': '11px'}
    )
    chip_buttons.append(btn)

chips_row = widgets.HBox(
    chip_buttons,
    layout=widgets.Layout(
        flex_flow='row wrap', 
        gap='8px',
        padding='12px 20px 16px',
        background_color='#0a1929'
    )
)

# ── Chat output area ──────────────────────────────────────────────
chat_out = widgets.Output(
    layout=widgets.Layout(
        min_height='400px',
        max_height='450px',
        overflow_y='auto',
        background_color='#050d1a',
        padding='0px'
    )
)

# ── Input row ─────────────────────────────────────────────────────
text_input = widgets.Text(
    placeholder='Ask anything about Ghana healthcare — press Enter or click Send',
    layout=widgets.Layout(
        flex='1', 
        height='40px',
        border='1px solid #1565c0',
        border_radius='8px',
        background_color='#0d2137',
        color='white'
    )
)
send_btn = widgets.Button(
    description='Send ➤',
    button_style='primary',
    layout=widgets.Layout(width='100px', height='40px'),
    style={'button_color': '#1565c0', 'font_weight': '700'}
)
clear_btn = widgets.Button(
    description='Clear Chat',
    layout=widgets.Layout(width='90px', height='28px'),
    style={'button_color': '#0d2137', 'font_color': '#546e7a'}
)

input_row = widgets.HBox(
    [text_input, send_btn],
    layout=widgets.Layout(
        padding='14px 18px', 
        gap='10px',
        background_color='#070f1c',
        align_items='center'
    )
)

footer = widgets.HBox(
    [
        clear_btn,
        widgets.HTML(
            '<div class="vf-wrap"><div class="vf-footer-text">'
            'Powered by GPT-4o + Databricks Delta Lake · '
            'All answers cite real facility data · '
            'Virtue Foundation Ghana Healthcare Intelligence © 2024'
            '</div></div>'
        )
    ],
    layout=widgets.Layout(
        padding='7px 18px',
        background_color='#040c18',
        justify_content='space-between',
        align_items='center'
    )
)

# ── Message renderers ─────────────────────────────────────────────
WELCOME_HTML = """
<div class="vf-wrap">
<div class="vf-chat">
<div class="vf-welcome">
  <div class="vf-welcome-icon">🌍</div>
  <div class="vf-welcome-title">Ghana Healthcare Planning Assistant</div>
  <div class="vf-welcome-sub">
    I have complete data on 797 facilities across Ghana's 16 regions.<br>
    Every answer includes findings with data citations,<br>
    actionable recommendations, and a confidence assessment.
  </div>
  <div class="vf-welcome-cards">
    <div class="vf-welcome-card">🔍 Find facilities by capability</div>
    <div class="vf-welcome-card">📊 Analyze regional gaps</div>
    <div class="vf-welcome-card">🎯 Plan resource deployment</div>
    <div class="vf-welcome-card">⚠️ Identify medical deserts</div>
  </div>
</div>
</div>
</div>
"""

def render_user_message(text):
    return f"""
<div class="vf-wrap">
<div class="vf-chat" style="padding-top:8px;padding-bottom:0;min-height:0;">
  <div class="vf-msg-user-label">You</div>
  <div class="vf-msg-user-wrap">
    <div class="vf-msg-user">{text}</div>
    <div class="vf-avatar-user">YOU</div>
  </div>
</div>
</div>"""

def render_thinking():
    return """
<div class="vf-wrap">
<div class="vf-chat" style="padding-top:6px;padding-bottom:8px;min-height:0;">
  <div class="vf-response-wrap">
    <div class="vf-response-header">
      <div class="vf-avatar-bot">🏥</div>
      <div class="vf-response-label">Healthcare Assistant · Analyzing data...</div>
    </div>
    <div class="vf-thinking">
      ⏳ &nbsp; Searching 797 facilities and 16 regions for your answer...
    </div>
  </div>
</div>
</div>"""

def render_bot_response(result):
    answer = result.get("answer", "")
    findings = result.get("findings", [])
    recommendations = result.get("recommendations", [])
    confidence = result.get("confidence", {})

    conf_level = confidence.get("level", "MEDIUM")
    conf_score = float(confidence.get("score", 0.5))
    conf_reason = confidence.get("reason", "")

    conf_color = {
        "HIGH": "#4caf50", "MEDIUM": "#ff9800", "LOW": "#f44336"
    }.get(conf_level, "#ff9800")

    conf_bar_pct = int(conf_score * 100)

    # Start with header
    html = """
<div class="vf-wrap">
<div class="vf-chat" style="padding-top:6px;padding-bottom:16px;min-height:0;">
  <div class="vf-response-wrap">
    <div class="vf-response-header">
      <div class="vf-avatar-bot">🏥</div>
      <div class="vf-response-label">Healthcare Assistant · Ghana Intelligence System</div>
    </div>
"""

    # CARD 1 - Answer
    html += f"""
    <div class="vf-card-answer">
      <div class="vf-card-header">🔵 &nbsp; ANSWER</div>
      <div class="vf-answer-text">{answer}</div>
    </div>
"""

    # CARD 2 - Findings
    if findings:
        findings_items = ""
        for i, f in enumerate(findings, 1):
            point = f.get("point", "")
            citation = f.get("citation", "")
            findings_items += f"""
            <div class="vf-finding-item">
              <div class="vf-finding-point">
                <span class="vf-finding-num">{i}</span>{point}
              </div>
              {f'<div class="vf-citation">📎 {citation}</div>' if citation else ''}
            </div>"""
        
        html += f"""
    <div class="vf-card-findings">
      <div class="vf-card-header">🔎 &nbsp; FINDINGS ({len(findings)} data points)</div>
      {findings_items}
    </div>
"""

    # CARD 3 - Recommendations
    if recommendations:
        recs_items = ""
        for i, rec in enumerate(recommendations, 1):
            recs_items += f"""
            <div class="vf-rec-item">
              <div class="vf-rec-num">{i}</div>
              <div>{rec}</div>
            </div>"""
        
        html += f"""
    <div class="vf-card-recommendations">
      <div class="vf-card-header">✅ &nbsp; RECOMMENDATIONS ({len(recommendations)} actions)</div>
      {recs_items}
    </div>
"""

    # CARD 4 - Confidence
    html += f"""
    <div class="vf-card-confidence">
      <div class="vf-conf-badge"
           style="background:{conf_color}22;border:1px solid {conf_color};color:{conf_color};">
        {conf_level}
      </div>
      <div style="flex:1;">
        <div class="vf-conf-bar-wrap">
          <div class="vf-conf-bar"
               style="width:{conf_bar_pct}%;background:{conf_color};"></div>
        </div>
      </div>
      <div class="vf-conf-score">{conf_score:.0%}</div>
      <div class="vf-conf-reason">{conf_reason}</div>
    </div>
"""

    # Close wrappers
    html += """
  </div>
</div>
</div>
"""
    
    return html

# ── Render helpers ────────────────────────────────────────────────
def show_welcome():
    chat_out.clear_output(wait=True)
    with chat_out:
        display(HTML(WELCOME_HTML))

def render_all_messages():
    chat_out.clear_output(wait=True)
    with chat_out:
        for html in all_messages_html:
            display(HTML(html))

def process_question(question):
    question = question.strip()
    if not question:
        return

    # Add user message
    all_messages_html.append(render_user_message(question))
    render_all_messages()

    # Show thinking
    all_messages_html.append(render_thinking())
    render_all_messages()

    # Get AI response
    history_for_api = []
    for h in conversation_history:
        history_for_api.append({
            "q": h["q"],
            "a": h["a"]
        })

    result = ask_planning_assistant(question, history_for_api)

    # Replace thinking with actual response
    all_messages_html.pop()
    all_messages_html.append(render_bot_response(result))

    # Save to history
    conversation_history.append({"q": question, "a": result})

    render_all_messages()

# ── Event handlers ────────────────────────────────────────────────
def on_send(b):
    q = text_input.value.strip()
    text_input.value = ''
    process_question(q)

def on_submit(widget):
    q = widget.value.strip()
    widget.value = ''
    process_question(q)

def on_clear(b):
    conversation_history.clear()
    all_messages_html.clear()
    show_welcome()

def make_chip_handler(question):
    def handler(b):
        process_question(question)
    return handler

send_btn.on_click(on_send)
text_input.on_submit(on_submit)
clear_btn.on_click(on_clear)
for btn, q in zip(chip_buttons, QUICK_QUESTIONS):
    btn.on_click(make_chip_handler(q))

# ── Assemble full UI ──────────────────────────────────────────────
full_ui = widgets.VBox(
    [header, stats_bar, chip_label, chips_row,
     chat_out, input_row, footer],
    layout=widgets.Layout(
        width='900px',
        border='1px solid #0d2137',
        border_radius='12px',
        overflow='hidden',
        background_color='#050d1a'
    )
)

show_welcome()
display(full_ui)

# COMMAND ----------

# DBTITLE 1,Export data package for Streamlit
# Run this cell to export everything Streamlit needs
import json, pandas as pd

gold_df = spark.table("virtue_foundation.ghana_health.gold_facilities").toPandas()

bool_caps = ['has_emergency','has_surgery','has_icu','has_maternity',
             'has_lab','has_imaging','has_blood_bank','has_dialysis']
for col in bool_caps:
    if col in gold_df.columns:
        gold_df[col] = gold_df[col].astype(str).str.lower().eq('true')

for col in ['facility_richness_score','deployment_priority_score',
            'clean_region_mdi','confidence_score']:
    gold_df[col] = pd.to_numeric(gold_df[col], errors='coerce')

REGIONS = ['Upper East','Upper West','Savannah','North East','Northern',
           'Oti','Volta','Ahafo','Bono','Bono East','Eastern',
           'Western North','Western','Central','Ashanti','Greater Accra']

pkg = {
    "stats": {
        "total": int(len(gold_df)),
        "hospitals": int((gold_df['facilityTypeId']=='hospital').sum()),
        "clinics": int((gold_df['facilityTypeId']=='clinic').sum()),
        "ngos": int((gold_df['organization_type']=='ngo').sum()),
        "emergency": int(gold_df['has_emergency'].sum()),
        "surgery": int(gold_df['has_surgery'].sum()),
        "icu": int(gold_df['has_icu'].sum()),
        "maternity": int(gold_df['has_maternity'].sum()),
        "lab": int(gold_df['has_lab'].sum()),
        "flagged": int(gold_df['needs_human_review'].astype(str).eq('True').sum()),
    },
    "regions": [],
    "top25": [],
    "all_hospitals": [],
    "gaps": {}
}

for region in REGIONS:
    g = gold_df[gold_df['official_region'] == region]
    missing = []
    for cap, label in [('has_emergency','Emergency'),('has_surgery','Surgery'),
                       ('has_icu','ICU'),('has_maternity','Maternity'),
                       ('has_lab','Lab'),('has_blood_bank','Blood Bank')]:
        if len(g) == 0 or not g[cap].any():
            missing.append(label)
    pkg["regions"].append({
        "name": region,
        "total": int(len(g)),
        "hospitals": int((g['facilityTypeId']=='hospital').sum()) if len(g)>0 else 0,
        "mdi": round(float(g['clean_region_mdi'].mean()),3) if len(g)>0 else 0.0,
        "alert": str(g['clean_alert_level'].mode()[0]) if len(g)>0 else 'CRITICAL',
        "emergency": int(g['has_emergency'].sum()) if len(g)>0 else 0,
        "surgery": int(g['has_surgery'].sum()) if len(g)>0 else 0,
        "icu": int(g['has_icu'].sum()) if len(g)>0 else 0,
        "maternity": int(g['has_maternity'].sum()) if len(g)>0 else 0,
        "lab": int(g['has_lab'].sum()) if len(g)>0 else 0,
        "missing": missing
    })
    pkg["gaps"][region] = missing

top25 = gold_df.nlargest(25, 'deployment_priority_score')
for _, r in top25.iterrows():
    try:
        procs = json.loads(str(r.get('final_procedure') or '[]'))
        equip = json.loads(str(r.get('final_equipment') or '[]'))
    except:
        procs, equip = [], []
    pkg["top25"].append({
        "name": str(r['name']),
        "city": str(r.get('address_city','')),
        "region": str(r.get('official_region','')),
        "type": str(r.get('facilityTypeId','')),
        "priority": round(float(r.get('deployment_priority_score') or 0), 3),
        "mdi": round(float(r.get('clean_region_mdi') or 0), 3),
        "alert": str(r.get('clean_alert_level','')),
        "richness": round(float(r.get('facility_richness_score') or 0), 3),
        "emergency": bool(r.get('has_emergency', False)),
        "surgery": bool(r.get('has_surgery', False)),
        "maternity": bool(r.get('has_maternity', False)),
        "icu": bool(r.get('has_icu', False)),
        "lab": bool(r.get('has_lab', False)),
        "procedures": procs[:5],
        "equipment": equip[:5],
        "confidence": round(float(r.get('confidence_score') or 0.5), 2)
    })

hospitals = gold_df[gold_df['facilityTypeId']=='hospital']
for _, r in hospitals.iterrows():
    try:
        procs = json.loads(str(r.get('final_procedure') or '[]'))
        equip = json.loads(str(r.get('final_equipment') or '[]'))
        caps = json.loads(str(r.get('final_capability') or '[]'))
    except:
        procs, equip, caps = [], [], []
    pkg["all_hospitals"].append({
        "name": str(r['name']),
        "city": str(r.get('address_city','')),
        "region": str(r.get('official_region','')),
        "emergency": bool(r.get('has_emergency',False)),
        "surgery": bool(r.get('has_surgery',False)),
        "icu": bool(r.get('has_icu',False)),
        "maternity": bool(r.get('has_maternity',False)),
        "lab": bool(r.get('has_lab',False)),
        "imaging": bool(r.get('has_imaging',False)),
        "blood_bank": bool(r.get('has_blood_bank',False)),
        "procedures": procs[:4],
        "equipment": equip[:4],
        "capabilities": caps[:4],
        "priority": round(float(r.get('deployment_priority_score') or 0), 3),
        "confidence": round(float(r.get('confidence_score') or 0.5), 2)
    })

with open('ghana_data_package.json', 'w') as f:
    json.dump(pkg, f, indent=2)

print(f"✅ Package saved to ghana_data_package.json")
print(f"   Regions: {len(pkg['regions'])}")
print(f"   Top 25: {len(pkg['top25'])}")
print(f"   Hospitals: {len(pkg['all_hospitals'])}")
print(f"   Total stats: {pkg['stats']['total']} facilities")