# 🏥 Ghana Healthcare Intelligence System
**Databricks × Accenture Hackathon 2024 | Virtue Foundation Track**

---

## 🎯 Problem Statement

By 2030, the world faces a shortage of over 10 million healthcare workers.
In Ghana, 797 healthcare facilities exist across 16 regions — but distribution
is critically uneven. Patients in Upper East, Savannah, and North East regions
have zero access to surgery, ICU, or emergency care within their region.

The Virtue Foundation needed an intelligent system to:
- Identify WHERE the gaps are
- Understand WHAT capabilities are missing
- Recommend WHERE to deploy limited medical resources

---

## 💡 Our Solution

An end-to-end **Intelligent Document Parsing (IDP) Agent** built on Databricks
that extracts, enriches, and analyzes Ghana's healthcare facility data using
AI to surface actionable medical desert intelligence.

---

## 🏗️ Architecture
Raw CSV (987 rows)
↓
[Bronze Delta Table] — Raw ingestion, null cleaning, typo fixes
↓
[Silver Delta Table] — Deduplication → 797 unique facilities + geocoding
↓
[IDP Agent] — GPT-4o extracts procedure/equipment/capability from unstructured text
↓
[Silver Enriched] — 260+ facilities now have structured clinical facts
↓
[Capability Flags] — 13 boolean flags per facility (has_emergency, has_surgery...)
↓
[Gold Delta Table] — MDI scores, deployment priority, distance calculations
↓
[Dashboard + Map + Planning Assistant]

---

## 📊 Key Results

| Metric | Value |
|--------|-------|
| Total facilities analyzed | 797 |
| Facilities with procedures (before IDP) | 183 |
| Facilities with procedures (after IDP) | 260 |
| Medical desert regions identified | 6+ critical |
| Anomalies detected by AI | 55 |
| Average AI confidence score | 0.72 |
| Regions with NO surgery capability | 6 |
| Regions with NO ICU | 10+ |

---

## 🔴 Critical Findings

- **Upper East, Savannah, North East** — no surgery, no ICU, no blood bank
- **North East Region** — MDI score of 0.15 (lowest in Ghana)
- **55 facilities** flagged with suspicious capability claims
- **Greater Accra** has 42 facilities clustered in 27km radius while rural areas remain unserved

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Data Platform | Databricks (Delta Lake, Notebooks, SQL Dashboards) |
| AI Extraction | OpenAI GPT-4o |
| Architecture | Medallion (Bronze → Silver → Gold) |
| Geospatial | Geopy, Folium, Haversine |
| Orchestration | PySpark + Pandas |
| Visualization | Databricks SQL Dashboards, Folium Maps |
| Planning UI | ipywidgets (notebook-native chat) |

---

## 🗂️ Notebooks Guide

| # | Notebook | What It Does |
|---|----------|-------------|
| 01 | data_profiling | Profiles raw CSV — finds gaps, typos, duplicates |
| 02 | create_bronze_table | Ingests raw CSV into Delta with cleaning |
| 03 | create_silver_table | Deduplicates 987→797, geocodes facilities |
| 04 | phase2_setup | Tests OpenAI API connection |
| 05 | extraction_functions | Defines IDP extraction functions |
| 06 | batch_processing | Runs IDP on all 797 facilities |
| 07 | save_enriched_silver | Merges IDP results back to Silver |
| 08 | capability_flags | Adds 13 capability boolean flags |
| 09 | medical_desert_analysis | Computes MDI per region |
| 10 | gold_table | Builds final Gold intelligence table |
| 11 | region_fix | Standardizes to 16 official Ghana regions |
| 12 | interactive_map | Builds Folium medical desert map |
| 13 | dashboard_queries | SQL queries powering the dashboard |
| 14-15 | planning_assistant | Natural language planning chat UI |
| 16 | final_test | Full system validation test suite |

---

## 🚀 How to Run

1. Upload `data/Virtue_Foundation_Ghana_v0_3.csv` to Databricks FileStore
2. Create a Databricks cluster (Runtime 14.3 LTS)
3. Run notebooks in order: 01 → 02 → 03 → 06 → 07 → 08 → 09 → 10 → 11
4. Run notebook 16 to validate everything
5. Run notebook 15 for the planning assistant
6. Set `OPENAI_API_KEY` environment variable in each notebook

---

## 📹 Demo

[Link to your 5-minute demo video]

---

## 🔗 Live Demo

Databricks Workspace: [your workspace URL]
Dashboard: [your dashboard URL]

---

## 👥 Team

- [Your Name] — [Your College/Institution]

---

## 📄 License

MIT License — see LICENSE file
