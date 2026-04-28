# Databricks notebook source
# DBTITLE 1,Verify all data is ready
# Verify all data is ready
display(spark.sql("""
SELECT 'gold_facilities' as table_name, COUNT(*) as rows 
FROM virtue_foundation.ghana_health.gold_facilities
UNION ALL
SELECT 'region_desert_analysis', COUNT(*) 
FROM virtue_foundation.ghana_health.region_desert_analysis
"""))

# COMMAND ----------

# DBTITLE 1,Medical Desert Overview by Region
# Query 1: Medical Desert Overview by Region
# This becomes your main bar chart in the dashboard
# Visualization: Bar chart | X = region, Y = mdi_score, Color = alert_level
# Color mapping: CRITICAL = #d32f2f (red), HIGH = #f57c00 (orange), MEDIUM = #fbc02d (yellow), LOW = #2e7d32 (green)
display(spark.sql("""
SELECT 
    official_region as region,
    ROUND(AVG(CAST(NULLIF(clean_region_mdi,'') AS DOUBLE)), 3) as mdi_score,
    MAX(clean_alert_level) as alert_level,
    COUNT(*) as total_facilities,
    SUM(CASE WHEN facilityTypeId = 'hospital' THEN 1 ELSE 0 END) as hospitals,
    SUM(CASE WHEN has_emergency = 'True' THEN 1 ELSE 0 END) as with_emergency,
    SUM(CASE WHEN has_surgery = 'True' THEN 1 ELSE 0 END) as with_surgery
FROM virtue_foundation.ghana_health.gold_facilities
WHERE official_region NOT IN ('Unknown','nan','','None')
GROUP BY official_region
ORDER BY mdi_score ASC
"""))

# COMMAND ----------

# DBTITLE 1,Top 20 Priority Facilities
# Query 2: Top 20 Priority Facilities for Intervention
# This is the actionable table — the list the Virtue Foundation actually uses
# Visualization: Table (NOT a chart)
display(spark.sql("""
SELECT 
    ROW_NUMBER() OVER (
        ORDER BY CAST(NULLIF(deployment_priority_score,'') AS DOUBLE) DESC
    ) as rank,
    name as facility_name,
    facilityTypeId as type,
    official_region as region,
    address_city as city,
    clean_alert_level as alert_level,
    ROUND(CAST(NULLIF(deployment_priority_score,'') AS DOUBLE), 2) as priority_score,
    ROUND(CAST(NULLIF(clean_region_mdi,'') AS DOUBLE), 2) as region_mdi,
    CASE WHEN has_emergency = 'True' THEN 'YES' ELSE '—' END as emergency,
    CASE WHEN has_surgery = 'True' THEN 'YES' ELSE '—' END as surgery,
    CASE WHEN has_maternity = 'True' THEN 'YES' ELSE '—' END as maternity,
    CASE WHEN has_icu = 'True' THEN 'YES' ELSE '—' END as icu,
    CASE WHEN has_lab = 'True' THEN 'YES' ELSE '—' END as lab,
    ROUND(CAST(NULLIF(facility_richness_score,'') AS DOUBLE), 2) as richness,
    CASE WHEN needs_human_review = 'True' 
         THEN 'REVIEW' ELSE 'CLEAN' END as data_quality
FROM virtue_foundation.ghana_health.gold_facilities
WHERE official_region NOT IN ('Unknown','nan','','None')
ORDER BY CAST(NULLIF(deployment_priority_score,'') AS DOUBLE) DESC
LIMIT 20
"""))

# COMMAND ----------

# DBTITLE 1,Specialty Coverage Gap Analysis
# Query 3: Specialty Coverage Gap Analysis (Capability Heatmap)
# Shows which specialties are missing across Ghana
# Visualization: Heatmap or Table
display(spark.sql("""
SELECT
    official_region as region,
    COUNT(*) as total_facilities,
    SUM(CASE WHEN has_emergency = 'True' THEN 1 ELSE 0 END) as emergency,
    SUM(CASE WHEN has_surgery = 'True' THEN 1 ELSE 0 END) as surgery,
    SUM(CASE WHEN has_maternity = 'True' THEN 1 ELSE 0 END) as maternity,
    SUM(CASE WHEN has_icu = 'True' THEN 1 ELSE 0 END) as icu,
    SUM(CASE WHEN has_lab = 'True' THEN 1 ELSE 0 END) as laboratory,
    SUM(CASE WHEN has_imaging = 'True' THEN 1 ELSE 0 END) as imaging,
    SUM(CASE WHEN has_blood_bank = 'True' THEN 1 ELSE 0 END) as blood_bank,
    SUM(CASE WHEN has_dialysis = 'True' THEN 1 ELSE 0 END) as dialysis,
    SUM(CASE WHEN has_dental = 'True' THEN 1 ELSE 0 END) as dental,
    SUM(CASE WHEN has_eye_care = 'True' THEN 1 ELSE 0 END) as eye_care
FROM virtue_foundation.ghana_health.gold_facilities
WHERE official_region NOT IN ('Unknown','nan','','None')
GROUP BY official_region
ORDER BY emergency DESC
"""))

# COMMAND ----------

# DBTITLE 1,Summary KPIs
# Query 4: Summary KPI numbers for dashboard headline
display(spark.sql("""
SELECT
    COUNT(*) as total_facilities,
    
    SUM(CASE WHEN clean_alert_level = 'CRITICAL' 
        THEN 1 ELSE 0 END) as critical_alert_facilities,
    
    SUM(CASE WHEN clean_alert_level IN ('CRITICAL','HIGH') 
        AND official_region != 'Unknown' 
        THEN 1 ELSE 0 END) as facilities_in_medical_deserts,
    
    ROUND(
        AVG(CAST(NULLIF(NULLIF(deployment_priority_score,''),'nan') AS DOUBLE))
    , 2) as avg_priority_score,
    
    SUM(CASE WHEN has_emergency = 'True' THEN 1 ELSE 0 END) as emergency_services,
    SUM(CASE WHEN has_surgery = 'True' THEN 1 ELSE 0 END) as surgical_services,
    SUM(CASE WHEN has_icu = 'True' THEN 1 ELSE 0 END) as icu_available,
    SUM(CASE WHEN has_maternity = 'True' THEN 1 ELSE 0 END) as maternity_services,
    SUM(CASE WHEN needs_human_review = 'True' THEN 1 ELSE 0 END) as anomaly_flagged,
    COUNT(DISTINCT official_region) - 1 as regions_analyzed
FROM virtue_foundation.ghana_health.gold_facilities
"""))

# COMMAND ----------

# DBTITLE 1,Facility Type Distribution
# Query 5: Facility Type Distribution
# Shows the breakdown of facility types across Ghana
# Visualization: Pie chart or Bar chart
display(spark.sql("""
SELECT
    CASE 
        WHEN organization_type = 'ngo' THEN 'NGO'
        WHEN facilityTypeId = 'hospital' THEN 'Hospital'
        WHEN facilityTypeId = 'clinic' THEN 'Clinic'
        WHEN facilityTypeId = 'pharmacy' THEN 'Pharmacy'
        WHEN facilityTypeId = 'dentist' THEN 'Dentist'
        WHEN facilityTypeId = 'health_center' THEN 'Health Center'
        WHEN facilityTypeId = 'maternity_home' THEN 'Maternity Home'
        WHEN facilityTypeId = 'doctor' THEN 'Doctor Practice'
        ELSE 'Unclassified'
    END as facility_type,
    COUNT(*) as count,
    ROUND(COUNT(*) * 100.0 / 797, 1) as percentage
FROM virtue_foundation.ghana_health.gold_facilities
GROUP BY 1
ORDER BY count DESC
"""))

# COMMAND ----------

# DBTITLE 1,Medical Desert Regions Summary
# Query 6: Medical Desert Regions Summary
# Focused view on CRITICAL and HIGH alert regions only
# This is for a dedicated "Medical Desert Focus" section in the dashboard
# Visualization: Table
display(spark.sql("""
SELECT
    official_region as region,
    MAX(clean_alert_level) as alert_level,
    COUNT(*) as facilities,
    ROUND(AVG(CAST(NULLIF(clean_region_mdi,'') AS DOUBLE)), 2) as mdi_score,
    SUM(CASE WHEN has_emergency = 'True' THEN 1 ELSE 0 END) as emergency_count,
    SUM(CASE WHEN has_surgery = 'True' THEN 1 ELSE 0 END) as surgery_count,
    SUM(CASE WHEN has_icu = 'True' THEN 1 ELSE 0 END) as icu_count,
    ROUND(AVG(CAST(NULLIF(deployment_priority_score,'') AS DOUBLE)), 2) as avg_priority
FROM virtue_foundation.ghana_health.gold_facilities
WHERE clean_alert_level IN ('CRITICAL','HIGH')
AND official_region NOT IN ('Unknown','nan','','None')
GROUP BY official_region
ORDER BY mdi_score ASC
"""))