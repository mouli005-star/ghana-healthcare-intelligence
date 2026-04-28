# Databricks notebook source
# DBTITLE 1,Load flagged data with type conversions
import pandas as pd
import json
import numpy as np

flagged_df = spark.table("virtue_foundation.ghana_health.silver_flagged").toPandas()

# Convert numeric columns back from string
numeric_cols = ['latitude', 'longitude', 'facility_richness_score', 'confidence_score']
for col in numeric_cols:
    flagged_df[col] = pd.to_numeric(flagged_df[col], errors='coerce')

bool_caps = [
    'has_emergency', 'has_icu', 'has_surgery', 'has_maternity',
    'has_pediatrics', 'has_lab', 'has_imaging', 'has_blood_bank',
    'has_pharmacy', 'has_dialysis', 'has_dental', 'has_eye_care', 'has_mental_health'
]
for col in bool_caps:
    flagged_df[col] = flagged_df[col].astype(str).str.lower().eq('true')

print(f"Loaded {len(flagged_df)} facilities")
print(f"With coordinates: {flagged_df['latitude'].notna().sum()}")

# COMMAND ----------

# DBTITLE 1,Region-level aggregation
# Ghana's 16 administrative regions (as of 2019)
# We group facilities by their stated region and compute scores

def compute_region_stats(region_facilities):
    """
    Given all facilities in a region, compute the medical desert metrics.
    """
    total = len(region_facilities)
    if total == 0:
        return None
    
    geocoded = region_facilities['latitude'].notna().sum()
    
    stats = {
        'total_facilities': total,
        'geocoded_facilities': int(geocoded),
        'hospitals': int((region_facilities['facilityTypeId'] == 'hospital').sum()),
        'clinics': int((region_facilities['facilityTypeId'] == 'clinic').sum()),
        'ngos': int((region_facilities['organization_type'] == 'ngo').sum()),
        
        # Capability presence in region
        'has_any_emergency': bool(region_facilities['has_emergency'].any()),
        'has_any_icu': bool(region_facilities['has_icu'].any()),
        'has_any_surgery': bool(region_facilities['has_surgery'].any()),
        'has_any_maternity': bool(region_facilities['has_maternity'].any()),
        'has_any_blood_bank': bool(region_facilities['has_blood_bank'].any()),
        'has_any_imaging': bool(region_facilities['has_imaging'].any()),
        'has_any_lab': bool(region_facilities['has_lab'].any()),
        'has_any_dialysis': bool(region_facilities['has_dialysis'].any()),
        
        # Count of facilities with each capability
        'count_with_emergency': int(region_facilities['has_emergency'].sum()),
        'count_with_surgery': int(region_facilities['has_surgery'].sum()),
        'count_with_maternity': int(region_facilities['has_maternity'].sum()),
        'count_with_lab': int(region_facilities['has_lab'].sum()),
        'count_with_imaging': int(region_facilities['has_imaging'].sum()),
        
        # Average richness
        'avg_facility_richness': round(region_facilities['facility_richness_score'].mean(), 3),
        'max_facility_richness': round(region_facilities['facility_richness_score'].max(), 3),
        'avg_confidence': round(region_facilities['confidence_score'].mean(), 2),
    }
    
    return stats


# Group by region
region_groups = flagged_df.groupby('address_stateOrRegion')

region_stats_list = []
for region_name, group in region_groups:
    if not region_name or region_name in ('', 'null', 'None'):
        region_name = 'Unknown Region'
    
    stats = compute_region_stats(group)
    if stats:
        stats['region'] = region_name
        region_stats_list.append(stats)

region_df = pd.DataFrame(region_stats_list)
print(f"Analyzed {len(region_df)} regions")
print(region_df[['region', 'total_facilities', 'hospitals', 'has_any_surgery', 
                  'avg_facility_richness']].to_string())

# COMMAND ----------

# DBTITLE 1,Compute Medical Desert Index (MDI) per region
def compute_mdi(row):
    """
    Medical Desert Index: 0.0 = complete desert, 1.0 = fully covered.
    
    Built from 6 components weighted by importance for Ghana context.
    """
    score = 0.0
    
    # Component 1: Basic facility presence (20 points)
    # Even 1 facility is better than nothing
    if row['total_facilities'] > 0:
        facility_score = min(row['total_facilities'] / 20.0, 1.0)  # Caps at 20 facilities
        score += 0.20 * facility_score
    
    # Component 2: Hospital presence (20 points)
    # Hospitals signal higher-level care than clinics
    if row['hospitals'] > 0:
        hospital_score = min(row['hospitals'] / 5.0, 1.0)  # Caps at 5 hospitals
        score += 0.20 * hospital_score
    
    # Component 3: Critical capability coverage (30 points)
    # This is the most important — do they have emergency/surgery/maternity?
    critical_caps = [
        'has_any_emergency', 'has_any_surgery', 
        'has_any_maternity', 'has_any_blood_bank'
    ]
    critical_coverage = sum(1 for cap in critical_caps if row.get(cap, False))
    score += 0.30 * (critical_coverage / len(critical_caps))
    
    # Component 4: Diagnostic capability (15 points)
    # Lab + imaging are essential for proper diagnosis
    diagnostic_caps = ['has_any_lab', 'has_any_imaging']
    diagnostic_coverage = sum(1 for cap in diagnostic_caps if row.get(cap, False))
    score += 0.15 * (diagnostic_coverage / len(diagnostic_caps))
    
    # Component 5: Average quality of facilities present (15 points)
    score += 0.15 * row.get('avg_facility_richness', 0)
    
    return round(min(score, 1.0), 3)


def get_alert_level(mdi_score):
    if mdi_score < 0.15:
        return 'CRITICAL'
    elif mdi_score < 0.30:
        return 'HIGH'
    elif mdi_score < 0.50:
        return 'MEDIUM'
    else:
        return 'LOW'


def get_missing_capabilities(row):
    """List which critical capabilities are completely absent in this region"""
    missing = []
    capability_map = {
        'has_any_emergency': 'Emergency Care',
        'has_any_surgery': 'Surgery',
        'has_any_maternity': 'Maternity Services',
        'has_any_blood_bank': 'Blood Bank',
        'has_any_icu': 'ICU',
        'has_any_imaging': 'Medical Imaging',
        'has_any_lab': 'Laboratory',
        'has_any_dialysis': 'Dialysis/Kidney Care'
    }
    for col, label in capability_map.items():
        if not row.get(col, False):
            missing.append(label)
    return ', '.join(missing) if missing else 'None'


region_df['mdi_score'] = region_df.apply(compute_mdi, axis=1)
region_df['alert_level'] = region_df['mdi_score'].apply(get_alert_level)
region_df['is_medical_desert'] = region_df['mdi_score'] < 0.30
region_df['missing_capabilities'] = region_df.apply(get_missing_capabilities, axis=1)

print("\nMedical Desert Analysis by Region:")
print("=" * 80)
display_cols = ['region', 'total_facilities', 'hospitals', 'mdi_score', 
                'alert_level', 'is_medical_desert', 'missing_capabilities']
print(region_df[display_cols].sort_values('mdi_score').to_string())

# COMMAND ----------

# DBTITLE 1,Identify the specific medical deserts
deserts = region_df[region_df['is_medical_desert']].sort_values('mdi_score')
well_served = region_df[~region_df['is_medical_desert']].sort_values('mdi_score', ascending=False)

print(f"MEDICAL DESERT REGIONS ({len(deserts)}):")
print("=" * 60)
for _, row in deserts.iterrows():
    print(f"\n⚠️  {row['region']}")
    print(f"   Alert Level:       {row['alert_level']}")
    print(f"   MDI Score:         {row['mdi_score']} / 1.0")
    print(f"   Total facilities:  {row['total_facilities']}")
    print(f"   Hospitals:         {row['hospitals']}")
    print(f"   Missing:           {row['missing_capabilities']}")

print(f"\n\nWELL-SERVED REGIONS ({len(well_served)}):")
print("=" * 60)
for _, row in well_served.head(5).iterrows():
    print(f"✅ {row['region']} — MDI: {row['mdi_score']} | Facilities: {row['total_facilities']}")

# COMMAND ----------

# DBTITLE 1,Compute geographic spread (urban clustering detection)
from math import radians, cos, sin, asin, sqrt

def haversine_km(lat1, lon1, lat2, lon2):
    """Calculate distance in km between two GPS points"""
    R = 6371  # Earth radius
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return 2 * R * asin(sqrt(a))


def compute_geographic_spread(group):
    """
    Returns the max distance between any two facilities in a region.
    High spread = facilities cover a wide area (good)
    Low spread = all clustered in one city (might leave rural areas uncovered)
    """
    coords = group[['latitude', 'longitude']].dropna()
    if len(coords) < 2:
        return 0.0
    
    max_dist = 0.0
    coords_list = coords.values.tolist()
    # Sample max 20 points to avoid O(n²) slowness for large regions
    sample = coords_list[:20]
    for i in range(len(sample)):
        for j in range(i+1, len(sample)):
            dist = haversine_km(sample[i][0], sample[i][1], sample[j][0], sample[j][1])
            max_dist = max(max_dist, dist)
    
    return round(max_dist, 1)


print("Computing geographic spread per region...")
spread_data = {}
for region_name, group in flagged_df.groupby('address_stateOrRegion'):
    spread_data[region_name] = compute_geographic_spread(group)

region_df['geographic_spread_km'] = region_df['region'].map(spread_data)

# Flag regions where spread is low but many facilities exist
# These likely have urban clustering — rural parts of these regions may be deserts
region_df['likely_urban_clustered'] = (
    (region_df['total_facilities'] > 10) & 
    (region_df['geographic_spread_km'] < 30)
)

print("\nRegions with potential urban clustering (facilities may not cover rural areas):")
clustered = region_df[region_df['likely_urban_clustered']]
if len(clustered) > 0:
    print(clustered[['region', 'total_facilities', 'geographic_spread_km']].to_string())
else:
    print("None detected")

# COMMAND ----------

# DBTITLE 1,Save region desert analysis
import json

# Save as a Delta table for use in dashboard
region_df_str = region_df.astype(str)
spark_regions = spark.createDataFrame(region_df_str)

(spark_regions.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("virtue_foundation.ghana_health.region_desert_analysis"))

print(f"✅ Region desert analysis saved: {len(region_df)} regions")

# Also save summary stats
critical_regions = region_df[region_df['alert_level'] == 'CRITICAL']
high_regions = region_df[region_df['alert_level'] == 'HIGH']

print(f"\nDESERT SUMMARY:")
print(f"CRITICAL alert regions: {len(critical_regions)}")
print(f"HIGH alert regions:     {len(high_regions)}")
print(f"Total desert regions:   {len(region_df[region_df['is_medical_desert']])}")
print(f"Lowest MDI region:      {region_df.loc[region_df['mdi_score'].idxmin(), 'region']}")
print(f"Highest MDI region:     {region_df.loc[region_df['mdi_score'].idxmax(), 'region']}")