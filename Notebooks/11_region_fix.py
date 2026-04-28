# Databricks notebook source
# DBTITLE 1,Load gold and understand the problem
import pandas as pd
import json

gold_df = spark.table("virtue_foundation.ghana_health.gold_facilities").toPandas()

print(f"Total facilities: {len(gold_df)}")
print(f"Facilities with no region: {(gold_df['address_stateOrRegion'].isin(['', 'nan', 'None', 'null'])).sum()}")
print(f"Unique region values currently: {gold_df['address_stateOrRegion'].nunique()}")
print(f"\nAll current region values:")
print(gold_df['address_stateOrRegion'].value_counts().to_string())

# COMMAND ----------

# DBTITLE 1,Build the city-to-region mapping
# Comprehensive city-to-region mapping for Ghana
# Covers all major cities and towns in the dataset
CITY_TO_REGION = {
    # Greater Accra
    'accra': 'Greater Accra', 'tema': 'Greater Accra', 'ashaiman': 'Greater Accra',
    'madina': 'Greater Accra', 'achimota': 'Greater Accra', 'adenta': 'Greater Accra',
    'east legon': 'Greater Accra', 'airport': 'Greater Accra', 'osu': 'Greater Accra',
    'labadi': 'Greater Accra', 'dansoman': 'Greater Accra', 'kaneshie': 'Greater Accra',
    'spintex': 'Greater Accra', 'lapaz': 'Greater Accra', 'kasoa': 'Greater Accra',
    'adabraka': 'Greater Accra', 'north labone': 'Greater Accra', 'ridge': 'Greater Accra',
    'dzorwulu': 'Greater Accra', 'teshie': 'Greater Accra', 'nungua': 'Greater Accra',
    'lashibi': 'Greater Accra', 'community': 'Greater Accra', 'kotobabi': 'Greater Accra',
    'kanda': 'Greater Accra', 'okponglo': 'Greater Accra', 'haatso': 'Greater Accra',
    'ashongman': 'Greater Accra', 'abokobi': 'Greater Accra', 'dome': 'Greater Accra',
    'pokuase': 'Greater Accra', 'anyaa': 'Greater Accra', 'ablekuma': 'Greater Accra',
    'chorkor': 'Greater Accra', 'weija': 'Greater Accra', 'botianor': 'Greater Accra',
    'dodowa': 'Greater Accra', 'prampram': 'Greater Accra', 'kpone': 'Greater Accra',
    'afienya': 'Greater Accra', 'shai osudoku': 'Greater Accra',
    
    # Ashanti
    'kumasi': 'Ashanti', 'obuasi': 'Ashanti', 'bekwai': 'Ashanti', 'konongo': 'Ashanti',
    'mampong': 'Ashanti', 'agogo': 'Ashanti', 'ashanti mampong': 'Ashanti',
    'ejisu': 'Ashanti', 'juaben': 'Ashanti', 'effiduase': 'Ashanti',
    'asokwa': 'Ashanti', 'suame': 'Ashanti', 'adum': 'Ashanti', 'nhyiaeso': 'Ashanti',
    'bantama': 'Ashanti', 'kwadaso': 'Ashanti', 'tafo': 'Ashanti', 'asawase': 'Ashanti',
    'manhyia': 'Ashanti', 'krofrom': 'Ashanti', 'abuakwa': 'Ashanti',
    
    # Western
    'takoradi': 'Western', 'sekondi': 'Western', 'tarkwa': 'Western', 'prestea': 'Western',
    'bogoso': 'Western', 'agona nkwanta': 'Western', 'dixcove': 'Western',
    'axim': 'Western', 'half assini': 'Western', 'elubo': 'Western',
    'shama': 'Western', 'mpohor': 'Western', 'ahanta west': 'Western',
    
    # Western North
    'sefwi wiawso': 'Western North', 'bibiani': 'Western North', 'enchi': 'Western North',
    'juaboso': 'Western North', 'bodi': 'Western North',
    
    # Central
    'cape coast': 'Central', 'winneba': 'Central', 'kasoa central': 'Central',
    'saltpond': 'Central', 'mankessim': 'Central', 'elmina': 'Central',
    'agona swedru': 'Central', 'assin fosu': 'Central', 'dunkwa': 'Central',
    'twifo praso': 'Central', 'assin bereku': 'Central', 'anomabo': 'Central',
    'abura dunkwa': 'Central',
    
    # Eastern
    'koforidua': 'Eastern', 'nkawkaw': 'Eastern', 'suhum': 'Eastern',
    'nsawam': 'Eastern', 'asamankese': 'Eastern', 'oda': 'Eastern',
    'akwatia': 'Eastern', 'abetifi': 'Eastern', 'somanya': 'Eastern',
    'mpraeso': 'Eastern', 'kwahu': 'Eastern', 'kade': 'Eastern',
    'nkurakan': 'Eastern', 'aburi': 'Eastern', 'manfe': 'Eastern',
    
    # Volta
    'ho': 'Volta', 'hohoe': 'Volta', 'keta': 'Volta', 'aflao': 'Volta',
    'anloga': 'Volta', 'kpando': 'Volta', 'dambai': 'Volta', 'jasikan': 'Volta',
    'nkwanta': 'Volta', 'worawora': 'Volta', 'kadjebi': 'Volta',
    'biakoye': 'Volta', 'krachi': 'Volta',
    
    # Oti
    'dambai oti': 'Oti', 'nkwanta north': 'Oti', 'nkwanta south': 'Oti',
    
    # Bono
    'sunyani': 'Bono', 'berekum': 'Bono', 'dormaa ahenkro': 'Bono',
    'dormaa': 'Bono', 'jaman north': 'Bono', 'jaman south': 'Bono',
    
    # Bono East
    'techiman': 'Bono East', 'kintampo': 'Bono East', 'nkoranza': 'Bono East',
    'wenchi': 'Bono East', 'atebubu': 'Bono East', 'yeji': 'Bono East',
    
    # Ahafo
    'goaso': 'Ahafo', 'kukuom': 'Ahafo', 'bechem': 'Ahafo', 'hwidiem': 'Ahafo',
    
    # Northern
    'tamale': 'Northern', 'yendi': 'Northern', 'savelugu': 'Northern',
    'gushegu': 'Northern', 'karaga': 'Northern', 'tolon': 'Northern',
    'kumbungu': 'Northern', 'sagnarigu': 'Northern', 'nanton': 'Northern',
    
    # North East
    'nalerigu': 'North East', 'gambaga': 'North East', 'walewale': 'North East',
    'bunkpurugu': 'North East', 'chereponi': 'North East',
    
    # Savannah
    'damongo': 'Savannah', 'bole': 'Savannah', 'sawla': 'Savannah',
    'tuna': 'Savannah', 'east gonja': 'Savannah', 'west gonja': 'Savannah',
    
    # Upper East
    'bolgatanga': 'Upper East', 'navrongo': 'Upper East', 'bawku': 'Upper East',
    'zebilla': 'Upper East', 'sandema': 'Upper East', 'paga': 'Upper East',
    'chiana': 'Upper East', 'tongo': 'Upper East', 'binduri': 'Upper East',
    'garu': 'Upper East', 'tempane': 'Upper East', 'pusiga': 'Upper East',
    
    # Upper West
    'wa': 'Upper West', 'lawra': 'Upper West', 'nandom': 'Upper West',
    'jirapa': 'Upper West', 'tumu': 'Upper West', 'gwollu': 'Upper West',
    'sissala': 'Upper West', 'lambussie': 'Upper West', 'nadowli': 'Upper West',
}

# Also map the messy existing region values to the 16 official regions
REGION_NORMALIZER = {
    # Ashanti variations
    'ashanti': 'Ashanti', 'ashanti region': 'Ashanti', 'ASHANTI': 'Ashanti',
    'ASHANTI REGION': 'Ashanti', 'asokwa-kumasi': 'Ashanti', 'ejisu municipal': 'Ashanti',
    'ahafo ano south-east': 'Ashanti',
    
    # Greater Accra variations
    'greater accra': 'Greater Accra', 'greater accra region': 'Greater Accra',
    'accra': 'Greater Accra', 'accra east': 'Greater Accra', 'accra north': 'Greater Accra',
    'east legon': 'Greater Accra', 'ledzokuku-krowor': 'Greater Accra',
    'ga east municipality, greater accra region': 'Greater Accra',
    
    # Western variations
    'western': 'Western', 'western region': 'Western', 'takoradi': 'Western',
    
    # Western North
    'western north': 'Western North', 'western north region': 'Western North',
    
    # Central variations
    'central': 'Central', 'central region': 'Central', 'central ghana': 'Central',
    'central tongu district': 'Volta',  # Central Tongu is actually in Volta
    
    # Eastern variations  
    'eastern': 'Eastern', 'eastern region': 'Eastern',
    
    # Volta variations
    'volta': 'Volta', 'volta region': 'Volta',
    
    # Oti
    'oti': 'Oti', 'oti region': 'Oti',
    
    # Bono variations
    'bono': 'Bono', 'brong ahafo': 'Bono', 'brong ahafo region': 'Bono',
    'dormaa east': 'Bono',
    
    # Bono East
    'bono east region': 'Bono East', 'techiman municipal': 'Bono East',
    
    # Ahafo
    'ahafo': 'Ahafo', 'ahafo region': 'Ahafo',
    
    # Northern variations
    'northern': 'Northern', 'northern region': 'Northern',
    
    # Savannah
    'savannah': 'Savannah',
    
    # Upper East
    'upper east': 'Upper East', 'upper east region': 'Upper East',
    
    # Upper West
    'upper west': 'Upper West', 'upper west region': 'Upper West',
    'sissala west district': 'Upper West',
    
    # KEEA is a district in Central Region
    'keea': 'Central',
    
    # Tema West is Greater Accra
    'tema west municipal': 'Greater Accra',
    
    # SH - likely Accra area (Kasoa/SH is near Accra)
    'sh': 'Greater Accra',
    
    # Generic Ghana-wide
    'ghana': 'Greater Accra',  # Default to Greater Accra if just "Ghana"
}

print("Mapping tables ready ✅")
print(f"City mappings: {len(CITY_TO_REGION)} cities")
print(f"Region normalizations: {len(REGION_NORMALIZER)} mappings")

# COMMAND ----------

# DBTITLE 1,Apply region assignment logic
def assign_official_region(row):
    """
    Determine the correct official Ghana region for a facility.
    Priority order:
    1. Normalize existing region value if it exists
    2. Look up city in our city-to-region map
    3. Keep as 'Unknown' if nothing matches
    """
    existing_region = str(row.get('address_stateOrRegion', '') or '').strip().lower()
    city = str(row.get('address_city', '') or '').strip().lower()
    
    # Step 1: Try to normalize existing region name
    if existing_region and existing_region not in ('', 'nan', 'none', 'null'):
        normalized = REGION_NORMALIZER.get(existing_region)
        if normalized:
            return normalized
        # If it starts with a known region word, use that
        for key, val in REGION_NORMALIZER.items():
            if existing_region.startswith(key[:5]) and len(key) > 4:
                return val
    
    # Step 2: Try to assign from city name
    if city and city not in ('', 'nan', 'none', 'null'):
        # Direct city lookup
        region = CITY_TO_REGION.get(city)
        if region:
            return region
        # Partial match — city string contains a known city
        for known_city, region in CITY_TO_REGION.items():
            if known_city in city or city in known_city:
                return region
    
    # Step 3: Try address_line1 for city clues
    line1 = str(row.get('address_line1', '') or '').strip().lower()
    for known_city, region in CITY_TO_REGION.items():
        if known_city in line1:
            return region
    
    return 'Unknown'


# Load gold table fresh
gold_df = spark.table("virtue_foundation.ghana_health.gold_facilities").toPandas()

# Apply the assignment
gold_df['official_region'] = gold_df.apply(assign_official_region, axis=1)

# See the results
region_assignment_summary = gold_df['official_region'].value_counts()
unknown_count = (gold_df['official_region'] == 'Unknown').sum()
assigned_count = len(gold_df) - unknown_count

print(f"Total facilities:     {len(gold_df)}")
print(f"Successfully assigned: {assigned_count} ({assigned_count/len(gold_df)*100:.1f}%)")
print(f"Still unknown:         {unknown_count} ({unknown_count/len(gold_df)*100:.1f}%)")
print(f"\nFacilities per official region:")
print(region_assignment_summary.to_string())

# COMMAND ----------

# DBTITLE 1,Re-run desert analysis on clean 16 regions
from math import radians, cos, sin, asin, sqrt

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    a = sin((lat2-lat1)/2)**2 + cos(lat1)*cos(lat2)*sin((lon2-lon1)/2)**2
    return 2 * R * asin(sqrt(a))

# Convert bool columns
bool_caps = ['has_emergency', 'has_icu', 'has_surgery', 'has_maternity',
             'has_pediatrics', 'has_lab', 'has_imaging', 'has_blood_bank',
             'has_pharmacy', 'has_dialysis', 'has_dental', 'has_eye_care']
for col in bool_caps:
    if col in gold_df.columns:
        gold_df[col] = gold_df[col].astype(str).str.lower().eq('true')

gold_df['facility_richness_score'] = pd.to_numeric(gold_df['facility_richness_score'], errors='coerce').fillna(0)
gold_df['latitude'] = pd.to_numeric(gold_df['latitude'], errors='coerce')
gold_df['longitude'] = pd.to_numeric(gold_df['longitude'], errors='coerce')

# Now compute MDI per official region
GHANA_16_REGIONS = [
    'Greater Accra', 'Ashanti', 'Western', 'Western North', 'Central',
    'Eastern', 'Volta', 'Oti', 'Bono', 'Bono East', 'Ahafo',
    'Northern', 'North East', 'Savannah', 'Upper East', 'Upper West'
]

def compute_clean_mdi(region_facilities):
    total = len(region_facilities)
    if total == 0:
        return 0.0
    
    hospitals = (region_facilities['facilityTypeId'] == 'hospital').sum()
    
    # Component 1: Facility count (20%)
    c1 = min(total / 20.0, 1.0) * 0.20
    
    # Component 2: Hospital presence (20%)
    c2 = min(hospitals / 5.0, 1.0) * 0.20
    
    # Component 3: Critical capabilities (30%)
    critical = ['has_emergency', 'has_surgery', 'has_maternity', 'has_blood_bank']
    crit_coverage = sum(1 for c in critical if region_facilities[c].any())
    c3 = (crit_coverage / 4) * 0.30
    
    # Component 4: Diagnostics (15%)
    diag = ['has_lab', 'has_imaging']
    diag_coverage = sum(1 for c in diag if region_facilities[c].any())
    c4 = (diag_coverage / 2) * 0.15
    
    # Component 5: Average richness (15%)
    c5 = region_facilities['facility_richness_score'].mean() * 0.15
    
    return round(min(c1 + c2 + c3 + c4 + c5, 1.0), 3)

clean_region_stats = []
for region in GHANA_16_REGIONS:
    group = gold_df[gold_df['official_region'] == region]
    if len(group) == 0:
        mdi = 0.0
        total = 0
        hospitals = 0
    else:
        mdi = compute_clean_mdi(group)
        total = len(group)
        hospitals = int((group['facilityTypeId'] == 'hospital').sum())
    
    clean_region_stats.append({
        'official_region': region,
        'total_facilities': total,
        'hospitals': total and hospitals,
        'mdi_score': mdi,
        'alert_level': 'CRITICAL' if mdi < 0.15 else 'HIGH' if mdi < 0.30 else 'MEDIUM' if mdi < 0.50 else 'LOW',
        'is_desert': mdi < 0.30,
        'has_emergency': len(group) > 0 and bool(group['has_emergency'].any()),
        'has_surgery': len(group) > 0 and bool(group['has_surgery'].any()),
        'has_icu': len(group) > 0 and bool(group['has_icu'].any()),
        'has_maternity': len(group) > 0 and bool(group['has_maternity'].any()),
        'has_lab': len(group) > 0 and bool(group['has_lab'].any()),
        'has_imaging': len(group) > 0 and bool(group['has_imaging'].any()),
    })

clean_region_df = pd.DataFrame(clean_region_stats).sort_values('mdi_score')

print("CLEAN MEDICAL DESERT ANALYSIS — 16 OFFICIAL GHANA REGIONS")
print("=" * 70)
print(f"{'Region':<20} {'Facilities':>10} {'Hospitals':>9} {'MDI':>6} {'Alert':>10}")
print("-" * 70)
for _, r in clean_region_df.iterrows():
    icon = "🔴" if r['alert_level'] in ('CRITICAL','HIGH') else "🟡" if r['alert_level'] == 'MEDIUM' else "🟢"
    print(f"{icon} {r['official_region']:<18} {r['total_facilities']:>10} {r['hospitals']:>9} {r['mdi_score']:>6} {r['alert_level']:>10}")

# COMMAND ----------

# DBTITLE 1,Recompute deployment priority with clean regions, save final Gold
# Join clean region MDI back onto each facility
gold_df = gold_df.merge(
    clean_region_df[['official_region', 'mdi_score', 'alert_level', 'is_desert']].rename(columns={
        'mdi_score': 'clean_region_mdi',
        'alert_level': 'clean_alert_level',
        'is_desert': 'clean_is_desert'
    }),
    on='official_region',
    how='left'
)

# Recompute priority score using clean MDI
def recompute_priority(row):
    isolation = float(row.get('isolation_score') or 0.5)
    richness = float(row.get('facility_richness_score') or 0)
    region_mdi = float(row.get('clean_region_mdi') or 0.5)
    return round(min((isolation * 0.35) + ((1 - richness) * 0.35) + ((1 - region_mdi) * 0.30), 1.0), 3)

gold_df['deployment_priority_score'] = gold_df.apply(recompute_priority, axis=1)

# Save the final clean gold table — overwrite previous version
spark_gold_clean = spark.createDataFrame(gold_df.astype(str))
(spark_gold_clean.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("virtue_foundation.ghana_health.gold_facilities"))

# Save the clean region table too
spark_regions_clean = spark.createDataFrame(clean_region_df.astype(str))
(spark_regions_clean.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("virtue_foundation.ghana_health.region_desert_analysis"))

print(f"✅ Final Gold table saved: {len(gold_df)} facilities")
print(f"✅ Clean region analysis saved: {len(clean_region_df)} regions")
print(f"\nDesert regions: {clean_region_df['is_desert'].sum()} / 16")
print(f"Facilities now assigned to a region: {(gold_df['official_region'] != 'Unknown').sum()}")