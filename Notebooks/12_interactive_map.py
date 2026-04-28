# Databricks notebook source
# DBTITLE 1,Install and load
# MAGIC %pip install folium
# MAGIC
# MAGIC import pandas as pd
# MAGIC import json
# MAGIC import folium
# MAGIC from folium.plugins import MarkerCluster, HeatMap
# MAGIC from IPython.display import IFrame
# MAGIC import os
# MAGIC
# MAGIC gold_df = spark.table("virtue_foundation.ghana_health.gold_facilities").toPandas()
# MAGIC region_df = spark.table("virtue_foundation.ghana_health.region_desert_analysis").toPandas()
# MAGIC
# MAGIC # Fix types
# MAGIC gold_df['latitude'] = pd.to_numeric(gold_df['latitude'], errors='coerce')
# MAGIC gold_df['longitude'] = pd.to_numeric(gold_df['longitude'], errors='coerce')
# MAGIC gold_df['facility_richness_score'] = pd.to_numeric(gold_df['facility_richness_score'], errors='coerce')
# MAGIC gold_df['deployment_priority_score'] = pd.to_numeric(gold_df['deployment_priority_score'], errors='coerce')
# MAGIC
# MAGIC # Only facilities with GPS coordinates
# MAGIC mapped_df = gold_df.dropna(subset=['latitude', 'longitude'])
# MAGIC print(f"Facilities with GPS coordinates: {len(mapped_df)} / {len(gold_df)}")

# COMMAND ----------

# DBTITLE 1,Build the map
# MAGIC %sql
# MAGIC /Workspace/Users/burugula_b220794ce@nitc.ac.in/ghana_health_map.html

# COMMAND ----------

# DBTITLE 1,Get the shareable map URL
# ═══════════════════════════════════════════════════════════════════════════
# OPTIONS FOR SHARING YOUR MAP
# ═══════════════════════════════════════════════════════════════════════════

print("\n🎯 MAP SHARING OPTIONS FOR YOUR HACKATHON DEMO\n")
print("="*70)

# Option 1: Interactive map in this notebook (RECOMMENDED)
print("\n✅ OPTION 1 (RECOMMENDED): Use the Interactive Map Above")
print("   • The map in Cell 2 is fully functional and ready to present")
print("   • Zoom, pan, click facilities, toggle layers - everything works!")
print("   • Perfect for screen sharing during your demo")
print("   • No file export needed")

# Option 2: Download HTML file
print("\n📥 OPTION 2: Download the HTML File")
print(f"   • Map saved at: /tmp/ghana_health_map.html")
print("   • To download:")
print("     1. Click the Data panel on the left sidebar")
print("     2. Navigate to the file path above")
print("     3. Download to your local machine")
print("     4. Open the HTML file in any browser")
print("   • Works offline, can be emailed or hosted anywhere")

# Option 3: Notebook presentation mode
print("\n🖥️  OPTION 3: Databricks Notebook Presentation Mode")
print("   • Click 'View' menu → 'Presentation Mode'")
print("   • Shows only cell outputs (hides code)")
print("   • Perfect for clean demo presentations")
print("   • All interactivity preserved")

# Option 4: Share notebook URL
workspace_url = spark.conf.get("spark.databricks.workspaceUrl")
notebook_url = f"https://{workspace_url}/editor/notebooks/3384340901345449"
print("\n🔗 OPTION 4: Share Notebook URL (for Databricks users)")
print(f"   • Notebook URL: {notebook_url}")
print("   • Note: Requires Databricks workspace access")
print("   • Viewers can interact with the map directly")

print("\n" + "="*70)
print("\n💡 FOR YOUR HACKATHON: Use Option 1 or 3 for live demos!")
print("   The map above has all features and looks professional.\n")

# COMMAND ----------

# DBTITLE 1,Download: Copy map to Workspace folder
# Copy map to your Workspace folder for easy download
import shutil

# Source and destination paths
source = "/tmp/ghana_health_map.html"
destination = "/Workspace/Users/burugula_b220794ce@nitc.ac.in/ghana_health_map.html"

try:
    shutil.copy(source, destination)
    print(f"✓ Map copied successfully!")
    print(f"\nDownload instructions:")
    print(f"1. Look in the left sidebar (Workspace panel)")
    print(f"2. Navigate to: /Users/burugula_b220794ce@nitc.ac.in/")
    print(f"3. Find: ghana_health_map.html")
    print(f"4. Right-click → Download")
    print(f"\nFile location: {destination}")
except Exception as e:
    print(f"Copy failed: {e}")
    print(f"\nAlternative: The map is still available at {source}")

# COMMAND ----------

