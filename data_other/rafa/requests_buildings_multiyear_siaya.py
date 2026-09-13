# -*- coding: utf-8 -*-
"""
Exports Open Buildings 2.5D Temporal (building_height + building_presence)
for Siaya County, Kenya - one GeoTIFF per requested year.

Adapted from requests_buildings_multiyear_jakarta.py.

Years exported: 2016 (earliest available in the dataset), 2019, 2023.
  - 2016 is the first year of coverage.
  - 2019 matches the Google Static Maps imagery used in Huang, Hsiang and
    Gonzalez-Navarro, so it is the direct comparison point.
  - 2023 is the last year of coverage and gives a longer run follow up.

Changes vs. the Jakarta version:
  - AOI: Siaya County, Kenya. Three ways to define it, in priority order:
      1. the extent of the beyond-nightlight treatment map csv (tightest,
         matches the GiveDirectly study area exactly),
      2. the FAO GAUL admin boundary, optionally restricted to the three
         subcounties GiveDirectly actually worked in,
      3. a hardcoded bounding box fallback.
  - CRS: EPSG:32636 (UTM zone 36N) instead of EPSG:32748.
  - Explicit year list instead of a stride, since 2016/2019/2023 is not a
    regular interval.
  - Prints AOI area so you know what you are about to export.

Dataset coverage runs 2016 to 2023. Years outside that range are skipped
automatically.
"""

import os

import ee

ee.Initialize(project="edificios-489500")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

YEARS = [2016, 2019, 2023]

SCALE = 4                      # metres per pixel, native resolution
CRS = "EPSG:32636"             # UTM zone 36N, covers western Kenya
DRIVE_FOLDER = "EarthEngine_Siaya"
BANDS = ["building_height", "building_presence"]
# Add "building_fractional_count" to BANDS if you want a count based measure
# rather than presence probability.

# Option 1: derive the AOI from the treatment map you already have locally.
# fig-map.csv is the Figure 2 raster from the beyond-nightlight repo, at
# 0.005 degrees. Using its extent keeps the export to the actual study area
# instead of the whole county. Columns are lon, lat, treatment_intensity,
# building_footprint, tin_roof_area, night_light.
USE_TREATMENT_MAP_EXTENT = True
TREATMENT_MAP_CSV = r"G:\Mi unidad\RESEARCH\beyond-nightlight\fig-map.csv"
PAD_DEGREES = 0.02             # buffer around the map extent, roughly 2 km

# Option 2: admin boundary.
USE_ADMIN_BOUNDARY = True
# Set to True to clip to the three subcounties GiveDirectly worked in
# (Alego Usonga, Ugunja, Ugenya) rather than all of Siaya. Cuts the export
# roughly in half and drops the lakeside subcounties that are out of sample.
STUDY_AREA_ONLY = False
SUBCOUNTIES = ["Alego Usonga", "Ugunja", "Ugenya"]

# Option 3: fallback bounding box for Siaya County.
SIAYA_BBOX = ee.Geometry.Rectangle([33.90, -0.45, 34.65, 0.40])

# ---------------------------------------------------------------------------
# Area of interest
# ---------------------------------------------------------------------------


def aoi_from_treatment_map(path, pad):
    """Bounding box of the GiveDirectly treatment map, padded."""
    if not os.path.exists(path):
        print(f"Treatment map not found at {path}")
        return None

    try:
        import pandas as pd
    except ImportError:
        print("pandas not installed, cannot read the treatment map")
        return None

    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}

    lon_col = next((cols[c] for c in ("lon", "longitude", "x", "long") if c in cols), None)
    lat_col = next((cols[c] for c in ("lat", "latitude", "y") if c in cols), None)

    if lon_col is None or lat_col is None:
        print(f"Could not find coordinate columns. Available: {list(df.columns)}")
        return None

    west, east = df[lon_col].min() - pad, df[lon_col].max() + pad
    south, north = df[lat_col].min() - pad, df[lat_col].max() + pad

    print(f"AOI: treatment map extent [{west:.3f}, {south:.3f}, {east:.3f}, {north:.3f}]")
    return ee.Geometry.Rectangle([west, south, east, north])


def aoi_from_admin(study_area_only):
    """Siaya County, or just the three GiveDirectly subcounties."""
    if study_area_only:
        admin = (
            ee.FeatureCollection("FAO/GAUL/2015/level2")
            .filter(ee.Filter.eq("ADM0_NAME", "Kenya"))
            .filter(ee.Filter.inList("ADM2_NAME", SUBCOUNTIES))
        )
        label = "GiveDirectly subcounties"
    else:
        admin = (
            ee.FeatureCollection("FAO/GAUL/2015/level1")
            .filter(ee.Filter.eq("ADM0_NAME", "Kenya"))
            .filter(ee.Filter.eq("ADM1_NAME", "Siaya"))
        )
        label = "Siaya County admin boundary"

    if admin.size().getInfo() == 0:
        # GAUL naming for Kenya is inconsistent across vintages: some versions
        # use the old provinces at level1 rather than the 47 counties.
        available = (
            ee.FeatureCollection("FAO/GAUL/2015/level1")
            .filter(ee.Filter.eq("ADM0_NAME", "Kenya"))
            .aggregate_array("ADM1_NAME")
            .getInfo()
        )
        print(f"Admin unit not found. ADM1 names available for Kenya: {available}")
        return None

    print(f"AOI: {label}")
    return admin.geometry()


siaya = None

if USE_TREATMENT_MAP_EXTENT:
    siaya = aoi_from_treatment_map(TREATMENT_MAP_CSV, PAD_DEGREES)

if siaya is None and USE_ADMIN_BOUNDARY:
    siaya = aoi_from_admin(STUDY_AREA_ONLY)

if siaya is None:
    siaya = SIAYA_BBOX
    print("AOI: fallback bounding box for Siaya County")

area_km2 = siaya.area(maxError=1).divide(1e6).getInfo()
print(f"AOI area: {area_km2:,.0f} km2")
print(f"Approx pixels per band at {SCALE} m: {area_km2 * 1e6 / SCALE ** 2:,.0f}")

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

dataset = ee.ImageCollection("GOOGLE/Research/open-buildings-temporal/v1")

for year in YEARS:
    start = f"{year}-01-01"
    end = f"{year + 1}-01-01"

    subset = dataset.filterDate(start, end).filterBounds(siaya)

    n_tiles = subset.size().getInfo()
    if n_tiles == 0:
        print(f"Skipping {year} - no tiles over Siaya")
        continue

    image = (
        subset
        .mosaic()
        .select(BANDS)
        .clip(siaya)
        .unmask(-9999)
    )

    task = ee.batch.Export.image.toDrive(
        image=image,
        description=f"siaya_buildings_{year}",
        folder=DRIVE_FOLDER,
        fileNamePrefix=f"siaya_buildings_{year}",
        region=siaya,
        scale=SCALE,
        crs=CRS,
        maxPixels=1e13,
        fileFormat="GeoTIFF",
    )
    task.start()
    print(f"Export started for {year} ({n_tiles} tiles) - siaya_buildings_{year}")

print("\nAll tasks submitted - check https://code.earthengine.google.com/tasks")
