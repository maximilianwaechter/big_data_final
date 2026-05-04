# ================================================
# Philadelphia Police Complaint + Crime Analysis
# Big Data Final Project
# ================================================
# Purpose:
# 1. Download Philadelphia police district, census tract, complaint,
#    and complaint demographic data
# 2. Allocate tract populations to police districts using areal allocation
# 3. Calculate district-level complaint rates and disparity indicators
# 4. Pull crime incidents, classify as violent/nonviolent, and allocate
#    counts to districts and PSAs
# 5. Export GeoJSON + CSV for an interactive web dashboard (district + PSA)
# 6. Create supporting static figures for the report
# 7. Build building centroids with nearby 2024 crime counts from
#    Philadelphia building footprints
# ================================================

import os
import re
import json
import math
import zipfile
import urllib.request
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# Optional deps for the heat raster step. Imported lazily so the script
# still runs if these are not installed (the heat step will be skipped).
try:
    import rasterio  # noqa: F401
    from rasterio.warp import calculate_default_transform, reproject, Resampling  # noqa: F401
    _HAS_RASTERIO = True
except Exception:
    _HAS_RASTERIO = False

# Planetary Computer STAC client for fetching real Landsat surface
# temperature scenes. If unavailable, the heat step falls back to the
# direct USGS URL attempt and then to the proxy surface.
try:
    from pystac_client import Client as _PCClient  # noqa: F401
    import planetary_computer as _PC  # noqa: F401
    _HAS_PC = True
except Exception:
    _HAS_PC = False


# ------------------------------------------------
# USER SETTINGS
# ------------------------------------------------
WORKING_DIR = r"C:\Users\tuj97893\Desktop\geospatial_final\final"

DATA_URLS = {
    "police_stations": "https://services.arcgis.com/fLeGjb7u4uXqeF9q/arcgis/rest/services/PPD_Districts_HQ/FeatureServer/0/query?outFields=*&where=1%3D1&f=geojson",
    "police_districts": "https://hub.arcgis.com/api/v3/datasets/62ec63afb8824a15953399b1fa819df2_0/downloads/data?format=shp&spatialRefId=3857&where=1%3D1",
    "census_tracts": "https://hub.arcgis.com/api/v3/datasets/20332a074f0446b3b3190ba9d68b863e_0/downloads/data?format=shp&spatialRefId=3857&where=1%3D1",
    "police_psa": "https://services.arcgis.com/fLeGjb7u4uXqeF9q/arcgis/rest/services/Boundaries_PSA/FeatureServer/0/query?outFields=*&where=1%3D1&f=geojson",
    "building_footprints": "https://services.arcgis.com/fLeGjb7u4uXqeF9q/arcgis/rest/services/LI_BUILDING_FOOTPRINTS/FeatureServer/0/query?outFields=*&where=1%3D1&returnGeometry=true&f=geojson",
    "complaints": "https://phl.carto.com/api/v2/sql?q=SELECT+*+FROM+ppd_complaints&filename=ppd_complaints&format=csv&skipfields=cartodb_id,the_geom,the_geom_webmercator",
    "complaint_demographics": "https://phl.carto.com/api/v2/sql?q=SELECT+*+FROM+ppd_complainant_demographics&filename=ppd_complainant_demographics&format=csv&skipfields=cartodb_id,the_geom,the_geom_webmercator",
    "crime_incidents": "https://phl.carto.com/api/v2/sql?q=SELECT+*+FROM+incidents_part1_part2&filename=incidents_part1_part2&format=csv&skipfields=cartodb_id,the_geom,the_geom_webmercator"
}

OUTPUT_DIR = "output"
WEB_DIR = "web"


# ------------------------------------------------
# SETUP
# ------------------------------------------------
def setup_project():
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.chdir(WORKING_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(WEB_DIR, exist_ok=True)
    print(f"Working directory set to: {os.getcwd()}")


# ------------------------------------------------
# HELPER: Download one file
# ------------------------------------------------
def download_data(urls_dict):
    """
    Downloads shapefiles, CSV files, and GeoJSON files.
    Returns a dictionary of local file paths or URLs.
    """
    print("\nDOWNLOADING DATA")
    downloaded_files = {}

    for name, url in urls_dict.items():
        print(f"\nDownloading {name}...")

        try:
            if name in ("police_psa", "police_stations"):
                downloaded_files[name] = url
                print(f"  Using {name} service/GeoJSON URL directly: {url}")
                continue

            if "format=shp" in url:
                zip_path = f"{name}.zip"

                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req) as response, open(zip_path, "wb") as out_file:
                    out_file.write(response.read())

                extract_folder = name
                os.makedirs(extract_folder, exist_ok=True)

                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(extract_folder)

                shp_path = None
                for root, dirs, files in os.walk(extract_folder):
                    for file in files:
                        if file.endswith(".shp"):
                            shp_path = os.path.join(root, file)
                            break
                    if shp_path:
                        break

                if shp_path is None:
                    raise FileNotFoundError(f"No .shp file found in extracted folder for {name}")

                downloaded_files[name] = shp_path
                print(f"  Saved shapefile: {shp_path}")

            elif "f=geojson" in url:
                geojson_path = f"{name}.geojson"
                urllib.request.urlretrieve(url, geojson_path)
                downloaded_files[name] = geojson_path
                print(f"  Saved GeoJSON: {geojson_path}")

            elif "format=csv" in url or url.endswith(".csv"):
                csv_path = f"{name}.csv"
                urllib.request.urlretrieve(url, csv_path)
                downloaded_files[name] = csv_path
                print(f"  Saved CSV: {csv_path}")

        except Exception as e:
            print(f"ERROR downloading {name}: {e}")
            return None

    print(f"\nDownload complete: {len(downloaded_files)} datasets ready.")
    return downloaded_files


# ------------------------------------------------
# HELPER: standardize district ids
# ------------------------------------------------
def standardize_district_code(value):
    if pd.isna(value):
        return np.nan

    s = str(value).strip()

    if s == "" or s.lower() in ["na", "n/a", "none", "null", "<na>"]:
        return np.nan

    digits = re.sub(r"\D", "", s)

    if digits == "":
        return np.nan

    if len(digits) > 2:
        if digits.endswith("00"):
            digits = str(int(digits) // 100)
        else:
            digits = digits[-2:]

    return str(int(digits)).zfill(2)


# ------------------------------------------------
# HELPER: choose column by candidate list
# ------------------------------------------------
def find_column(columns, candidates):
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]

    for col in columns:
        col_l = col.lower()
        for cand in candidates:
            if cand.lower() in col_l:
                return col
    return None


# ------------------------------------------------
# FUNCTION 1: Spatial analysis - areal allocation to districts
# ------------------------------------------------
def spatial_analysis(districts_shp, tracts_shp):
    """
    Allocates census tract populations to police districts using area-weighted allocation.
    """
    print("\nSTEP 1: SPATIAL ANALYSIS - AREAL ALLOCATION")

    districts = gpd.read_file(districts_shp)
    tracts = gpd.read_file(tracts_shp)

    tracts = tracts.to_crs(districts.crs)

    if "dist_numc" not in districts.columns:
        alt_dist = find_column(districts.columns, ["dist_num", "district", "district_1", "dist"])
        if alt_dist is None:
            raise ValueError("Could not find district code field in district shapefile.")
        districts = districts.rename(columns={alt_dist: "dist_numc"})

    districts["dist_numc"] = districts["dist_numc"].astype(str).str.strip().str.zfill(2)

    pop_columns = [col for col in tracts.columns if col.lower().startswith("count_")]

    pop_mapping = {}
    for col in pop_columns:
        col_lower = col.lower()
        if "all" in col_lower or "race" in col_lower:
            pop_mapping[col] = "Total"
        elif "black" in col_lower or "blac" in col_lower:
            pop_mapping[col] = "Black"
        elif "white" in col_lower or "whit" in col_lower:
            pop_mapping[col] = "White"
        elif "hisp" in col_lower:
            pop_mapping[col] = "Hispanic"
        elif "asia" in col_lower:
            pop_mapping[col] = "Asian"
        elif "mult" in col_lower:
            pop_mapping[col] = "Multiracial"

    if len(pop_mapping) == 0:
        raise ValueError("No usable tract population columns were found.")

    print(f"Population variables found: {pop_mapping}")

    tracts["tract_area"] = tracts.geometry.area

    fragments = gpd.overlay(tracts, districts, how="intersection")
    fragments["frag_area"] = fragments.geometry.area

    for pop_col in pop_mapping.keys():
        if pop_col in fragments.columns:
            fragments[f"{pop_col}_alloc"] = (
                fragments[pop_col] * (fragments["frag_area"] / fragments["tract_area"])
            )

    alloc_cols = [col for col in fragments.columns if col.endswith("_alloc")]
    pop_by_district = fragments.groupby("dist_numc")[alloc_cols].sum().reset_index()

    districts = districts.merge(pop_by_district, on="dist_numc", how="left")

    print(f"Allocated population to {len(districts)} police districts.")
    return districts, pop_mapping


# ------------------------------------------------
# FUNCTION 2: Load PSA geography
# ------------------------------------------------
def load_psa_geography(psa_source, districts_crs):
    """
    Loads police service areas (PSAs) and standardizes key fields.
    Accepts either a local file path or a GeoJSON/REST URL.
    """
    print("\nSTEP 2: LOAD PSA GEOGRAPHY")

    psa = gpd.read_file(psa_source)
    psa = psa.to_crs(districts_crs)

    psa_id_col = find_column(psa.columns, ["psa_num", "psa", "service_area", "area"])
    psa_dist_col = find_column(psa.columns, ["dist_num", "district", "dist"])

    if psa_id_col is None:
        raise ValueError("Could not identify PSA ID field in PSA data.")

    psa = psa.rename(columns={psa_id_col: "psa_id"})

    if psa_dist_col is not None and psa_dist_col != "psa_id":
        psa = psa.rename(columns={psa_dist_col: "dist_numc"})
    else:
        psa["dist_numc"] = np.nan

    psa["psa_id"] = psa["psa_id"].astype(str).str.strip()

    if "dist_numc" in psa.columns:
        psa["dist_numc"] = psa["dist_numc"].apply(standardize_district_code)

    print(f"Loaded {len(psa)} PSA polygons.")
    return psa


# ------------------------------------------------
# FUNCTION 3: Standardize complainant race
# ------------------------------------------------
def standardize_race(value):
    if pd.isna(value):
        return "Unknown"

    v = str(value).strip()
    if v == "" or v.lower() in ["unknown", "unk", "na", "n/a", "none"]:
        return "Unknown"

    v_lower = v.lower()

    if v_lower in ["black", "african american", "african-american"]:
        return "Black"
    elif v_lower in ["white", "caucasian"]:
        return "White"
    elif v_lower in ["asian", "asian/pacific islander", "asian pacific islander"]:
        return "Asian"
    elif v_lower in ["latino", "hispanic", "latino / hispanic", "hispanic/latino"]:
        return "Hispanic"
    else:
        return "Other"


# ------------------------------------------------
# FUNCTION 4: Complaint rates by district (legacy logic restored)
# ------------------------------------------------
def calculate_complaint_rates(districts, complaints_csv, demographics_csv, pop_mapping):
    """
    Calculates:
    - overall complaint rates
    - race-specific rates for Black, White, Asian, Hispanic
    - counts for Other and Unknown
    - complaint type counts and rates
    - disparity metrics using White rate as baseline
    """
    print("\nSTEP 3: CALCULATE DISTRICT COMPLAINT RATES")

    complaints = pd.read_csv(complaints_csv)
    complaint_demog = pd.read_csv(demographics_csv)

    complaints_full = complaints.merge(complaint_demog, on="complaint_id", how="left")

    complaints_full["complainant_race"] = (
        complaints_full.get("complainant_race", pd.Series(index=complaints_full.index))
        .fillna("Unknown")
        .astype(str)
        .str.strip()
    )
    complaints_full["complainant_race_clean"] = complaints_full["complainant_race"].apply(standardize_race)

    complaints_full["district_occurrence"] = pd.to_numeric(
        complaints_full["district_occurrence"], errors="coerce"
    )
    complaints_full["district_occurrence"] = (
        (complaints_full["district_occurrence"] / 100)
        .astype("Int64")
        .astype(str)
        .str.zfill(2)
    )

    districts["dist_numc"] = districts["dist_numc"].astype(str).str.strip().str.zfill(2)

    excluded = ["77", "00", "99", "na", "", "<NA>"]
    complaints_full = complaints_full[
        ~complaints_full["district_occurrence"].isin(excluded)
    ].copy()

    if "77" in districts["dist_numc"].values:
        districts = districts[districts["dist_numc"] != "77"].copy()

    valid_districts = set(districts["dist_numc"].unique())
    complaints_full = complaints_full[
        complaints_full["district_occurrence"].isin(valid_districts)
    ].copy()

    print(f"Processed valid complaints: {len(complaints_full):,}")

    total_pop_candidates = [col for col, label in pop_mapping.items() if label == "Total"]
    if len(total_pop_candidates) == 0:
        raise ValueError("Could not identify total population column from census tract data.")
    total_pop_col = total_pop_candidates[0] + "_alloc"

    rate_groups = {}
    for col, label in pop_mapping.items():
        if label in ["Black", "White", "Asian", "Hispanic"]:
            rate_groups[label] = f"{col}_alloc"

    print(f"Rate groups available: {rate_groups}")

    all_complaint_types = []
    if "general_cap_classification" in complaints_full.columns:
        all_complaint_types = sorted(
            complaints_full["general_cap_classification"].dropna().astype(str).unique()
        )

    results = []

    for dist_id in districts["dist_numc"].unique():
        row = {"district_id": dist_id}

        dist_complaints = complaints_full[
            complaints_full["district_occurrence"] == dist_id
        ].copy()

        row["total_complaints"] = len(dist_complaints)

        dist_data = districts[districts["dist_numc"] == dist_id]
        total_pop = dist_data[total_pop_col].values[0] if len(dist_data) > 0 else 0

        row["total_population"] = round(float(total_pop), 0)
        row["overall_rate_per_1000"] = round(
            (len(dist_complaints) / total_pop * 1000) if total_pop > 0 else np.nan, 2
        )

        for race_label, pop_col in rate_groups.items():
            race_complaints = dist_complaints[
                dist_complaints["complainant_race_clean"] == race_label
            ]
            count = len(race_complaints)
            pop = dist_data[pop_col].values[0] if len(dist_data) > 0 and pop_col in dist_data.columns else 0
            rate = (count / pop * 1000) if pop > 0 else np.nan

            row[f"{race_label}_complaints"] = count
            row[f"{race_label}_population"] = round(float(pop), 0) if pd.notnull(pop) else np.nan
            row[f"{race_label}_rate_per_1000"] = round(rate, 2) if pd.notnull(rate) else np.nan

        for extra_group in ["Other", "Unknown"]:
            extra_count = len(
                dist_complaints[dist_complaints["complainant_race_clean"] == extra_group]
            )
            row[f"{extra_group}_complaints"] = extra_count

        for ctype in all_complaint_types:
            type_count = len(
                dist_complaints[
                    dist_complaints["general_cap_classification"] == ctype
                ]
            )
            safe_name = (
                str(ctype)
                .replace(" ", "_")
                .replace("/", "_")
                .replace("-", "_")
                .replace(",", "")
                .replace("(", "")
                .replace(")", "")
            )

            row[f"{safe_name}_count"] = type_count
            row[f"{safe_name}_rate_per_1000"] = round(
                (type_count / total_pop * 1000) if total_pop > 0 else np.nan, 2
            ) if total_pop > 0 else np.nan

        results.append(row)

    summary = pd.DataFrame(results)

    if "White_rate_per_1000" in summary.columns:
        summary["black_white_ratio"] = np.where(
            summary["White_rate_per_1000"] > 0,
            summary.get("Black_rate_per_1000", np.nan) / summary["White_rate_per_1000"],
            np.nan
        )
        summary["hispanic_white_ratio"] = np.where(
            summary["White_rate_per_1000"] > 0,
            summary.get("Hispanic_rate_per_1000", np.nan) / summary["White_rate_per_1000"],
            np.nan
        )
        summary["asian_white_ratio"] = np.where(
            summary["White_rate_per_1000"] > 0,
            summary.get("Asian_rate_per_1000", np.nan) / summary["White_rate_per_1000"],
            np.nan
        )

        summary["black_white_flag"] = np.where(
            (summary.get("Black_rate_per_1000", np.nan).notna()) & (summary["White_rate_per_1000"] > 0),
            summary.get("Black_rate_per_1000", 0) > (2 * summary["White_rate_per_1000"]),
            False
        )
        summary["hispanic_white_flag"] = np.where(
            (summary.get("Hispanic_rate_per_1000", np.nan).notna()) & (summary["White_rate_per_1000"] > 0),
            summary.get("Hispanic_rate_per_1000", 0) > (2 * summary["White_rate_per_1000"]),
            False
        )
        summary["asian_white_flag"] = np.where(
            (summary.get("Asian_rate_per_1000", np.nan).notna()) & (summary["White_rate_per_1000"] > 0),
            summary.get("Asian_rate_per_1000", 0) > (2 * summary["White_rate_per_1000"]),
            False
        )

        summary["disparity_flag"] = summary["black_white_flag"]
    else:
        summary["black_white_ratio"] = np.nan
        summary["hispanic_white_ratio"] = np.nan
        summary["asian_white_ratio"] = np.nan
        summary["black_white_flag"] = False
        summary["hispanic_white_flag"] = False
        summary["asian_white_flag"] = False
        summary["disparity_flag"] = False

    summary["high_complaint_flag"] = (
        summary["overall_rate_per_1000"] > summary["overall_rate_per_1000"].median()
    )

    districts_map = districts.merge(
        summary, left_on="dist_numc", right_on="district_id", how="left"
    )

    return districts_map, summary, complaints_full, rate_groups


# ------------------------------------------------
# FUNCTION 5: Build crime points geodataframe
# ------------------------------------------------
def build_crime_geodataframe(crime_csv):
    """
    Builds a crime GeoDataFrame from incident CSV using lon/lat or xy fields.
    Filters to 2024 incidents only.
    """
    print("\nSTEP 4: LOAD CRIME INCIDENTS")

    crime = pd.read_csv(crime_csv)
    cols = list(crime.columns)

    date_col = find_column(cols, ["dispatch_date", "dispatch_date_time", "occur_date", "date"])

    if date_col is not None:
        crime[date_col] = pd.to_datetime(crime[date_col], errors="coerce")
        before = len(crime)
        crime = crime[crime[date_col].dt.year == 2024].copy()
        print(f"Filtered to 2024 incidents: {len(crime):,} of {before:,} rows")

    lon_col = find_column(cols, ["lng", "lon", "longitude", "point_x"])
    lat_col = find_column(cols, ["lat", "latitude", "point_y"])

    if lon_col is None or lat_col is None:
        raise ValueError("Could not find longitude/latitude fields in crime incident CSV.")

    crime[lon_col] = pd.to_numeric(crime[lon_col], errors="coerce")
    crime[lat_col] = pd.to_numeric(crime[lat_col], errors="coerce")

    crime = crime[
        crime[lon_col].notna() & crime[lat_col].notna()
    ].copy()

    crime_gdf = gpd.GeoDataFrame(
        crime,
        geometry=gpd.points_from_xy(crime[lon_col], crime[lat_col]),
        crs="EPSG:4326"
    )

    print(f"Crime incidents with valid coordinates (2024 only if date field found): {len(crime_gdf):,}")
    return crime_gdf


# ------------------------------------------------
# FUNCTION 6: Classify violent vs nonviolent crime
# ------------------------------------------------
def classify_crime_category(row, offense_col):
    """
    Very simple offense classification into violent/nonviolent.
    """
    if offense_col is None or pd.isna(row[offense_col]):
        return "Unknown"

    txt = str(row[offense_col]).lower()

    violent_terms = [
        "homicide", "murder", "rape", "robbery", "aggravated assault",
        "assault", "shooting", "weapon", "kidnapping", "manslaughter"
    ]

    for term in violent_terms:
        if term in txt:
            return "Violent"

    return "Nonviolent"


# ------------------------------------------------
# FUNCTION 7: Spatially assign crime to districts and PSAs
# ------------------------------------------------
def process_crime_data(crime_gdf, districts_gdf, psa_gdf):
    """
    Spatially joins crime incidents to district and PSA polygons,
    then aggregates violent/nonviolent totals.
    """
    print("\nSTEP 5: SPATIALLY ASSIGN CRIME TO DISTRICTS AND PSAS")

    crime_gdf = crime_gdf.to_crs(districts_gdf.crs)

    offense_col = find_column(crime_gdf.columns, [
        "text_general_code", "ucr_general", "offense", "offense_description",
        "dispatch_date_time", "crime_type"
    ])

    if offense_col is not None:
        crime_gdf["crime_category_simple"] = crime_gdf.apply(
            classify_crime_category, axis=1, offense_col=offense_col
        )
    else:
        crime_gdf["crime_category_simple"] = "Unknown"

    district_join = gpd.sjoin(
        crime_gdf,
        districts_gdf[["dist_numc", "geometry"]],
        how="left",
        predicate="within"
    )

    psa_keep = ["psa_id", "geometry"]
    if "dist_numc" in psa_gdf.columns:
        psa_keep.insert(1, "dist_numc")

    psa_join = gpd.sjoin(
        crime_gdf,
        psa_gdf[psa_keep],
        how="left",
        predicate="within"
    )

    district_summary = (
        district_join.groupby("dist_numc")
        .agg(
            total_crime=("crime_category_simple", "size"),
            violent_crime=("crime_category_simple", lambda x: (x == "Violent").sum()),
            nonviolent_crime=("crime_category_simple", lambda x: (x == "Nonviolent").sum()),
            unknown_crime=("crime_category_simple", lambda x: (x == "Unknown").sum())
        )
        .reset_index()
    )

    psa_summary = (
        psa_join.groupby("psa_id")
        .agg(
            total_crime=("crime_category_simple", "size"),
            violent_crime=("crime_category_simple", lambda x: (x == "Violent").sum()),
            nonviolent_crime=("crime_category_simple", lambda x: (x == "Nonviolent").sum()),
            unknown_crime=("crime_category_simple", lambda x: (x == "Unknown").sum())
        )
        .reset_index()
    )

    district_summary["violent_share"] = np.where(
        district_summary["total_crime"] > 0,
        district_summary["violent_crime"] / district_summary["total_crime"],
        np.nan
    )

    psa_summary["violent_share_from_crime"] = np.where(
        psa_summary["total_crime"] > 0,
        psa_summary["violent_crime"] / psa_summary["total_crime"],
        np.nan
    )

    districts_crime = districts_gdf.merge(district_summary, on="dist_numc", how="left")
    psa_crime = psa_gdf.merge(psa_summary, on="psa_id", how="left")

    for col in ["total_crime", "violent_crime", "nonviolent_crime", "unknown_crime"]:
        if col in districts_crime.columns:
            districts_crime[col] = districts_crime[col].fillna(0)
        if col in psa_crime.columns:
            psa_crime[col] = psa_crime[col].fillna(0)

    print("Crime counts assigned to district and PSA geographies.")
    return districts_crime, psa_crime, district_join, psa_join, crime_gdf


# ------------------------------------------------
# FUNCTION 7B: Export building footprints for web
# ------------------------------------------------
def export_building_footprints_for_web(buildings_geojson_path):
    """
    Copies/exports building footprints to web/building_footprints.geojson
    so they can be used later in the HTML/Cesium workflow.
    """
    print("\nSTEP 5B: EXPORT BUILDING FOOTPRINTS FOR WEB")

    buildings = gpd.read_file(buildings_geojson_path).to_crs(epsg=4326)
    out_path = os.path.join(WEB_DIR, "building_footprints.geojson")
    buildings.to_file(out_path, driver="GeoJSON")

    print(f"Saved building footprints GeoJSON: {out_path}")
    return out_path


# ------------------------------------------------
# FUNCTION 7C: Build building centroids with nearby 2024 crime counts
# ------------------------------------------------
def build_building_centroids_with_crime(crime_gdf, buildings_geojson_path, radius_m=100):
    """
    Creates building centroid points with 2024 crime counts within radius_m meters.
    Saves result to web/building_centroids_2024_crime.geojson.
    """
    print("\nSTEP 5C: BUILD BUILDING CENTROIDS WITH 2024 CRIME COUNTS")

    buildings = gpd.read_file(buildings_geojson_path)

    id_col = find_column(buildings.columns, ["objectid", "bin", "id", "building_id"])
    if id_col is None:
        buildings["building_id"] = buildings.index.astype(str)
    else:
        buildings["building_id"] = buildings[id_col].astype(str)

    target_crs = "EPSG:26918"

    buildings = buildings.to_crs(target_crs)
    crime_local = crime_gdf.to_crs(target_crs)

    centroids = buildings.copy()
    centroids["geometry"] = centroids.geometry.centroid

    centroid_buffers = centroids[["building_id", "geometry"]].copy()
    centroid_buffers["geometry"] = centroid_buffers.geometry.buffer(radius_m)

    joined = gpd.sjoin(
        crime_local,
        centroid_buffers,
        how="inner",
        predicate="within"
    )

    crime_counts = (
        joined.groupby("building_id")
        .size()
        .reset_index(name="crime_2024_nearby")
    )

    centroids = centroids.merge(crime_counts, on="building_id", how="left")
    centroids["crime_2024_nearby"] = centroids["crime_2024_nearby"].fillna(0).astype(int)

    centroids_wgs84 = centroids.to_crs("EPSG:4326")
    centroids_wgs84["lon"] = centroids_wgs84.geometry.x
    centroids_wgs84["lat"] = centroids_wgs84.geometry.y

    keep_cols = ["building_id", "crime_2024_nearby", "lon", "lat", "geometry"]
    centroids_wgs84 = centroids_wgs84[keep_cols]

    out_path = os.path.join(WEB_DIR, "building_centroids_2024_crime.geojson")
    centroids_wgs84.to_file(out_path, driver="GeoJSON")

    print(f"Saved building centroid GeoJSON: {out_path}")
    return out_path


# ------------------------------------------------
# FUNCTION 8: Combine district outputs
# ------------------------------------------------
def combine_district_outputs(districts_complaints_gdf, districts_crime_gdf):
    """
    Combines district complaint and crime metrics into one GeoDataFrame
    and adds district-level crime rate columns that mirror PSA columns:
    - violent_share_from_crime
    - crime_rate_per_1000
    - violent_crime_rate_per_1000
    - nonviolent_crime_rate_per_1000
    """
    print("\nSTEP 6: COMBINE DISTRICT OUTPUTS")

    merged = districts_complaints_gdf.merge(
        districts_crime_gdf[
            [
                "dist_numc",
                "total_crime",
                "violent_crime",
                "nonviolent_crime",
                "unknown_crime",
                "violent_share"
            ]
        ],
        on="dist_numc",
        how="left"
    )

    merged["violent_share_from_crime"] = merged["violent_share"]

    if "total_population" in merged.columns:
        merged["crime_rate_per_1000"] = np.where(
            merged["total_population"] > 0,
            merged["total_crime"] / merged["total_population"] * 1000,
            np.nan
        )
        merged["violent_crime_rate_per_1000"] = np.where(
            merged["total_population"] > 0,
            merged["violent_crime"] / merged["total_population"] * 1000,
            np.nan
        )
        merged["nonviolent_crime_rate_per_1000"] = np.where(
            merged["total_population"] > 0,
            merged["nonviolent_crime"] / merged["total_population"] * 1000,
            np.nan
        )
    else:
        merged["crime_rate_per_1000"] = np.nan
        merged["violent_crime_rate_per_1000"] = np.nan
        merged["nonviolent_crime_rate_per_1000"] = np.nan

    return merged


# ------------------------------------------------
# FUNCTION 9: Allocate population + complaints to PSAs by area
# ------------------------------------------------
def allocate_complaints_to_psa(districts_dashboard_gdf, psa_gdf):
    """
    Area-weighted allocation of district population and complaint totals to PSAs.
    Assumes population and complaints are uniformly distributed within districts.
    """
    print("\nSTEP 7: ALLOCATE POPULATION & COMPLAINTS TO PSAS (AREAL)")

    psa = psa_gdf.to_crs(districts_dashboard_gdf.crs)

    districts = districts_dashboard_gdf[
        ["dist_numc", "geometry", "total_population", "total_complaints"]
    ].copy()
    districts["district_area"] = districts.geometry.area

    psa_fragments = gpd.overlay(psa, districts, how="intersection")
    psa_fragments["frag_area"] = psa_fragments.geometry.area

    psa_fragments = psa_fragments[psa_fragments["district_area"] > 0].copy()

    psa_fragments["pop_alloc"] = (
        psa_fragments["total_population"] *
        (psa_fragments["frag_area"] / psa_fragments["district_area"])
    )
    psa_fragments["complaints_alloc"] = (
        psa_fragments["total_complaints"] *
        (psa_fragments["frag_area"] / psa_fragments["district_area"])
    )

    psa_alloc = (
        psa_fragments.groupby("psa_id")[["pop_alloc", "complaints_alloc"]]
        .sum()
        .reset_index()
    )

    psa_with_alloc = psa.merge(psa_alloc, on="psa_id", how="left")
    psa_with_alloc[["pop_alloc", "complaints_alloc"]] = psa_with_alloc[
        ["pop_alloc", "complaints_alloc"]
    ].fillna(0)

    psa_with_alloc["complaint_rate_per_1000"] = np.where(
        psa_with_alloc["pop_alloc"] > 0,
        psa_with_alloc["complaints_alloc"] / psa_with_alloc["pop_alloc"] * 1000,
        np.nan
    )

    print("Allocated district population and complaints to PSAs using area weights.")
    return psa_with_alloc


# ------------------------------------------------
# FUNCTION 10: Export dashboard data
# ------------------------------------------------
def export_web_data(districts_dashboard_gdf, psa_dashboard_gdf, district_summary_csv, psa_summary_csv):
    """
    Exports GeoJSON and CSV for interactive district and PSA dashboard views.
    """
    print("\nSTEP 8: EXPORT WEB DATA")

    district_web = districts_dashboard_gdf.copy().to_crs(epsg=4326)
    psa_web = psa_dashboard_gdf.copy().to_crs(epsg=4326)

    district_geojson_path = os.path.join(WEB_DIR, "districts_results.geojson")
    psa_geojson_path = os.path.join(WEB_DIR, "psa_results.geojson")

    district_csv_path = os.path.join(OUTPUT_DIR, "district_summary.csv")
    psa_csv_path = os.path.join(OUTPUT_DIR, "psa_summary.csv")

    district_web.to_file(district_geojson_path, driver="GeoJSON")
    psa_web.to_file(psa_geojson_path, driver="GeoJSON")

    district_summary_csv.to_csv(district_csv_path, index=False)
    psa_summary_csv.to_csv(psa_csv_path, index=False)

    print(f"Saved GeoJSON: {district_geojson_path}")
    print(f"Saved GeoJSON: {psa_geojson_path}")
    print(f"Saved CSV: {district_csv_path}")
    print(f"Saved CSV: {psa_csv_path}")

    return district_geojson_path, psa_geojson_path, district_csv_path, psa_csv_path


# ------------------------------------------------
# FUNCTION 11: Supporting static figures
# ------------------------------------------------
def generate_supporting_maps(districts_dashboard_gdf, psa_dashboard_gdf):
    """
    Creates supporting static maps for the final report.
    """
    print("\nSTEP 9: GENERATE SUPPORTING MAPS")

    if "overall_rate_per_1000" in districts_dashboard_gdf.columns:
        fig, ax = plt.subplots(figsize=(10, 9))
        districts_dashboard_gdf.plot(
            column="overall_rate_per_1000",
            ax=ax,
            legend=True,
            cmap="YlOrBr",
            edgecolor="black",
            linewidth=0.6
        )
        ax.set_title("Overall Complaint Rate per 1,000 Residents", fontsize=15, fontweight="bold")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(
            os.path.join(OUTPUT_DIR, "district_complaint_rate.png"),
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

    if "violent_crime_rate_per_1000" in districts_dashboard_gdf.columns:
        fig, ax = plt.subplots(figsize=(10, 9))
        districts_dashboard_gdf.plot(
            column="violent_crime_rate_per_1000",
            ax=ax,
            legend=True,
            cmap="Reds",
            edgecolor="black",
            linewidth=0.6,
            missing_kwds={"color": "lightgrey", "label": "No data"}
        )
        ax.set_title("Violent Crime Rate per 1,000 Residents", fontsize=15, fontweight="bold")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(
            os.path.join(OUTPUT_DIR, "district_violent_crime_rate.png"),
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

    if "violent_share_from_crime" in psa_dashboard_gdf.columns:
        fig, ax = plt.subplots(figsize=(10, 9))
        psa_dashboard_gdf.plot(
            column="violent_share_from_crime",
            ax=ax,
            legend=True,
            cmap="Purples",
            edgecolor="black",
            linewidth=0.4,
            missing_kwds={"color": "lightgrey", "label": "No data"}
        )
        ax.set_title("PSA Violent Crime Share", fontsize=15, fontweight="bold")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(
            os.path.join(OUTPUT_DIR, "psa_violent_share.png"),
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

    if "disparity_flag" in districts_dashboard_gdf.columns:
        temp = districts_dashboard_gdf.copy()
        temp["disparity_label"] = temp["disparity_flag"].map(
            {True: "Flagged", False: "Not flagged"}
        )

        fig, ax = plt.subplots(figsize=(10, 9))
        temp.plot(
            column="disparity_label",
            ax=ax,
            categorical=True,
            legend=True,
            cmap="RdYlGn_r",
            edgecolor="black",
            linewidth=0.6
        )
        ax.set_title(
            "Districts Flagged for Black-White Complaint Rate Disparity",
            fontsize=15,
            fontweight="bold"
        )
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(
            os.path.join(OUTPUT_DIR, "district_disparity_flag.png"),
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

    print("Supporting maps saved to output/.")


# ------------------------------------------------
# FUNCTION: Export thinned 2024 crime points for the web overlay
# ------------------------------------------------
def export_crime_points_for_web(crime_gdf, max_points_per_category=12000):
    """
    Writes three thinned 2024 incident point files for the web overlay,
    split by the crime_category_simple field set in process_crime_data:
        web/crime_points_violent_2024.geojson
        web/crime_points_nonviolent_2024.geojson
        web/crime_points_2024.geojson           (combined, kept for backwards compat)

    Each file keeps lon/lat plus a small set of useful fields if present
    (offense category, simplified violent/nonviolent label, dispatch hour).
    Random-samples each category down to max_points_per_category if needed
    so the browser stays responsive.
    """
    print("\nSTEP: EXPORT CRIME POINTS FOR WEB (split violent/nonviolent)")

    if crime_gdf is None or len(crime_gdf) == 0:
        print("  No crime points available, skipping export.")
        return None

    gdf = crime_gdf.copy()
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif str(gdf.crs).lower() not in ("epsg:4326", "wgs84"):
        gdf = gdf.to_crs("EPSG:4326")

    # Pick a clean subset of useful columns if they exist.
    keep_cols = ["geometry"]

    # Detailed offense text (e.g. "Aggravated Assault Firearm")
    cat_col = find_column(list(gdf.columns),
                          ["text_general_code", "ucr_general", "offense_description", "offense"])
    if cat_col is not None:
        gdf = gdf.rename(columns={cat_col: "offense"})
        keep_cols.append("offense")

    # Simplified Violent / Nonviolent / Unknown label produced by
    # classify_crime_category() inside process_crime_data().
    if "crime_category_simple" in gdf.columns:
        gdf = gdf.rename(columns={"crime_category_simple": "category"})
        keep_cols.append("category")
    else:
        gdf["category"] = "Unknown"
        keep_cols.append("category")
        print("  WARNING: crime_category_simple not found - run process_crime_data first.")

    date_col = find_column(list(gdf.columns),
                           ["dispatch_date_time", "dispatch_date", "occur_date", "date"])
    if date_col is not None:
        gdf[date_col] = pd.to_datetime(gdf[date_col], errors="coerce")
        gdf["hour"] = gdf[date_col].dt.hour
        keep_cols.append("hour")

    gdf = gdf[[c for c in keep_cols if c in gdf.columns]].copy()

    # Drop anything outside Philly bbox just in case.
    minx, miny, maxx, maxy = -75.30, 39.85, -74.95, 40.15
    gdf["_x"] = gdf.geometry.x
    gdf["_y"] = gdf.geometry.y
    before = len(gdf)
    gdf = gdf[(gdf["_x"] >= minx) & (gdf["_x"] <= maxx) &
              (gdf["_y"] >= miny) & (gdf["_y"] <= maxy)].copy()
    gdf = gdf.drop(columns=["_x", "_y"])
    print(f"  Filtered to Philly bbox: {len(gdf):,} of {before:,} rows")

    paths = {}

    def _write_subset(subset_gdf, label, fname):
        if len(subset_gdf) > max_points_per_category:
            subset_gdf = subset_gdf.sample(
                n=max_points_per_category, random_state=42
            ).copy()
        out_path = os.path.join(WEB_DIR, fname)
        subset_gdf.to_file(out_path, driver="GeoJSON")
        print(f"  {label:<12} {len(subset_gdf):>6,} points -> {out_path}")
        return out_path

    violent_gdf = gdf[gdf["category"] == "Violent"].copy()
    nonviolent_gdf = gdf[gdf["category"] == "Nonviolent"].copy()

    paths["violent"] = _write_subset(
        violent_gdf, "Violent:", "crime_points_violent_2024.geojson"
    )
    paths["nonviolent"] = _write_subset(
        nonviolent_gdf, "Nonviolent:", "crime_points_nonviolent_2024.geojson"
    )

    # Combined file (capped at 2x the per-category limit) for any consumer
    # that wants the union without re-merging two files.
    combined = pd.concat([violent_gdf, nonviolent_gdf], ignore_index=True)
    combined_gdf = gpd.GeoDataFrame(combined, geometry="geometry", crs=gdf.crs)
    paths["combined"] = _write_subset(
        combined_gdf, "Combined:",
        "crime_points_2024.geojson"
    )

    return paths


# ------------------------------------------------
# CONSTANT: Philadelphia Police District station roster
# ------------------------------------------------
# All 21 active PPD district stations, with verified street addresses
# (cross-checked against the official PPD contacts page, Apple Maps,
# Waze, and Facebook listings). Coordinates were geocoded once via
# OpenStreetMap Nominatim using the full street address (intersections
# alone do not geocode reliably). The 25th District shares the building
# at 3901 Whitaker Ave with the 24th, so its marker is offset ~30 m east
# for clickability.
#
# This roster is the SINGLE SOURCE OF TRUTH for police station markers
# on the web map. We deliberately do not fetch from the city's ArcGIS
# feature service anymore because the PPD_Districts_HQ layer was retired
# and the previous fallback produced fake "Station 1...21" centroids.
PPD_STATION_ROSTER = [
    # (district_id, captain, division, address_display, street_address,
    #  phone, email, twitter_url, page_url, lon, lat)
    ("01", "Capt. Kelly Robbins", "South Police Division",
     "24th St. & Wolf St.", "2301 S 24th St, Philadelphia, PA 19145",
     "215-686-3010", "police.co_01@phila.gov",
     "https://x.com/PPD01Dist", "https://www.phillypolice.com/districts/1st-district/",
     -75.18640, 39.92325),
    ("02", "Capt. Andrew DiSanto", "Northeast Police Division",
     "7306 Castor Ave", "7306 Castor Ave, Philadelphia, PA 19152",
     "215-686-3020", "police.co_02@phila.gov",
     "https://x.com/PPD02Dist", "https://www.phillypolice.com/districts/2nd-district/",
     -75.06595, 40.05143),
    ("03", "Capt. George Mullen", "South Police Division",
     "11th St. & Wharton St.", "1100 Wharton St, Philadelphia, PA 19147",
     "215-686-3030", "police.co_03@phila.gov",
     "https://x.com/PPD03Dist", "https://www.phillypolice.com/districts/3rd-district/",
     -75.16274, 39.93326),
    ("05", "Capt. James Kimrey", "Northwest Police Division",
     "Ridge Ave. & Cinnaminson St.", "6666 Ridge Ave, Philadelphia, PA 19128",
     "215-686-3050", "police.co_05@phila.gov",
     "https://x.com/PPD05Dist", "https://www.phillypolice.com/districts/5th-district/",
     -75.22460, 40.03992),
    ("07", "Capt. Steven O'Brien", "Northeast Police Division",
     "Bustleton Ave. & Bowler St.", "9685 Bustleton Ave, Philadelphia, PA 19115",
     "215-686-3070", "police.co_07@phila.gov",
     "https://x.com/PPD07Dist", "https://www.phillypolice.com/districts/7th-district/",
     -75.03473, 40.08801),
    ("08", "Capt. Nicholas Deblasis", "Northeast Police Division",
     "Academy Rd. & Red Lion Rd.", "3100 Red Lion Rd, Philadelphia, PA 19114",
     "215-686-3080", "police.co_08@phila.gov",
     "https://x.com/PPD08Dist", "https://www.phillypolice.com/districts/8th-district/",
     -74.99873, 40.08117),
    ("09", "Capt. David Read", "Central Police Division",
     "401 N. 21st St.", "401 N 21st St, Philadelphia, PA 19103",
     "215-686-3090", "police.co_09@phila.gov",
     "https://x.com/PPD09Dist", "https://www.phillypolice.com/districts/9th-district/",
     -75.17324, 39.96178),
    ("12", "Capt. Matthew Johnson", "Southwest Police Division",
     "65th St. & Woodland Ave.", "6448 Woodland Ave, Philadelphia, PA 19142",
     "215-686-3120", "police.co_12@phila.gov",
     "https://x.com/PPD12Dist", "https://www.phillypolice.com/districts/12th-district/",
     -75.23386, 39.92580),
    ("14", "Capt. Stuart McCoullum", "Northwest Police Division",
     "Haines St. & Germantown Ave.", "43 W Haines St, Philadelphia, PA 19144",
     "215-686-3140", "police.co_14@phila.gov",
     "https://x.com/PPD14Dist", "https://www.phillypolice.com/districts/14th-district/",
     -75.17772, 40.03784),
    ("15", "Capt. Marques Newsome", "Northeast Police Division",
     "Harbison Ave. & Levick St.", "2831 Levick St, Philadelphia, PA 19149",
     "215-686-3150", "police.co_15@phila.gov",
     "https://x.com/PPD15Dist", "https://www.phillypolice.com/districts/15th-district/",
     -75.06414, 40.03241),
    ("16", "Capt. Amina Brown", "Southwest Police Division",
     "39th St. & Lancaster Ave.", "3900 Lancaster Ave, Philadelphia, PA 19104",
     "215-686-3160", "police.co_16@phila.gov",
     "https://x.com/PPD16Dist", "https://www.phillypolice.com/districts/16th-district/",
     -75.20045, 39.96166),
    ("17", "Capt. Kenneth McKinney", "South Police Division",
     "20th St. & Federal St.", "1201 S 20th St, Philadelphia, PA 19146",
     "215-686-3170", "police.co_17@phila.gov",
     "https://x.com/PPD17Dist", "https://www.phillypolice.com/districts/17th-district/",
     -75.17698, 39.93682),
    ("18", "Capt. Joseph Waters", "Southwest Police Division",
     "55th St. & Pine St.", "5510 Pine St, Philadelphia, PA 19143",
     "215-686-3180", "police.co_18@phila.gov",
     "https://x.com/PPD18Dist", "https://www.phillypolice.com/districts/18th-district/",
     -75.23253, 39.95423),
    ("19", "Capt. Lawrence Nuble", "Southwest Police Division",
     "61st St. & Thompson St.", "6100 W Thompson St, Philadelphia, PA 19151",
     "215-686-3190", "police.co_19@phila.gov",
     "https://x.com/PPD19Dist", "https://www.phillypolice.com/districts/19th-district/",
     -75.24019, 39.97141),
    ("22", "Capt. Brian Sprowal", "Central Police Division",
     "17th St. & Montgomery Ave.", "1747 W Montgomery Ave, Philadelphia, PA 19121",
     "215-686-3220", "police.co_22@phila.gov",
     "https://x.com/PPD22Dist", "https://www.phillypolice.com/districts/22nd-district/",
     -75.16345, 39.98106),
    # 24th and 25th Districts share the building at 3901 Whitaker Ave;
    # we represent them as a single combined marker.
    ("24-25", "Capt. Christopher Bullick (24th) / Capt. Stephen Bennis Jr. (25th)",
     "East Police Division",
     "3901 Whitaker Ave. (24th & 25th shared)",
     "3901 Whitaker Ave, Philadelphia, PA 19124",
     "215-686-3240 (24th) / 215-686-3250 (25th)",
     "police.co_24@phila.gov / police.co_25@phila.gov",
     "https://x.com/PPD24Dist", "https://www.phillypolice.com/districts/24th-district/",
     -75.12239, 40.00869),
    ("26", "Capt. Victoria Casale", "East Police Division",
     "E. Girard Ave. & Montgomery Ave.", "615 E Girard Ave, Philadelphia, PA 19125",
     "215-686-3260", "police.co_26@phila.gov",
     "https://x.com/PPD26Dist", "https://www.phillypolice.com/districts/26th-district/",
     -75.13246, 39.96958),
    ("35", "Capt. Walter Burks", "Northwest Police Division",
     "N. Broad St. & Champlost St.", "5960 N Broad St, Philadelphia, PA 19141",
     "215-686-3350", "police.co_35@phila.gov",
     "https://x.com/PPD35Dist", "https://www.phillypolice.com/districts/35th-district/",
     -75.14378, 40.04382),
    ("39", "Capt. Dana Bradley", "Northwest Police Division",
     "22nd St. & Hunting Park Ave.", "2201 W Hunting Park Ave, Philadelphia, PA 19140",
     "215-686-3390", "police.co_39@phila.gov",
     "https://x.com/PPD39Dist", "https://www.phillypolice.com/districts/39th-district/",
     -75.16475, 40.01096),
    ("77", "Capt. Samantha Brown", "Southwest Police Division",
     "8800 Essington Ave. (Airport)", "8800 Essington Ave, Philadelphia, PA 19153",
     "215-937-6918", "police.co_airport@phila.gov",
     None, "https://www.phillypolice.com/districts/77th-district/",
     -75.22148, 39.91427),
]


def _ordinal_suffix(n):
    if 11 <= (n % 100) <= 13:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


# ------------------------------------------------
# FUNCTION: Write police station GeoJSON for the web
# ------------------------------------------------
def export_police_stations_for_web(stations_url=None, districts_gdf=None):
    """Writes web/police_stations.geojson from the hardcoded PPD_STATION_ROSTER.

    The `stations_url` and `districts_gdf` arguments are kept for backwards
    compatibility with the call site in main(), but are no longer used.
    """
    print("\nSTEP: EXPORT POLICE STATIONS FOR WEB (hardcoded roster)")

    out_path = os.path.join(WEB_DIR, "police_stations.geojson")
    features = []
    for (did, captain, division, addr_disp, street_addr, phone, email,
         twitter, page, lon, lat) in PPD_STATION_ROSTER:
        if did == "77":
            name = "77th District (Airport)"
        elif did == "24-25":
            name = "24th & 25th Districts HQ"
        else:
            n = int(did)
            name = f"{n}{_ordinal_suffix(n)} District HQ"
        props = {
            "station_name": name,
            "district_id": did,
            "captain": captain,
            "division": division,
            "address": addr_disp,
            "street_address": street_addr,
            "phone": phone,
            "email": email,
            "twitter_url": twitter,
            "page_url": page,
            "geocode_source": "manual_verified",
        }
        features.append({
            "type": "Feature",
            "properties": props,
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
        })

    geojson = {"type": "FeatureCollection", "features": features}
    with open(out_path, "w") as f:
        json.dump(geojson, f, indent=2)
    print(f"  Saved {out_path}  ({len(features)} stations)")
    return out_path


# ------------------------------------------------
# FUNCTION: Build a heat-index raster for the web
# ------------------------------------------------
def build_heat_index_raster_for_web(districts_gdf, crime_gdf, grid_size=200):
    """
    Builds a Philadelphia heat-index surface and writes:
      - web/heat_index.png       (RGBA color-ramped raster)
      - web/heat_index_bounds.json  (bbox + min/max temp metadata)

    Strategy: try a real Landsat 8/9 LST tile first via rasterio (if
    rasterio is installed and the COG endpoint is reachable). If that
    fails, fall back to a deterministic proxy heat surface built from
    a smoothed kernel of crime density + a north-south thermal gradient.
    The proxy is clearly labeled as proxy in the bounds JSON so the
    HTML can show that note in the legend.
    """
    print("\nSTEP: BUILD HEAT INDEX RASTER FOR WEB")

    # Philly bounding box (lon/lat).
    minx, miny, maxx, maxy = -75.280, 39.870, -74.955, 40.140

    temp_grid = None
    source = "proxy"
    note = ("Proxy heat surface: smoothed kernel of 2024 incident density "
            "plus a north-south gradient. Replace with real LST when available.")

    # ---- Attempt 1: Microsoft Planetary Computer STAC search for Landsat LST ----
    # Searches the landsat-c2-l2 collection for low-cloud summer scenes
    # over Philadelphia and pulls the lwir11 (ST_B10) band of the best match.
    if _HAS_RASTERIO and _HAS_PC:
        try:
            print("  Searching Planetary Computer for Landsat surface temperature scenes...")
            catalog = _PCClient.open(
                "https://planetarycomputer.microsoft.com/api/stac/v1",
                modifier=_PC.sign_inplace,
            )

            # Try summer 2024 first, then fall back to summer 2023, then 2022.
            search_seasons = [
                ("2024-06-01", "2024-09-15", "summer 2024"),
                ("2023-06-01", "2023-09-15", "summer 2023"),
                ("2022-06-01", "2022-09-15", "summer 2022"),
            ]

            chosen_item = None
            chosen_label = None
            for start, end, label in search_seasons:
                search = catalog.search(
                    collections=["landsat-c2-l2"],
                    bbox=[minx, miny, maxx, maxy],
                    datetime=f"{start}/{end}",
                    query={
                        "eo:cloud_cover": {"lt": 25},
                        "platform": {"in": ["landsat-8", "landsat-9"]},
                    },
                    max_items=20,
                )
                items = list(search.items())
                if items:
                    # Pick the lowest-cloud-cover scene.
                    items.sort(key=lambda it: it.properties.get("eo:cloud_cover", 100))
                    chosen_item = items[0]
                    chosen_label = label
                    print(f"  Found {len(items)} {label} scene(s); using {chosen_item.id} "
                          f"({chosen_item.properties.get('eo:cloud_cover', 'NA')}% cloud cover)")
                    break
                else:
                    print(f"  No matching {label} scenes; trying older season...")

            if chosen_item is not None and "lwir11" in chosen_item.assets:
                lwir_href = chosen_item.assets["lwir11"].href
                with rasterio.open(lwir_href) as src:
                    dst_height = grid_size
                    dst_width = grid_size
                    dst_transform = rasterio.transform.from_bounds(
                        minx, miny, maxx, maxy, dst_width, dst_height
                    )
                    dst_array = np.zeros((dst_height, dst_width), dtype=np.float32)
                    reproject(
                        source=rasterio.band(src, 1),
                        destination=dst_array,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=dst_transform,
                        dst_crs="EPSG:4326",
                        resampling=Resampling.bilinear,
                        src_nodata=0,
                        dst_nodata=0,
                    )
                    # Landsat Collection-2 Level-2 ST_B10 scaling:
                    # temp_K = DN * 0.00341802 + 149.0
                    valid = dst_array > 0
                    if valid.sum() > (0.05 * dst_array.size):
                        kelvin = dst_array * 0.00341802 + 149.0
                        celsius = kelvin - 273.15
                        fahrenheit = celsius * 9.0 / 5.0 + 32.0
                        bg = float(np.nanmean(fahrenheit[valid]))
                        fahrenheit[~valid] = bg
                        temp_grid = fahrenheit
                        source = "landsat_st_b10"
                        note = (f"Surface temperature from Landsat Collection 2 ST_B10 "
                                f"({chosen_label} scene {chosen_item.id}, "
                                f"{chosen_item.properties.get('eo:cloud_cover', 'NA')}% cloud cover), "
                                f"fetched via Microsoft Planetary Computer. Units: deg F.")
                        print(f"  Decoded LST: {valid.sum()} valid pixels, "
                              f"range {fahrenheit[valid].min():.1f}-{fahrenheit[valid].max():.1f} F")
            else:
                print("  No usable Landsat scene found in any season window.")

        except Exception as e:
            print(f"  Planetary Computer fetch failed ({e}); will use proxy.")
    else:
        if not _HAS_PC:
            print("  pystac-client / planetary-computer not installed; "
                  "install with: pip install pystac-client planetary-computer")
        if not _HAS_RASTERIO:
            print("  rasterio not installed; skipping real LST fetch.")

    # ---- Fallback: proxy heat surface ----
    if temp_grid is None:
        print("  Building proxy heat surface from crime density + gradient.")
        gx = np.linspace(minx, maxx, grid_size)
        gy = np.linspace(miny, maxy, grid_size)
        XX, YY = np.meshgrid(gx, gy)

        # North-south gradient (south is hotter in Philly UHI patterns).
        gradient = (maxy - YY) / (maxy - miny)  # 0 at north, 1 at south

        # Crime density kernel as a stand-in for activity-driven heat.
        density = np.zeros_like(XX, dtype=np.float32)
        if crime_gdf is not None and len(crime_gdf) > 0:
            cg = crime_gdf
            if cg.crs is None:
                cg = cg.set_crs("EPSG:4326")
            elif str(cg.crs).lower() not in ("epsg:4326", "wgs84"):
                cg = cg.to_crs("EPSG:4326")
            xs = cg.geometry.x.values
            ys = cg.geometry.y.values
            mask = (xs >= minx) & (xs <= maxx) & (ys >= miny) & (ys <= maxy)
            xs = xs[mask]
            ys = ys[mask]
            if len(xs) > 50000:
                idx = np.random.RandomState(42).choice(len(xs), 50000, replace=False)
                xs = xs[idx]; ys = ys[idx]
            # 2D histogram on the grid, then a fast box-blur.
            H, _, _ = np.histogram2d(
                ys, xs, bins=[grid_size, grid_size],
                range=[[miny, maxy], [minx, maxx]]
            )
            # Simple separable smoothing (5x5 mean) to fake a Gaussian.
            for _ in range(3):
                H = (H +
                     np.roll(H, 1, axis=0) + np.roll(H, -1, axis=0) +
                     np.roll(H, 1, axis=1) + np.roll(H, -1, axis=1)) / 5.0
            if H.max() > 0:
                density = H / H.max()

        # Combine: base 78F, +12F from gradient, +10F from density.
        temp_grid = 78.0 + 12.0 * gradient.astype(np.float32) + 10.0 * density.astype(np.float32)

    # Mask to Philly boundary so we don't paint heat over rivers/NJ.
    if districts_gdf is not None and len(districts_gdf) > 0:
        try:
            from rasterio.features import geometry_mask  # type: ignore
            d = districts_gdf.copy()
            if d.crs is None:
                d = d.set_crs("EPSG:3857")
            d_wgs = d.to_crs("EPSG:4326")
            transform = rasterio.transform.from_bounds(
                minx, miny, maxx, maxy, temp_grid.shape[1], temp_grid.shape[0]
            )
            mask = geometry_mask(
                [g.__geo_interface__ for g in d_wgs.geometry if g is not None],
                out_shape=temp_grid.shape,
                transform=transform,
                invert=True,
            )
        except Exception:
            mask = np.ones_like(temp_grid, dtype=bool)
    else:
        mask = np.ones_like(temp_grid, dtype=bool)

    # Build RGBA image with a yellow->orange->red ramp + transparent outside.
    valid_temps = temp_grid[mask]
    if valid_temps.size == 0:
        valid_temps = temp_grid.flatten()
    vmin = float(np.percentile(valid_temps, 2))
    vmax = float(np.percentile(valid_temps, 98))
    if vmax - vmin < 1e-3:
        vmax = vmin + 1.0
    norm = (np.clip(temp_grid, vmin, vmax) - vmin) / (vmax - vmin)

    # Custom 5-stop ramp: cool yellow -> orange -> red -> deep red.
    ramp = mcolors.LinearSegmentedColormap.from_list(
        "heat_ramp",
        ["#fff7bc", "#fed976", "#fd8d3c", "#e31a1c", "#800026"],
    )
    rgba = ramp(norm)  # shape (H, W, 4), float in [0,1]
    rgba[..., 3] = np.where(mask, 0.75, 0.0)
    rgba_uint8 = (rgba * 255).astype(np.uint8)

    # Save PNG with matplotlib (no extra deps).
    out_png = os.path.join(WEB_DIR, "heat_index.png")
    fig = plt.figure(figsize=(rgba_uint8.shape[1] / 100.0, rgba_uint8.shape[0] / 100.0), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    # Flip vertically so the PNG north-up matches Leaflet bounds expectation.
    ax.imshow(np.flipud(rgba_uint8), interpolation="nearest")
    fig.savefig(out_png, dpi=100, transparent=True)
    plt.close(fig)

    bounds_meta = {
        "bounds": [[miny, minx], [maxy, maxx]],  # Leaflet [[south,west],[north,east]]
        "bbox_lonlat": [minx, miny, maxx, maxy],
        "temp_min_f": round(vmin, 2),
        "temp_max_f": round(vmax, 2),
        "grid_size": int(temp_grid.shape[0]),
        "source": source,
        "units": "deg_F",
        "note": note,
    }
    out_json = os.path.join(WEB_DIR, "heat_index_bounds.json")
    with open(out_json, "w") as f:
        json.dump(bounds_meta, f, indent=2)

    print(f"  Saved {out_png}")
    print(f"  Saved {out_json}")
    print(f"  Source: {source}, range {vmin:.1f}-{vmax:.1f} deg F")
    return temp_grid, (minx, miny, maxx, maxy), source


# ------------------------------------------------
# FUNCTION: Build Heat-Vulnerability x Crime hex grid for the web
# ------------------------------------------------
def build_heat_crime_hex_grid_for_web(
    temp_grid,
    bbox,
    heat_source,
    crime_gdf,
    districts_gdf,
    hex_size_m=500,
):
    """
    Builds a flat-top hex grid covering Philly, then for each hex computes:
      - mean Landsat surface temperature (from temp_grid),
      - a Heat-Vulnerability Index (0-100) = percentile rank of mean temp
        across all Philly-overlapping hexes (so 0 = coolest in Philly,
        100 = hottest — a relative urban-heat-exposure score, not raw F),
      - violent / nonviolent / total 2024 crime counts,
      - a crime-density score (total per sq km),
      - heat tier (0/1/2 = low/mid/high) by HVI tercile,
      - crime tier (0/1/2) by total-crime-density tercile,
      - separate violent/nonviolent crime tiers for the toggle,
      - a 3x3 bivariate class index = heat_tier*3 + crime_tier (0-8).

    Writes web/heat_crime_hexes.geojson with all of those properties so the
    front-end can color hexes directly without re-doing any math.
    """
    print("\nSTEP: BUILD HEAT-VULNERABILITY x CRIME HEX GRID")

    if temp_grid is None or bbox is None:
        print("  No heat surface available; skipping hex grid build.")
        return None

    minx, miny, maxx, maxy = bbox

    # ---- 1. Build flat-top hex grid in EPSG:3857 (meters) ----
    # Using shapely + a simple offset lattice. Flat-top hex with circumradius R:
    #   width  = 2R, height = sqrt(3)*R, horizontal step = 1.5R, vertical step = sqrt(3)*R
    from shapely.geometry import Polygon as ShpPolygon

    R = float(hex_size_m)  # circumradius in meters
    h = math.sqrt(3.0) * R

    # Project Philly bbox to 3857 to get meter bounds
    bbox_gdf = gpd.GeoDataFrame(
        geometry=[ShpPolygon([(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)])],
        crs="EPSG:4326",
    ).to_crs("EPSG:3857")
    mxmin, mymin, mxmax, mymax = bbox_gdf.total_bounds

    hexes = []
    col = 0
    x = mxmin - R
    while x < mxmax + R:
        y_offset = (h / 2.0) if (col % 2 == 1) else 0.0
        y = mymin - h + y_offset
        while y < mymax + h:
            cx, cy = x, y
            verts = [
                (cx + R * math.cos(math.radians(60.0 * k)),
                 cy + R * math.sin(math.radians(60.0 * k)))
                for k in range(6)
            ]
            hexes.append(ShpPolygon(verts))
            y += h
        x += 1.5 * R
        col += 1

    hex_gdf = gpd.GeoDataFrame({"hex_id": range(len(hexes))}, geometry=hexes, crs="EPSG:3857")
    hex_gdf_wgs = hex_gdf.to_crs("EPSG:4326")
    print(f"  Generated {len(hex_gdf)} candidate hexes (R={R} m).")

    # ---- 2. Clip to Philly districts so we don't shade rivers/NJ ----
    if districts_gdf is not None and len(districts_gdf) > 0:
        d = districts_gdf.copy()
        if d.crs is None:
            d = d.set_crs("EPSG:3857")
        d_wgs = d.to_crs("EPSG:4326")
        philly_union = d_wgs.geometry.union_all() if hasattr(d_wgs.geometry, "union_all") else d_wgs.geometry.unary_union
        # Keep hexes whose centroid is inside Philly
        centroids = hex_gdf_wgs.geometry.centroid
        mask = centroids.within(philly_union)
        hex_gdf_wgs = hex_gdf_wgs.loc[mask].reset_index(drop=True)
        hex_gdf_wgs["hex_id"] = range(len(hex_gdf_wgs))
        print(f"  Kept {len(hex_gdf_wgs)} hexes whose centroid lies inside Philly.")

    if len(hex_gdf_wgs) == 0:
        print("  No hexes after clipping; aborting.")
        return None

    # ---- 3. Mean surface temperature per hex from the temp_grid ----
    grid_h, grid_w = temp_grid.shape
    centroids = hex_gdf_wgs.geometry.centroid
    mean_temps = np.full(len(hex_gdf_wgs), np.nan, dtype=np.float32)
    for i, c in enumerate(centroids):
        lon, lat = c.x, c.y
        if lon < minx or lon > maxx or lat < miny or lat > maxy:
            continue
        # Image is north-up: row 0 = north (maxy)
        fx = (lon - minx) / (maxx - minx)
        fy = (maxy - lat) / (maxy - miny)
        # Sample a small 5x5 window centered on the centroid pixel
        px = int(round(fx * (grid_w - 1)))
        py = int(round(fy * (grid_h - 1)))
        x0 = max(0, px - 2); x1 = min(grid_w, px + 3)
        y0 = max(0, py - 2); y1 = min(grid_h, py + 3)
        window = temp_grid[y0:y1, x0:x1]
        if window.size > 0:
            mean_temps[i] = float(np.nanmean(window))
    hex_gdf_wgs["mean_temp_f"] = mean_temps

    # ---- 4. Heat Vulnerability Index = percentile rank within Philly hexes ----
    valid_mask = ~np.isnan(mean_temps)
    valid_temps = mean_temps[valid_mask]
    if valid_temps.size > 0:
        # rank 1..N then map to 0..100
        order = valid_temps.argsort().argsort().astype(np.float32)
        hvi = np.full_like(mean_temps, np.nan, dtype=np.float32)
        hvi[valid_mask] = (order / max(1, valid_temps.size - 1)) * 100.0
    else:
        hvi = np.full_like(mean_temps, np.nan, dtype=np.float32)
    hex_gdf_wgs["heat_vuln_index"] = hvi

    # Heat tier (0/1/2) by terciles of HVI
    if valid_temps.size > 0:
        valid_hvi = hvi[valid_mask]
        sorted_hvi = np.sort(valid_hvi)
        n = sorted_hvi.size
        t1 = sorted_hvi[n // 3]
        t2 = sorted_hvi[(2 * n) // 3]
        heat_tier = np.zeros_like(hvi, dtype=np.int8)
        heat_tier[hvi >= t1] = 1
        heat_tier[hvi >= t2] = 2
        heat_tier[~valid_mask] = 0
        hex_gdf_wgs["heat_tier"] = heat_tier
        hex_gdf_wgs.attrs["heat_tier_cuts"] = (float(t1), float(t2))
        print(f"  HVI tercile cuts: {t1:.1f} / {t2:.1f}")
    else:
        hex_gdf_wgs["heat_tier"] = 0

    # ---- 5. Spatially join crimes to hexes (violent + nonviolent counts) ----
    if crime_gdf is None or len(crime_gdf) == 0:
        hex_gdf_wgs["violent_count"] = 0
        hex_gdf_wgs["nonviolent_count"] = 0
        hex_gdf_wgs["total_count"] = 0
    else:
        cg = crime_gdf.copy()
        if cg.crs is None:
            cg = cg.set_crs("EPSG:4326")
        elif str(cg.crs).lower() not in ("epsg:4326", "wgs84"):
            cg = cg.to_crs("EPSG:4326")
        if "crime_category_simple" not in cg.columns:
            cg["crime_category_simple"] = "Unknown"

        joined = gpd.sjoin(
            cg[["crime_category_simple", "geometry"]],
            hex_gdf_wgs[["hex_id", "geometry"]],
            how="inner",
            predicate="within",
        )

        counts = (
            joined.groupby(["hex_id", "crime_category_simple"])
            .size()
            .unstack(fill_value=0)
        )
        counts = counts.reindex(hex_gdf_wgs["hex_id"], fill_value=0)
        v = counts["Violent"].astype(int) if "Violent" in counts.columns else pd.Series(0, index=counts.index)
        nv = counts["Nonviolent"].astype(int) if "Nonviolent" in counts.columns else pd.Series(0, index=counts.index)
        hex_gdf_wgs["violent_count"] = v.values
        hex_gdf_wgs["nonviolent_count"] = nv.values
        hex_gdf_wgs["total_count"] = (v + nv).values
        print(f"  Joined crimes -> hexes: "
              f"violent total={int(v.sum())}, nonviolent total={int(nv.sum())}.")

    # ---- 6. Crime tiers (terciles) for total / violent / nonviolent ----
    def tercile_tiers(values):
        v = np.asarray(values, dtype=np.float32)
        if v.size == 0 or v.max() == 0:
            return np.zeros_like(v, dtype=np.int8), (0.0, 0.0)
        # Use only nonzero hexes for cuts so we don't waste a tier on "no crime"
        nz = v[v > 0]
        if nz.size < 3:
            return np.where(v > 0, 2, 0).astype(np.int8), (0.5, 0.5)
        sorted_nz = np.sort(nz)
        n = sorted_nz.size
        c1 = sorted_nz[n // 3]
        c2 = sorted_nz[(2 * n) // 3]
        tiers = np.zeros_like(v, dtype=np.int8)
        tiers[(v > 0) & (v < c1)] = 0
        tiers[(v >= c1) & (v < c2)] = 1
        tiers[v >= c2] = 2
        return tiers, (float(c1), float(c2))

    total_tier, total_cuts = tercile_tiers(hex_gdf_wgs["total_count"].values)
    violent_tier, violent_cuts = tercile_tiers(hex_gdf_wgs["violent_count"].values)
    nonviolent_tier, nonviolent_cuts = tercile_tiers(hex_gdf_wgs["nonviolent_count"].values)
    hex_gdf_wgs["crime_tier_total"] = total_tier
    hex_gdf_wgs["crime_tier_violent"] = violent_tier
    hex_gdf_wgs["crime_tier_nonviolent"] = nonviolent_tier

    # 3x3 bivariate class for the default (combined) view: heat_tier * 3 + crime_tier_total
    hex_gdf_wgs["biv_class_total"] = (
        hex_gdf_wgs["heat_tier"].astype(int) * 3 + hex_gdf_wgs["crime_tier_total"].astype(int)
    )
    hex_gdf_wgs["biv_class_violent"] = (
        hex_gdf_wgs["heat_tier"].astype(int) * 3 + hex_gdf_wgs["crime_tier_violent"].astype(int)
    )
    hex_gdf_wgs["biv_class_nonviolent"] = (
        hex_gdf_wgs["heat_tier"].astype(int) * 3 + hex_gdf_wgs["crime_tier_nonviolent"].astype(int)
    )

    # ---- 7. Save GeoJSON + side-car metadata ----
    out_path = os.path.join(WEB_DIR, "heat_crime_hexes.geojson")
    keep_cols = [
        "hex_id", "mean_temp_f", "heat_vuln_index", "heat_tier",
        "violent_count", "nonviolent_count", "total_count",
        "crime_tier_total", "crime_tier_violent", "crime_tier_nonviolent",
        "biv_class_total", "biv_class_violent", "biv_class_nonviolent",
        "geometry",
    ]
    hex_gdf_wgs[keep_cols].to_file(out_path, driver="GeoJSON")

    meta = {
        "hex_count": int(len(hex_gdf_wgs)),
        "hex_size_m": float(hex_size_m),
        "heat_source": heat_source,
        "heat_tier_cuts_hvi": list(hex_gdf_wgs.attrs.get("heat_tier_cuts", (33.3, 66.7))),
        "crime_tier_cuts_total": list(total_cuts),
        "crime_tier_cuts_violent": list(violent_cuts),
        "crime_tier_cuts_nonviolent": list(nonviolent_cuts),
        "violent_total": int(hex_gdf_wgs["violent_count"].sum()),
        "nonviolent_total": int(hex_gdf_wgs["nonviolent_count"].sum()),
        "note": (
            "Heat-Vulnerability Index (0-100) = percentile rank of mean Landsat "
            "surface temperature across Philly hexes. 0 = coolest area in Philly, "
            "100 = hottest. Bivariate class = heat_tier*3 + crime_tier (0-8)."
        ),
    }
    meta_path = os.path.join(WEB_DIR, "heat_crime_hexes_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Saved {out_path}")
    print(f"  Saved {meta_path}")
    return out_path


# ------------------------------------------------
# MAIN
# ------------------------------------------------
def main():
    print("\n" + "=" * 70)
    print("PHILADELPHIA POLICE COMPLAINT + CRIME ANALYSIS")
    print("BIG DATA FINAL PROJECT VERSION")
    print("=" * 70)

    setup_project()

    data_files = download_data(DATA_URLS)
    if data_files is None:
        print("ERROR: Download failed. Exiting.")
        return

    districts_with_pop, pop_mapping = spatial_analysis(
        data_files["police_districts"],
        data_files["census_tracts"]
    )

    psa_gdf = load_psa_geography(
        data_files["police_psa"],
        districts_with_pop.crs
    )

    districts_complaints_gdf, complaint_summary_df, complaints_full, rate_groups = (
        calculate_complaint_rates(
            districts_with_pop,
            data_files["complaints"],
            data_files["complaint_demographics"],
            pop_mapping
        )
    )

    crime_gdf = build_crime_geodataframe(data_files["crime_incidents"])

    districts_crime_gdf, psa_crime_gdf, district_join_points, psa_join_points, crime_gdf = (
        process_crime_data(
            crime_gdf,
            districts_with_pop,
            psa_gdf
        )
    )

    # Web overlays for the Heat x Crime tab
    export_crime_points_for_web(crime_gdf, max_points_per_category=12000)
    export_police_stations_for_web(
        data_files.get("police_stations"),
        districts_gdf=districts_with_pop
    )
    heat_temp_grid, heat_bbox, heat_source = build_heat_index_raster_for_web(
        districts_with_pop,
        crime_gdf,
        grid_size=200
    )
    build_heat_crime_hex_grid_for_web(
        temp_grid=heat_temp_grid,
        bbox=heat_bbox,
        heat_source=heat_source,
        crime_gdf=crime_gdf,
        districts_gdf=districts_with_pop,
        hex_size_m=500,
    )

    districts_dashboard_gdf = combine_district_outputs(
        districts_complaints_gdf,
        districts_crime_gdf
    )

    psa_with_complaints = allocate_complaints_to_psa(
        districts_dashboard_gdf,
        psa_crime_gdf
    )

    crime_cols = [
        "psa_id",
        "total_crime",
        "violent_crime",
        "nonviolent_crime",
        "unknown_crime",
        "violent_share_from_crime"
    ]
    psa_with_complaints = psa_with_complaints.merge(
        psa_crime_gdf[crime_cols],
        on="psa_id",
        how="left",
        suffixes=("", "_from_crime")
    )

    for col in ["total_crime", "violent_crime", "nonviolent_crime", "unknown_crime"]:
        if col in psa_with_complaints.columns:
            psa_with_complaints[col] = psa_with_complaints[col].fillna(0)

    psa_with_complaints["crime_rate_per_1000"] = np.where(
        psa_with_complaints["pop_alloc"] > 0,
        psa_with_complaints["total_crime"] / psa_with_complaints["pop_alloc"] * 1000,
        np.nan
    )
    psa_with_complaints["violent_crime_rate_per_1000"] = np.where(
        psa_with_complaints["pop_alloc"] > 0,
        psa_with_complaints["violent_crime"] / psa_with_complaints["pop_alloc"] * 1000,
        np.nan
    )
    psa_with_complaints["nonviolent_crime_rate_per_1000"] = np.where(
        psa_with_complaints["pop_alloc"] > 0,
        psa_with_complaints["nonviolent_crime"] / psa_with_complaints["pop_alloc"] * 1000,
        np.nan
    )

    psa_summary_df = pd.DataFrame(psa_with_complaints.drop(columns="geometry"))
    district_summary_for_csv = districts_dashboard_gdf.drop(columns="geometry")

    export_web_data(
        districts_dashboard_gdf,
        psa_with_complaints,
        district_summary_for_csv,
        psa_summary_df
    )

    generate_supporting_maps(districts_dashboard_gdf, psa_with_complaints)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("Main web files created:")
    print("  web/districts_results.geojson")
    print("  web/psa_results.geojson")
    print("  web/crime_points_2024.geojson")
    print("  web/police_stations.geojson")
    print("  web/heat_index.png")
    print("  web/heat_index_bounds.json")
    print("  web/heat_crime_hexes.geojson")
    print("  web/heat_crime_hexes_meta.json")
    print("Summary tables created:")
    print("  output/district_summary.csv")
    print("  output/psa_summary.csv")
    print("Supporting maps created:")
    print("  output/district_complaint_rate.png")
    print("  output/district_violent_crime_rate.png")
    print("  output/psa_violent_share.png")
    print("  output/district_disparity_flag.png")
    print("=" * 70)


if __name__ == "__main__":
    main()