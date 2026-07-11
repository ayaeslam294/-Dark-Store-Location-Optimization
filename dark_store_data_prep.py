"""Data preparation utilities for a dark store location optimization engine.

This module provides reusable loading and preparation functions for:
- population raster points extracted from a constrained GeoTIFF
- projected road network edges with imputed travel times
- competitor locations extracted from OpenStreetMap
- CRS harmonization for downstream optimization models

The functions are intentionally defensive: they validate inputs, preserve
geospatial metadata, and surface clear exceptions for missing or invalid data.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
import rasterio
from geopandas import GeoDataFrame
from pyproj import CRS
from rasterio.transform import xy

LOGGER = logging.getLogger(__name__)

# Configure a reasonable default logging handler for standalone use.
if not LOGGER.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


def _validate_path(filepath: str | Path, expected_suffix: str | None = None) -> Path:
    """Validate that a path exists and optionally matches an expected suffix."""

    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if expected_suffix is not None and path.suffix.lower() != expected_suffix.lower():
        raise ValueError(f"Expected a {expected_suffix} file, got: {path.suffix}")
    return path


def _ensure_geodataframe(obj: object, name: str) -> GeoDataFrame:
    """Validate that an object is a non-empty GeoDataFrame."""

    if not isinstance(obj, gpd.GeoDataFrame):
        raise TypeError(f"{name} must be a GeoDataFrame.")
    if obj.empty:
        raise ValueError(f"{name} is empty.")
    if obj.geometry is None:
        raise ValueError(f"{name} has no active geometry column.")
    return obj


def _pointify_geometry(geometry) -> object:
    """Convert polygons/lines to a representative point for competitor datasets."""

    if geometry is None or geometry.is_empty:
        return None
    geom_type = geometry.geom_type
    if geom_type == "Point":
        return geometry
    if geom_type in {"MultiPoint", "GeometryCollection"}:
        return geometry.centroid
    if geom_type in {"LineString", "MultiLineString"}:
        return geometry.interpolate(0.5, normalized=True)
    return geometry.representative_point()


def load_population_data(filepath: str | Path) -> GeoDataFrame:
    """Load a cropped 100 m constrained population GeoTIFF into point weights.

    The raster is read with rasterio. All non-zero pixels are converted into point
    geometries using pixel center coordinates. The returned GeoDataFrame contains
    a numeric 'population' column that can be used as a demand weight in
    location-allocation or genetic algorithm workflows.
    """

    path = _validate_path(filepath, expected_suffix=".tif")

    try:
        with rasterio.open(path) as src:
            if src.count < 1:
                raise ValueError(f"Raster has no bands: {path}")

            band = src.read(1, masked=True)
            if band.size == 0:
                raise ValueError(f"Raster band is empty: {path}")

            # Keep only strictly positive demand pixels.
            valid_mask = (~np.ma.getmaskarray(band)) & (band > 0)
            row_indices, col_indices = np.where(valid_mask)
            if len(row_indices) == 0:
                raise ValueError(f"No non-zero population pixels found in {path}")

            values = band[row_indices, col_indices].astype(float)
            x_coords, y_coords = xy(src.transform, row_indices, col_indices, offset="center")
            geometry = gpd.points_from_xy(x_coords, y_coords, crs=src.crs)

            population_gdf = gpd.GeoDataFrame(
                {"population": values},
                geometry=geometry,
                crs=src.crs,
            )

            LOGGER.info("Loaded population points: %s", population_gdf.shape)
            return population_gdf
    except rasterio.errors.RasterioError as exc:
        raise RuntimeError(f"Failed to read population raster {path}: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"Unexpected error while loading population raster {path}: {exc}") from exc


def load_road_network(filepath: str | Path) -> GeoDataFrame:
    """Load a projected GeoPackage containing street edges and travel times."""

    path = _validate_path(filepath, expected_suffix=".gpkg")

    try:
        road_gdf = gpd.read_file(path)
    except Exception as exc:
        raise RuntimeError(f"Failed to read road network GeoPackage {path}: {exc}") from exc

    road_gdf = _ensure_geodataframe(road_gdf, "road_gdf")
    if road_gdf.crs is None:
        raise ValueError("Road network GeoDataFrame must have a defined CRS.")

    LOGGER.info("Loaded road network edges: %s", road_gdf.shape)
    return road_gdf


def extract_competitors(place_name: str) -> GeoDataFrame:
    """Extract existing supermarkets and convenience stores from OpenStreetMap.

    The function uses osmnx.features_from_place to query shop tags commonly used
    for competitors in dark store placement problems. The resulting geometries are
    normalized to point locations to make them easier to integrate with discrete
    optimization models.
    """

    if not isinstance(place_name, str) or not place_name.strip():
        raise ValueError("place_name must be a non-empty string.")

    try:
        tags = {"shop": ["supermarket", "convenience"]}
        features = ox.features_from_place(place_name, tags=tags)
    except Exception as exc:
        raise RuntimeError(f"Failed to query OpenStreetMap for '{place_name}': {exc}") from exc

    if features.empty:
        raise ValueError(f"No supermarket or convenience store features found for '{place_name}'.")

    if features.geometry is None:
        raise ValueError("OSM competitor data does not contain geometries.")

    competitor_gdf = features.copy()
    competitor_gdf = competitor_gdf[competitor_gdf.geometry.notna()].copy()
    competitor_gdf["geometry"] = competitor_gdf.geometry.apply(_pointify_geometry)
    competitor_gdf = competitor_gdf[competitor_gdf.geometry.notna()].copy()
    competitor_gdf = competitor_gdf.set_geometry("geometry")

    if competitor_gdf.crs is None:
        # OSMnx usually returns WGS84; set it explicitly if metadata is missing.
        competitor_gdf = competitor_gdf.set_crs("EPSG:4326", allow_override=True)

    LOGGER.info("Loaded competitor locations: %s", competitor_gdf.shape)
    return competitor_gdf


def prepare_ai_inputs(
    pop_gdf: GeoDataFrame,
    road_gdf: GeoDataFrame,
    comp_gdf: GeoDataFrame,
    target_crs: str | CRS = "EPSG:32636",
) -> Tuple[GeoDataFrame, GeoDataFrame, GeoDataFrame]:
    """Project all inputs to the same metric CRS for optimization workflows.

    A projected metric CRS is required so distance and travel-time calculations
    are meaningful in downstream location-allocation models and genetic algorithms.
    This wrapper harmonizes the CRS before any spatial joins, buffers, or distance
    operations are performed.
    """

    pop_gdf = _ensure_geodataframe(pop_gdf, "pop_gdf")
    road_gdf = _ensure_geodataframe(road_gdf, "road_gdf")
    comp_gdf = _ensure_geodataframe(comp_gdf, "comp_gdf")

    if pop_gdf.crs is None:
        raise ValueError("pop_gdf must have a defined CRS before projection.")
    if road_gdf.crs is None:
        raise ValueError("road_gdf must have a defined CRS before projection.")
    if comp_gdf.crs is None:
        raise ValueError("comp_gdf must have a defined CRS before projection.")

    try:
        target_crs_obj = CRS.from_user_input(target_crs)
        if not target_crs_obj.is_projected:
            raise ValueError(f"target_crs must be metric/projected, got: {target_crs_obj.to_string()}")

        # All inputs are reprojected into the same projected CRS so distances are
        # computed in linear units (meters for EPSG:32636), not degrees.
        pop_prepared = pop_gdf.to_crs(target_crs_obj)
        road_prepared = road_gdf.to_crs(target_crs_obj)
        comp_prepared = comp_gdf.to_crs(target_crs_obj)
    except Exception as exc:
        raise RuntimeError(f"Failed to project GeoDataFrames to {target_crs}: {exc}") from exc

    LOGGER.info("Prepared population dataframe shape: %s", pop_prepared.shape)
    LOGGER.info("Prepared road dataframe shape: %s", road_prepared.shape)
    LOGGER.info("Prepared competitor dataframe shape: %s", comp_prepared.shape)

    return pop_prepared, road_prepared, comp_prepared


__all__ = [
    "load_population_data",
    "load_road_network",
    "extract_competitors",
    "prepare_ai_inputs",
]


if __name__ == "__main__":
    print("Initializing Data Preparation Pipeline...")

    # 1. Define your local file paths 
    # (Update these if your downloaded files are named differently)
    POPULATION_RASTER = "nasr_city_constrained_population_100m.tif"
    ROAD_NETWORK = "target_area_roads.gpkg"
    TARGET_CITY = "Nasr City, Cairo, Egypt"

    try:
        # 2. Extract and Load
        print("\n--- Phase 1: Loading Datasets ---")
        raw_pop = load_population_data(POPULATION_RASTER)
        raw_roads = load_road_network(ROAD_NETWORK)
        raw_comps = extract_competitors(TARGET_CITY)

        # 3. Harmonize the CRS to EPSG:32636
        print("\n--- Phase 2: Harmonizing Coordinate Systems ---")
        pop_ready, roads_ready, comps_ready = prepare_ai_inputs(
            raw_pop, raw_roads, raw_comps, target_crs="EPSG:32636"
        )

        # 4. Final Output Verification
        print("\n--- Pipeline Execution Successful ---")
        print(f"Total Population Grid Points: {len(pop_ready)}")
        print(f"Total Navigable Road Edges:   {len(roads_ready)}")
        print(f"Total Competitor Locations:   {len(comps_ready)}")
        print("\nReady for Location-Allocation Optimization.")

    except FileNotFoundError as e:
        print(f"\n[ERROR] Missing File: Make sure you have downloaded and cropped the required data. Details: {e}")
    except Exception as e:
        print(f"\n[ERROR] Pipeline execution failed: {e}")