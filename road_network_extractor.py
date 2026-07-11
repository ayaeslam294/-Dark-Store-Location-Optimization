"""Extract and process a drivable road network for Nasr City, Cairo, Egypt.

The script fetches the drivable street network from OpenStreetMap with osmnx,
imputes free-flow speeds, calculates edge travel times, converts the graph to
GeoDataFrames, projects the edges to EPSG:32636, and saves the projected edges
as a GeoPackage in the current working directory.
"""

from __future__ import annotations

import time
from pathlib import Path

import geopandas as gpd
import osmnx as ox
from pyproj import CRS

TARGET_PLACE = "Nasr City, Cairo, Egypt"
TARGET_CRS = "EPSG:32636"
OUTPUT_FILENAME = "target_area_roads.gpkg"
REQUEST_TIMEOUT_SECONDS = 60
MAX_FETCH_ATTEMPTS = 3
RETRY_WAIT_SECONDS = 5


def _fetch_drivable_graph(place_name: str) -> object:
    """Fetch the drivable graph with a bounded timeout and retry logic."""

    original_timeout = ox.settings.requests_timeout
    ox.settings.requests_timeout = REQUEST_TIMEOUT_SECONDS

    try:
        last_error: Exception | None = None

        for attempt in range(1, MAX_FETCH_ATTEMPTS + 1):
            print(f"OSM fetch attempt {attempt} of {MAX_FETCH_ATTEMPTS}...")
            try:
                return ox.graph_from_place(place_name, network_type="drive")
            except Exception as exc:
                last_error = exc
                print(f"Attempt {attempt} failed: {exc}")

                if attempt < MAX_FETCH_ATTEMPTS:
                    print(f"Waiting {RETRY_WAIT_SECONDS} seconds before retrying...")
                    time.sleep(RETRY_WAIT_SECONDS)

        raise RuntimeError(
            f"Unable to fetch drivable network for '{place_name}' after {MAX_FETCH_ATTEMPTS} attempts."
        ) from last_error
    finally:
        ox.settings.requests_timeout = original_timeout


def extract_drivable_road_network(
    place_name: str,
    target_crs: str = TARGET_CRS,
    output_filename: str = OUTPUT_FILENAME,
) -> Path:
    """Fetch, process, project, and save a drivable road network."""

    print(f"Defining target area: {place_name}")
    print("Fetching drivable road network from OpenStreetMap...")
    graph = _fetch_drivable_graph(place_name)

    print("Imputing free-flow travel speeds on edges...")
    graph = ox.add_edge_speeds(graph)

    print("Calculating travel times for edges...")
    graph = ox.add_edge_travel_times(graph)

    print("Converting graph to node and edge GeoDataFrames...")
    nodes_gdf, edges_gdf = ox.graph_to_gdfs(graph, nodes=True, edges=True)

    print(f"Projecting edge GeoDataFrame to metric CRS ({target_crs})...")
    metric_crs = CRS.from_user_input(target_crs)
    if not metric_crs.is_projected:
        raise ValueError(f"Target CRS must be projected, got: {metric_crs.to_string()}")

    if edges_gdf.crs is None:
        raise ValueError("Edges GeoDataFrame has no CRS and cannot be projected safely.")

    projected_edges = edges_gdf.to_crs(metric_crs)
    projected_nodes = nodes_gdf.to_crs(metric_crs)
    projected_nodes["x"] = projected_nodes.geometry.x
    projected_nodes["y"] = projected_nodes.geometry.y

    output_path = Path.cwd() / output_filename
    print(f"Saving projected edges to {output_path.name}...")
    projected_edges.to_file(output_path, layer="edges", driver="GPKG")
    print("Saving projected nodes to the same GeoPackage...")
    projected_nodes.to_file(output_path, layer="nodes", driver="GPKG")

    print("Road network extraction complete.")
    print(f"Nodes extracted: {len(nodes_gdf):,}")
    print(f"Edges extracted: {len(edges_gdf):,}")
    print(f"Projected edges saved to: {output_path}")

    return output_path


def main() -> None:
    """Main execution entry point."""

    print("Starting road network extraction pipeline...")
    try:
        extract_drivable_road_network(TARGET_PLACE)
    except Exception as exc:
        print("[ERROR] Road network extraction failed.")
        print(f"Details: {exc}")


if __name__ == "__main__":
    main()
