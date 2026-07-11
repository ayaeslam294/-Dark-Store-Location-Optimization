from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd

from dark_store_data_prep import load_population_data

DEFAULT_RASTER = Path("nasr_city_constrained_population_100m.tif")
DEFAULT_SAMPLE_SIZE = 10


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect the cropped Nasr City population raster as point data."
    )
    parser.add_argument(
        "--raster",
        type=Path,
        default=DEFAULT_RASTER,
        help=f"Path to the cropped population raster (default: {DEFAULT_RASTER})",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=f"Number of population points to display (default: {DEFAULT_SAMPLE_SIZE})",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional path to export the points as a CSV file.",
    )
    parser.add_argument(
        "--gpkg",
        type=Path,
        default=None,
        help="Optional path to export the points as a GeoPackage.",
    )
    return parser


def _population_summary(pop_gdf: gpd.GeoDataFrame) -> None:
    total_points = len(pop_gdf)
    population_series = pop_gdf["population"]

    print("\nPopulation Summary")
    print("------------------")
    print(f"Points:   {total_points:,}")
    print(f"CRS:      {pop_gdf.crs}")
    print(f"Bounds:   {pop_gdf.total_bounds}")
    print(f"Min:      {population_series.min():.3f}")
    print(f"Max:      {population_series.max():.3f}")
    print(f"Mean:     {population_series.mean():.3f}")
    print(f"Median:   {population_series.median():.3f}")
    print(f"Sum:      {population_series.sum():.3f}")
    print(f"Nonzero:  {(population_series > 0).sum():,}")


def _print_sample(pop_gdf: gpd.GeoDataFrame, sample_size: int) -> None:
    sample_size = max(1, sample_size)
    sample = pop_gdf[["population", "geometry"]].head(sample_size).copy()
    sample["x"] = sample.geometry.x
    sample["y"] = sample.geometry.y
    sample = sample.drop(columns=["geometry"])

    print("\nSample Points")
    print("-------------")
    print(sample.to_string(index=False))


def _export_csv(pop_gdf: gpd.GeoDataFrame, csv_path: Path) -> None:
    export_df = pd.DataFrame(
        {
            "population": pop_gdf["population"].astype(float),
            "x": pop_gdf.geometry.x,
            "y": pop_gdf.geometry.y,
        }
    )
    export_df.to_csv(csv_path, index=False)
    print(f"Exported CSV: {csv_path}")


def _export_gpkg(pop_gdf: gpd.GeoDataFrame, gpkg_path: Path) -> None:
    gpkg_path.parent.mkdir(parents=True, exist_ok=True)
    pop_gdf.to_file(gpkg_path, layer="population_points", driver="GPKG")
    print(f"Exported GeoPackage: {gpkg_path}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    pop_gdf = load_population_data(args.raster)

    print(f"Loaded raster: {args.raster}")
    _population_summary(pop_gdf)
    _print_sample(pop_gdf, args.sample_size)

    if args.csv is not None:
        _export_csv(pop_gdf, args.csv)
    if args.gpkg is not None:
        _export_gpkg(pop_gdf, args.gpkg)


if __name__ == "__main__":
    main()
