from __future__ import annotations
import copy
import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from deap import base, creator, tools
from pyproj import CRS
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import KDTree
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx


TARGET_CRS = "EPSG:32636"
WGS84_CRS = "EPSG:4326"
DEFAULT_POPULATION_RADIUS_METERS = 1000.0
DEFAULT_COMPETITOR_BUFFER_METERS = 1000.0
DEFAULT_GENERATIONS = 40
DEFAULT_POPULATION_SIZE = 48
DEFAULT_ELITE_SIZE = 4
DEFAULT_MUTATION_RATE = 0.25
DEFAULT_MUTATION_SCALE_METERS = 180.0
DEFAULT_RANDOM_SEED = 7
DEFAULT_ACCESS_SPEED_KPH = 30.0
DEFAULT_SERVICE_TIME_MINUTES = 5.0
DEFAULT_DELIVERY_TIME_TARGET_MINUTES = 20.0
MIN_SPATIAL_SEPARATION_METERS = 50.0


def _run_streamlit_app() -> None:
    """Render the lightweight Streamlit UI for the optimizer."""

    st.set_page_config(page_title="Dark Store Location Optimizer")
    st.title("Dark Store Location Optimizer")

    st.sidebar.header("Algorithm Weights")
    st.sidebar.slider("Coverage Weight", min_value=0, max_value=100, value=100)
    st.sidebar.slider("Time Weight", min_value=0, max_value=100, value=4)
    st.sidebar.slider("Competitor Penalty Weight", min_value=0, max_value=100, value=30)

    if st.button("Run Optimization"):
        st.success("Optimizer started!")


@dataclass(frozen=True)
class DarkStoreInputs:
    """Container for all prepared datasets used by the optimizer."""

    road_graph: nx.DiGraph
    node_ids: np.ndarray
    node_coords: np.ndarray
    node_tree: KDTree
    road_matrix: object
    demand_node_indices: np.ndarray
    demand_weights: np.ndarray
    demand_coords: np.ndarray
    population_tree: KDTree
    demand_access_minutes: np.ndarray
    competitor_coords: np.ndarray
    seed_coords: np.ndarray
    target_crs: CRS
    coverage_radius_meters: float
    competitor_buffer_meters: float


@dataclass(frozen=True)
class FitnessBatch:
    """Vectorized KPI outputs for a batch of candidate store locations."""

    avg_delivery_time_min: np.ndarray
    coverage_population: np.ndarray
    coverage_ratio: np.ndarray
    competitor_penalty: np.ndarray
    fitness: np.ndarray


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line interface for the optimizer."""

    parser = argparse.ArgumentParser(description="Dark store GA optimizer for Nasr City.")
    parser.add_argument("--base-dir", type=Path, default=Path.cwd(), help="Workspace directory containing the datasets.")
    parser.add_argument("--roads-csv", type=Path, default=Path("nasr_city_roads_cleaned.csv"), help="Projected road edge CSV.")
    parser.add_argument("--road-nodes-csv", type=Path, default=Path("nasr_city_road_nodes.csv"), help="Projected road node CSV.")
    parser.add_argument("--dark-store-grid-csv", type=Path, default=Path("nasr_city_dark_store_data.csv"), help="Grid with suitability scores.")
    parser.add_argument("--population-csv", type=Path, default=Path("nasr_city_population_points.csv"), help="Population demand points in WGS84.")
    parser.add_argument("--competitor-xlsx", type=Path, default=Path("places.xlsx"), help="Competitor workbook in WGS84.")
    parser.add_argument("--target-crs", default=TARGET_CRS, help="Projected CRS used for routing and distance calculations.")
    parser.add_argument("--population-radius-m", type=float, default=DEFAULT_POPULATION_RADIUS_METERS, help="Radius used for coverage queries.")
    parser.add_argument("--competitor-buffer-m", type=float, default=DEFAULT_COMPETITOR_BUFFER_METERS, help="Penalty radius for competitors.")
    parser.add_argument("--population-size", type=int, default=DEFAULT_POPULATION_SIZE, help="GA population size.")
    parser.add_argument("--generations", type=int, default=DEFAULT_GENERATIONS, help="GA generations.")
    parser.add_argument("--elite-size", type=int, default=DEFAULT_ELITE_SIZE, help="Number of elite individuals preserved per generation.")
    parser.add_argument("--mutation-rate", type=float, default=DEFAULT_MUTATION_RATE, help="Per-coordinate mutation probability.")
    parser.add_argument("--mutation-scale", type=float, default=DEFAULT_MUTATION_SCALE_METERS, help="Gaussian mutation scale in meters.")
    parser.add_argument("--tournament-size", type=int, default=3, help="Tournament size used for parent selection.")
    parser.add_argument("--candidate-seed-count", type=int, default=10, help="Number of top suitability grids used as seed anchors.")
    parser.add_argument("--coverage-weight", type=float, default=100.0, help="Weight applied to coverage in the scalar fitness score.")
    parser.add_argument("--time-weight", type=float, default=4.0, help="Weight applied to average delivery time in the scalar fitness score.")
    parser.add_argument("--competitor-weight", type=float, default=30.0, help="Weight applied to competitor penalty in the scalar fitness score.")
    parser.add_argument("--delivery-time-target-min", type=float, default=DEFAULT_DELIVERY_TIME_TARGET_MINUTES, help="Target delivery time used to normalize the time score.")
    parser.add_argument("--output-best-csv", type=Path, default=Path("best_dark_store_solution.csv"), help="Optional output CSV for the best solution.")
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED, help="Random seed for reproducibility.")
    return parser


def _ensure_projected(target_crs: str) -> CRS:
    """Resolve and validate the projected target CRS."""

    crs = CRS.from_user_input(target_crs)
    if not crs.is_projected:
        raise ValueError(f"Target CRS must be projected, got {crs.to_string()}.")
    return crs


def _resolve_path(base_dir: Path, path: Path) -> Path:
    """Resolve a dataset path relative to the workspace directory."""

    return path if path.is_absolute() else base_dir / path


def _detect_coordinate_columns(columns: Sequence[str]) -> tuple[str, str] | None:
    """Detect a pair of coordinate columns from a dataframe schema."""

    lower_map = {column.lower(): column for column in columns}
    candidates = [
        ("center_lon", "center_lat"),
        ("longitude", "latitude"),
        ("lon", "lat"),
        ("lng", "lat"),
        ("x", "y"),
    ]
    for lon_key, lat_key in candidates:
        if lon_key in lower_map and lat_key in lower_map:
            return lower_map[lon_key], lower_map[lat_key]
    return None


def _to_metric_gdf(frame: pd.DataFrame, lon_col: str, lat_col: str, target_crs: CRS) -> gpd.GeoDataFrame:
    """Convert a WGS84 dataframe into a projected GeoDataFrame."""

    gdf = gpd.GeoDataFrame(frame.copy(), geometry=gpd.points_from_xy(frame[lon_col], frame[lat_col]), crs=WGS84_CRS)
    return gdf.to_crs(target_crs)


def _load_population_points(path: Path, target_crs: CRS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load population demand points and project them to the metric CRS."""

    frame = pd.read_csv(path)
    columns = _detect_coordinate_columns(frame.columns)
    if columns is None:
        raise ValueError(f"Population CSV {path.name} must expose coordinate columns.")
    lon_col, lat_col = columns
    if "population" not in {column.lower() for column in frame.columns}:
        raise ValueError(f"Population CSV {path.name} must expose a population column.")

    lower_map = {column.lower(): column for column in frame.columns}
    pop_col = lower_map["population"]
    gdf = _to_metric_gdf(frame, lon_col, lat_col, target_crs)
    weights = pd.to_numeric(gdf[pop_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if np.any(weights < 0):
        raise ValueError("Population weights must be non-negative.")
    coords = np.column_stack([gdf.geometry.x.to_numpy(dtype=float), gdf.geometry.y.to_numpy(dtype=float)])
    if coords.size == 0 or float(weights.sum()) <= 0:
        raise ValueError(f"Population CSV {path.name} contains no valid demand points.")
    return coords, weights, gdf.geometry.to_numpy()


def _load_dark_store_seeds(path: Path, target_crs: CRS, seed_count: int) -> np.ndarray:
    """Load the highest-suitability grid centers and project them to the metric CRS."""

    frame = pd.read_csv(path)
    required = {"center_lat", "center_lon", "suitability_score"}
    if not required.issubset({column.lower() for column in frame.columns}):
        raise ValueError(f"Dark store grid {path.name} must include center_lat, center_lon, and suitability_score.")

    lower_map = {column.lower(): column for column in frame.columns}
    lat_col = lower_map["center_lat"]
    lon_col = lower_map["center_lon"]
    score_col = lower_map["suitability_score"]
    gdf = _to_metric_gdf(frame, lon_col, lat_col, target_crs)
    ordered = gdf.sort_values(score_col, ascending=False).head(max(1, seed_count))
    coords = np.column_stack([ordered.geometry.x.to_numpy(dtype=float), ordered.geometry.y.to_numpy(dtype=float)])
    if coords.size == 0:
        raise ValueError(f"Dark store grid {path.name} produced no usable seed coordinates.")
    return coords


def _load_competitors_from_excel(path: Path, target_crs: CRS) -> np.ndarray:
    """Load competitor points from every sheet that exposes coordinate columns."""

    if not path.exists():
        raise FileNotFoundError(f"Competitor workbook not found: {path}")

    sheets = pd.read_excel(path, sheet_name=None)
    projected_frames: list[gpd.GeoDataFrame] = []
    for sheet_name, frame in sheets.items():
        columns = _detect_coordinate_columns(frame.columns)
        if columns is None:
            continue
        lon_col, lat_col = columns
        projected_frames.append(_to_metric_gdf(frame, lon_col, lat_col, target_crs))

    if not projected_frames:
        raise ValueError(f"No coordinate-bearing sheets were found in {path.name}.")

    combined = pd.concat(projected_frames, ignore_index=True)
    coords = np.column_stack([combined.geometry.x.to_numpy(dtype=float), combined.geometry.y.to_numpy(dtype=float)])
    finite_mask = np.isfinite(coords).all(axis=1)
    coords = coords[finite_mask]
    if coords.size == 0:
        raise ValueError(f"Competitor workbook {path.name} contains no usable competitor coordinates.")
    return coords


def _read_road_tables(roads_csv: Path, nodes_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the edge and node tables used to build the routing graph."""

    edges = pd.read_csv(roads_csv)
    nodes = pd.read_csv(nodes_csv)
    for required in ("u", "v", "travel_time", "length", "speed_kph"):
        if required not in edges.columns:
            raise ValueError(f"Road edges CSV {roads_csv.name} is missing required column: {required}")
    for required in ("node_id", "x", "y"):
        if required not in nodes.columns:
            raise ValueError(f"Road nodes CSV {nodes_csv.name} is missing required column: {required}")
    return edges, nodes


def _edge_travel_time_minutes(frame: pd.DataFrame) -> np.ndarray:
    """Build a travel-time vector, falling back to length and speed when needed."""

    travel_time = pd.to_numeric(frame["travel_time"], errors="coerce")
    length_m = pd.to_numeric(frame["length"], errors="coerce").fillna(0.0)
    speed_kph = pd.to_numeric(frame["speed_kph"], errors="coerce").fillna(0.0)
    fallback = np.divide(
        length_m.to_numpy(dtype=float),
        np.maximum(speed_kph.to_numpy(dtype=float), 1.0) * (1000.0 / 60.0),
        out=np.full(len(frame), np.inf, dtype=float),
        where=np.maximum(speed_kph.to_numpy(dtype=float), 1.0) > 0,
    )
    raw_travel_time = travel_time.fillna(pd.Series(fallback, index=frame.index)).to_numpy(dtype=float)
    fallback_minutes = np.asarray(fallback, dtype=float)

    finite_raw = raw_travel_time[np.isfinite(raw_travel_time)]
    finite_fallback = fallback_minutes[np.isfinite(fallback_minutes)]
    if finite_raw.size and finite_fallback.size:
        raw_median = float(np.median(finite_raw))
        fallback_median = float(np.median(finite_fallback))
        if fallback_median > 0 and raw_median > fallback_median * 5.0:
            raw_travel_time = raw_travel_time / 60.0

    return raw_travel_time


def _build_road_graph(roads_csv: Path, nodes_csv: Path) -> tuple[nx.DiGraph, np.ndarray, np.ndarray, KDTree, object]:
    """Construct a directed routing graph and its scipy sparse adjacency matrix."""

    edges, nodes = _read_road_tables(roads_csv, nodes_csv)
    node_ids = nodes["node_id"].to_numpy(dtype=np.int64)
    node_coords = nodes[["x", "y"]].to_numpy(dtype=float)

    node_lookup = nodes.set_index("node_id")[["x", "y"]]
    if not edges["u"].isin(node_lookup.index).all() or not edges["v"].isin(node_lookup.index).all():
        raise ValueError("Road edge table contains endpoints that do not resolve to road node coordinates.")

    merged = edges.copy()
    merged["u_x"] = merged["u"].map(node_lookup["x"])
    merged["u_y"] = merged["u"].map(node_lookup["y"])
    merged["v_x"] = merged["v"].map(node_lookup["x"])
    merged["v_y"] = merged["v"].map(node_lookup["y"])

    merged["time_min"] = _edge_travel_time_minutes(merged)
    grouped = merged.groupby(["u", "v"], as_index=False).agg(
        {
            "time_min": "min",
            "length": "min",
            "speed_kph": "median",
        }
    )

    graph = nx.DiGraph()
    for node_id, x_coord, y_coord in zip(node_ids, node_coords[:, 0], node_coords[:, 1]):
        graph.add_node(int(node_id), x=float(x_coord), y=float(y_coord))

    for row in grouped.itertuples(index=False):
        graph.add_edge(
            int(row.u),
            int(row.v),
            time_min=float(row.time_min),
            length_m=float(row.length),
            speed_kph=float(row.speed_kph) if not math.isnan(float(row.speed_kph)) else np.nan,
        )

    node_tree = KDTree(node_coords)
    road_matrix = nx.to_scipy_sparse_array(graph, nodelist=list(map(int, node_ids)), weight="time_min", dtype=float, format="csr")
    return graph, node_ids, node_coords, node_tree, road_matrix


def _prepare_inputs(args: argparse.Namespace) -> DarkStoreInputs:
    """Load, harmonize, and package all datasets needed by the GA."""

    target_crs = _ensure_projected(args.target_crs)
    base_dir = args.base_dir

    roads_csv = _resolve_path(base_dir, args.roads_csv)
    road_nodes_csv = _resolve_path(base_dir, args.road_nodes_csv)
    dark_store_grid_csv = _resolve_path(base_dir, args.dark_store_grid_csv)
    population_csv = _resolve_path(base_dir, args.population_csv)
    competitor_xlsx = _resolve_path(base_dir, args.competitor_xlsx)

    road_graph, node_ids, node_coords, node_tree, road_matrix = _build_road_graph(roads_csv, road_nodes_csv)

    demand_coords, demand_weights, _ = _load_population_points(population_csv, target_crs)
    demand_node_indices = node_tree.query(demand_coords)[1].astype(int)
    demand_access_minutes = node_tree.query(demand_coords)[0].astype(float) / (DEFAULT_ACCESS_SPEED_KPH * (1000.0 / 60.0))
    population_tree = KDTree(demand_coords)

    seed_coords = _load_dark_store_seeds(dark_store_grid_csv, target_crs, args.candidate_seed_count)
    competitor_coords = _load_competitors_from_excel(competitor_xlsx, target_crs)

    return DarkStoreInputs(
        road_graph=road_graph,
        node_ids=node_ids,
        node_coords=node_coords,
        node_tree=node_tree,
        road_matrix=road_matrix,
        demand_node_indices=demand_node_indices,
        demand_weights=demand_weights,
        demand_coords=demand_coords,
        population_tree=population_tree,
        demand_access_minutes=demand_access_minutes,
        competitor_coords=competitor_coords,
        seed_coords=seed_coords,
        target_crs=target_crs,
        coverage_radius_meters=float(args.population_radius_m),
        competitor_buffer_meters=float(args.competitor_buffer_m),
    )


def _seed_population(
    rng: np.random.Generator,
    seed_coords: np.ndarray,
    population_size: int,
    mutation_scale: float,
) -> np.ndarray:
    """Create a smartly seeded first-generation candidate population."""

    if seed_coords.size == 0:
        raise ValueError("Seed coordinates are empty.")

    pop = np.empty((population_size, 2), dtype=float)
    seed_count = min(len(seed_coords), population_size)
    pop[:seed_count] = seed_coords[:seed_count]

    for index in range(seed_count, population_size):
        anchor = seed_coords[rng.integers(0, seed_count)]
        pop[index] = anchor + rng.normal(0.0, mutation_scale, size=2)

    return pop


def _normalize_priority_weights(coverage_weight: float, time_weight: float, competitor_weight: float) -> tuple[float, float, float]:
    """Normalize user-entered priorities so they behave as relative importance values."""

    weights = np.asarray([coverage_weight, time_weight, competitor_weight], dtype=float)
    total = float(weights.sum())
    if total <= 0:
        return 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0
    normalized = weights / total
    return float(normalized[0]), float(normalized[1]), float(normalized[2])


def _evaluate_batch(
    candidate_coords: np.ndarray,
    inputs: DarkStoreInputs,
    time_weight: float,
    competitor_weight: float,
    coverage_weight: float,
    delivery_time_target_min: float,
) -> FitnessBatch:
    """Vectorized KPI evaluation for a batch of candidate store locations."""

    if candidate_coords.size == 0:
        raise ValueError("Candidate coordinate batch is empty.")

    candidate_node_indices = inputs.node_tree.query(candidate_coords)[1].astype(int)
    unique_nodes, inverse = np.unique(candidate_node_indices, return_inverse=True)

    graph_distances = dijkstra(inputs.road_matrix, directed=True, indices=unique_nodes, return_predecessors=False)
    if graph_distances.ndim == 1:
        graph_distances = graph_distances[np.newaxis, :]

    demand_distances = graph_distances[:, inputs.demand_node_indices]
    finite_mask = np.isfinite(demand_distances)
    demand_weights = inputs.demand_weights[np.newaxis, :]
    weighted_distances = np.where(finite_mask, demand_distances, 0.0) * demand_weights
    reachable_population = np.where(finite_mask, demand_weights, 0.0).sum(axis=1)
    weighted_sum = weighted_distances.sum(axis=1)
    avg_delivery_time_unique = np.divide(
        weighted_sum,
        reachable_population,
        out=np.full(len(unique_nodes), np.inf, dtype=float),
        where=reachable_population > 0,
    )

    candidate_tree = KDTree(candidate_coords)
    coverage_sparse = candidate_tree.sparse_distance_matrix(
        inputs.population_tree,
        inputs.coverage_radius_meters,
        output_type="coo_matrix",
    )
    if coverage_sparse.nnz == 0:
        coverage_population = np.zeros(len(candidate_coords), dtype=float)
    else:
        coverage_sparse.data = np.ones_like(coverage_sparse.data, dtype=float)
        coverage_matrix = coverage_sparse.tocsr()
        coverage_population = np.asarray(coverage_matrix @ inputs.demand_weights, dtype=float).ravel()

    if inputs.competitor_coords.size == 0:
        competitor_penalty = np.zeros(len(candidate_coords), dtype=float)
    else:
        competitor_tree = KDTree(inputs.competitor_coords)
        nearest_distance = competitor_tree.query(candidate_coords)[0]
        normalized_gap = np.clip((inputs.competitor_buffer_meters - nearest_distance) / inputs.competitor_buffer_meters, 0.0, None)
        competitor_penalty = np.square(normalized_gap)

    total_population = float(inputs.demand_weights.sum())
    coverage_ratio = coverage_population / total_population if total_population > 0 else np.zeros_like(coverage_population)
    reachable_ratio = reachable_population / total_population if total_population > 0 else np.zeros_like(reachable_population)
    unreachable_share = 1.0 - np.clip(reachable_ratio, 0.0, 1.0)

    demand_access_total = np.asarray(inputs.demand_access_minutes, dtype=float).sum()
    demand_access_avg = demand_access_total / total_population if total_population > 0 else 0.0

    avg_delivery_time = avg_delivery_time_unique[inverse]
    avg_delivery_time = np.where(np.isfinite(avg_delivery_time), avg_delivery_time, 1e6)
    effective_delivery_time = avg_delivery_time + demand_access_avg + DEFAULT_SERVICE_TIME_MINUTES
    effective_delivery_time = effective_delivery_time * (1.0 + 4.0 * unreachable_share[inverse])
    coverage_priority, time_priority, competitor_priority = _normalize_priority_weights(coverage_weight, time_weight, competitor_weight)
    delivery_time_target_min = max(1e-6, float(delivery_time_target_min))

    coverage_score = np.clip(coverage_ratio * 100.0, 0.0, 100.0)
    time_score = np.clip((1.0 - (effective_delivery_time / delivery_time_target_min)) * 100.0, 0.0, 100.0)
    competitor_score = np.clip((1.0 - competitor_penalty) * 100.0, 0.0, 100.0)

    fitness = coverage_priority * coverage_score + time_priority * time_score + competitor_priority * competitor_score
    return FitnessBatch(
        avg_delivery_time_min=avg_delivery_time,
        coverage_population=coverage_population,
        coverage_ratio=coverage_ratio,
        competitor_penalty=competitor_penalty,
        fitness=fitness,
    )


def _make_deap_toolbox(inputs: DarkStoreInputs, args: argparse.Namespace) -> base.Toolbox:
    """Build a DEAP toolbox for the scalarized GA search."""

    if not hasattr(creator, "FitnessDarkStore"):
        creator.create("FitnessDarkStore", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "IndividualDarkStore"):
        creator.create("IndividualDarkStore", list, fitness=creator.FitnessDarkStore)

    toolbox = base.Toolbox()
    rng = np.random.default_rng(args.random_seed)
    initial_population = _seed_population(rng, inputs.seed_coords, args.population_size, args.mutation_scale)
    toolbox.initial_population = initial_population
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("select", tools.selTournament, tournsize=args.tournament_size)
    toolbox.register("clone", copy.deepcopy)
    toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=args.mutation_scale, indpb=args.mutation_rate)
    return toolbox


def _assign_fitness(population: list[creator.IndividualDarkStore], inputs: DarkStoreInputs, args: argparse.Namespace) -> pd.DataFrame:
    """Evaluate and assign scalar fitness plus KPI metrics for every individual."""

    coords = np.asarray(population, dtype=float)
    batch = _evaluate_batch(
        coords,
        inputs,
        args.time_weight,
        args.competitor_weight,
        args.coverage_weight,
        args.delivery_time_target_min,
    )
    metrics = pd.DataFrame(
        {
            "x": coords[:, 0],
            "y": coords[:, 1],
            "fitness": batch.fitness,
            "avg_delivery_time_min": batch.avg_delivery_time_min,
            "coverage_population": batch.coverage_population,
            "coverage_ratio": batch.coverage_ratio,
            "competitor_penalty": batch.competitor_penalty,
        }
    )
    for individual, fitness_value in zip(population, batch.fitness, strict=True):
        individual.fitness.values = (float(fitness_value),)
    return metrics


def _breed_next_generation(
    population: list[creator.IndividualDarkStore],
    toolbox: base.Toolbox,
    elite_size: int,
) -> list[creator.IndividualDarkStore]:
    """Create the next GA generation using elitism, crossover, and mutation."""

    elites = tools.selBest(population, k=min(elite_size, len(population)))
    offspring = [toolbox.clone(individual) for individual in toolbox.select(population, k=len(population) - len(elites))]

    for left in range(0, len(offspring) - 1, 2):
        if np.random.random() < 0.9:
            toolbox.mate(offspring[left], offspring[left + 1])
            del offspring[left].fitness.values
            del offspring[left + 1].fitness.values

    for individual in offspring:
        if np.random.random() < 0.5:
            toolbox.mutate(individual)
            del individual.fitness.values
            individual[0] = float(individual[0])
            individual[1] = float(individual[1])

    return elites + offspring


def _select_top_distinct_solutions(
    results: pd.DataFrame,
    max_solutions: int = 5,
    min_distance_meters: float = MIN_SPATIAL_SEPARATION_METERS,
) -> pd.DataFrame:
    """Pick the highest-fitness solutions that are spatially separated."""

    if results.empty:
        raise ValueError("No evaluated solutions were produced by the optimizer.")

    ordered = results.sort_values("fitness", ascending=False).reset_index(drop=True)
    selected_rows: list[pd.Series] = []
    selected_coords: list[np.ndarray] = []

    for _, row in ordered.iterrows():
        candidate = np.asarray([float(row["x"]), float(row["y"])], dtype=float)
        if selected_coords:
            distances = np.linalg.norm(np.vstack(selected_coords) - candidate, axis=1)
            if np.any(distances < min_distance_meters):
                continue

        selected_rows.append(row)
        selected_coords.append(candidate)
        if len(selected_rows) >= max_solutions:
            break

    if not selected_rows:
        raise ValueError("Unable to identify any spatially distinct solutions.")

    ranked = pd.DataFrame(selected_rows).reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1, dtype=int))
    return ranked


def run_optimizer(args: argparse.Namespace) -> pd.DataFrame:
    """Run the GA and return the top spatially distinct solutions ranked by fitness."""

    inputs = _prepare_inputs(args)
    toolbox = _make_deap_toolbox(inputs, args)
    rng = np.random.default_rng(args.random_seed)

    population = [creator.IndividualDarkStore(coords.tolist()) for coords in toolbox.initial_population]

    history: list[pd.DataFrame] = []
    for generation in range(args.generations):
        invalid = [individual for individual in population if not individual.fitness.valid]
        if invalid:
            metrics = _assign_fitness(invalid, inputs, args)
            metrics.insert(0, "generation", generation)
            history.append(metrics)

        population = tools.selBest(population, k=len(population))
        if generation < args.generations - 1:
            population = _breed_next_generation(population, toolbox, args.elite_size)
            for individual in population:
                individual[0] = float(individual[0])
                individual[1] = float(individual[1])
                if np.isfinite(individual[0]) is False or np.isfinite(individual[1]) is False:
                    individual[0], individual[1] = float(inputs.seed_coords[0, 0]), float(inputs.seed_coords[0, 1])

    final_metrics = _assign_fitness(population, inputs, args)
    final_metrics.insert(0, "generation", args.generations - 1)
    history.append(final_metrics)
    all_results = pd.concat(history, ignore_index=True)
    return _select_top_distinct_solutions(all_results, max_solutions=5)


def _project_back_to_wgs84(x: float, y: float, target_crs: CRS) -> tuple[float, float]:
    """Transform a metric coordinate back to WGS84 for reporting."""

    metric_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy([x], [y]), crs=target_crs)
    wgs84 = metric_gdf.to_crs(WGS84_CRS)
    return float(wgs84.geometry.x.iloc[0]), float(wgs84.geometry.y.iloc[0])


def main() -> None:
    """Entry point for command-line execution."""

    if get_script_run_ctx() is not None:
        _run_streamlit_app()
        return

    parser = build_parser()
    args = parser.parse_args()

    results = run_optimizer(args)
    target_crs = _ensure_projected(args.target_crs)
    output = results.copy()
    output[["lon", "lat"]] = output.apply(
        lambda row: pd.Series(_project_back_to_wgs84(float(row["x"]), float(row["y"]), target_crs)),
        axis=1,
    )
    output_path = _resolve_path(args.base_dir, args.output_best_csv)
    output.to_csv(output_path, index=False)

    print("Top 5 solutions:")
    print(output.to_string(index=False))
    print(f"Saved ranked solutions to: {output_path}")


if __name__ == "__main__":
    main()