"""
Utility functions for network node classification and snapping.

Exports:

    nodes_on(gdf_nodes, gdf_links, query)
        Boolean Series — True if node N appears in A or B of any link
        matching the pandas .query() string.

        e.g. gdf_nodes["Freeway"] = nodes_on(gdf_nodes, gdf_links, "FT_2023 in [20, 22, 23]")

    count_node_links(gdf_nodes, gdf_links)
    snap_nodes(gdf_nodes, gdf_centerlines_filtered, node_mask, max_distance_m, label, ...)
    snap_to_gtfs_stops(gdf_nodes, gdf_stops, node_mask, max_distance_m, ...)

All filtering logic lives entirely in the calling notebook — not here.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
from shapely.strtree import STRtree

# ---------------------------------------------------------------------------
# Classification helpers — monkey-patched onto GeoDataFrame
# ---------------------------------------------------------------------------


def nodes_on(gdf_nodes: gpd.GeoDataFrame, gdf_links: gpd.GeoDataFrame, query: str) -> pd.Series:
    """
    Return a boolean Series: True if the node's N appears in A or B of
    any link matching the pandas .query() string.

    Parameters
    ----------
    gdf_nodes : GeoDataFrame
        Nodes layer. Must contain column N.
    gdf_links : GeoDataFrame
        Full links layer. Must contain columns A and B, plus any columns
        referenced in query.
    query : str
        pandas .query() string to filter links.
        e.g. "FT_2023 in [20, 22, 23]"
        e.g. "FT_2023 == 1"

    Returns
    -------
    pd.Series
        Boolean Series aligned to gdf_nodes index.

    Example
    -------
    gdf_nodes["Freeway"] = nodes_on(gdf_nodes, gdf_links, "FT_2023 in [20, 22, 23]")
    gdf_nodes["FixedTransit"] = (
        nodes_on(gdf_nodes, gdf_links, "FT_2023 in [70, 80]")
        | gdf_nodes["N"].between(10_000, 19_999)
    )

    """
    filtered = gdf_links.query(query)
    connected = set(filtered["A"]).union(filtered["B"])
    return gdf_nodes["N"].isin(connected)


def count_node_links(gdf_nodes: gpd.GeoDataFrame, gdf_links: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Add a LinkCount column to the nodes GeoDataFrame, counting how many
    link endpoints (A or B) match each node's N value.

    Parameters
    ----------
    gdf_nodes : GeoDataFrame
        Nodes layer. Must contain column N.
    gdf_links : GeoDataFrame
        Links layer. Must contain columns A and B.

    Returns
    -------
    GeoDataFrame
        Copy of gdf_nodes with LinkCount column appended.

    """
    result = gdf_nodes.copy()
    link_counts = pd.concat([gdf_links["A"], gdf_links["B"]]).value_counts()
    result["LinkCount"] = result["N"].map(link_counts).fillna(0).astype(int)
    return result


# ---------------------------------------------------------------------------
# Snapping helpers (private)
# ---------------------------------------------------------------------------


def _extract_endpoints(gdf_lines: gpd.GeoDataFrame, precision: int = 7) -> gpd.GeoSeries:
    """
    Extract deduplicated start and end points from a LineString GeoDataFrame.

    MultiLineString geometries are exploded into simple LineStrings first.
    Coordinates are rounded to `precision` decimal places before deduplication
    to eliminate floating-point noise from source data.

    Parameters
    ----------
    gdf_lines : GeoDataFrame
        LineString or MultiLineString layer.
    precision : int, optional
        Decimal places to round coordinates before deduplication.
        Default is 7 (~1cm precision in UTM).

    Returns
    -------
    GeoSeries
        Deduplicated endpoint Points in the same CRS as gdf_lines.

    """
    lines = gdf_lines.explode(index_parts=False)

    def rounded_endpoints(geom):
        start = geom.coords[0]
        end = geom.coords[-1]
        return [
            Point(round(start[0], precision), round(start[1], precision)),
            Point(round(end[0], precision), round(end[1], precision)),
        ]

    all_points = [pt for geom in lines.geometry for pt in rounded_endpoints(geom)]
    return gpd.GeoSeries(all_points, crs=gdf_lines.crs).drop_duplicates()


def _snap_nodes_to_points(
    gdf_nodes: gpd.GeoDataFrame,
    snap_targets: gpd.GeoSeries,
    max_distance_m: float,
    crs_projected: str,
) -> tuple[list, list, list]:
    """
    Snap nodes to the nearest point in snap_targets using a spatial index.

    Reprojects to crs_projected for metric distance calculations.
    Returns snapped geometries in the original CRS of gdf_nodes.

    Returns
    -------
    tuple of (snapped_geoms, snap_distances_m, snapped_flags)
        snapped_geoms    : list of Point geometries (original CRS)
        snap_distances_m : list of float distances in metres
        snapped_flags    : list of bool — False if node exceeded threshold

    """
    nodes_proj = gdf_nodes.to_crs(crs_projected)
    targets_proj = snap_targets.to_crs(crs_projected)
    tree = STRtree(targets_proj.values)

    snapped_geoms = []
    snap_distances_m = []
    snapped_flags = []

    # Zip original and projected geometries so exceeded-threshold nodes
    # return their original geometry without any index gymnastics
    for orig_geom, proj_geom in zip(gdf_nodes.geometry, nodes_proj.geometry):
        nearest_idx = tree.nearest(proj_geom)
        nearest_geom = targets_proj.iloc[nearest_idx]
        distance_m = proj_geom.distance(nearest_geom)

        if distance_m <= max_distance_m:
            snapped_orig = (
                gpd.GeoSeries([nearest_geom], crs=crs_projected).to_crs(gdf_nodes.crs).iloc[0]
            )
            snapped_geoms.append(snapped_orig)
            snapped_flags.append(True)
        else:
            snapped_geoms.append(orig_geom)  # original CRS, no reprojection needed
            snapped_flags.append(False)

        snap_distances_m.append(round(distance_m, 2))

    return snapped_geoms, snap_distances_m, snapped_flags


# ---------------------------------------------------------------------------
# Public snapping functions
# ---------------------------------------------------------------------------


def snap_nodes(
    gdf_nodes: gpd.GeoDataFrame,
    gdf_centerlines_filtered: gpd.GeoDataFrame,
    node_mask: pd.Series,
    max_distance_m: float,
    label: str,
    precision: int = 7,
    crs_projected: str = "EPSG:26912",
) -> gpd.GeoDataFrame:
    """
    Snap a subset of nodes to the nearest endpoint of a pre-filtered
    centerline GeoDataFrame. Nodes already snapped by a previous call
    are automatically skipped (first-call-wins).

    Adds three audit columns on the first call; updates them on subsequent calls:
        snap_rule        : str   — label applied, 'none' if unmatched,
                           'exceeded_threshold' if beyond max_distance_m
        snap_distance_m  : float — distance moved in metres
        snapped          : bool  — True if geometry was actually moved

    Parameters
    ----------
    gdf_nodes : GeoDataFrame
        Nodes layer. Pass the result of the previous snap_nodes() call to
        chain multiple rules.
    gdf_centerlines_filtered : GeoDataFrame
        Centerlines already filtered to the relevant subset (e.g. only
        Interstate rows, only DOT_RTNAME ramps, etc.). Filtering is the
        caller's responsibility — this function receives the result.
    node_mask : pd.Series
        Boolean Series aligned to gdf_nodes.index selecting candidate nodes.
        e.g. gdf_nodes["Freeway"] or gdf_nodes["Arterial"] | gdf_nodes["Local"]
    max_distance_m : float
        Nodes further than this threshold are not moved.
    label : str
        Written to the snap_rule audit column for snapped nodes.
    precision : int, optional
        Coordinate rounding for endpoint deduplication. Default 7 (~1cm UTM).
    crs_projected : str, optional
        Projected CRS for metric distance calculations.
        Default EPSG:26912 (UTM Zone 12N — Utah).

    Returns
    -------
    GeoDataFrame
        Copy of gdf_nodes with updated geometry and audit columns.

    Example
    -------
    # Chain multiple rules — first call wins per node
    gdf = snap_nodes(gdf, active_cl[active_cl["DOT_FCLASS"] == "Interstate"],
                     node_mask=gdf["Freeway"], max_distance_m=500, label="Freeway")

    gdf = snap_nodes(gdf, active_cl[active_cl["DOT_RTNAME"].str[5] == "R"],
                     node_mask=gdf["Ramp"], max_distance_m=300, label="Ramp")

    """
    result = gdf_nodes.copy()

    # Initialise audit columns on first call
    if "snap_rule" not in result.columns:
        result["snap_rule"] = "none"
        result["snap_distance_m"] = np.nan
        result["snapped"] = False

    if len(gdf_centerlines_filtered) == 0:
        print(f"  [{label}] No centerlines provided — skipping.")
        return result

    # Only consider nodes that match the mask AND haven't been snapped yet
    candidate_idx = node_mask[node_mask].index
    candidate_idx = candidate_idx[result.loc[candidate_idx, "snap_rule"] == "none"]

    if len(candidate_idx) == 0:
        print(f"  [{label}] No unsnapped candidate nodes — skipping.")
        return result

    endpoints = _extract_endpoints(gdf_centerlines_filtered, precision=precision)
    print(f"  [{label}] {len(candidate_idx):,} nodes → {len(endpoints):,} endpoints")

    candidate_nodes = result.loc[candidate_idx]
    snapped_geoms, distances, flags = _snap_nodes_to_points(
        candidate_nodes, endpoints, max_distance_m, crs_projected
    )

    for idx, geom, dist, snapped_flag in zip(candidate_idx, snapped_geoms, distances, flags):
        result.at[idx, "geometry"] = geom
        result.at[idx, "snap_distance_m"] = dist
        result.at[idx, "snapped"] = snapped_flag
        result.at[idx, "snap_rule"] = label if snapped_flag else "exceeded_threshold"

    snapped = sum(flags)
    exceeded = len(flags) - snapped
    print(f"         → {snapped:,} snapped | {exceeded:,} exceeded {max_distance_m}m threshold")

    return result


def snap_to_gtfs_stops(
    gdf_nodes: gpd.GeoDataFrame,
    gdf_stops: gpd.GeoDataFrame,
    node_mask: pd.Series,
    max_distance_m: float = 200,
    label: str = "FixedTransit_GTFS",
    crs_projected: str = "EPSG:26912",
) -> gpd.GeoDataFrame:
    """
    Snap a subset of nodes to the nearest GTFS stop point.

    Follows the same first-call-wins pattern as snap_nodes() — nodes already
    snapped by a previous call are automatically skipped.

    Parameters
    ----------
    gdf_nodes : GeoDataFrame
        Nodes layer. Pass the result of previous snap calls to chain.
    gdf_stops : GeoDataFrame
        GTFS stops layer with Point geometry.
    node_mask : pd.Series
        Boolean Series aligned to gdf_nodes.index selecting candidate nodes.
        e.g. gdf_nodes["FixedTransit"]
    max_distance_m : float, optional
        Nodes further than this threshold are not moved. Default 200m.
    label : str, optional
        Written to the snap_rule audit column. Default 'FixedTransit_GTFS'.
    crs_projected : str, optional
        Projected CRS for metric distances. Default EPSG:26912 (UTM 12N).

    Returns
    -------
    GeoDataFrame
        Copy of gdf_nodes with updated geometry and audit columns.

    """
    result = gdf_nodes.copy()

    if "snap_rule" not in result.columns:
        result["snap_rule"] = "none"
        result["snap_distance_m"] = np.nan
        result["snapped"] = False

    candidate_idx = node_mask[node_mask].index
    candidate_idx = candidate_idx[result.loc[candidate_idx, "snap_rule"] == "none"]

    if len(candidate_idx) == 0:
        print(f"  [{label}] No unsnapped candidate nodes — skipping.")
        return result

    print(f"  [{label}] {len(candidate_idx):,} nodes → {len(gdf_stops):,} stops")

    candidate_nodes = result.loc[candidate_idx]
    snapped_geoms, distances, flags = _snap_nodes_to_points(
        candidate_nodes, gdf_stops.geometry, max_distance_m, crs_projected
    )

    for idx, geom, dist, snapped_flag in zip(candidate_idx, snapped_geoms, distances, flags):
        result.at[idx, "geometry"] = geom
        result.at[idx, "snap_distance_m"] = dist
        result.at[idx, "snapped"] = snapped_flag
        result.at[idx, "snap_rule"] = label if snapped_flag else "exceeded_threshold"

    snapped = sum(flags)
    exceeded = len(flags) - snapped
    print(f"         → {snapped:,} snapped | {exceeded:,} exceeded {max_distance_m}m threshold")

    return result
