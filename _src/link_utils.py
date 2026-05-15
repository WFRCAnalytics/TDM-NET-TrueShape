"""
Utility functions for WFRC true-shape link generation.

Design contract
---------------
- Filtering / topology definitions live in the calling notebook.
- This module only accepts pre-filtered data and implements the mechanics
  of geometric operations, splitting, and graph assembly.
- All heavy-lifting uses vectorised Shapely 2.x APIs (shapely.get_coordinates,
  shapely.points, STRtree bulk queries); row-level Python loops are limited to
  the piece-splitting step where sequential state is unavoidable.

Stage 2 — Public API
--------------------
    resolve_snap_coords(gdf_nodes_snapped, gdf_centerlines, tolerance=0.05)
        For each snapped node, resolve the rounded snap coordinate to the
        exact centerline vertex coordinate using STRtree nearest-vertex lookup.
        Adds x_exact, y_exact, snap_resolved columns.

    merge_centerlines(gdf_centerlines)
        Merge all centerline geometries into a single combined geometry via
        linemerge on the raw segments. Segments touching at degree-2 vertices
        are fused; intersections and flyover crossings are preserved as-is.

    split_at_nodes(cl_merged, snap_points_exact)
        Split the merged centerline geometry at all snapped node positions.
        Uses M-value projection + substring to handle multiple split points per
        line in one pass. Returns a list of LineString pieces.

Stage 3 — Public API (implemented in 03_transfer_attributes.qmd)
-----------------------------------------------------------------
    dissolve_pseudonodes, build_piece_graph, assemble_chain
"""

import warnings
from collections import defaultdict

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import linemerge, substring
from shapely.strtree import STRtree

# =============================================================================
# Stage 2 helpers
# =============================================================================


def resolve_snap_coords(
    gdf_nodes_snapped: gpd.GeoDataFrame,
    gdf_centerlines: gpd.GeoDataFrame,
    tolerance: float = 0.05,
) -> gpd.GeoDataFrame:
    """
    Resolve rounded snap coordinates to exact centerline vertex coordinates.

    nodes_snapped.gpkg stores snapped_x_round / snapped_y_round rounded to
    2 decimal places (cm precision). Shapely split/substring operations require
    coordinates that match the actual vertices stored in the centerline geometry.
    This function finds, for each snapped node, the nearest unique centerline
    vertex within `tolerance` metres and records its exact coordinate.

    Parameters
    ----------
    gdf_nodes_snapped : GeoDataFrame
        Output of 01_node_classification.qmd. Must have columns snapped (bool),
        snapped_x_round, snapped_y_round.
    gdf_centerlines : GeoDataFrame
        Centerline layer in the same CRS. Used only for vertex extraction.
    tolerance : float
        Maximum distance in metres between the rounded snap coord and the
        nearest exact vertex. Nodes exceeding this emit a warning.

    Returns
    -------
    GeoDataFrame
        Copy of gdf_nodes_snapped with three new columns:
        - x_exact, y_exact : float  — exact vertex coordinate (NaN if unresolved)
        - snap_resolved     : bool  — True when a vertex was found within tolerance
    """
    # ── Extract all unique centerline vertices ────────────────────────────────
    raw_coords = shapely.get_coordinates(gdf_centerlines.geometry.values)
    # Round to 4dp before dedup to avoid floating-point noise creating false dupes
    unique_coords = np.unique(np.round(raw_coords, 4), axis=0)
    vertex_geoms = shapely.points(unique_coords[:, 0], unique_coords[:, 1])

    tree = STRtree(vertex_geoms)

    # ── Build query points from snapped nodes ────────────────────────────────
    snapped_mask = gdf_nodes_snapped["snapped"].fillna(False).astype(bool)
    df_snapped = gdf_nodes_snapped[snapped_mask]

    x_exact = np.full(len(gdf_nodes_snapped), np.nan)
    y_exact = np.full(len(gdf_nodes_snapped), np.nan)
    snap_resolved = np.zeros(len(gdf_nodes_snapped), dtype=bool)

    if df_snapped.empty:
        result = gdf_nodes_snapped.copy()
        result["x_exact"] = x_exact
        result["y_exact"] = y_exact
        result["snap_resolved"] = snap_resolved
        return result

    query_pts = shapely.points(
        df_snapped["snapped_x_round"].to_numpy(dtype=float),
        df_snapped["snapped_y_round"].to_numpy(dtype=float),
    )

    # nearest() returns one closest vertex index per query point
    nearest_idxs = tree.nearest(query_pts)
    nearest_verts = vertex_geoms[nearest_idxs]
    dists = shapely.distance(query_pts, nearest_verts)
    resolved = dists <= tolerance

    n_unresolved = int((~resolved).sum())
    if n_unresolved:
        warnings.warn(
            f"resolve_snap_coords: {n_unresolved} snapped node(s) had no "
            f"centerline vertex within {tolerance} m — x_exact/y_exact left NaN."
        )

    nearest_xy = shapely.get_coordinates(nearest_verts)

    # Map results back to the full-GDF positional index
    snapped_positions = np.where(snapped_mask)[0]
    x_exact[snapped_positions] = np.where(resolved, nearest_xy[:, 0], np.nan)
    y_exact[snapped_positions] = np.where(resolved, nearest_xy[:, 1], np.nan)
    snap_resolved[snapped_positions] = resolved

    result = gdf_nodes_snapped.copy()
    result["x_exact"] = x_exact
    result["y_exact"] = y_exact
    result["snap_resolved"] = snap_resolved
    return result


def merge_centerlines(gdf_centerlines: gpd.GeoDataFrame) -> MultiLineString | LineString:
    """
    Merge all centerline geometries into a single combined geometry.

    Passes the raw segment collection directly to linemerge (no unary_union).
    linemerge only joins lines at pre-existing shared endpoints — it never
    inserts new vertices at interior crossings. This means:
    - Flyovers/underpasses that cross geometrically but share no endpoint are
      left as independent components (no false split points).
    - Ramps that physically connect to another road at a shared endpoint are
      still merged through that point regardless of vertical level.

    Real intersections (degree >= 3, where 3+ segments share an endpoint)
    remain as component boundaries and are not merged.

    Parameters
    ----------
    gdf_centerlines : GeoDataFrame
        Centerline layer. All rows are merged; apply any FT/class filtering in
        the notebook before calling.

    Returns
    -------
    MultiLineString or LineString
        Combined geometry. Iterate over .geoms for individual components.
    """
    # Flatten any MultiLineStrings to simple LineStrings before merging;
    # linemerge requires simple geometries (it accesses .coords on each element).
    parts = shapely.get_parts(gdf_centerlines.geometry.values)
    return linemerge(parts.tolist())


def split_at_nodes(
    cl_merged: MultiLineString | LineString,
    snap_points_exact: list[Point],
) -> list[LineString]:
    """
    Split the merged centerline geometry at all snapped node positions.

    For each component LineString in cl_merged, finds all snap points that lie
    on it (within 1 cm tolerance), projects them to M-values, and uses
    shapely.ops.substring to extract the sub-segments between consecutive
    cut points. Points at or beyond a line's endpoints (M ≈ 0 or M ≈ total
    length) are silently skipped — those lines are already bounded there.

    Parameters
    ----------
    cl_merged : MultiLineString or LineString
        Output of merge_centerlines(). Must be in the same CRS as snap points.
    snap_points_exact : list[Point]
        Exact-coordinate snap points, one per snapped node (from x_exact/y_exact
        columns of resolve_snap_coords output). Points that do not lie on any
        line are silently ignored.

    Returns
    -------
    list[LineString]
        Flat list of LineString pieces. Every piece's endpoints are either a
        snapped node position or a physical intersection preserved by linemerge.
    """
    # ── Flatten to list of component LineStrings ─────────────────────────────
    if cl_merged.geom_type == "LineString":
        components = [cl_merged]
    elif cl_merged.geom_type == "MultiLineString":
        components = list(cl_merged.geoms)
    else:
        # GeometryCollection fallback — extract any LineStrings present
        components = [g for g in cl_merged.geoms if g.geom_type == "LineString"]

    n_components = len(components)

    # ── Assign each snap point to the component(s) it lies on ────────────────
    # Use a 1 cm buffer for the query, then verify with distance check.
    _QUERY_TOL = 0.01  # metres
    tree = STRtree(components)

    piece_to_pts: defaultdict[int, list[Point]] = defaultdict(list)
    n_assigned = 0
    for pt in snap_points_exact:
        if pt is None:
            continue
        candidate_idxs = tree.query(pt.buffer(_QUERY_TOL))
        for idx in candidate_idxs:
            if idx < n_components and components[idx].distance(pt) <= _QUERY_TOL:
                piece_to_pts[idx].append(pt)
                n_assigned += 1
                break  # a point lies on at most one component after linemerge

    # ── Split each component at its assigned points ───────────────────────────
    result: list[LineString] = []
    for i, line in enumerate(components):
        pts = piece_to_pts.get(i)
        if not pts:
            result.append(line)
        else:
            result.extend(_split_line_at_points(line, pts))

    return result


# =============================================================================
# Internal helpers
# =============================================================================


def _split_line_at_points(line: LineString, points: list[Point]) -> list[LineString]:
    """
    Split a single LineString at one or more interior points using M-values.

    Points that project to M <= eps or M >= (length - eps) are treated as
    endpoint coincidences and do not generate a split.
    """
    total_len = line.length
    if total_len < 1e-9:
        return [line]

    _EPS = 0.50  # 50 cm — skip splits within 50 cm of an endpoint; drop sub-segments shorter than 50 cm

    # Project each point to an M-value along the line; deduplicate
    m_set: set[float] = set()
    for pt in points:
        m = line.project(pt)
        if _EPS < m < total_len - _EPS:
            m_set.add(m)

    if not m_set:
        return [line]

    cuts = [0.0] + sorted(m_set) + [total_len]

    segments: list[LineString] = []
    for i in range(len(cuts) - 1):
        seg = substring(line, cuts[i], cuts[i + 1])
        if seg is not None and not seg.is_empty and seg.length > _EPS:
            segments.append(seg)

    return segments if segments else [line]
