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

Stage 3 — Public API
--------------------
    dissolve_pseudonodes(gdf_links, pseudo_node_ids)
        Collapse chains of model links connected through pseudonodes into single
        dissolved links spanning from one real node to another.
        Returns a DataFrame with A, B, FT_2027, LN_2027, DIRECTION, ONEWAY,
        n_constituents, constituent_ab_pairs.

    build_piece_graph(gdf_pieces, gdf_nodes_snapped)
        Build a networkx Graph from physical link pieces. Each node is labelled
        "real_junction", "pseudonode", or "internal" based on snap status.

    assemble_chain(piece_graph, a_coord, b_coord)
        Constrained BFS from a_coord to b_coord through non-real-junction nodes.
        Returns an ordered list of LineString pieces forming the path.
"""

import json
import warnings
from collections import defaultdict, deque

import geopandas as gpd
import networkx as nx
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


# =============================================================================
# Stage 3 helpers
# =============================================================================


def dissolve_pseudonodes(
    gdf_links: gpd.GeoDataFrame,
    pseudo_node_ids: set,
) -> pd.DataFrame:
    """
    Collapse chains of model links connected through pseudonodes into dissolved links.

    A pseudonode is a degree-2 node lying in the interior of what should be a
    single logical link (all connected links share the same FT). This function
    merges consecutive links into a single dissolved link spanning from one real
    node (is_pseudo=False) to another.

    Parameters
    ----------
    gdf_links : GeoDataFrame
        Filtered model links to dissolve. Must have columns A, B, FT_2027,
        LN_2027, DIRECTION, ONEWAY.
    pseudo_node_ids : set
        Set of node N-values that are pseudonodes (is_pseudo=True).

    Returns
    -------
    DataFrame with one row per dissolved link:
        A, B               : int  — real start/end node N-values
        FT_2027, LN_2027   : int  — from first constituent link in chain
        DIRECTION, ONEWAY  : str/int — from first constituent link
        n_constituents     : int  — number of original links collapsed
        constituent_ab_pairs : str — JSON list of [A, B] pairs in traversal order

    Notes
    -----
    Uses a directed MultiDiGraph so that opposite-direction links (A→P and P→A)
    on the same road segment produce two separate chains (A→B and B→A) rather
    than collapsing into a single self-loop.
    """
    G: nx.MultiDiGraph = nx.MultiDiGraph()
    for _, row in gdf_links.iterrows():
        a, b = int(row["A"]), int(row["B"])
        G.add_edge(
            a, b,
            orig_A=a,
            orig_B=b,
            FT_2027=int(row["FT_2027"]),
            LN_2027=int(row["LN_2027"]),
            DIRECTION=str(row["DIRECTION"]),
            ONEWAY=int(row["ONEWAY"]),
        )

    real_nodes = set(G.nodes()) - pseudo_node_ids
    dissolved: list[dict] = []
    visited_edges: set[tuple] = set()

    for start in sorted(real_nodes):
        # out_edges: only edges leaving `start` (directed)
        for _, nbr, key, edata in G.out_edges(start, keys=True, data=True):
            ek = (start, nbr, key)
            if ek in visited_edges:
                continue
            visited_edges.add(ek)

            chain_pairs: list[list[int]] = [[edata["orig_A"], edata["orig_B"]]]
            first_edata = edata
            curr = nbr

            while curr in pseudo_node_ids:
                # Prefer edges not returning to `start`; fall back to any unvisited edge.
                # This prevents a pseudonode's back-edge to the entry real node from being
                # chosen when a forward edge to a different real node is also available.
                next_found = None
                fallback = None
                for _, cnbr, ckey, cedata in G.out_edges(curr, keys=True, data=True):
                    cek = (curr, cnbr, ckey)
                    if cek in visited_edges:
                        continue
                    if cnbr != start:
                        next_found = (cnbr, cedata, cek)
                        break
                    if fallback is None:
                        fallback = (cnbr, cedata, cek)
                if next_found is None:
                    next_found = fallback
                if next_found is None:
                    break
                next_nbr, next_edata, next_ek = next_found
                visited_edges.add(next_ek)
                chain_pairs.append([next_edata["orig_A"], next_edata["orig_B"]])
                curr = next_nbr

            dissolved.append(
                {
                    "A": start,
                    "B": curr,
                    "FT_2027": first_edata["FT_2027"],
                    "LN_2027": first_edata["LN_2027"],
                    "DIRECTION": first_edata["DIRECTION"],
                    "ONEWAY": first_edata["ONEWAY"],
                    "n_constituents": len(chain_pairs),
                    "constituent_ab_pairs": json.dumps(chain_pairs),
                }
            )

    return pd.DataFrame(dissolved)


def build_piece_graph(
    gdf_pieces: gpd.GeoDataFrame,
    gdf_nodes_snapped: gpd.GeoDataFrame,
) -> nx.Graph:
    """
    Build a networkx Graph from physical link pieces for constrained chain traversal.

    Each node in the graph is a (x_round, y_round) coordinate tuple representing
    a piece endpoint. Nodes are classified by comparing endpoint coordinates to
    the snapped model node lookup:

    - "real_junction" — snapped AND is_pseudo=False: acts as a chain boundary
    - "pseudonode"    — snapped AND is_pseudo=True: traversable interior node
    - "internal"      — not in snapped nodes: physical road intersection

    Parameters
    ----------
    gdf_pieces : GeoDataFrame
        Output of 02_create_link.qmd. Must have x_start, y_start, x_end, y_end,
        piece_id, length_m, geometry.
    gdf_nodes_snapped : GeoDataFrame
        nodes_snapped layer. Must have snapped (bool), snapped_x_round,
        snapped_y_round, is_pseudo.

    Returns
    -------
    networkx.Graph
        Nodes: (x_round, y_round) tuples with node_type attribute.
        Edges: one per piece with piece_id, geometry, length_m attributes.
        When two pieces share the same endpoint pair, the shorter one is kept.
    """
    snapped_mask = gdf_nodes_snapped["snapped"].fillna(False).astype(bool)
    df_snapped = gdf_nodes_snapped[snapped_mask]

    x_arr = df_snapped["snapped_x_round"].to_numpy(dtype=float)
    y_arr = df_snapped["snapped_y_round"].to_numpy(dtype=float)
    pseudo_arr = df_snapped["is_pseudo"].fillna(False).astype(bool).to_numpy()
    coord_to_type: dict[tuple, str] = {
        (float(x), float(y)): ("pseudonode" if p else "real_junction")
        for x, y, p in zip(x_arr, y_arr, pseudo_arr)
    }

    G: nx.Graph = nx.Graph()

    x_start = gdf_pieces["x_start"].to_numpy(dtype=float)
    y_start = gdf_pieces["y_start"].to_numpy(dtype=float)
    x_end = gdf_pieces["x_end"].to_numpy(dtype=float)
    y_end = gdf_pieces["y_end"].to_numpy(dtype=float)
    pids = gdf_pieces["piece_id"].to_numpy()
    lens = gdf_pieces["length_m"].to_numpy(dtype=float)
    geoms = gdf_pieces["geometry"].values

    for i in range(len(gdf_pieces)):
        cs = (x_start[i], y_start[i])
        ce = (x_end[i], y_end[i])
        if cs == ce:
            continue
        for c in (cs, ce):
            if c not in G:
                G.add_node(c, node_type=coord_to_type.get(c, "internal"))
        if not G.has_edge(cs, ce):
            G.add_edge(
                cs, ce, piece_id=int(pids[i]), geometry=geoms[i], length_m=float(lens[i])
            )
        elif float(lens[i]) < G[cs][ce]["length_m"]:
            G[cs][ce].update(piece_id=int(pids[i]), geometry=geoms[i], length_m=float(lens[i]))

    return G


def assemble_chain(
    piece_graph: nx.Graph,
    a_coord: tuple,
    b_coord: tuple,
) -> list[LineString] | None:
    """
    Constrained BFS from a_coord to b_coord through non-real-junction nodes.

    Starting from a_coord (a real junction), the BFS traverses the piece graph
    but will not pass through any node labelled "real_junction" other than b_coord.
    This ensures the assembled path stays within the bounds of a single dissolved
    model link and does not bleed into adjacent links.

    Parameters
    ----------
    piece_graph : networkx.Graph
        Output of build_piece_graph. Nodes carry node_type attribute.
    a_coord : tuple
        (x_round, y_round) of start node — must be a real junction in the graph.
    b_coord : tuple
        (x_round, y_round) of end node — must be a real junction in the graph.

    Returns
    -------
    list[LineString] or None
        Ordered list of piece geometries forming the path from A to B.
        Empty list if a_coord == b_coord.
        None if B is unreachable under the real-junction constraint.
    """
    if a_coord not in piece_graph or b_coord not in piece_graph:
        return None
    if a_coord == b_coord:
        return _assemble_loop(piece_graph, a_coord)

    # BFS with parent pointer for memory-efficient path reconstruction
    parent: dict[tuple, tuple | None] = {a_coord: None}
    queue: deque[tuple] = deque([a_coord])

    while queue:
        curr = queue.popleft()
        for nbr, edata in piece_graph[curr].items():
            if nbr in parent:
                continue
            if nbr == b_coord:
                parent[nbr] = (curr, edata)
                geoms: list[LineString] = []
                node = nbr
                while parent[node] is not None:
                    prev_node, edge_data = parent[node]
                    geoms.append(edge_data["geometry"])
                    node = prev_node
                geoms.reverse()
                return geoms
            node_type = piece_graph.nodes[nbr].get("node_type", "internal")
            if node_type == "real_junction":
                continue
            parent[nbr] = (curr, edata)
            queue.append(nbr)

    return None


def _assemble_loop(piece_graph: nx.Graph, a_coord: tuple) -> list[LineString] | None:
    """
    Find a closed path from a_coord back to itself through non-real-junction nodes.

    Used for dissolved model links where A == B (stub turnarounds). The BFS
    takes one step away from a_coord, then searches for a path back.
    """
    # parent maps node -> (prev_node, edge_data), with a_coord as target (not in parent)
    parent: dict[tuple, tuple] = {}
    queue: deque[tuple] = deque()

    for first_nbr, first_edata in piece_graph[a_coord].items():
        if first_nbr == a_coord:
            continue  # skip degenerate self-loop edges
        nt = piece_graph.nodes[first_nbr].get("node_type", "internal")
        if nt == "real_junction":
            continue
        if first_nbr not in parent:
            parent[first_nbr] = (a_coord, first_edata)
            queue.append(first_nbr)

    while queue:
        curr = queue.popleft()
        for nbr, edata in piece_graph[curr].items():
            if nbr == a_coord:
                # Reconstruct closed path
                geoms: list[LineString] = [edata["geometry"]]
                node = curr
                while node != a_coord:
                    prev_node, edge_data = parent[node]
                    geoms.append(edge_data["geometry"])
                    node = prev_node
                geoms.reverse()
                return geoms
            if nbr in parent:
                continue
            nt = piece_graph.nodes[nbr].get("node_type", "internal")
            if nt == "real_junction":
                continue
            parent[nbr] = (curr, edata)
            queue.append(nbr)

    return None
