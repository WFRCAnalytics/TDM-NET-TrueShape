"""
Utility functions for network node classification and snapping.

Exports:

    nodes_on(gdf_nodes, gdf_links, query)
        Boolean Series — True if node N appears in A or B of any link
        matching the pandas .query() string.

        e.g. gdf_nodes["Freeway"] = nodes_on(gdf_nodes, gdf_links, "FT_2023 in [20, 22, 23]")

    count_links(gdf_nodes, gdf_links)
    snap_nodes(gdf_nodes, gdf_centerlines_filtered, node_mask, max_distance_m, label, ...)
    snap_transit(gdf_nodes, gdf_stops, node_mask, max_distance_m, ...)

All filtering logic lives entirely in the calling notebook — not here.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
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


def count_links(gdf_nodes: gpd.GeoDataFrame, gdf_links: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
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


def assign_directions(gdf_nodes: gpd.GeoDataFrame, gdf_links: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Add a 'link_directions' column to the nodes GeoDataFrame, containing a
    comma-separated string of all unique directions (from link 'DIRECTION' col)
    connected to that node.

    Parameters
    ----------
    gdf_nodes : GeoDataFrame
        Nodes layer. Must contain column N.
    gdf_links : GeoDataFrame
        Links layer. Must contain columns A, B, and DIRECTION.

    Returns
    -------
    GeoDataFrame
        Copy of gdf_nodes with 'link_directions' column appended.

    """
    result = gdf_nodes.copy()

    # Ensure DIRECTION column exists to avoid KeyError
    if "DIRECTION" not in gdf_links.columns:
        print("Warning: 'DIRECTION' column not found in links. Skipping direction assignment.")
        result["link_directions"] = ""
        return result

    # Stack A and B nodes so we have a flat list of (Node, Direction)
    links_a = gdf_links[["A", "DIRECTION"]].rename(columns={"A": "N"})
    links_b = gdf_links[["B", "DIRECTION"]].rename(columns={"B": "N"})

    # Combine, drop null directions, and strip whitespace just in case
    node_dirs = pd.concat([links_a, links_b]).dropna(subset=["DIRECTION"])
    node_dirs["DIRECTION"] = node_dirs["DIRECTION"].astype(str).str.strip()

    # Group by Node 'N', get unique directions, and join as a comma-separated string
    # e.g., {'NB', 'WB'} becomes "NB,WB"
    dir_strings = (
        node_dirs.groupby("N")["DIRECTION"]
        .unique()
        .apply(lambda x: ",".join(sorted([d for d in x if d and d.lower() != "nan"])))
    )

    # Map back to the nodes dataframe; fill missing with empty string
    result["link_directions"] = result["N"].map(dir_strings).fillna("")

    return result


# ---------------------------------------------------------------------------
# Snapping helpers (private)
# ---------------------------------------------------------------------------


def _assign_line_directions(gdf_lines: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Extract allowed directions from FULLNAME or DOT_RTNAME for LineStrings."""
    lines = gdf_lines.copy()
    lines["allowed_dirs"] = ""

    if "FULLNAME" in lines.columns:
        extracted = lines["FULLNAME"].astype(str).str.extract(r"\b(NB|SB|EB|WB)\b", expand=False)
        lines["allowed_dirs"] = extracted.fillna("")

    if "DOT_RTNAME" in lines.columns:
        mask_empty = lines["allowed_dirs"] == ""
        # The 5th character (index 4) is the direction (P or N)
        lrs_dir = lines.loc[mask_empty, "DOT_RTNAME"].astype(str).str[4:5]

        # P = Positive (Northbound or Eastbound)
        lines.loc[mask_empty & (lrs_dir == "P"), "allowed_dirs"] = "NB,EB"
        # N = Negative (Southbound or Westbound)
        lines.loc[mask_empty & (lrs_dir == "N"), "allowed_dirs"] = "SB,WB"

    return lines


def _spatial_snap(
    gdf_nodes: gpd.GeoDataFrame,
    snap_targets: gpd.GeoDataFrame | gpd.GeoSeries,
    max_distance_m: float,
    crs_projected: str,
    target_id_cols: list[str] = None,
) -> tuple[list, list, list, dict]:
    """
    Snap nodes using Segment-First (Point-to-Line-to-Point) logic.
    Extracts GERS lineage attributes and calculates precise ALRS mileposts.
    """
    nodes_proj = gdf_nodes.to_crs(crs_projected)
    num_nodes = len(gdf_nodes)

    # Prepare GERS attribute dictionary
    snapped_attrs = {}
    if target_id_cols and isinstance(snap_targets, gpd.GeoDataFrame):
        for col in target_id_cols:
            snapped_attrs[col] = [None] * num_nodes

    is_lines = False
    if isinstance(snap_targets, gpd.GeoDataFrame):
        geom_types = snap_targets.geometry.type.unique()
        if any("LineString" in t for t in geom_types):
            is_lines = True
            snap_targets = _assign_line_directions(snap_targets)
            targets_proj = snap_targets.to_crs(crs_projected)
            has_dirs = True
            # Setup dynamic milepost extraction if ALRS columns exist
            if "DOT_F_MILE" in snap_targets.columns and "DOT_T_MILE" in snap_targets.columns:
                snapped_attrs["milepost"] = [np.nan] * num_nodes
        else:
            targets_proj = snap_targets.to_crs(crs_projected)
            has_dirs = "allowed_dirs" in snap_targets.columns
    else:
        targets_proj = snap_targets.to_crs(crs_projected)
        has_dirs = False

    target_geoms_proj = targets_proj.geometry.values
    target_geoms_orig = snap_targets.geometry.values

    tree = STRtree(target_geoms_proj)
    DIR_GROUP = {"NB": "P", "EB": "P", "P": "P", "SB": "N", "WB": "N", "N": "N"}

    # ==========================================
    # PHASE 1: Bulk Spatial Query & Pre-Compute Sets
    # ==========================================
    query_pairs = tree.query(
        nodes_proj.geometry.values, predicate="dwithin", distance=max_distance_m
    )
    node_indices = query_pairs[0]
    target_indices = query_pairs[1]

    distances_to_target = shapely.distance(
        nodes_proj.geometry.values[node_indices], target_geoms_proj[target_indices]
    )

    node_dirs_exact = [
        set(d for d in str(d_str).split(",") if d)
        for d_str in gdf_nodes.get("link_directions", pd.Series([""] * len(gdf_nodes)))
    ]
    node_dirs_grp = [
        set(DIR_GROUP.get(d, d) for d in str(d_str).split(",") if d)
        for d_str in gdf_nodes.get("link_directions", pd.Series([""] * len(gdf_nodes)))
    ]

    if has_dirs:
        target_dirs_exact = [
            set(d for d in str(d_str).split(",") if d)
            for d_str in targets_proj.get("allowed_dirs", pd.Series([""] * len(targets_proj)))
        ]
        target_dirs_grp = [
            set(DIR_GROUP.get(d, d) for d in str(d_str).split(",") if d)
            for d_str in targets_proj.get("allowed_dirs", pd.Series([""] * len(targets_proj)))
        ]

    # ==========================================
    # PHASE 2: Generate Segment-First Bids
    # ==========================================
    all_candidates = []
    for i in range(len(node_indices)):
        n_idx = node_indices[i]
        t_idx = target_indices[i]
        dist_target = distances_to_target[i]
        proj_node = nodes_proj.geometry.values[n_idx]

        match_tier = 0
        is_compatible = True

        if has_dirs:
            n_exact, n_grp = node_dirs_exact[n_idx], node_dirs_grp[n_idx]
            t_exact, t_grp = target_dirs_exact[t_idx], target_dirs_grp[t_idx]
            if t_exact and n_exact:
                if n_exact.intersection(t_exact):
                    match_tier = 0
                elif n_grp.intersection(t_grp):
                    match_tier = 1
                else:
                    is_compatible = False

        if is_compatible:
            if is_lines:
                proj_line = target_geoms_proj[t_idx]
                orig_line = target_geoms_orig[t_idx]

                if proj_line.geom_type == "MultiLineString":
                    p_start, p_end = (
                        Point(proj_line.geoms[0].coords[0]),
                        Point(proj_line.geoms[-1].coords[-1]),
                    )
                    o_start, o_end = (
                        Point(orig_line.geoms[0].coords[0]),
                        Point(orig_line.geoms[-1].coords[-1]),
                    )
                else:
                    p_start, p_end = Point(proj_line.coords[0]), Point(proj_line.coords[-1])
                    o_start, o_end = Point(orig_line.coords[0]), Point(orig_line.coords[-1])

                dist_start = proj_node.distance(p_start)
                dist_end = proj_node.distance(p_end)

                id_start = (round(p_start.x, 3), round(p_start.y, 3))
                id_end = (round(p_end.x, 3), round(p_end.y, 3))

                # Tuple expanded to track target_index (t_idx) and whether it is the start (True/False)
                if dist_start <= max_distance_m:
                    all_candidates.append(
                        (match_tier, dist_target, dist_start, n_idx, id_start, o_start, t_idx, True)
                    )
                if dist_end <= max_distance_m:
                    all_candidates.append(
                        (match_tier, dist_target, dist_end, n_idx, id_end, o_end, t_idx, False)
                    )
            else:
                orig_pt = target_geoms_orig[t_idx]
                dist_pt = proj_node.distance(target_geoms_proj[t_idx])
                id_pt = (round(target_geoms_proj[t_idx].x, 3), round(target_geoms_proj[t_idx].y, 3))
                if dist_pt <= max_distance_m:
                    all_candidates.append(
                        (match_tier, dist_target, dist_pt, n_idx, id_pt, orig_pt, t_idx, None)
                    )

    # ==========================================
    # PHASE 3: Global Sort
    # ==========================================
    all_candidates.sort(key=lambda x: (x[0], x[1], x[2]))

    # ==========================================
    # PHASE 4: The Claiming Process & GERS Extraction
    # ==========================================
    claimed_nodes = set()
    claimed_endpoints = set()

    snapped_geoms = [None] * num_nodes
    snap_distances_m = [None] * num_nodes
    snapped_flags = [False] * num_nodes

    # Unpack bid with target index and start/end flag
    for (
        _,
        _,
        dist_endpoint,
        node_idx,
        endpoint_id,
        endpoint_geom_orig,
        t_idx,
        is_start,
    ) in all_candidates:
        if node_idx not in claimed_nodes and endpoint_id not in claimed_endpoints:
            claimed_nodes.add(node_idx)
            claimed_endpoints.add(endpoint_id)

            snapped_geoms[node_idx] = endpoint_geom_orig
            snap_distances_m[node_idx] = round(dist_endpoint, 2)
            snapped_flags[node_idx] = True

            # Extract requested GERS attributes
            if target_id_cols and isinstance(snap_targets, gpd.GeoDataFrame):
                for col in target_id_cols:
                    if col in snap_targets.columns:
                        snapped_attrs[col][node_idx] = snap_targets.iloc[t_idx][col]

            # Dynamically extract correct ALRS Milepost
            if is_lines and "milepost" in snapped_attrs:
                mp = (
                    snap_targets.iloc[t_idx]["DOT_F_MILE"]
                    if is_start
                    else snap_targets.iloc[t_idx]["DOT_T_MILE"]
                )
                snapped_attrs["milepost"][node_idx] = mp

    # Handle Leftovers
    for i, (orig_geom, proj_geom) in enumerate(zip(gdf_nodes.geometry, nodes_proj.geometry)):
        if i not in claimed_nodes:
            nearest_idx = tree.nearest(proj_geom)
            abs_distance_m = (
                proj_geom.distance(target_geoms_proj[nearest_idx])
                if nearest_idx is not None
                else np.nan
            )
            snapped_geoms[i] = orig_geom
            snap_distances_m[i] = round(abs_distance_m, 2) if pd.notna(abs_distance_m) else np.nan
            snapped_flags[i] = False

    return snapped_geoms, snap_distances_m, snapped_flags, snapped_attrs


# ---------------------------------------------------------------------------
# Public snapping functions
# ---------------------------------------------------------------------------


def snap_nodes(
    gdf_nodes: gpd.GeoDataFrame,
    gdf_centerlines_filtered: gpd.GeoDataFrame,
    node_mask: pd.Series,
    max_distance_m: float,
    label: str,
    target_id_cols: list[str] = None,
    crs_projected: str = "EPSG:26912",
) -> gpd.GeoDataFrame:
    """Snap nodes to the endpoints of pre-filtered centerlines."""
    result = gdf_nodes.copy()

    if "snap_rule" not in result.columns:
        result["snap_rule"] = "none"
        result["snap_distance_m"] = np.nan
        result["snapped"] = False

    if len(gdf_centerlines_filtered) == 0:
        return result

    candidate_idx = node_mask[node_mask].index
    candidate_idx = candidate_idx[result.loc[candidate_idx, "snap_rule"] == "none"]

    if len(candidate_idx) == 0:
        return result

    print(
        f"  [{label}] {len(candidate_idx):,} nodes → {len(gdf_centerlines_filtered):,} centerlines"
    )

    candidate_nodes = result.loc[candidate_idx]
    snapped_geoms, distances, flags, attrs = _spatial_snap(
        candidate_nodes, gdf_centerlines_filtered, max_distance_m, crs_projected, target_id_cols
    )

    # Pre-create attribute columns to avoid SettingWithCopy warnings
    for col in attrs.keys():
        col_name = f"snapped_{col}"
        if col_name not in result.columns:
            result[col_name] = None

    for i, (idx, geom, dist, snapped_flag) in enumerate(
        zip(candidate_idx, snapped_geoms, distances, flags)
    ):
        result.at[idx, "geometry"] = geom
        result.at[idx, "snap_distance_m"] = dist
        result.at[idx, "snapped"] = snapped_flag
        result.at[idx, "snap_rule"] = label if snapped_flag else "exceeded_threshold"

        # Apply GERS attributes
        for col, values_list in attrs.items():
            result.at[idx, f"snapped_{col}"] = values_list[i]

    snapped = sum(flags)
    exceeded = len(flags) - snapped
    print(f"         → {snapped:,} snapped | {exceeded:,} exceeded {max_distance_m}m threshold")

    return result


def snap_transit(
    gdf_nodes: gpd.GeoDataFrame,
    gdf_stops: gpd.GeoDataFrame,
    node_mask: pd.Series,
    max_distance_m: float = 200,
    label: str = "FixedTransit_GTFS",
    target_id_cols: list[str] = None,
    crs_projected: str = "EPSG:26912",
) -> gpd.GeoDataFrame:
    """Snap a subset of nodes to the nearest GTFS stop point."""
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
    snapped_geoms, distances, flags, attrs = _spatial_snap(
        candidate_nodes, gdf_stops.geometry, max_distance_m, crs_projected, target_id_cols
    )

    # Pre-create attribute columns
    for col in attrs.keys():
        col_name = f"snapped_{col}"
        if col_name not in result.columns:
            result[col_name] = None

    for i, (idx, geom, dist, snapped_flag) in enumerate(
        zip(candidate_idx, snapped_geoms, distances, flags)
    ):
        result.at[idx, "geometry"] = geom
        result.at[idx, "snap_distance_m"] = dist
        result.at[idx, "snapped"] = snapped_flag
        result.at[idx, "snap_rule"] = label if snapped_flag else "exceeded_threshold"

        for col, values_list in attrs.items():
            result.at[idx, f"snapped_{col}"] = values_list[i]

    snapped = sum(flags)
    exceeded = len(flags) - snapped
    print(f"         → {snapped:,} snapped | {exceeded:,} exceeded {max_distance_m}m threshold")

    return result
