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


def _extract_endpoints(gdf_lines: gpd.GeoDataFrame, precision: int = 7) -> gpd.GeoDataFrame:
    """
    Extract deduplicated start and end points from a LineString GeoDataFrame.
    Attaches allowed directions extracted from FULLNAME or DOT_RTNAME.

    Parameters
    ----------
    gdf_lines : GeoDataFrame
        LineString or MultiLineString layer.
    precision : int, optional
        Decimal places to round coordinates before deduplication.

    Returns
    -------
    GeoDataFrame
        Deduplicated endpoint Points with an 'allowed_dirs' column.

    """
    lines = gdf_lines.explode(index_parts=False).copy()
    lines["allowed_dirs"] = ""

    # 1. Try extracting explicit direction from FULLNAME (NB, SB, EB, WB)
    if "FULLNAME" in lines.columns:
        extracted = lines["FULLNAME"].astype(str).str.extract(r"\b(NB|SB|EB|WB)\b", expand=False)
        lines["allowed_dirs"] = extracted.fillna("")

    # 2. Fallback to DOT_RTNAME for missing directions
    if "DOT_RTNAME" in lines.columns:
        mask_empty = lines["allowed_dirs"] == ""
        # The 5th character (index 4) is the direction (P or N)
        lrs_dir = lines.loc[mask_empty, "DOT_RTNAME"].astype(str).str[4:5]

        # P = Positive (Northbound or Eastbound)
        lines.loc[mask_empty & (lrs_dir == "P"), "allowed_dirs"] = "NB,EB"
        # N = Negative (Southbound or Westbound)
        lines.loc[mask_empty & (lrs_dir == "N"), "allowed_dirs"] = "SB,WB"

    # 3. Extract endpoints
    records = []
    for geom, allowed in zip(lines.geometry, lines["allowed_dirs"]):
        start = geom.coords[0]
        end = geom.coords[-1]

        records.append(
            {
                "geometry": Point(round(start[0], precision), round(start[1], precision)),
                "allowed_dirs": allowed,
            }
        )
        records.append(
            {
                "geometry": Point(round(end[0], precision), round(end[1], precision)),
                "allowed_dirs": allowed,
            }
        )

    # 4. Create GeoDataFrame and deduplicate based on location AND direction
    endpoints_gdf = gpd.GeoDataFrame(records, crs=gdf_lines.crs)
    endpoints_gdf["geom_wkt"] = endpoints_gdf.geometry.to_wkt()
    endpoints_gdf = endpoints_gdf.drop_duplicates(subset=["geom_wkt", "allowed_dirs"]).drop(
        columns=["geom_wkt"]
    )

    return endpoints_gdf


def _spatial_snap(
    gdf_nodes: gpd.GeoDataFrame,
    snap_targets: gpd.GeoDataFrame | gpd.GeoSeries,
    max_distance_m: float,
    crs_projected: str,
) -> tuple[list, list, list]:
    """
    Snap nodes to the nearest directionally-compatible point in snap_targets.
    """
    nodes_proj = gdf_nodes.to_crs(crs_projected)
    targets_proj = snap_targets.to_crs(crs_projected)

    # Handle both GeoDataFrame (Centerlines with directions) and GeoSeries (GTFS stops)
    if isinstance(targets_proj, gpd.GeoDataFrame):
        target_geoms_proj = targets_proj.geometry.values
        target_geoms_orig = snap_targets.geometry.values
        has_dirs = "allowed_dirs" in snap_targets.columns
    else:
        target_geoms_proj = targets_proj.values
        target_geoms_orig = snap_targets.values
        has_dirs = False

    tree = STRtree(target_geoms_proj)

    snapped_geoms = []
    snap_distances_m = []
    snapped_flags = []

    # Mapping dictionary to group directions into Positive (P) and Negative (N)
    # This solves the "highway bends" issue where NB links meet EB centerlines.
    DIR_GROUP = {"NB": "P", "EB": "P", "P": "P", "SB": "N", "WB": "N", "N": "N"}

    for i, (orig_geom, proj_geom) in enumerate(zip(gdf_nodes.geometry, nodes_proj.geometry)):
        # Get allowed directions and map them to their P/N alias
        node_dir_str = gdf_nodes.iloc[i].get("link_directions", "")
        node_dirs = set(DIR_GROUP.get(d, d) for d in node_dir_str.split(",") if d)

        # Query the tree for ALL endpoints within the maximum threshold
        nearby_idx = tree.query(proj_geom, predicate="dwithin", distance=max_distance_m)

        best_dist = float("inf")
        best_geom = orig_geom

        # Check nearby points for directional compatibility
        if len(nearby_idx) > 0:
            for idx in nearby_idx:
                is_compatible = True

                # If we have directions, enforce the logic via the P/N groups
                if has_dirs and node_dirs:
                    target_dir_str = snap_targets.iloc[idx].get("allowed_dirs", "")
                    target_dirs = set(DIR_GROUP.get(d, d) for d in target_dir_str.split(",") if d)

                    # If target has specific directions, node must share the P or N group
                    if target_dirs and not node_dirs.intersection(target_dirs):
                        is_compatible = False

                # If compatible, see if it is the closest one we've found so far
                if is_compatible:
                    dist = proj_geom.distance(target_geoms_proj[idx])
                    if dist < best_dist:
                        best_dist = dist
                        best_geom = target_geoms_orig[idx]

        # If we successfully found a valid endpoint within the threshold
        if best_dist <= max_distance_m:
            snapped_geoms.append(best_geom)
            snap_distances_m.append(round(best_dist, 2))
            snapped_flags.append(True)
        else:
            # Fallback for audit purposes: find the absolute nearest point
            # (even if direction is wrong or distance is too far) just to
            # populate the 'snap_distance_m' column so you know how far away it was.
            nearest_idx = tree.nearest(proj_geom)
            abs_distance_m = proj_geom.distance(target_geoms_proj[nearest_idx])

            snapped_geoms.append(orig_geom)  # original CRS, no reprojection needed
            snap_distances_m.append(round(abs_distance_m, 2))
            snapped_flags.append(False)

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
    snapped_geoms, distances, flags = _spatial_snap(
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


def snap_transit(
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
    snapped_geoms, distances, flags = _spatial_snap(
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
