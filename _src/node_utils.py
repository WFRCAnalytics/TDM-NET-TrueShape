"""
Utility functions for WFRC network node classification and snapping.

Design contract
---------------
- Filtering / topology *definitions* live in the calling notebook.
- This module only accepts pre-filtered data and implements the *mechanics*
  of spatial math, indexing, and assignment algorithms.
- All heavy-lifting is vectorised (numpy / pandas); row-level Python loops
  are limited to the Gale-Shapley proposal loop where sequential state is
  mandatory.

Public API
----------
    nodes_on(gdf_nodes, gdf_links, ft_mask)
        Boolean Series — True if node N appears as endpoint A or B of any
        link whose FT code is in ft_mask.

    count_links(gdf_nodes, gdf_links) -> GeoDataFrame
        Append a LinkCount column (how many link endpoints touch each node).

    assign_node_directions(gdf_nodes, gdf_links, freeway_ft_codes) -> GeoDataFrame
        Append link_directions and fw_directions columns.

    assign_node_type(gdf_nodes, is_fwy_mask, is_ramp_mask, is_surface_mask) -> GeoDataFrame
        Append node_type column — one of:
        "fwy" | "fwy_sf" | "gore" | "gore_sf" | "ramp" | "ramp_sf" | "surface".

    extract_endpoints(gdf_centerlines) -> GeoDataFrame
        Vectorised extraction of segment start/end points with topology flags.

    assign_endpoint_directions(gdf_ep_unique, gdf_ep_raw) -> GeoDataFrame
        Append ep_allowed_dirs direction string per unique endpoint.

    assign_endpoint_type(gdf_ep_unique) -> GeoDataFrame
        Append ep_type column — one of: "fwy" | "gore" | "fwy_sf" | "ramp" | "ramp_sf" | "surface".

    snap_nodes(gdf_nodes, gdf_endpoints, node_mask, max_distance_m, label, ...) -> GeoDataFrame
        Snap a masked subset of nodes to the nearest compatible endpoint
        using Gale-Shapley stable matching with (type_tier, dir_tier, dist) sorting.

    snap_transit(gdf_nodes, gdf_stops, node_mask, max_distance_m, ...) -> GeoDataFrame
        Snap transit nodes to nearest GTFS stop (no direction matching).

    ep_claimed_coords(gdf_nodes_snapped) -> frozenset
        Return set of (x_round, y_round) tuples already claimed across all
        completed snap passes.

    filter_ep_claimed(gdf_ep, claimed_coords) -> GeoDataFrame
        Remove endpoint rows already claimed in prior passes so the same
        physical endpoint cannot be assigned to two nodes across separate passes.
"""

import re
from collections import defaultdict, deque

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.strtree import STRtree

# =============================================================================
# Node classification helpers
# =============================================================================


def nodes_on(
    gdf_nodes: gpd.GeoDataFrame,
    gdf_links: gpd.GeoDataFrame,
    ft_mask: set | list | None = None,
) -> pd.Series:
    """
    Boolean Series: True if a node's N appears as endpoint A or B of any
    link in gdf_links (optionally filtered to ft_mask FT codes).

    The functional-type *definition* (which FT codes constitute "freeway",
    "ramp", etc.) belongs in the notebook. Pass a pre-filtered gdf_links with
    ft_mask=None when the compound condition cannot be expressed as a simple
    FT-code set (e.g. FT=2 AND DIRECTION=1).

    Parameters
    ----------
    gdf_nodes : GeoDataFrame with column N.
    gdf_links : GeoDataFrame with columns A, B, and an FT column.
    ft_mask   : Collection of FT code integers to match against.
                Pass None (default) to use all rows in gdf_links as-is —
                useful when the caller has already applied a compound filter.

    Examples
    --------
    # Simple FT-code filter (legacy usage)
    gdf_nodes["Freeway"] = nodes_on(gdf_nodes, gdf_links, {32, 33, 34, 35, 36})

    # Pre-filtered DataFrame (compound condition handled in notebook)
    fwy_links = gdf_links[fwy_mask]
    gdf_nodes["Freeway"] = nodes_on(gdf_nodes, fwy_links)
    """
    if ft_mask is None:
        matched_links = gdf_links
    else:
        ft_set = set(ft_mask)
        ft_col = _detect_ft_col(gdf_links)
        matched_links = gdf_links[gdf_links[ft_col].isin(ft_set)]
    connected = set(matched_links["A"]) | set(matched_links["B"])
    return gdf_nodes["N"].isin(connected)


def count_links(gdf_nodes: gpd.GeoDataFrame, gdf_links: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Append a LinkCount column to gdf_nodes: how many link endpoint references
    (A or B) match each node's N value.
    """
    result = gdf_nodes.copy()
    link_counts = pd.concat([gdf_links["A"], gdf_links["B"]]).value_counts()
    result["LinkCount"] = result["N"].map(link_counts).fillna(0).astype(int)
    return result


def count_neighbors(gdf_nodes: gpd.GeoDataFrame, gdf_links: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Append a NeighborCount column to gdf_nodes: how many *distinct* neighboring
    nodes each node connects to, regardless of link direction.

    Unlike LinkCount, NeighborCount is invariant to whether bidirectional roads
    are stored as one shared record or two directional records. A pseudonode on
    a bidirectional road would get LinkCount=4 but NeighborCount=2.

    Note: NeighborCount alone is not sufficient for pseudonode detection. A node
    at a functional class boundary (e.g. freeway-to-surface transition) also has
    NeighborCount=2 but is a real junction. Pair with a link-class homogeneity
    check in the calling notebook — see A2 in 01_node_classification.qmd.
    """
    result = gdf_nodes.copy()
    pairs = pd.concat(
        [
            gdf_links[["A", "B"]].rename(columns={"A": "node", "B": "neighbor"}),
            gdf_links[["B", "A"]].rename(columns={"B": "node", "A": "neighbor"}),
        ],
        ignore_index=True,
    )
    neighbor_counts = pairs.groupby("node")["neighbor"].nunique()
    result["NeighborCount"] = result["N"].map(neighbor_counts).fillna(0).astype(int)
    return result


def assign_node_directions(
    gdf_nodes: gpd.GeoDataFrame, gdf_links: gpd.GeoDataFrame, freeway_ft_codes: set | list
) -> gpd.GeoDataFrame:
    """
    Append two direction columns to gdf_nodes.

    link_directions
        Freeway-priority cascade: pool DIRECTION from freeway links first;
        fall back to interchange links only when no freeway links are present.
        Used in passes (a) and (c).

    fw_directions
        Freeway links only. Majority-vote when opposing directions appear on
        multiple freeway links at the same node (e.g. a gore at a reversible
        section). Ties broken alphabetically. Used in pass (b) so that the
        directional signal is isolated to the mainline side of a mixed node.

    Parameters
    ----------
    freeway_ft_codes : FT codes that classify a link as freeway mainline.
                       e.g. set(range(32, 37))

    """
    result = gdf_nodes.copy()

    ft_col = _detect_ft_col(gdf_links)
    if ft_col is None or "DIRECTION" not in gdf_links.columns:
        result["link_directions"] = ""
        result["fw_directions"] = ""
        return result

    fw_set = set(freeway_ft_codes)
    CARDINAL = {"NB", "SB", "EB", "WB"}

    # Stack A/B so each link contributes to both its endpoint nodes.
    stacked = pd.concat(
        [
            gdf_links[["A", "DIRECTION", ft_col]].rename(columns={"A": "N"}),
            gdf_links[["B", "DIRECTION", ft_col]].rename(columns={"B": "N"}),
        ],
        ignore_index=True,
    )
    stacked["DIRECTION"] = stacked["DIRECTION"].astype(str).str.strip().str.upper()
    stacked["is_freeway"] = stacked[ft_col].isin(fw_set)

    # ── link_directions: freeway-first cascade ─────────────────────────────
    # For each node, collect cardinal directions from freeway links.
    # If none present, fall back to all other links.
    fw_stacked = stacked[stacked["is_freeway"] & stacked["DIRECTION"].isin(CARDINAL)]
    non_fw_stacked = stacked[~stacked["is_freeway"] & stacked["DIRECTION"].isin(CARDINAL)]

    fw_dirs_by_node = fw_stacked.groupby("N")["DIRECTION"].apply(lambda s: ",".join(sorted(set(s))))
    non_fw_dirs_by_node = non_fw_stacked.groupby("N")["DIRECTION"].apply(
        lambda s: ",".join(sorted(set(s)))
    )
    # Freeway wins; non-freeway fills gaps (nodes with no freeway links).
    link_dir_map = non_fw_dirs_by_node.to_dict()
    link_dir_map.update(fw_dirs_by_node.to_dict())  # freeway overrides
    result["link_directions"] = result["N"].map(link_dir_map).fillna("")

    # ── fw_directions: freeway-only, majority-vote ─────────────────────────
    fw_cardinal = stacked[stacked["is_freeway"] & stacked["DIRECTION"].isin(CARDINAL)]

    def majority_direction(group: pd.Series) -> str:
        counts = group.value_counts()
        max_count = counts.max()
        winners = sorted(counts[counts == max_count].index)
        return ",".join(winners)

    fw_dir_map = fw_cardinal.groupby("N")["DIRECTION"].apply(majority_direction).to_dict()
    result["fw_directions"] = result["N"].map(fw_dir_map).fillna("")

    return result


# =============================================================================
# Centerline endpoint extraction and classification
# =============================================================================


def assign_endpoint_directions(
    gdf_ep_unique: gpd.GeoDataFrame, gdf_ep_raw: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Append ep_allowed_dirs to each unique endpoint using the freeway-priority
    direction cascade.

    Priority per unique coordinate
    --------------------------------
    1. Any freeway segment terminates here →
       FULLNAME token (e.g. "I-15 NB FWY") → LRS P/N fallback (DOT_RTNAME[4]).
    2. Interchange-only (no freeway) →
       DOT_RTNAME[4] P/N → "NB,EB" or "SB,WB" group (corridor side).
    3. Surface / unresolved → empty string (direction-agnostic).

    Parameters
    ----------
    gdf_ep_unique : Deduplicated endpoint layer (from groupby x_round/y_round).
    gdf_ep_raw    : Raw endpoint records from extract_endpoints().

    """
    result = gdf_ep_unique.copy()

    # Build a lookup: coord_key → slice of raw records (vectorised groupby)
    raw_grouped = gdf_ep_raw.groupby(["x_round", "y_round"])

    ep_dirs = []
    for _, ep_row in result.iterrows():
        key = (ep_row["x_round"], ep_row["y_round"])
        if key not in raw_grouped.groups:
            ep_dirs.append("")
            continue
        raw_rows = gdf_ep_raw.loc[raw_grouped.groups[key]]
        ep_dirs.append(
            _resolve_direction(
                directions=[""] * len(raw_rows),  # DIRECTION not on centerlines
                fullnames=raw_rows["fullname"].tolist(),
                dot_rtnames=raw_rows["dot_rtname"].tolist(),
                is_freeway_flags=raw_rows["is_freeway"].tolist(),
                is_interchange_flags=raw_rows["is_interchange"].tolist(),
            )
        )

    result["ep_allowed_dirs"] = ep_dirs
    return result


# =============================================================================
# Direction resolution — private helpers
# =============================================================================

_FULLNAME_DIR_RE = re.compile(r"\b(NB|SB|EB|WB)\b")
_CARDINAL = {"NB", "SB", "EB", "WB"}
_DIR_FROM_LRS = {"P": {"NB", "EB"}, "N": {"SB", "WB"}}


def _extract_fullname_direction(fullname: str) -> str:
    """Return the first NB/SB/EB/WB token from a FULLNAME string, or ''."""
    m = _FULLNAME_DIR_RE.search(str(fullname))
    return m.group(1) if m else ""


def _resolve_direction(
    directions: list[str],
    fullnames: list[str],
    dot_rtnames: list[str],
    is_freeway_flags: list[bool],
    is_interchange_flags: list[bool],
) -> str:
    """
    Resolve a direction string from a collection of associated segments
    (network links or centerline segments).

    Priority cascade
    ----------------
    If ANY freeway segment is present:
      1a. Pool DIRECTION / FULLNAME tokens from freeway segments only.
      1b. LRS P/N fallback (DOT_RTNAME[4]) if 1a resolves nothing.
    If no freeway segment (interchange-only):
      2a. Pool DIRECTION / LRS P/N from interchange segments.
    Nothing resolves → "" (direction-agnostic).

    Returns
    -------
    Comma-separated cardinal string, e.g. "NB", "NB,EB", "SB,WB", "".

    """
    has_freeway = any(is_freeway_flags)
    collected: set[str] = set()

    if has_freeway:
        for d, fn, rt, is_fw in zip(directions, fullnames, dot_rtnames, is_freeway_flags):
            if not is_fw:
                continue
            d_up = str(d).strip().upper()
            if d_up in _CARDINAL:
                collected.add(d_up)
            elif fn:
                token = _extract_fullname_direction(fn)
                if token:
                    collected.add(token)
        # LRS P/N fallback — only when DIRECTION/FULLNAME both failed
        if not collected:
            for rt, is_fw in zip(dot_rtnames, is_freeway_flags):
                if not is_fw:
                    continue
                lrs_char = str(rt)[4:5] if rt and len(str(rt)) > 4 else ""
                collected.update(_DIR_FROM_LRS.get(lrs_char, set()))
    else:
        for d, fn, rt, is_ic in zip(directions, fullnames, dot_rtnames, is_interchange_flags):
            if not is_ic:
                continue
            d_up = str(d).strip().upper()
            if d_up in _CARDINAL:
                collected.add(d_up)
            elif fn:
                # Prefer FULLNAME cardinal token (e.g. "EB" from "I-215S EB X11 OFF STATE RAMP")
                # over LRS P/N — the ramp's own travel direction matters, not the
                # parent freeway corridor direction encoded in DOT_RTNAME[4].
                token = _extract_fullname_direction(fn)
                if token:
                    collected.add(token)
                elif rt:
                    # LRS P/N fallback only when FULLNAME has no cardinal token
                    lrs_char = str(rt)[4:5] if len(str(rt)) > 4 else ""
                    collected.update(_DIR_FROM_LRS.get(lrs_char, set()))
            elif rt:
                lrs_char = str(rt)[4:5] if len(str(rt)) > 4 else ""
                collected.update(_DIR_FROM_LRS.get(lrs_char, set()))

    return ",".join(sorted(collected))


# =============================================================================
# Type-tier lookup table (private)
# =============================================================================
#
# (node_type, ep_type) → int
#   0  = same topology class (preferred)
#   1  = adjacent class (permitted)
#   99 = hard reject (dropped before matching)
#
# Node types    : "fwy" | "fwy_sf" | "gore" | "gore_sf" | "ramp" | "ramp_sf" | "surface"
# Endpoint types: "fwy" | "fwy_sf" | "gore" | "ramp" | "ramp_sf" | "surface"
#
# Node types map the full combinatorics of the three TDM link classes:
#   fwy     Freeway ONLY             — pure mainline, no surface contact
#   fwy_sf  Freeway + Surface        — at-grade freeway crossing (e.g. Bangerter × local)
#   gore    Freeway + Ramp           — elevated fwy/ramp junction, no surface
#   gore_sf Freeway + Ramp + Surface — at-grade diamond interchange, all three classes
#   ramp    Ramp ONLY               — mid-ramp, no surface contact
#   ramp_sf Ramp + Surface          — ramp terminal at an arterial
#   surface Surface ONLY            — surface street node
#
# Endpoint types capture the same combinations at the physical centerline level
# (no "gore_sf" ep_type — the endpoint layer uses the same 6 types as before;
# the physical analog of gore_sf is a gore ep at an at-grade location).

_TYPE_TIER: dict[tuple[str, str], int] = {
    # ── fwy: pure freeway mainline ────────────────────────────────────────
    # Strict — ramp and surface endpoints hard-rejected to prevent
    # divided-highway collapse at parallel NB/SB carriageways.
    ("fwy", "fwy"):    0,
    ("fwy", "gore"):   1,
    ("fwy", "fwy_sf"): 1,

    # ── fwy_sf: at-grade freeway/surface crossing ─────────────────────────
    # Exact match is fwy_sf ep (same topology).
    # fwy and surface are both valid one-component fallbacks.
    # gore is adjacent via the freeway component.
    ("fwy_sf", "fwy_sf"):  0,
    ("fwy_sf", "fwy"):     1,
    ("fwy_sf", "surface"): 1,
    ("fwy_sf", "gore"):    1,

    # ── gore: elevated freeway/ramp junction ──────────────────────────────
    # gore ep is exact; fwy and ramp are one-component fallbacks.
    # fwy_sf allowed via freeway component (physically adjacent at most gores).
    # ramp_sf and surface hard-rejected — gore is not at grade.
    ("gore", "gore"):   0,
    ("gore", "fwy"):    1,
    ("gore", "ramp"):   1,
    ("gore", "fwy_sf"): 1,

    # ── gore_sf: at-grade freeway/ramp/surface interchange ────────────────
    # No ep_type covers all three components, so gore ep (fwy+ramp) is the
    # best single match (Tier-0). fwy_sf and ramp_sf each capture one of the
    # two at-grade component pairs and are valid Tier-1 fallbacks.
    ("gore_sf", "gore"):    0,
    ("gore_sf", "fwy_sf"):  1,
    ("gore_sf", "ramp_sf"): 1,
    ("gore_sf", "fwy"):     1,
    ("gore_sf", "ramp"):    1,

    # ── ramp: mid-ramp ────────────────────────────────────────────────────
    # Strict — only ramp/gore/ramp_sf compatible; fwy and surface rejected.
    ("ramp", "ramp"):    0,
    ("ramp", "gore"):    1,
    ("ramp", "ramp_sf"): 1,

    # ── ramp_sf: ramp terminal at arterial ───────────────────────────────
    # ramp_sf ep is exact; surface and ramp are one-component fallbacks.
    # gore permitted — a ramp terminal can sit near a freeway/ramp junction.
    ("ramp_sf", "ramp_sf"): 0,
    ("ramp_sf", "surface"): 1,
    ("ramp_sf", "ramp"):    1,
    ("ramp_sf", "gore"):    1,

    # ── surface: surface only ─────────────────────────────────────────────
    # fwy_sf shares the surface-intersection topology (Tier-0).
    # ramp_sf (ramp terminal at grade) is topologically adjacent (Tier-1).
    ("surface", "surface"): 0,
    ("surface", "fwy_sf"):  0,
    ("surface", "ramp_sf"): 1,
}
_TYPE_TIER_REJECT = 99


# =============================================================================
# Snapping core (private)
# =============================================================================


def _spatial_snap(
    gdf_nodes: gpd.GeoDataFrame,
    snap_targets: gpd.GeoDataFrame,
    max_distance_m: float,
    crs_projected: str,
    direction_col: str = "link_directions",
    target_id_cols: list[str] = None,
    node_type_col: str = "node_type",
) -> tuple[list, list, list, dict]:
    """
    Snap nodes to target points using Gale-Shapley Stable Matching.

    !! DO NOT replace with nearest-neighbour or segment-first approaches !!
    See CLAUDE.md §3 for the rationale. The Point-to-Point Greedy pool is
    mandatory to avoid Pigeonhole failures at complex interchanges.

    Algorithm
    ---------
    Phase 1 — Bulk STRtree dwithin query (vectorised).
    Phase 2 — Two-dimensional tier assignment per (node, target) pair:
              type_tier  : topology compatibility (0=same, 1=adjacent, 99=reject).
              dir_tier   : direction compatibility (0=exact, 1=same P/N group).
                           Interchange-only endpoints are direction-agnostic
                           (P/N encodes corridor side, not approach direction),
                           so their P/N-group matches are promoted to dir_tier=0.
              Sort key   : (type_tier, dir_tier, dist)
    Phase 3 — Node-proposing Gale-Shapley stable match.
    Phase 4 — Apply assignments; unmatched nodes retain original geometry.

    Parameters
    ----------
    direction_col  : Column on gdf_nodes used for directional matching.
                     "link_directions" for passes (a) and (c).
                     "fw_directions"   for pass (b) gore nodes.
    node_type_col  : Column on gdf_nodes containing the topology type label.
                     If absent, type-tier matching is skipped (e.g. transit).

    """
    nodes_proj = gdf_nodes.to_crs(crs_projected)
    targets_proj = snap_targets.to_crs(crs_projected)
    num_nodes = len(gdf_nodes)

    target_geoms_proj = targets_proj.geometry.values
    target_geoms_orig = snap_targets.geometry.values
    tree = STRtree(target_geoms_proj)

    # Only cardinal directions are valid group keys.
    # Raw LRS characters "P" / "N" must NOT appear here — if a DIRECTION column
    # contains raw LRS values instead of cardinals, they fall through as unresolved
    # rather than creating a false cross-side group match.
    DIR_GROUP = {"NB": "P", "EB": "P", "SB": "N", "WB": "N"}

    # ── GERS attribute storage ─────────────────────────────────────────────
    snapped_attrs: dict[str, list] = {}
    if target_id_cols and isinstance(snap_targets, gpd.GeoDataFrame):
        for col in target_id_cols:
            if col in snap_targets.columns:
                snapped_attrs[col] = [None] * num_nodes

    # ── Pre-compute type labels ────────────────────────────────────────────
    has_type = node_type_col in gdf_nodes.columns and "ep_type" in snap_targets.columns
    node_type_labels = gdf_nodes[node_type_col].tolist() if has_type else []
    target_type_labels = snap_targets["ep_type"].tolist() if has_type else []

    # ── Direction-agnostic flag per target endpoint ────────────────────────
    # Interchange-only endpoints: LRS P/N is a corridor side, not approach
    # direction → treat same-P/N group as exact match (dir_tier = 0).
    is_ic_only_target = np.zeros(len(targets_proj), dtype=bool)
    if all(c in snap_targets.columns for c in ["is_interchange", "is_freeway"]):
        is_ic_only_target = snap_targets["is_interchange"].values.astype(bool) & ~snap_targets[
            "is_freeway"
        ].values.astype(bool)

    # ── Pre-compute node direction sets ───────────────────────────────────
    has_dirs = "ep_allowed_dirs" in snap_targets.columns
    node_dir_series = gdf_nodes.get(direction_col, pd.Series([""] * num_nodes))

    def _dir_sets(series):
        exact = [set(d for d in str(v).split(",") if d) for v in series]
        group = [{DIR_GROUP.get(d, d) for d in s} for s in exact]
        return exact, group

    node_dirs_exact, node_dirs_grp = _dir_sets(node_dir_series)
    if has_dirs:
        target_dirs_exact, target_dirs_grp = _dir_sets(snap_targets["ep_allowed_dirs"])

    # ==========================================
    # PHASE 1: Bulk Spatial Query
    # ==========================================
    pairs = tree.query(nodes_proj.geometry.values, predicate="dwithin", distance=max_distance_m)
    node_indices, target_indices = pairs[0], pairs[1]
    distances = shapely.distance(
        nodes_proj.geometry.values[node_indices], target_geoms_proj[target_indices]
    )

    # ==========================================
    # PHASE 2: Compatibility Check & Bid Generation
    # ==========================================
    all_candidates = []

    for i in range(len(node_indices)):
        n_idx = int(node_indices[i])
        t_idx = int(target_indices[i])
        dist_pt = distances[i]

        # — Type tier —
        if has_type:
            type_tier = _TYPE_TIER.get(
                (node_type_labels[n_idx], target_type_labels[t_idx]), _TYPE_TIER_REJECT
            )
            if type_tier == _TYPE_TIER_REJECT:
                continue
        else:
            type_tier = 0

        # — Direction tier —
        dir_tier = 0
        if has_dirs:
            n_exact, n_grp = node_dirs_exact[n_idx], node_dirs_grp[n_idx]
            t_exact, t_grp = target_dirs_exact[t_idx], target_dirs_grp[t_idx]
            t_is_ic_only = bool(is_ic_only_target[t_idx])

            if t_exact and n_exact:
                if n_exact & t_exact:
                    dir_tier = 0  # exact cardinal match
                elif n_grp & t_grp:
                    # Same P/N group: promote interchange endpoints to dir_tier=0
                    # (their P/N encodes corridor side, not strict approach direction).
                    # Freeway/gore endpoints keep dir_tier=1 for overpass protection.
                    dir_tier = 0 if t_is_ic_only else 1
                else:
                    continue  # opposite P/N groups → reject

        orig_pt = target_geoms_orig[t_idx]
        ep_id = (round(target_geoms_proj[t_idx].x, 3), round(target_geoms_proj[t_idx].y, 3))
        all_candidates.append((type_tier, dir_tier, dist_pt, n_idx, ep_id, orig_pt, t_idx))

    # ==========================================
    # PHASE 3: Gale-Shapley Stable Matching
    # ==========================================

    # Build preference structures
    # node_prefs[n_idx]  = sorted list of bid payloads
    # ep_scores[ep_id][n_idx] = (type_tier, dir_tier, dist, n_idx) — endpoint ranks nodes
    temp_node_prefs: dict[int, dict] = defaultdict(dict)
    ep_scores: dict[tuple, dict[int, tuple]] = defaultdict(dict)

    for type_tier, dir_tier, dist_pt, n_idx, ep_id, geom_orig, t_idx in all_candidates:
        payload = {
            "type_tier": type_tier,
            "dir_tier": dir_tier,
            "dist_ep": dist_pt,
            "ep_id": ep_id,
            "geom_orig": geom_orig,
            "t_idx": t_idx,
        }
        # Keep best bid per (node, endpoint) pair
        existing = temp_node_prefs[n_idx].get(ep_id)
        if existing is None or (type_tier, dir_tier, dist_pt) < (
            existing["type_tier"],
            existing["dir_tier"],
            existing["dist_ep"],
        ):
            temp_node_prefs[n_idx][ep_id] = payload

        new_score = (type_tier, dir_tier, dist_pt, n_idx)
        if n_idx not in ep_scores[ep_id] or new_score < ep_scores[ep_id][n_idx]:
            ep_scores[ep_id][n_idx] = new_score

    node_prefs: dict[int, list] = {
        n_idx: sorted(bids.values(), key=lambda x: (x["type_tier"], x["dir_tier"], x["dist_ep"]))
        for n_idx, bids in temp_node_prefs.items()
    }

    # Node-proposing Gale-Shapley loop
    next_proposal = dict.fromkeys(node_prefs, 0)
    ep_holder: dict[tuple, dict] = {}
    free_nodes: deque = deque(node_prefs.keys())

    while free_nodes:
        n_idx = free_nodes.popleft()
        prefs = node_prefs[n_idx]
        if next_proposal[n_idx] >= len(prefs):
            continue  # exhausted — falls through to leftover handler

        proposal = prefs[next_proposal[n_idx]]
        next_proposal[n_idx] += 1
        ep_id = proposal["ep_id"]

        if ep_id not in ep_holder:
            ep_holder[ep_id] = {"n_idx": n_idx, "bid": proposal}
        else:
            current_n = ep_holder[ep_id]["n_idx"]
            if ep_scores[ep_id][n_idx] < ep_scores[ep_id][current_n]:
                ep_holder[ep_id] = {"n_idx": n_idx, "bid": proposal}
                free_nodes.append(current_n)  # displaced node re-queues
            else:
                free_nodes.append(n_idx)  # rejected node re-queues

    # ==========================================
    # PHASE 4: Apply Assignments
    # ==========================================
    claimed: set = set()
    snapped_geoms = [None] * num_nodes
    snap_distances_m = [None] * num_nodes
    snapped_flags = [False] * num_nodes

    for accepted in ep_holder.values():
        n_idx = accepted["n_idx"]
        bid = accepted["bid"]
        claimed.add(n_idx)
        snapped_geoms[n_idx] = bid["geom_orig"]
        snap_distances_m[n_idx] = round(bid["dist_ep"], 2)
        snapped_flags[n_idx] = True

        t_idx = bid["t_idx"]
        for col in snapped_attrs:
            if col in snap_targets.columns:
                snapped_attrs[col][n_idx] = snap_targets.iloc[t_idx][col]

    # Unclaimed nodes: retain original geometry, record nearest distance
    unclaimed = [i for i in range(num_nodes) if i not in claimed]
    if unclaimed:
        unclaimed_geoms = nodes_proj.geometry.values[unclaimed]
        nearest_idxs = tree.nearest(unclaimed_geoms)
        nearest_dists = shapely.distance(unclaimed_geoms, target_geoms_proj[nearest_idxs])
        for i, (orig_i, dist) in enumerate(zip(unclaimed, nearest_dists)):
            snapped_geoms[orig_i] = gdf_nodes.geometry.iloc[orig_i]
            snap_distances_m[orig_i] = round(float(dist), 2)
            snapped_flags[orig_i] = False

    return snapped_geoms, snap_distances_m, snapped_flags, snapped_attrs


# =============================================================================
# Public snapping functions
# =============================================================================


def snap_nodes(
    gdf_nodes: gpd.GeoDataFrame,
    gdf_endpoints: gpd.GeoDataFrame,
    node_mask: pd.Series,
    max_distance_m: float,
    label: str,
    direction_col: str = "link_directions",
    target_id_cols: list[str] = None,
    crs_projected: str = "EPSG:26912",
    node_type_col: str = "node_type",
) -> gpd.GeoDataFrame:
    """
    Snap a masked subset of nodes to the nearest compatible endpoint using
    Gale-Shapley stable matching with (type_tier, dir_tier, dist) sorting.

    First-call-wins: nodes whose snap_rule is already set to a successful label
    are not re-attempted. Nodes with snap_rule == "none" or "exceeded_threshold"
    are eligible — the latter allows nodes that failed an earlier pass to fall
    through to subsequent passes.

    Parameters
    ----------
    gdf_nodes     : Full node layer. Must have snap_rule, link_directions,
                    fw_directions, and node_type columns (built by the notebook
                    before calling this function).
    gdf_endpoints : Pre-classified endpoint layer (output of Step E in the
                    notebook) with ep_allowed_dirs, ep_type, is_freeway,
                    is_interchange, is_surface columns.
    node_mask     : Boolean Series selecting which nodes to attempt.
    max_distance_m: Search radius in metres.
    label         : snap_rule value written for successfully snapped nodes.
    direction_col : "link_directions" for passes (a)/(c); "fw_directions" for (b).
    target_id_cols: Columns on gdf_endpoints to propagate to snapped_<col> output.
    node_type_col : Column containing topology type label. If absent, type-tier
                    matching is skipped (backward-compatible with transit).

    """
    result = gdf_nodes.copy()
    if "snap_rule" not in result.columns:
        result["snap_rule"] = "none"
        result["snap_distance_m"] = np.nan
        result["snapped"] = False

    if len(gdf_endpoints) == 0:
        print(f"  [{label}] No target endpoints — skipping.")
        return result

    candidate_idx = node_mask[node_mask].index
    candidate_idx = candidate_idx[
        result.loc[candidate_idx, "snap_rule"].isin(["none", "exceeded_threshold"])
    ]
    if len(candidate_idx) == 0:
        print(f"  [{label}] No unsnapped candidate nodes — skipping.")
        return result

    print(
        f"  [{label}] {len(candidate_idx):,} nodes → {len(gdf_endpoints):,} endpoints"
        f"  (dir={direction_col}, type={node_type_col}, max={max_distance_m}m)"
    )

    for col in target_id_cols or []:
        if f"snapped_{col}" not in result.columns:
            result[f"snapped_{col}"] = None

    candidate_nodes = result.loc[candidate_idx]
    geoms, dists, flags, attrs = _spatial_snap(
        candidate_nodes,
        gdf_endpoints,
        max_distance_m,
        crs_projected,
        direction_col=direction_col,
        target_id_cols=target_id_cols,
        node_type_col=node_type_col,
    )

    for i, (idx, geom, dist, snapped_flag) in enumerate(zip(candidate_idx, geoms, dists, flags)):
        result.at[idx, "geometry"] = geom
        result.at[idx, "snap_distance_m"] = dist
        result.at[idx, "snapped"] = snapped_flag
        result.at[idx, "snap_rule"] = label if snapped_flag else "exceeded_threshold"
        for col, values in attrs.items():
            result.at[idx, f"snapped_{col}"] = values[i]

    snapped = sum(flags)
    exceeded = len(flags) - snapped
    print(f"         → {snapped:,} snapped | {exceeded:,} exceeded {max_distance_m}m threshold")
    return result


def snap_transit(
    gdf_nodes: gpd.GeoDataFrame,
    gdf_stops: gpd.GeoDataFrame,
    node_mask: pd.Series,
    max_distance_m: float = 200,
    label: str = "FixedTransit_Rail",
    target_id_cols: list[str] = None,
    crs_projected: str = "EPSG:26912",
) -> gpd.GeoDataFrame:
    """
    Snap transit nodes to the nearest GTFS stop point.

    Direction and type-tier matching are not applied — gdf_stops is not
    expected to have ep_allowed_dirs or ep_type columns. Only nodes with
    snap_rule == "none" are attempted (transit is the final pass).
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

    for col in target_id_cols or []:
        if f"snapped_{col}" not in result.columns:
            result[f"snapped_{col}"] = None

    candidate_nodes = result.loc[candidate_idx]
    geoms, dists, flags, attrs = _spatial_snap(
        candidate_nodes,
        gdf_stops,
        max_distance_m,
        crs_projected,
        direction_col="link_directions",
        target_id_cols=target_id_cols,
        node_type_col="node_type",  # absent on gdf_stops → type-tier skipped
    )

    for i, (idx, geom, dist, snapped_flag) in enumerate(zip(candidate_idx, geoms, dists, flags)):
        result.at[idx, "geometry"] = geom
        result.at[idx, "snap_distance_m"] = dist
        result.at[idx, "snapped"] = snapped_flag
        result.at[idx, "snap_rule"] = label if snapped_flag else "exceeded_threshold"
        for col, values in attrs.items():
            result.at[idx, f"snapped_{col}"] = values[i]

    snapped = sum(flags)
    exceeded = len(flags) - snapped
    print(f"         → {snapped:,} snapped | {exceeded:,} exceeded {max_distance_m}m threshold")
    return result


def ep_claimed_coords(gdf_nodes_snapped: gpd.GeoDataFrame) -> frozenset:
    """
    Return the set of (x_round, y_round) endpoint coordinates that have already
    been claimed across all completed snap passes.

    Call this after each snap_nodes() call and pass the result to
    filter_ep_claimed() before the next pass.  This ensures each physical
    endpoint is assigned to at most one model node even when the same endpoint
    row appears in multiple per-pass pools (e.g. a gore endpoint that lives in
    both ep_gore and ep_surface).

    Requires target_id_cols=["x_round", "y_round"] on all prior snap_nodes()
    calls so that snapped_x_round / snapped_y_round columns are present.
    """
    if "snapped_x_round" not in gdf_nodes_snapped.columns:
        return frozenset()
    claimed = gdf_nodes_snapped.loc[
        gdf_nodes_snapped["snapped"], ["snapped_x_round", "snapped_y_round"]
    ].dropna()
    return frozenset(zip(claimed["snapped_x_round"], claimed["snapped_y_round"]))


def filter_ep_claimed(
    gdf_ep: gpd.GeoDataFrame,
    claimed_coords: frozenset,
) -> gpd.GeoDataFrame:
    """
    Remove endpoint rows whose (x_round, y_round) coordinate is already in
    claimed_coords.  Endpoints are identified by their rounded coordinate pair
    (written by snap_nodes when target_id_cols=["x_round","y_round"]), so the
    lookup is exact for values produced by np.round(x, 2).

    Parameters
    ----------
    gdf_ep        : Endpoint pool GeoDataFrame.  Must have x_round, y_round.
    claimed_coords: Frozenset from ep_claimed_coords().
    """
    if not claimed_coords:
        return gdf_ep
    keys = list(zip(gdf_ep["x_round"], gdf_ep["y_round"]))
    keep = [k not in claimed_coords for k in keys]
    return gdf_ep[keep].copy()


# =============================================================================
# Private helpers
# =============================================================================


def _detect_ft_col(gdf_links: gpd.GeoDataFrame) -> str | None:
    """Return the first FT column found (FT_2027 preferred, then FT_2023)."""
    for col in ["FT_2027", "FT_2023"]:
        if col in gdf_links.columns:
            return col
    return None
