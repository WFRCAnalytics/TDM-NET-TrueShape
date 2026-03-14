"""
Utility functions for network node classification.

This module contains only the mechanical implementation of node classification.
All business logic (which FT codes map to which road type, which N ranges
indicate transit nodes) is defined in the calling notebook and passed in as
arguments. See classify_nodes() for the expected config format.
"""

import geopandas as gpd
import pandas as pd


def _nodes_in_ft_values(links: pd.DataFrame, ft_col: str, ft_values: list[int]) -> set:
    """
    Return the set of node IDs connected to links whose ft_col is in ft_values.
    """
    mask = links[ft_col].isin(ft_values)
    return set(links.loc[mask, "A"]).union(links.loc[mask, "B"])


def _nodes_in_n_ranges(n_series: pd.Series, n_ranges: list[tuple[int, int]]) -> pd.Series:
    """
    Return a boolean Series: True if N falls within any of the given ranges.
    Ranges are inclusive on both ends (pandas .between() default).
    """
    mask = pd.Series(False, index=n_series.index)
    for lo, hi in n_ranges:
        mask |= n_series.between(lo, hi)
    return mask


def _evaluate_criterion(
    criterion: dict, n: pd.Series, links: pd.DataFrame, ft_col: str
) -> pd.Series:
    """
    Evaluate a single criterion dict and return a boolean Series.

    Supported criterion types:
        {"type": "ft", "values": [...]}
            True if node N appears in A or B of links with ft_col in values.

        {"type": "n_range", "ranges": [(lo, hi), ...]}
            True if node N falls within any of the given ranges (inclusive).
    """
    ctype = criterion["type"]

    if ctype == "ft":
        connected = _nodes_in_ft_values(links, ft_col, criterion["values"])
        return n.isin(connected)

    if ctype == "n_range":
        return _nodes_in_n_ranges(n, criterion["ranges"])

    raise ValueError(f"Unknown criterion type '{ctype}'. Supported types: 'ft', 'n_range'.")


def classify_nodes(
    gdf_nodes: gpd.GeoDataFrame,
    gdf_links: gpd.GeoDataFrame,
    classification: dict,
    ft_year: int = 2023,
) -> gpd.GeoDataFrame:
    """
    Add classification flag columns and a LinkCount column to the nodes
    GeoDataFrame, based on a user-supplied classification config.

    Parameters
    ----------
    gdf_nodes : GeoDataFrame
        Nodes layer. Must contain column N.
    gdf_links : GeoDataFrame
        Links layer. Must contain columns A, B, and FT_<ft_year>.
    classification : dict
        Mapping of output column name to a classification spec. Each spec is
        a dict with the following keys:

            criteria : list of criterion dicts
                Each criterion has a "type" key and type-specific keys:
                    {"type": "ft",      "values": [...]}
                    {"type": "n_range", "ranges": [(lo, hi), ...]}

            combine : str, optional
                How to combine multiple criteria: "AND" or "OR".
                Defaults to "OR" if omitted.

        Example:
            {
                "Freeway": {
                    "criteria": [{"type": "ft", "values": [20, 21, 22]}],
                },
                "FixedTransit": {
                    "criteria": [
                        {"type": "ft",      "values": [70, 80]},
                        {"type": "n_range", "ranges": [(10_000, 19_999)]},
                    ],
                    "combine": "AND",
                },
            }

    ft_year : int, optional
        Year suffix for the functional type column. Default is 2023,
        resolving to FT_2023.

    Returns
    -------
    GeoDataFrame
        Copy of gdf_nodes with classification columns and LinkCount appended.

    """
    ft_col = f"FT_{ft_year}"
    if ft_col not in gdf_links.columns:
        ft_cols = [c for c in gdf_links.columns if c.startswith("FT_")]
        raise ValueError(f"Column '{ft_col}' not found in links. Available FT_ columns: {ft_cols}")

    result = gdf_nodes.copy()
    n = result["N"]
    links = gdf_links[["A", "B", ft_col]]

    # -- Classification flags -----------------------------------------------
    for col_name, spec in classification.items():
        criteria = spec["criteria"]
        combine = spec.get("combine", "OR").upper()

        masks = [_evaluate_criterion(criterion, n, links, ft_col) for criterion in criteria]

        if combine == "OR":
            result[col_name] = pd.concat(masks, axis=1).any(axis=1)
        elif combine == "AND":
            result[col_name] = pd.concat(masks, axis=1).all(axis=1)
        else:
            raise ValueError(
                f"Unknown combine value '{combine}' for column '{col_name}'. Use 'AND' or 'OR'."
            )

    # -- LinkCount ----------------------------------------------------------
    link_counts = pd.concat([links["A"], links["B"]]).value_counts()
    result["LinkCount"] = n.map(link_counts).fillna(0).astype(int)

    return result
