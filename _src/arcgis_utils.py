"""
Utility functions for downloading ArcGIS feature layers.
"""

import geopandas as gpd
from arcgis.features import FeatureLayer


def fetch_feature_layer(
    service_url: str, gis, out_sr: int = 4326, where: str = "1=1", out_fields: str = "*"
) -> gpd.GeoDataFrame:
    """
    Fetch an ArcGIS feature layer and return as a GeoDataFrame.

    Parameters
    ----------
    service_url : str
        Full URL to the ArcGIS FeatureServer layer (must end with layer index, e.g. /0).
    gis : arcgis.gis.GIS
        Authenticated or anonymous GIS connection.
    out_sr : int, optional
        Output spatial reference EPSG code. Default is 4326 (WGS84).
    where : str, optional
        SQL where clause to filter features. Default is '1=1' (all features).
    out_fields : str, optional
        Comma-separated field names to return. Default is '*' (all fields).

    Returns
    -------
    geopandas.GeoDataFrame
        The fetched data as a GeoDataFrame.

    """
    print(f"Fetching: {service_url}")
    layer = FeatureLayer(service_url, gis=gis)

    try:
        feature_set = layer.query(
            where=where, out_fields=out_fields, return_geometry=True, out_sr=out_sr
        )
        print(f"Success! Fetched {len(feature_set.features)} features.")
    except Exception as e:
        print(f"Error querying layer: {e}")
        raise

    gdf = gpd.GeoDataFrame(feature_set.sdf, geometry="SHAPE").set_crs(epsg=out_sr)
    return gdf
