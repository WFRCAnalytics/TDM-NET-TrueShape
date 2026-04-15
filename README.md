# TDM-NET-TrueShape

> [!WARNING]
> **Work in Progress:** This project currently contains significant bugs and logic inconsistencies. Proceed with caution!

`TDM-NET-TrueShape` is a working pipeline for converting a travel demand model (TDM) network into a more realistic true-shape network using:

- UGRC roadway centerlines as the primary geometric reference
- WFRC MasterNet links and nodes as the source network topology
- GTFS data, where useful, to place fixed-guideway transit nodes more accurately

The current implementation is focused on automating node classification and node snapping as much as possible. In practice, this means taking abstract model nodes from the TDM network and moving them onto defensible real-world geometry, using a repeatable rule set based on roadway functional class, link connectivity, and transit stop locations.

## Project Goal

The overall objective of this project is to move the TDM network away from schematic geometry and toward a true-shape representation that is:

- spatially aligned with UGRC roadway centerlines
- aware of the network's existing functional-class logic
- capable of using GTFS where it improves transit node placement
- reproducible enough to rerun as source data changes

At the moment, the repository implements the node side of that workflow. Link true-shape reconstruction is not yet packaged as a standalone script or documented notebook in this repo.

## Current Workflow

The repository is organized as a notebook-first workflow with a small `_src/` helper library.

### 1. Data preparation

[`00_data_preparation.qmd`](./00_data_preparation.qmd) downloads and stores source inputs under `_data/raw/`:

- `UtahRoads.gpkg` from the UGRC `UtahRoads` FeatureServer
- `WFv1000_MasterNet - Link.gpkg` from the WFRC MasterNet link service
- `WFv1000_MasterNet - Node.gpkg` from the WFRC MasterNet node service
- `uta_gtfs.zip` from the UTA GTFS feed

### 2. Node classification

[`01_node_classification.qmd`](./01_node_classification.qmd) loads the network and assigns boolean node flags based on the attached link functional types and node-number ranges. The notebook currently classifies nodes into categories such as:

- `Freeway`
- `CD`
- `Expressway`
- `Arterial`
- `Collector`
- `Local`
- `Ramp`
- `CentroidConnector`
- `CentroidNode`
- `ExternalNode`
- `ConnectorOnly`
- `HOV`
- `FixedTransit`

It also computes `LinkCount`, which is used to identify likely pseudonodes and shape-point artifacts.

### 3. Node snapping

The same notebook then snaps eligible nodes to real-world reference geometry in ordered stages. The snapping helpers use a "first call wins" rule, so once a node is snapped by one rule it is skipped by later rules.

Current snapping stages in the notebook are:

1. `Freeway`
   Snap freeway nodes to endpoints of active centerlines classified as `Interstate` or `Other Freeway`, using a 500 m threshold.
2. `CD`
   Snap collector-distributor nodes to centerlines where `DOT_RTNAME` position 6 is `C`, using a 300 m threshold.
3. `Ramp`
   Snap ramp nodes to centerlines where `DOT_RTNAME` position 6 is `R`, using a 300 m threshold.
4. `Surface`
   Snap arterial, collector, and local nodes to active surface-street centerlines using `CARTOCODE` values `4`, `5`, `8`, `10`, and `11`, using a 200 m threshold.
5. `FixedTransit_Rail`
   Snap fixed-transit nodes to GTFS rail stops only. Bus stops are intentionally excluded. The current threshold is 200 m.

Nodes beyond the threshold are not moved and are tagged as `exceeded_threshold`.

### 4. Export

The notebook writes processed outputs to `_data/processed/`:

- `nodes_classified.gpkg`
- `nodes_snapped.gpkg`

## Repository Structure

```text
.
|-- 00_data_preparation.qmd
|-- 01_node_classification.qmd
|-- index.qmd
|-- pyproject.toml
|-- uv.lock
|-- _src/
|   |-- arcgis_utils.py
|   `-- node_utils.py
|-- _data/
|   |-- raw/
|   `-- processed/
`-- README.md
```

## Key Source Files

### [`_src/arcgis_utils.py`](./_src/arcgis_utils.py)

Contains `fetch_feature_layer()`, a utility that queries an ArcGIS FeatureServer layer and returns the result as a GeoDataFrame.

### [`_src/node_utils.py`](./_src/node_utils.py)

Contains the main reusable node-processing helpers:

- `nodes_on()` to flag nodes connected to links matching a pandas query
- `count_links()` to count how many links touch each node
- `snap_nodes()` to snap nodes to centerline endpoints
- `snap_transit()` to snap nodes directly to GTFS stop points

The snapping helpers also maintain audit fields:

- `snap_rule`
- `snap_distance_m`
- `snapped`

## Data Inputs

### Network inputs

The workflow expects WFRC MasterNet links and nodes with fields such as:

- node ID `N`
- link endpoint fields `A` and `B`
- functional type fields such as `FT_2023`

### Roadway geometry inputs

The snapping workflow depends on UGRC/roadway centerline geometry and several roadway classification fields, especially:

- `DOT_FCLASS`
- `DOT_RTNAME`
- `CARTOCODE`
- `STATUS`

### GTFS inputs

GTFS is used selectively:

- rail stops are used for fixed-guideway transit snapping
- bus stops are currently excluded from snapping

## Environment and Setup

### Python

The project is configured for Python `3.11` and currently pins dependencies in [`pyproject.toml`](./pyproject.toml) and [`uv.lock`](./uv.lock).

### Recommended setup

If you are using `uv`:

```powershell
uv sync --dev
```

If the existing local virtual environment is already provisioned, activate and use `.venv` instead.

### ArcGIS credentials

An example environment file is included:

```powershell
Copy-Item .env.example .env
```

`ARCGIS_URL`, `ARCGIS_USERNAME`, and `ARCGIS_PASSWORD` are only needed when you want an authenticated ArcGIS connection. The current public download steps use an anonymous `GIS()` connection.

### Quarto / notebook execution

The analysis is authored as Quarto notebooks (`.qmd`). You can run the workflow either:

- interactively in an editor that supports Quarto notebooks
- by rendering the notebooks with the Quarto CLI

Example:

```powershell
quarto render 00_data_preparation.qmd
quarto render 01_node_classification.qmd
```

If you prefer to work inside Python tooling, the repo also includes Jupyter-related dependencies for notebook-oriented development.

## Expected Local Data Layout

The code currently assumes a local `_data/` directory like this:

```text
_data/
|-- raw/
|   |-- BangerterEdits.gdb.zip
|   |-- UtahRoads.gpkg
|   |-- WFv1000_MasterNet - Link.gpkg
|   |-- WFv1000_MasterNet - Node.gpkg
|   `-- uta_gtfs.zip
`-- processed/
    |-- nodes_classified.gpkg
    `-- nodes_snapped.gpkg
```

`_data/` is ignored by git, so source and processed datasets are treated as local working files rather than versioned artifacts.

## Current Outputs

### `nodes_classified.gpkg`

This is the network node layer with added classification fields and `LinkCount`.

### `nodes_snapped.gpkg`

This is the classified node layer with updated geometry plus snapping audit fields:

- `snap_rule`
- `snap_distance_m`
- `snapped`

These outputs are intended to support both downstream geometry work and QA/QC.

## Assumptions and Caveats

The repo is useful now, but it is still clearly an in-progress workflow. The main caveats visible from the current code are:

1. `00_data_preparation.qmd` downloads `UtahRoads.gpkg`, but `01_node_classification.qmd` currently reads `_data/raw/BangerterEdits.gdb.zip` with layer `RoadsBangerterEtc` as its centerline source.
2. That means the preparation notebook and classification notebook are not yet fully aligned on the roadway input used for snapping.
3. `00_data_preparation.qmd` references `load_dotenv()` and `urllib.request.urlretrieve()` but does not currently import `load_dotenv` or `urllib.request` in the notebook.
4. The snapping workflow is only partially automated; the notebook explicitly includes map-based validation steps and expects the analyst to review results between snapping stages.
5. GTFS is currently used only for rail stop snapping, not for full transit line shape construction or bus stop integration.
6. The repository is not yet packaged as a CLI or Python library with a single end-to-end entrypoint.

## Suggested End-to-End Run Order

For the workflow that exists today:

1. Prepare a Python 3.11 environment.
2. Install dependencies from `pyproject.toml` / `uv.lock`.
3. Add `.env` only if authenticated ArcGIS access is needed.
4. Run `00_data_preparation.qmd` to download the public datasets.
5. Confirm the roadway centerline input you want to use for node snapping, because the next notebook currently expects `BangerterEdits.gdb.zip`.
6. Run `01_node_classification.qmd`.
7. Review the pre-snap and post-snap maps before accepting each stage.
8. Use the GeoPackage outputs in `_data/processed/` for downstream true-shape work.

## Status

This repository currently functions as:

- a reproducible exploratory workflow for classifying TDM nodes
- a rule-based snapping pipeline for moving nodes onto roadway and rail geometry
- a foundation for a fuller TDM-to-true-shape network conversion process

It should be treated as an active working project rather than a finalized production tool.

## License

See [`LICENSE`](./LICENSE).
