# Isidion NASA Space Apps

A terrain-analog search tool for comparing lunar DEM data against Earth elevation data. The project preprocesses GeoTIFF rasters, normalizes their values, builds spatial metadata, and then tests candidate Earth windows against a lunar template using elevation, slope, and roughness similarity.

This repo was built to help identify the Earth analog most visually and statistically similar to a Moon terrain sample, using a fast approximate search pipeline suited to large rasters.

## Project goals

- Load and normalize Earth and Moon DEMs from GeoTIFF inputs
- Reproject and resample rasters to a common metric grid
- Compute terrain-aware features: elevation, slope, and roughness
- Search for nearby Earth windows matching a lunar template
- Save ranked results and comparison plots for the best matches

## Repository layout

```text
.
├── README.md
├── LICENSE
├── analyze_data.py
├── data_processing.py
├── data/
│   ├── raw_earth/
│   ├── raw_moon/
│   └── processed/
│       ├── spatial_metadata.json
│       ├── normalized_earth_*.npy
│       ├── normalized_moon_*.npy
│       └── analog comparison outputs
└── top_terrain_analogs.json
```

## What each script does

### `data_processing.py`

Preprocesses raw DEM GeoTIFF files:

- scans `data/raw_earth/` and `data/raw_moon/`
- reprojects Earth rasters to a metric grid
- resamples lunar rasters to a matching meter-based grid
- normalizes each DEM to `[0, 1]` with NaN preserved for nodata
- writes processed arrays and JSON metadata to `data/processed/`

This script is the preprocessing pipeline for the terrain matching workflow.

### `analyze_data.py`

Runs the fast terrain-analog search:

- loads the normalized Earth and Moon arrays
- computes per-pixel slope and roughness features
- samples candidate Earth windows across the grid
- scores candidates using weighted RMSE across elevation, slope, and roughness
- keeps the top results and generates comparison plots
- writes ranked output to `top_terrain_analogs.json`

This is the main analysis script for producing top analog matches.

## Installation

Use Python 3.10+ and install the required dependencies:

```bash
python -m pip install numpy scipy matplotlib rasterio
```

## Data preparation

Place your DEM files in the expected folders:

```text
data/
├── raw_earth/
│   └── your_earth_dem.tif
├── raw_moon/
│   └── your_moon_dem.tif
└── processed/
```

The project expects GeoTIFFs with single-band elevation data. The Earth DEM is projected to EPSG:32616 by default, while lunar DEMs are resampled to a meter-based projected grid in their source CRS.

## Running the preprocessing step

From the repository root:

```bash
python data_processing.py
```

This produces processed arrays and updates `data/processed/spatial_metadata.json` with metadata like:

- CRS
- affine transform
- width / height
- resolution
- source file paths
- processed file paths

## Running the analog search

```bash
python analyze_data.py
```

What this produces:

- `top_terrain_analogs.json`
- plot files such as `analog_comparison_rank_1.png`
- ranked output printed to the terminal

## Output interpretation

The top results are sorted by score, where lower is better. The score combines weighted RMSE values for:

- elevation
- slope
- roughness

This is a similarity score, not a physical probability. It indicates how closely a candidate region resembles the Moon template under the current feature weighting.

### Example output entry

```json
{
  "row": 2960,
  "col": 5120,
  "scale_factor": 4,
  "score": 0.089094,
  "weights": {
    "elevation": 0.5,
    "slope": 0.25,
    "roughness": 0.25
  },
  "rmse": {
    "elevation": 0.073,
    "slope": 0.102,
    "roughness": 0.088
  }
}
```

The plot side-by-side compares the Moon template and the best Earth analog found in the sampled search grid.

## Search configuration

The current analysis script uses a fast approximate search strategy. The following settings are important:

- `STRIDE`: spacing between candidate windows
- `SCALE_FACTOR`: downsampling factor used for the quick search
- `MAX_CANDIDATES`: maximum number of candidate windows to score
- `CHUNK_ROWS`: width of working chunks for feature evaluation
- `WORKERS`: number of multiprocessing workers

These are defined near the top of `analyze_data.py`.

### Typical quick-search settings

```python
STRIDE = 100
SCALE_FACTOR = 4
MAX_CANDIDATES = 128
WORKERS = min(2, os.cpu_count() or 1)
```

This allows a fast first pass over a large dataset without evaluating every possible location.

## Notes and limitations

- The search is intentionally approximate for speed.
- It does not guarantee the global optimum over the full Earth DEM.
- Lower scores are better because they represent smaller feature differences.
- The plot color scale is normalized per image, so the colors are a visual aid rather than a direct physical measurement.
- Performance can depend heavily on DEM size and available RAM.

## License

This project is licensed under the MIT License. See `LICENSE` for details.

## Acknowledgements

This project was developed for NASA Space Apps and demonstrates a terrain-analog matching workflow using normalized DEMs, terrain features, and ranked Earth-to-Moon comparison plots.
