# Isidion NASA Space Apps

A terrain-analog search workflow that compares a lunar digital elevation model (DEM) with Earth terrain. It preprocesses elevation rasters, calculates terrain features, searches Earth windows for similar patterns, and saves ranked results, comparison images, and an optional interactive map.

> **The data are not included in this repository.** DEM files—especially large, high-resolution Earth rasters and processed NumPy arrays—can be many gigabytes. Committing them would make cloning and updating the code unnecessarily difficult, and data sources may have their own licensing or attribution requirements. You download or obtain appropriate DEMs separately and keep them on your computer. The repository contains the code and instructions, not a ready-to-run copy of the datasets.

## What the project does

1. Reads single-band Earth and lunar elevation GeoTIFFs.
2. Reprojects Earth data to a projected, meter-based coordinate system and resamples lunar data to a meter-based grid.
3. Normalizes valid elevation values and preserves nodata cells as `NaN`.
4. Compares a lunar terrain template against sampled Earth locations using elevation, slope, and local roughness.
5. Writes ranked matches and side-by-side comparison plots. An optional script maps the Earth matches.

The search is a terrain-pattern comparison, not a claim that lunar and terrestrial geology or gravity are equivalent.

## Repository contents

```text
.
├── analyze_data.py           # Fast, approximate terrain-analog search
├── data_processing.py        # GeoTIFF preprocessing pipeline
├── map_analogs.py            # Optional interactive map of Earth matches
├── README.md
├── LICENSE
└── data/                     # Create locally; datasets are not included
    ├── raw_earth/            # Input Earth DEM GeoTIFFs
    ├── raw_moon/             # Input lunar DEM GeoTIFFs
    └── processed/            # Generated .npy arrays and metadata
```

Generated files are written locally, including `top_terrain_analogs.json`, `analog_comparison_rank_*.png`, and (if you run the mapping script) `terrain_analogs_map.html`.

## Requirements

- Python 3.10 or newer
- A computer with enough free disk space for the source DEMs and generated arrays. The Earth array can be very large; memory mapping reduces startup memory use but does not eliminate disk, I/O, or processing requirements.
- An internet connection to load the background tiles in the interactive map.

Install the core dependencies from the repository root:

```bash
python -m pip install numpy scipy matplotlib rasterio
```

For the optional interactive map, also install Folium:

```bash
python -m pip install folium
```

## Input data: exactly what to use

Use **digital elevation models (DEMs)** that store terrain elevation, not ordinary photographs, RGB satellite imagery, hillshade images, land-cover classifications, or rendered map images.

Accepted inputs are:

- **File format:** GeoTIFF, with a `.tif` or `.tiff` extension.
- **Raster bands:** exactly one band containing elevation values. RGB or multi-band images are not accepted.
- **Georeferencing:** each GeoTIFF must have a valid coordinate reference system (CRS). The lunar DEM must use a projected CRS whose linear units are meters. Missing CRS information or an unprojected lunar CRS will cause preprocessing to stop for that file.
- **Nodata:** a GeoTIFF nodata value or mask is recommended; it is retained as `NaN` in processed arrays.
- **Coverage and resolution:** choose a lunar DEM area you want to compare and an Earth DEM with a sufficiently large, useful area. Processing defaults to a 30-meter pixel size. A larger area or finer source data can generate very large files and take substantially longer.

Place the input files directly in these folders (the scripts scan these folders, not nested subfolders):

```text
data/
├── raw_earth/
│   └── earth_dem.tif
└── raw_moon/
    └── moon_dem.tif
```

Use DEMs from sources whose download and reuse terms are suitable for your project. Keep any required source citations and attribution with your project materials.

### Coordinate system note

Earth data are reprojected to `EPSG:32616` (UTM zone 16N) by default at 30-meter resolution. **That default is only appropriate for Earth locations covered by UTM zone 16N.** If your Earth DEM is elsewhere, choose a suitable projected CRS for that location using `--target-crs` when preprocessing. Earth data must use a projected CRS with meter units after reprojection.

The lunar input must already have a projected, meter-based CRS; the preprocessor resamples it while preserving its CRS. The metadata records the CRS and pixel-to-coordinate affine transform. Whether lunar coordinates can be converted to meaningful lunar latitude/longitude depends on the source DEM's CRS definition and datum.

## Step-by-step instructions

Run commands from the repository root—the folder containing `data_processing.py` and `analyze_data.py`.

### 1. Add your DEM files

Put at least one valid Earth `.tif`/`.tiff` in `data/raw_earth/` and one valid lunar `.tif`/`.tiff` in `data/raw_moon/`. Create these directories if they do not exist.

### 2. Preprocess the DEMs

Preprocess the inputs and write arrays plus metadata, without starting the older full-resolution matcher:

```bash
python data_processing.py --no-match
```

Optional arguments:

```bash
python data_processing.py --no-match --resolution 30 --target-crs EPSG:32616
```

Change `EPSG:32616` to a suitable projected meter-based CRS for your Earth dataset. The default resolution is 30 meters. The preprocessing script creates `data/processed/` and writes files named from the source basenames, for example:

```text
data/processed/
├── earth_dem_earth.npy
├── moon_dem_moon.npy
└── spatial_metadata.json
```

The metadata file records the processed paths, CRS, affine transform, grid dimensions, resolution, source filenames, and normalization details. Keep it alongside the generated arrays; the analysis and mapping scripts rely on it.

> Running `python data_processing.py` **without** `--no-match` also starts the older exhaustive matcher after preprocessing. It can take a very long time and use substantial resources. For the recommended quick workflow, use `--no-match` and then run `analyze_data.py` separately.

### 3. Run the fast approximate search

```bash
python analyze_data.py
```

The analysis script reads the first Earth and Moon entries listed in `spatial_metadata.json`. It opens the arrays using NumPy memory mapping and writes:

- `top_terrain_analogs.json` — ranked candidates, scores, weights, feature RMSEs, UTM coordinates, and plot paths.
- `analog_comparison_rank_1.png`, etc. — side-by-side plots of the downsampled lunar template and each Earth match.

The default search checks up to 128 sampled candidate windows, at a downsampling factor of 4 and stride of 100. This is intentionally a quick approximate pass; it does **not** examine every possible Earth location or guarantee the global best match.

### 4. (Optional) Create an interactive map

After `top_terrain_analogs.json` has been created, run:

```bash
python -m pip install folium
python map_analogs.py
```

Open `terrain_analogs_map.html` in a browser. It displays numbered markers for the ranked Earth matches. Select a marker to see its score and coordinate details. Background map tiles require an internet connection. This script reads the results and spatial metadata and writes a new HTML file; it does not change the DEMs or analysis results.

## Understanding the results

Each result includes a candidate's pixel row and column, the scale factor used, a combined score, the feature weights and RMSEs, and Earth coordinates (`utm_x_m`, `utm_y_m`). The Earth CRS comes from the Earth entry in `spatial_metadata.json`; the default preprocessing CRS is UTM zone 16N, EPSG:32616.

**A lower score means a closer match among the candidate windows that were actually tested.** The score combines weighted root-mean-square errors (RMSEs) for:

- **Elevation** — normalized elevation values
- **Slope** — local change in elevation
- **Roughness** — local variation in elevation

The score is a relative similarity measure, not a probability, a physical distance, or proof that the landscapes are geologically identical. The default weights are defined in `analyze_data.py`.

The plots use a terrain color map. In the quick-search plots, each panel may be autoscaled independently, so the same color in the two panels does not necessarily mean the same elevation. The displayed lunar patch is the downsampled template used by the search.

## Adjusting search speed and coverage

The main controls are near the top of `analyze_data.py`:

- `SCALE_FACTOR = 4`: samples every fourth pixel in each direction. Lowering it compares more detail but greatly increases work.
- `STRIDE = 100`: spacing between tested candidate positions on the downsampled Earth grid. Lowering it tests more locations.
- `MAX_CANDIDATES = 128`: caps the number of candidate windows. Increase it for a broader sample or set it to `None` to test every position generated by the stride. This can take much longer.
- `EARTH_BBOX = None`: searches the full Earth array. Set it to `(row_start, row_stop, col_start, col_stop)` to limit the search to a region in the original Earth array's pixel coordinates.
- `CHUNK_ROWS`: controls chunk size for feature processing. Smaller values can reduce temporary memory use but may increase processing overhead.
- `WORKERS`: number of parallel worker processes; the default uses at most two.

For a finer follow-up search, focus on a smaller region around an interesting candidate and reduce `STRIDE` and/or `SCALE_FACTOR`. A full-resolution exhaustive search over a huge DEM can be impractical.

## Troubleshooting

- **Missing processed array:** check that `spatial_metadata.json` points to the exact `.npy` filenames in `data/processed/`. Do not rename or substitute an array unless it is the corresponding processed DEM.
- **No DEMs processed:** confirm that the GeoTIFFs are directly inside the correct raw-data folder, are single-band elevation rasters, and contain a CRS. The lunar CRS must be projected and use meters.
- **Large disk or slow processing:** use a smaller source DEM or smaller region, and verify available disk space before preprocessing. Generated `.npy` arrays can be large.
- **Map does not display its background:** open the HTML file with an internet connection; the map tiles are loaded online.

## License and data attribution

The code is licensed under the terms in [`LICENSE`](LICENSE). That license does not automatically grant rights to any DEM data you obtain. Follow the terms and attribution requirements of each data provider.

## Acknowledgements

Developed for NASA Space Apps as a demonstration of lunar-to-Earth terrain-pattern comparison using digital elevation models.
