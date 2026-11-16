# Isidion: Quantitative Planetary Analog Discovery Engine

A terrain-analog search workflow that compares a lunar digital elevation model (DEM) with Earth terrain. It preprocesses elevation rasters, calculates terrain features, searches Earth windows for similar patterns, and saves ranked matches, comparison images, and an optional interactive map.

> [!NOTE]
> **Mars DEMs can be used as the non-Earth template, with a naming caveat.** The current code does not have a dedicated Mars input folder or planet selector: it calls the template input `moon` and looks for DEMs in `data/raw_moon/`. A compatible Mars DEM can be supplied there and compared with an Earth DEM using the existing workflow. It will still be recorded/labeled as a `moon` input in filenames and metadata. This is terrain-pattern comparison only: the code does not verify that a DEM comes from Mars or provide Mars-specific coordinate conversion or planetary interpretation. See [Using a Mars DEM](#using-a-mars-dem).

> [!WARNING]
> **CRITICAL: choose the correct Earth coordinate system for your region before preprocessing.** The current code does **not** automatically choose a CRS based on your DEM. Its default, `EPSG:32616` (UTM zone 16N), is only appropriate for locations covered by that UTM zone. For a different Earth region, pass an appropriate projected CRS with meter units using `--target-crs`. Using the wrong CRS can make reprojection, reported coordinates, and map markers inaccurate. A single UTM CRS is not suitable for a global dataset; process global coverage as regional tiles, each with a suitable CRS.

> [!WARNING]
> **CRITICAL: do not run the full exhaustive matcher by accident.** Running `python data_processing.py` without `--no-match` preprocesses the DEMs and then starts the older exhaustive matcher. That routine can be extremely slow and memory-heavy because it checks many candidate windows at full resolution. For the intended workflow, use:
>
> ```bash
> python data_processing.py --no-match
> python analyze_data.py
> ```
>
> Use `--no-match` unless you intentionally want the older slow matcher.

> **The data are not included in this repository.** High-resolution DEMs and generated arrays can be many gigabytes. Obtain suitable DEMs separately and keep them locally; this repository contains code and instructions, not a preloaded dataset. Follow data-provider licensing and attribution terms.

## What the project does

1. Reads single-band Earth and lunar elevation GeoTIFFs.
2. Reprojects Earth data to a selected projected, meter-based CRS and resamples both grids to a requested pixel size.
3. Normalizes valid elevation values to `[0, 1]`, preserving nodata as `NaN`.
4. Compares a lunar terrain template against sampled Earth locations using elevation, slope, and local roughness.
5. Writes ranked matches and side-by-side plots. An optional script maps the Earth matches.

This is a numerical terrain-pattern comparison, not a claim that lunar and terrestrial geology or gravity are equivalent.

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
    ├── raw_moon/             # Input lunar or Mars template DEM GeoTIFFs
    └── processed/            # Generated .npy arrays and metadata
```

Generated files are local: `top_terrain_analogs.json`, `analog_comparison_rank_*.png`, and, if the mapping script is run, `terrain_analogs_map.html`.

## Requirements and installation

- Python 3.10 or newer
- Enough free disk space for input DEMs and processed arrays. Memory mapping reduces startup memory use but does not eliminate storage, I/O, or processing requirements.
- An internet connection to load background tiles in the optional map.

Install core dependencies from the repository root:

```bash
python -m pip install numpy scipy matplotlib rasterio
```

For the optional interactive map:

```bash
python -m pip install folium
```

## Input data: exactly what to use

Use **digital elevation models (DEMs)** that store terrain elevation—not ordinary photographs, RGB satellite imagery, hillshades, land-cover classifications, or rendered map images.

Accepted inputs:

- **Format:** GeoTIFF (`.tif` or `.tiff`).
- **Bands:** exactly one elevation band; multi-band RGB imagery is not accepted.
- **CRS:** each GeoTIFF must have valid CRS metadata. The lunar or Mars template CRS must be projected and use meters. The Earth input must have a valid CRS so it can be reprojected.
- **Nodata:** a GeoTIFF nodata value or mask is recommended and is represented as `NaN` in processed arrays.
- **Coverage/resolution:** select a planetary template DEM area to compare and an Earth DEM covering the Earth region you want to search. Preprocessing defaults to 30-meter pixels. Larger areas and finer resolution create larger arrays and take longer.

Place files directly in these folders; the scripts do not scan nested folders:

```text
data/
├── raw_earth/
│   └── earth_dem.tif
└── raw_moon/
    └── moon_or_mars_dem.tif
```

Use data whose provider terms permit your intended use. Keep any required source citations and attribution with your project materials.

## Using a Mars DEM

The processing and search code uses generic raster operations, so it can use a Mars elevation DEM as the template if it meets the same input requirements: single-band GeoTIFF, valid projected CRS with meter units, and usable elevation data. Put the Mars DEM in `data/raw_moon/` because that is the directory the code currently scans. The generated filename and metadata entry will still use the `_moon` suffix and the `moon` metadata key; this is a code naming convention, not a claim that the input is lunar.

The program does not identify a DEM's planetary body, validate Mars-specific datums, or calculate Mars geographic coordinates. It compares normalized terrain patterns and reports the matched **Earth** location. Check the source DEM's CRS/datum and pixel scale before use; matching does not make coordinate systems or physical elevations equivalent. In this repository, “Mars support” means a compatible Mars DEM can serve as the comparison template—not that there is a separate, fully Mars-aware workflow.

## Coordinate systems and regions

The Earth preprocessing default is `EPSG:32616`, UTM zone 16N. **This is not a universal Earth CRS.** Choose a suitable projected CRS for the area covered by your Earth DEM, in meters. For a local or regional DEM, an appropriate UTM zone is often a practical choice. The code currently requires the user to select the CRS; it does not automatically infer one from the data.

For example, `EPSG:32616` is appropriate only for a region in UTM zone 16N. Replace it in the command below with the correct EPSG code for your region. If your Earth dataset spans multiple UTM zones or the whole planet, process regional tiles separately with suitable CRSs rather than projecting the globe into one local UTM CRS. The template retains its own projected CRS; Earth and template CRSs are not required to be the same because this workflow compares terrain patterns, not cross-planet geographic coordinates.

The lunar or Mars input must already have a projected, meter-based CRS. It is resampled while retaining its source CRS. Metadata records each grid's CRS and affine transform. Whether template grid coordinates can be converted into meaningful planetary latitude/longitude depends on the source CRS and datum definition; this code does not perform that planetary-coordinate conversion.

## Step-by-step instructions

Run commands from the repository root—the directory containing `data_processing.py` and `analyze_data.py`.

### 1. Add your DEMs

Put at least one valid Earth GeoTIFF in `data/raw_earth/` and one valid lunar or Mars template GeoTIFF in `data/raw_moon/`. Create those directories if they do not exist.

### 2. Preprocess

Use `--no-match` to prepare arrays and metadata without starting the older exhaustive matcher:

```bash
python data_processing.py --no-match --resolution 30 --target-crs EPSG:32616
```

Replace `EPSG:32616` with the appropriate projected, meter-based CRS for your Earth region. The default resolution is 30 meters; change it if appropriate for the datasets and your analysis. The preprocessing script creates `data/processed/` and writes files named from the source basenames, for example:

```text
data/processed/
├── earth_dem_earth.npy
├── template_dem_moon.npy
└── spatial_metadata.json
```

The template output keeps the `_moon.npy` naming convention even if the input is a Mars DEM. `spatial_metadata.json` records processed paths, CRS, affine transform, grid dimensions, resolution, source filenames, and normalization details. Keep it with the arrays; the analysis and mapping scripts rely on it.

> [!WARNING]
> **Do not forget `--no-match` unless you intentionally want the slow exhaustive search.** The default `python data_processing.py` command preprocesses and then launches the older full-resolution matcher. That routine explores many more candidate windows and can be extremely slow and memory-heavy. The intended path is:
>
> ```bash
> python data_processing.py --no-match
> python analyze_data.py
> ```

### 3. Run the fast approximate search

```bash
python analyze_data.py
```

The analysis script reads the first Earth and Moon entries listed in `spatial_metadata.json`. It opens the arrays using NumPy memory mapping and writes:

- `top_terrain_analogs.json` — ranked candidates, scores, weights, feature RMSEs, UTM coordinates, and plot paths.
- `analog_comparison_rank_1.png`, etc. — side-by-side plots of the downsampled template and each Earth match.

The default search checks up to 128 sampled candidate windows, at a downsampling factor of 4 and stride of 100. It does **not** examine every possible Earth location or guarantee the global best match. The input Earth DEM must cover the place you want to search.

### 4. (Optional) Create an interactive map

After `top_terrain_analogs.json` has been created:

```bash
python -m pip install folium
python map_analogs.py
```

Open `terrain_analogs_map.html` in a browser. It displays numbered markers for ranked Earth matches. Select a marker for its score and coordinate details. Background map tiles require an internet connection. This script uses the Earth CRS and transform recorded in `spatial_metadata.json`.

## Understanding the results and score

Each result includes a candidate's pixel row and column, scale factor, combined score, feature weights and RMSEs, and Earth coordinates (`utm_x_m`, `utm_y_m`). The Earth CRS comes from the Earth entry in metadata and should match the CRS selected during preprocessing.

The fast search calculates an RMSE for each of elevation, slope, and roughness over finite overlapping samples, then combines them:

```text
score = elevation_weight * elevation_RMSE
      + slope_weight * slope_RMSE
      + roughness_weight * roughness_RMSE
```

The default weight combinations are defined near the top of `analyze_data.py`; each has three non-negative values summing to one. **Lower scores mean smaller weighted differences among candidates actually tested.** The score is a relative similarity measure, not a probability, physical distance, or proof of geological identity. The fast search has no maximum-score cutoff or minimum valid-overlap fraction; a candidate is skipped when any feature has no valid overlap.

The terrain-color plots are visual aids. Each panel can be scaled independently, so the same color in the two panels does not necessarily mean the same numeric elevation. The score uses underlying array values, not the displayed colors.

### Important: top five may not mean five unique places

The script evaluates multiple weight combinations for each candidate and retains `TOP_N = 5` result records overall. The same Earth location can therefore appear more than once with different weights. The output is the top five **location-and-weight combinations**, not guaranteed to be five distinct locations.

The saved UTM coordinate is calculated from the candidate's starting pixel location (pixel center), not the center of the entire matched window. The plot displays the whole window.

## Search settings: speed versus coverage

The primary controls are near the top of `analyze_data.py`:

- `SCALE_FACTOR = 4`: samples every fourth pixel in each direction. Lower values retain more detail but substantially increase work.
- `STRIDE = 100`: spacing between tested starts on the downsampled Earth grid. Lower values test more positions; larger values are faster but skip more locations. At a 30-meter source pixel size, the defaults imply roughly 120-meter sample spacing and roughly 12-kilometer spacing between regular candidate starts.
- `MAX_CANDIDATES = 128`: caps the tested windows. Increase it for broader sampling or set it to `None` to test every position generated by the stride. This may take much longer.
- `EARTH_BBOX = None`: searches the full Earth array. Set `(row_start, row_stop, col_start, col_stop)` to restrict the search using original Earth-array pixel coordinates; stop indices are exclusive.
- `CHUNK_ROWS = 64`: controls chunk size for feature processing. Smaller values can reduce temporary memory but add processing overhead.
- `WORKERS`: number of parallel worker processes; the default uses at most two.
- `TOP_N = 5`: number of result records retained.

For a finer follow-up, restrict `EARTH_BBOX` around a promising area and reduce `STRIDE` and/or `SCALE_FACTOR`. A full-resolution exhaustive search over a huge DEM can be impractical.

## Troubleshooting

- **No DEMs processed:** make sure the GeoTIFFs are directly in the correct raw folder, are single-band elevation rasters, and contain CRS metadata. The lunar or Mars template CRS must be projected and use meters.
- **Wrong or implausible Earth coordinates:** verify that `--target-crs` is appropriate for the Earth DEM region. Do not rely on the default UTM zone 16N for other regions.
- **Missing processed array:** check that `spatial_metadata.json` refers to existing `.npy` files in `data/processed/`.
- **Multiple input DEMs:** the fast analysis currently selects the first Earth and first Moon metadata entries. Keep one of each in the workflow or verify which entries appear first in metadata.
- **Large disk use or slow processing:** use a smaller source region or a coarser resolution and check free disk space before preprocessing. Generated arrays can be large.
- **Map does not display its background:** open the HTML file with an internet connection; background map tiles are loaded online.

## License and data attribution

The code is licensed under the terms in [`LICENSE`](LICENSE). That license does not automatically grant rights to DEM data you obtain. Follow each data provider's license and attribution requirements.

## Acknowledgements

Developed for NASA Space Apps as a demonstration of lunar-to-Earth terrain-pattern comparison using digital elevation models.
