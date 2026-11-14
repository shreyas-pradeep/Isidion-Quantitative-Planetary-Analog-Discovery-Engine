# Isidion-Nasa-Space-Apps

## Running the terrain analog matcher

The matcher opens the large Earth `.npy` elevation array read-only with NumPy memory mapping. It computes Earth terrain features in bounded chunks and keeps only the best matches. The top results are written to `top_terrain_analogs.json`, with a comparison PNG for each result.

Expected inputs:

- `data/processed/normalized_earth_*.npy`
- `data/processed/normalized_moon_*.npy`
- `data/processed/spatial_metadata.json` (Earth transform at the top level or under an `earth` key)

Install dependencies and run from the repository root:

```bash
python -m pip install numpy scipy matplotlib rasterio
python analyze_data.py
```

### Speed and region-of-interest controls

`STRIDE` defaults to 100 pixels. Candidate windows are separated by this many pixels in both directions. Larger values run faster but may skip the true best match; reduce it for a finer search.

For an initial small-area test, set `EARTH_BBOX` in `analyze_data.py` to `(row_start, row_stop, col_start, col_stop)` using pixel coordinates in the original Earth array. Stop indices are exclusive. The box must be at least as large as the Moon template. Leave it as `None` for a full-grid search. Returned row/column coordinates remain relative to the original Earth grid, so the existing affine transform remains valid.

`CHUNK_ROWS` controls the working set used while scoring a candidate. Lower values reduce temporary memory at the cost of more Python overhead. The Moon template's terrain features are computed once and reused for all candidates. For initial testing, choose a small bounding box and a large stride, then refine the region and stride once the pipeline runs successfully.

Memory mapping avoids loading the full DEM at startup, but the operating system may cache recently accessed file pages in available RAM. A global search can still take a long time because it evaluates every sampled candidate window.
