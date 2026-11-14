# Isidion-Nasa-Space-Apps

## Running the terrain analog matcher

The matcher opens the large Earth `.npy` elevation array read-only with NumPy memory mapping and computes terrain features in small row chunks. It does not create full-size Earth slope or roughness arrays. The top five window/weight results are saved to `top_terrain_analogs.json`, with one downsampled comparison PNG per result.

Expected inputs:

- `data/processed/normalized_earth_*.npy`
- `data/processed/normalized_moon_*.npy`
- `data/processed/spatial_metadata.json` (Earth transform at the top level or under an `earth` key)

Install dependencies and run from the repository root:

```bash
python -m pip install numpy scipy matplotlib rasterio
python analyze_data.py
```

Adjust `STRIDE` and `CHUNK_ROWS` in `analyze_data.py` to trade runtime for candidate coverage and working memory. Smaller chunks reduce peak working memory; smaller stride evaluates more locations and can substantially increase runtime. Memory mapping avoids loading the entire DEM at startup, but the operating system may still use available RAM for its file cache.
