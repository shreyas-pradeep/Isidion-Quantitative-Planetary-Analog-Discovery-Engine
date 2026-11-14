from __future__ import annotations

"""Memory-conscious terrain analog search for large NumPy DEM files.

The Earth DEM is opened read-only using memory mapping. Search is performed on
sampled candidate windows and computes Earth features in row chunks. Configure
STRIDE and EARTH_BBOX below to trade search coverage for runtime.
"""

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rasterio import Affine
from scipy import ndimage

PROCESSED_DIR = Path("data/processed")
RESULTS_FILE = Path("top_terrain_analogs.json")
OUTPUT_IMAGE_TEMPLATE = "analog_comparison_rank_{rank}.png"
TOP_N = 5
STRIDE = 100  # Increase for speed; smaller values sample more candidate locations.
CHUNK_ROWS = 64
PLOT_MAX_DIMENSION = 1200
# Optional (row_start, row_stop, col_start, col_stop), pixel bounds in the full
# Earth array. Set to None for a global scan. Example: (10000, 20000, 30000, 40000)
EARTH_BBOX: tuple[int, int, int, int] | None = None
WEIGHT_COMBINATIONS = [
    (0.50, 0.25, 0.25),
    (0.60, 0.20, 0.20),
    (0.40, 0.30, 0.30),
    (0.70, 0.15, 0.15),
    (0.30, 0.35, 0.35),
]


def _load_single_array(pattern: str) -> np.ndarray:
    """Open the unique matching .npy file as a read-only memory map."""
    files = sorted(PROCESSED_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern!r} in {PROCESSED_DIR}")
    if len(files) > 1:
        raise RuntimeError(
            f"Expected one file matching {pattern!r}, found: "
            + ", ".join(str(path) for path in files)
        )
    return np.load(files[0], mmap_mode="r", allow_pickle=False)


def load_data_and_metadata() -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    earth_grid = _load_single_array("normalized_earth_*.npy")
    moon_grid = _load_single_array("normalized_moon_*.npy")
    if earth_grid.ndim != 2 or moon_grid.ndim != 2:
        raise ValueError("Earth and Moon arrays must both be two-dimensional.")

    metadata_path = PROCESSED_DIR / "spatial_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Spatial metadata not found: {metadata_path}")
    with metadata_path.open(encoding="utf-8") as file:
        metadata = json.load(file)
    return earth_grid, moon_grid, metadata


def _read_feature_chunk(
    grid: np.ndarray,
    row_start: int,
    row_stop: int,
    col_start: int,
    col_stop: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a bounded chunk and calculate elevation, slope and roughness."""
    height, width = grid.shape
    if not (0 <= row_start < row_stop <= height and 0 <= col_start < col_stop <= width):
        raise ValueError("Invalid feature-chunk bounds.")

    # A one-pixel halo gives chunk-edge derivatives/statistics their neighbors.
    src_r0, src_r1 = max(0, row_start - 1), min(height, row_stop + 1)
    src_c0, src_c1 = max(0, col_start - 1), min(width, col_stop + 1)
    source = np.asarray(grid[src_r0:src_r1, src_c0:src_c1], dtype=np.float32)
    finite = np.isfinite(source)
    padded = np.pad(source, 1, mode="constant", constant_values=np.nan)

    core_r0, core_c0 = row_start - src_r0 + 1, col_start - src_c0 + 1
    out_h, out_w = row_stop - row_start, col_stop - col_start
    rows = slice(core_r0, core_r0 + out_h)
    cols = slice(core_c0, core_c0 + out_w)
    center = padded[rows, cols]
    left = padded[rows, slice(core_c0 - 1, core_c0 - 1 + out_w)]
    right = padded[rows, slice(core_c0 + 1, core_c0 + 1 + out_w)]
    up = padded[slice(core_r0 - 1, core_r0 - 1 + out_h), cols]
    down = padded[slice(core_r0 + 1, core_r0 + 1 + out_h), cols]

    center_ok = np.isfinite(center)
    dx = np.full(center.shape, np.nan, dtype=np.float32)
    dy = np.full(center.shape, np.nan, dtype=np.float32)

    left_ok, right_ok = np.isfinite(left), np.isfinite(right)
    both = center_ok & left_ok & right_ok
    dx[both] = (right[both] - left[both]) * 0.5
    use = center_ok & ~left_ok & right_ok
    dx[use] = right[use] - center[use]
    use = center_ok & left_ok & ~right_ok
    dx[use] = center[use] - left[use]

    up_ok, down_ok = np.isfinite(up), np.isfinite(down)
    both = center_ok & up_ok & down_ok
    dy[both] = (down[both] - up[both]) * 0.5
    use = center_ok & ~up_ok & down_ok
    dy[use] = down[use] - center[use]
    use = center_ok & up_ok & ~down_ok
    dy[use] = center[use] - up[use]

    slope = np.hypot(dx, dy)
    slope[~(np.isfinite(dx) & np.isfinite(dy))] = np.nan

    # Nodata-aware 3x3 local population standard deviation.
    values = np.where(finite, source, 0.0).astype(np.float32, copy=False)
    mask = finite.astype(np.float32, copy=False)
    sums = ndimage.uniform_filter(values, size=3, mode="constant", cval=0) * 9
    squares = ndimage.uniform_filter(values * values, size=3, mode="constant", cval=0) * 9
    counts = ndimage.uniform_filter(mask, size=3, mode="constant", cval=0) * 9
    core = (slice(row_start - src_r0, row_stop - src_r0),
            slice(col_start - src_c0, col_stop - src_c0))
    count = counts[core]
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = sums[core] / count
        variance = np.maximum(squares[core] / count - mean * mean, 0)
        roughness = np.sqrt(variance).astype(np.float32, copy=False)
    roughness[(count == 0) | ~center_ok] = np.nan
    return center.copy(), slope, roughness


def compute_terrain_features(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute full-grid features chunkwise; use only for grids that fit in RAM."""
    slope = np.empty(grid.shape, dtype=np.float32)
    roughness = np.empty(grid.shape, dtype=np.float32)
    for row in range(0, grid.shape[0], CHUNK_ROWS):
        stop = min(row + CHUNK_ROWS, grid.shape[0])
        _, slope[row:stop], roughness[row:stop] = _read_feature_chunk(
            grid, row, stop, 0, grid.shape[1]
        )
    return slope, roughness


def _validate_weights(weights: list[tuple[float, float, float]]) -> None:
    if not weights:
        raise ValueError("At least one weight combination is required.")
    for item in weights:
        if len(item) != 3 or any(weight < 0 for weight in item):
            raise ValueError(f"Weights must be three non-negative values: {item!r}")
        if not np.isclose(sum(item), 1.0):
            raise ValueError(f"Weights must sum to 1.0: {item!r}")


def _window_starts(length: int, window: int, stride: int) -> list[int]:
    if window > length:
        return []
    starts = list(range(0, length - window + 1, stride))
    final = length - window
    if not starts or starts[-1] != final:
        starts.append(final)
    return starts


def _score_candidate(
    earth_grid: np.ndarray,
    moon_grid: np.ndarray,
    moon_features: tuple[np.ndarray, np.ndarray],
    row: int,
    col: int,
    chunk_rows: int,
) -> tuple[float, float, float] | None:
    """Accumulate the three RMSE metrics over stripes to bound temporary memory."""
    template_rows, template_cols = moon_grid.shape
    squared_error = np.zeros(3, dtype=np.float64)
    valid_counts = np.zeros(3, dtype=np.int64)
    moon_slope, moon_roughness = moon_features

    for offset in range(0, template_rows, chunk_rows):
        stop = min(offset + chunk_rows, template_rows)
        earth_values, earth_slope, earth_roughness = _read_feature_chunk(
            earth_grid, row + offset, row + stop, col, col + template_cols
        )
        moon_values, moon_slope_chunk, moon_rough_chunk = _read_feature_chunk(
            moon_grid, offset, stop, 0, template_cols
        )
        # Use the precomputed Moon terrain features; only its elevation stripe is read.
        moon_slope_chunk = moon_slope[offset:stop]
        moon_rough_chunk = moon_roughness[offset:stop]

        for index, (first, second) in enumerate((
            (earth_values, moon_values),
            (earth_slope, moon_slope_chunk),
            (earth_roughness, moon_rough_chunk),
        )):
            valid = np.isfinite(first) & np.isfinite(second)
            if valid.any():
                delta = first[valid].astype(np.float64) - second[valid].astype(np.float64)
                squared_error[index] += np.dot(delta, delta)
                valid_counts[index] += delta.size

    if np.any(valid_counts == 0):
        return None
    return tuple(np.sqrt(squared_error / valid_counts).tolist())


def sliding_window_multimetric_search(
    earth_grid: np.ndarray,
    moon_grid: np.ndarray,
    stride: int = STRIDE,
    weights: list[tuple[float, float, float]] | None = None,
    top_n: int = TOP_N,
    chunk_rows: int = CHUNK_ROWS,
    bounding_box: tuple[int, int, int, int] | None = None,
) -> list[dict[str, Any]]:
    """Search window/weight pairs, optionally restricting search to an Earth ROI.

    bounding_box is (row_start, row_stop, col_start, col_stop), with stop
    coordinates exclusive. Results contain pixel indices in the original grid.
    """
    if stride <= 0 or chunk_rows <= 0 or top_n <= 0:
        raise ValueError("stride, chunk_rows, and top_n must be positive.")
    if earth_grid.ndim != 2 or moon_grid.ndim != 2:
        raise ValueError("Earth and Moon grids must be two-dimensional.")
    weights = WEIGHT_COMBINATIONS if weights is None else weights
    _validate_weights(weights)

    if bounding_box is None:
        r0, r1, c0, c1 = 0, earth_grid.shape[0], 0, earth_grid.shape[1]
    else:
        if len(bounding_box) != 4:
            raise ValueError("bounding_box must be (row_start, row_stop, col_start, col_stop).")
        r0, r1, c0, c1 = map(int, bounding_box)
        if not (0 <= r0 < r1 <= earth_grid.shape[0] and 0 <= c0 < c1 <= earth_grid.shape[1]):
            raise ValueError(f"Bounding box {bounding_box} is outside Earth grid {earth_grid.shape}.")

    template_rows, template_cols = moon_grid.shape
    if r1 - r0 < template_rows or c1 - c0 < template_cols:
        raise ValueError("Bounding box must be at least as large as the Moon template.")
    row_starts = [r0 + offset for offset in _window_starts(r1 - r0, template_rows, stride)]
    col_starts = [c0 + offset for offset in _window_starts(c1 - c0, template_cols, stride)]

    # Moon is small; compute its slope/roughness once, not once per Earth candidate.
    moon_slope, moon_roughness = compute_terrain_features(moon_grid)
    moon_features = (moon_slope, moon_roughness)
    best: list[dict[str, Any]] = []
    candidate_count = len(row_starts) * len(col_starts)
    print(
        f"Scanning {candidate_count:,} windows, {len(weights)} weight sets, "
        f"stride={stride}, ROI=rows[{r0}:{r1}] cols[{c0}:{c1}]...",
        flush=True,
    )

    for index, (row, col) in enumerate(
        ((r, c) for r in row_starts for c in col_starts), start=1
    ):
        rmse = _score_candidate(
            earth_grid, moon_grid, moon_features, row, col, chunk_rows
        )
        if rmse is not None:
            elev_rmse, slope_rmse, rough_rmse = rmse
            for w_elev, w_slope, w_rough in weights:
                score = w_elev * elev_rmse + w_slope * slope_rmse + w_rough * rough_rmse
                best.append({
                    "row": row,
                    "col": col,
                    "score": float(score),
                    "weights": {
                        "elevation": w_elev,
                        "slope": w_slope,
                        "roughness": w_rough,
                    },
                    "rmse": {
                        "elevation": elev_rmse,
                        "slope": slope_rmse,
                        "roughness": rough_rmse,
                    },
                })
            best.sort(key=lambda result: (result["score"], result["row"], result["col"]))
            del best[top_n:]

        if index % 100 == 0 or index == candidate_count:
            print(f"Scanned {index:,}/{candidate_count:,} windows", flush=True)

    if not best:
        raise ValueError("No candidate windows had valid data for all metrics.")
    return best


def _earth_metadata_entry(metadata: dict[str, Any]) -> dict[str, Any]:
    for key in ("earth", "Earth", "earth_metadata", "earth_raster"):
        entry = metadata.get(key)
        if isinstance(entry, dict) and "transform" in entry:
            return entry
    if "transform" in metadata:
        return metadata
    raise KeyError("Earth metadata must include a six-value 'transform'.")


def convert_pixel_to_geographic(
    row: int, col: int, metadata_entry: dict[str, Any]
) -> tuple[float, float]:
    transform = metadata_entry.get("transform")
    if transform is None or len(transform) < 6:
        raise ValueError("Metadata entry must contain a six-value 'transform' list.")
    affine = Affine(*transform[:6])
    x, y = affine * (col + 0.5, row + 0.5)
    return float(x), float(y)


def _plot_sample(grid: np.ndarray, max_dimension: int) -> np.ndarray:
    row_step = max(1, int(np.ceil(grid.shape[0] / max_dimension)))
    col_step = max(1, int(np.ceil(grid.shape[1] / max_dimension)))
    return np.asarray(grid[::row_step, ::col_step], dtype=np.float32)


def plot_terrain_analog(
    moon_grid: np.ndarray,
    earth_grid: np.ndarray,
    result: dict[str, Any],
    rank: int,
) -> str:
    row, col = result["row"], result["col"]
    h, w = moon_grid.shape
    earth_window = earth_grid[row:row + h, col:col + w]
    if earth_window.shape != moon_grid.shape:
        raise ValueError(f"Earth window for rank {rank} does not match Moon template.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    axes[0].imshow(_plot_sample(moon_grid, PLOT_MAX_DIMENSION), cmap="terrain")
    axes[0].set_title("Lunar template")
    axes[1].imshow(_plot_sample(earth_window, PLOT_MAX_DIMENSION), cmap="terrain")
    axes[1].set_title(f"Earth analog #{rank}: row={row}, col={col}\nScore={result['score']:.6f}")
    for axis in axes:
        axis.set_axis_off()
    output = OUTPUT_IMAGE_TEMPLATE.format(rank=rank)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    earth_grid, moon_grid, metadata = load_data_and_metadata()
    print(
        f"Earth grid {earth_grid.shape}, dtype={earth_grid.dtype}, "
        f"memory-mapped={isinstance(earth_grid, np.memmap)}"
    )
    print(f"Moon template {moon_grid.shape}, dtype={moon_grid.dtype}")

    results = sliding_window_multimetric_search(
        earth_grid,
        moon_grid,
        stride=STRIDE,
        weights=WEIGHT_COMBINATIONS,
        top_n=TOP_N,
        chunk_rows=CHUNK_ROWS,
        bounding_box=EARTH_BBOX,
    )
    earth_metadata = _earth_metadata_entry(metadata)
    for rank, result in enumerate(results, start=1):
        result["rank"] = rank
        result["utm_x_m"], result["utm_y_m"] = convert_pixel_to_geographic(
            result["row"], result["col"], earth_metadata
        )
        result["plot_file"] = plot_terrain_analog(moon_grid, earth_grid, result, rank)

    with RESULTS_FILE.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    print(f"\nTop {len(results)} matches:")
    for item in results:
        w = item["weights"]
        print(
            f"#{item['rank']} row={item['row']} col={item['col']} "
            f"score={item['score']:.6f} weights="
            f"({w['elevation']:.2f},{w['slope']:.2f},{w['roughness']:.2f}) "
            f"UTM=({item['utm_x_m']:.2f}, {item['utm_y_m']:.2f}) m "
            f"plot={item['plot_file']}"
        )
    print(f"Saved results to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
