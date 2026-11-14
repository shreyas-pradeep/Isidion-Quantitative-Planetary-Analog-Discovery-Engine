from __future__ import annotations

"""Memory-conscious terrain analog search for large NumPy DEM files.

The Earth DEM is opened read-only with NumPy memory mapping. Search candidates
are scored in row chunks; the script never converts the full Earth DEM or its
terrain features into in-memory arrays. Runtime can still be substantial because
it compares every sampled candidate window with the Moon template.
"""

import heapq
import json
from pathlib import Path
from typing import Any

import matplotlib

# Use a non-interactive backend so the script works in terminals and headless runs.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rasterio import Affine
from scipy import ndimage


PROCESSED_DIR = Path("data/processed")
RESULTS_FILE = Path("top_terrain_analogs.json")
OUTPUT_IMAGE_TEMPLATE = "analog_comparison_rank_{rank}.png"
TOP_N = 5
STRIDE = 20
CHUNK_ROWS = 64
PLOT_MAX_DIMENSION = 1200
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
        raise FileNotFoundError(
            f"No files matching {pattern!r} found in {PROCESSED_DIR}"
        )
    if len(files) > 1:
        raise RuntimeError(
            f"Expected one file matching {pattern!r}, found: "
            + ", ".join(str(path) for path in files)
        )

    # Do not wrap this in np.asarray: preserving the memmap avoids an eager copy.
    return np.load(files[0], mmap_mode="r", allow_pickle=False)


def load_data_and_metadata() -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Memory-map the Earth and Moon grids and load their spatial metadata."""
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
    """Read and derive features for a rectangle, using only a one-cell halo.

    row_stop and col_stop are exclusive. NaNs are excluded from local
    statistics. A derivative is computed when the center and at least one
    finite neighbor are available in that direction.
    """
    height, width = grid.shape
    if not (0 <= row_start < row_stop <= height):
        raise ValueError("Invalid row bounds for feature chunk.")
    if not (0 <= col_start < col_stop <= width):
        raise ValueError("Invalid column bounds for feature chunk.")

    src_r0 = max(0, row_start - 1)
    src_r1 = min(height, row_stop + 1)
    src_c0 = max(0, col_start - 1)
    src_c1 = min(width, col_stop + 1)
    source = np.asarray(grid[src_r0:src_r1, src_c0:src_c1], dtype=np.float32)
    finite = np.isfinite(source)
    padded = np.pad(source, 1, mode="constant", constant_values=np.nan)

    core_r0 = row_start - src_r0 + 1
    core_c0 = col_start - src_c0 + 1
    rows = slice(core_r0, core_r0 + row_stop - row_start)
    cols = slice(core_c0, core_c0 + col_stop - col_start)
    center = padded[rows, cols]
    center_valid = np.isfinite(center)

    left, right = padded[rows, slice(core_c0 - 1, core_c0 - 1 + col_stop - col_start)], padded[
        rows, slice(core_c0 + 1, core_c0 + 1 + col_stop - col_start)
    ]
    up = padded[slice(core_r0 - 1, core_r0 - 1 + row_stop - row_start), cols]
    down = padded[slice(core_r0 + 1, core_r0 + 1 + row_stop - row_start), cols]

    left_ok, right_ok = np.isfinite(left), np.isfinite(right)
    up_ok, down_ok = np.isfinite(up), np.isfinite(down)
    dx = np.full(center.shape, np.nan, dtype=np.float32)
    dy = np.full(center.shape, np.nan, dtype=np.float32)

    both = center_valid & left_ok & right_ok
    dx[both] = (right[both] - left[both]) * 0.5
    only_right = center_valid & ~left_ok & right_ok
    dx[only_right] = right[only_right] - center[only_right]
    only_left = center_valid & left_ok & ~right_ok
    dx[only_left] = center[only_left] - left[only_left]

    both = center_valid & up_ok & down_ok
    dy[both] = (down[both] - up[both]) * 0.5
    only_down = center_valid & ~up_ok & down_ok
    dy[only_down] = down[only_down] - center[only_down]
    only_up = center_valid & up_ok & ~down_ok
    dy[only_up] = center[only_up] - up[only_up]

    slope = np.hypot(dx, dy)
    slope[~(np.isfinite(dx) & np.isfinite(dy))] = np.nan

    # Compute nodata-aware 3x3 population standard deviation. The source
    # includes a halo, so these neighborhood statistics are valid for the core.
    source_values = np.where(finite, source, 0.0).astype(np.float32, copy=False)
    source_mask = finite.astype(np.float32, copy=False)
    value_sum = ndimage.uniform_filter(
        source_values, size=3, mode="constant", cval=0.0
    ) * 9.0
    square_sum = ndimage.uniform_filter(
        source_values * source_values, size=3, mode="constant", cval=0.0
    ) * 9.0
    count = ndimage.uniform_filter(
        source_mask, size=3, mode="constant", cval=0.0
    ) * 9.0

    core = (slice(row_start - src_r0, row_stop - src_r0),
            slice(col_start - src_c0, col_stop - src_c0))
    core_count = count[core]
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = value_sum[core] / core_count
        variance = np.maximum(square_sum[core] / core_count - mean * mean, 0.0)
        roughness = np.sqrt(variance).astype(np.float32, copy=False)

    roughness[(core_count == 0) | ~center_valid] = np.nan
    return center.copy(), slope, roughness


def compute_terrain_features(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute full-grid features in chunks (for small arrays/API use).

    This helper returns full feature arrays, so do not call it on the 11.6 GB
    Earth DEM. The matcher below uses _read_feature_chunk instead.
    """
    slope = np.empty(grid.shape, dtype=np.float32)
    roughness = np.empty(grid.shape, dtype=np.float32)
    for row in range(0, grid.shape[0], CHUNK_ROWS):
        row_stop = min(row + CHUNK_ROWS, grid.shape[0])
        _, slope_chunk, rough_chunk = _read_feature_chunk(
            grid, row, row_stop, 0, grid.shape[1]
        )
        slope[row:row_stop] = slope_chunk
        roughness[row:row_stop] = rough_chunk
    return slope, roughness


def _validate_weights(weights_list: list[tuple[float, float, float]]) -> None:
    if not weights_list:
        raise ValueError("At least one weight combination is required.")
    for weights in weights_list:
        if len(weights) != 3 or any(weight < 0 for weight in weights):
            raise ValueError(f"Weights must be three non-negative values: {weights!r}")
        if not np.isclose(sum(weights), 1.0):
            raise ValueError(f"Weights must sum to 1.0: {weights!r}")


def _window_starts(length: int, window: int, stride: int) -> list[int]:
    if window > length:
        return []
    starts = list(range(0, length - window + 1, stride))
    final_start = length - window
    if not starts or starts[-1] != final_start:
        starts.append(final_start)
    return starts


def _score_candidate(
    earth_grid: np.ndarray,
    moon_grid: np.ndarray,
    row: int,
    col: int,
    chunk_rows: int,
) -> tuple[float, float, float] | None:
    """Accumulate three RMSEs in row stripes without materializing a window."""
    template_rows, template_cols = moon_grid.shape
    sums = np.zeros(3, dtype=np.float64)
    counts = np.zeros(3, dtype=np.int64)

    for offset in range(0, template_rows, chunk_rows):
        height = min(chunk_rows, template_rows - offset)
        earth_values, earth_slope, earth_roughness = _read_feature_chunk(
            earth_grid,
            row + offset,
            row + offset + height,
            col,
            col + template_cols,
        )
        moon_values, moon_slope, moon_roughness = _read_feature_chunk(
            moon_grid,
            offset,
            offset + height,
            0,
            template_cols,
        )

        for index, (first, second) in enumerate(
            (
                (earth_values, moon_values),
                (earth_slope, moon_slope),
                (earth_roughness, moon_roughness),
            )
        ):
            valid = np.isfinite(first) & np.isfinite(second)
            if valid.any():
                difference = first[valid].astype(np.float64) - second[valid].astype(
                    np.float64
                )
                sums[index] += np.dot(difference, difference)
                counts[index] += difference.size

    if np.any(counts == 0):
        return None
    return tuple(np.sqrt(sums / counts).tolist())


def sliding_window_multimetric_search(
    earth_grid: np.ndarray,
    moon_grid: np.ndarray,
    stride: int = STRIDE,
    weights: list[tuple[float, float, float]] | None = None,
    top_n: int = TOP_N,
    chunk_rows: int = CHUNK_ROWS,
) -> list[dict[str, Any]]:
    """Search window/weight pairs and retain only the globally best results."""
    if stride <= 0 or chunk_rows <= 0 or top_n <= 0:
        raise ValueError("stride, chunk_rows, and top_n must be positive.")
    if earth_grid.ndim != 2 or moon_grid.ndim != 2:
        raise ValueError("Earth and Moon grids must be two-dimensional.")
    weights = WEIGHT_COMBINATIONS if weights is None else weights
    _validate_weights(weights)

    template_rows, template_cols = moon_grid.shape
    row_starts = _window_starts(earth_grid.shape[0], template_rows, stride)
    col_starts = _window_starts(earth_grid.shape[1], template_cols, stride)
    if not row_starts or not col_starts:
        raise ValueError("Moon template must fit inside the Earth grid.")

    # Small bounded list: never retain the potentially enormous set of matches.
    best: list[dict[str, Any]] = []
    candidate_count = len(row_starts) * len(col_starts)
    print(
        f"Scanning {candidate_count:,} windows across {len(weights)} weight sets "
        f"(chunk_rows={chunk_rows}, stride={stride})...",
        flush=True,
    )

    for candidate_index, (row, col) in enumerate(
        ((r, c) for r in row_starts for c in col_starts), start=1
    ):
        rmse = _score_candidate(earth_grid, moon_grid, row, col, chunk_rows)
        if rmse is None:
            continue
        rmse_elev, rmse_slope, rmse_rough = rmse

        for w_elev, w_slope, w_rough in weights:
            score = (
                w_elev * rmse_elev
                + w_slope * rmse_slope
                + w_rough * rmse_rough
            )
            best.append(
                {
                    "row": row,
                    "col": col,
                    "score": float(score),
                    "weights": {
                        "elevation": w_elev,
                        "slope": w_slope,
                        "roughness": w_rough,
                    },
                    "rmse": {
                        "elevation": rmse_elev,
                        "slope": rmse_slope,
                        "roughness": rmse_rough,
                    },
                }
            )
        best.sort(key=lambda result: (result["score"], result["row"], result["col"]))
        del best[top_n:]

        if candidate_index % 100 == 0 or candidate_index == candidate_count:
            print(
                f"Scanned {candidate_index:,}/{candidate_count:,} windows",
                flush=True,
            )

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
    raise KeyError(
        "Earth spatial metadata must contain a 'transform' list, either at the "
        "top level or under an 'earth' entry."
    )


def convert_pixel_to_geographic(
    row: int, col: int, metadata_entry: dict[str, Any]
) -> tuple[float, float]:
    """Convert a pixel center to projected raster coordinates (UTM meters if UTM)."""
    transform_values = metadata_entry.get("transform")
    if transform_values is None or len(transform_values) < 6:
        raise ValueError("Metadata entry must contain a six-value 'transform' list.")
    affine = Affine(*transform_values[:6])
    x_coord, y_coord = affine * (col + 0.5, row + 0.5)
    return float(x_coord), float(y_coord)


def _plot_sample(grid: np.ndarray, max_dimension: int) -> np.ndarray:
    """Read a bounded, strided view for plotting instead of copying a large DEM."""
    row_step = max(1, int(np.ceil(grid.shape[0] / max_dimension)))
    col_step = max(1, int(np.ceil(grid.shape[1] / max_dimension)))
    return np.asarray(grid[::row_step, ::col_step], dtype=np.float32)


def plot_terrain_analog(
    moon_grid: np.ndarray,
    earth_grid: np.ndarray,
    result: dict[str, Any],
    rank: int,
) -> str:
    """Save a downsampled side-by-side comparison without loading full rasters."""
    row, col = result["row"], result["col"]
    template_rows, template_cols = moon_grid.shape
    earth_window = earth_grid[
        row : row + template_rows,
        col : col + template_cols,
    ]
    if earth_window.shape != moon_grid.shape:
        raise ValueError(f"Earth window for rank {rank} does not match Moon template.")

    moon_image = _plot_sample(moon_grid, PLOT_MAX_DIMENSION)
    earth_image = _plot_sample(earth_window, PLOT_MAX_DIMENSION)
    output_path = OUTPUT_IMAGE_TEMPLATE.format(rank=rank)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    axes[0].imshow(moon_image, cmap="terrain", interpolation="nearest")
    axes[0].set_title("Lunar template")
    axes[1].imshow(earth_image, cmap="terrain", interpolation="nearest")
    axes[1].set_title(
        f"Earth analog #{rank} (row={row}, col={col})\n"
        f"Score={result['score']:.6f}"
    )
    for axis in axes:
        axis.set_axis_off()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    earth_grid, moon_grid, metadata = load_data_and_metadata()
    print(
        f"Earth grid: {earth_grid.shape}, dtype={earth_grid.dtype}, "
        f"memory-mapped={isinstance(earth_grid, np.memmap)}"
    )
    print(f"Moon template: {moon_grid.shape}, dtype={moon_grid.dtype}")

    results = sliding_window_multimetric_search(
        earth_grid,
        moon_grid,
        stride=STRIDE,
        weights=WEIGHT_COMBINATIONS,
        top_n=TOP_N,
        chunk_rows=CHUNK_ROWS,
    )
    earth_metadata = _earth_metadata_entry(metadata)

    for rank, result in enumerate(results, start=1):
        x_coord, y_coord = convert_pixel_to_geographic(
            result["row"], result["col"], earth_metadata
        )
        result["rank"] = rank
        result["utm_x_m"] = x_coord
        result["utm_y_m"] = y_coord
        result["plot_file"] = plot_terrain_analog(
            moon_grid, earth_grid, result, rank
        )

    with RESULTS_FILE.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print(f"\nTop {len(results)} terrain analogs:")
    for result in results:
        weights = result["weights"]
        rmse = result["rmse"]
        print(
            f"\n#{result['rank']} at row={result['row']}, col={result['col']}"
            f" | score={result['score']:.6f}"
            f" | weights E/S/R={weights['elevation']:.2f}/"
            f"{weights['slope']:.2f}/{weights['roughness']:.2f}"
            f" | RMSE E/S/R={rmse['elevation']:.6f}/"
            f"{rmse['slope']:.6f}/{rmse['roughness']:.6f}"
            f" | UTM X={result['utm_x_m']:.2f} m, Y={result['utm_y_m']:.2f} m"
            f" | plot={result['plot_file']}"
        )
    print(f"\nSaved full results to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
