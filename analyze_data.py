from __future__ import annotations

"""Fast, memory-mapped terrain analog search.

This is tuned for a quick approximate pass that usually finishes in a couple of
minutes on a local machine. It only evaluates a sampled subset of candidate
windows and keeps the top-ranked matches.
"""

import json
import multiprocessing as mp
import os
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
STRIDE = 100
SCALE_FACTOR = 4
MAX_CANDIDATES: int | None = 128
CHUNK_ROWS = 64
WORKERS = min(2, os.cpu_count() or 1)
PLOT_MAX_DIMENSION = 1200
EARTH_BBOX: tuple[int, int, int, int] | None = None
WEIGHT_COMBINATIONS = [
    (0.50, 0.25, 0.25),
    (0.60, 0.20, 0.20),
    (0.40, 0.30, 0.30),
    (0.70, 0.15, 0.15),
    (0.30, 0.35, 0.35),
]

_WORK_EARTH: np.ndarray | None = None
_WORK_MOON: np.ndarray | None = None
_WORK_MOON_SLOPE: np.ndarray | None = None
_WORK_MOON_ROUGHNESS: np.ndarray | None = None
_WORK_CHUNK_ROWS = CHUNK_ROWS


def _load_single_array(pattern: str) -> np.ndarray:
    files = sorted(PROCESSED_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern!r} in {PROCESSED_DIR}")
    if len(files) != 1:
        raise RuntimeError(f"Expected one file matching {pattern!r}, found {len(files)}")
    return np.load(files[0], mmap_mode="r", allow_pickle=False)


def load_data_and_metadata() -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    metadata_path = PROCESSED_DIR / "spatial_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Spatial metadata not found: {metadata_path}")

    with metadata_path.open(encoding="utf-8") as file:
        metadata = json.load(file)

    earth_entries = metadata.get("earth", {})
    moon_entries = metadata.get("moon", {})
    if not earth_entries or not moon_entries:
        raise KeyError("Metadata must contain non-empty 'earth' and 'moon' entries.")

    earth_info = next(iter(earth_entries.values()))
    moon_info = next(iter(moon_entries.values()))

    earth_path = Path(__file__).resolve().parent / earth_info["processed_file"]
    moon_path = Path(__file__).resolve().parent / moon_info["processed_file"]

    if not earth_path.exists():
        raise FileNotFoundError(f"Earth array not found: {earth_path}")
    if not moon_path.exists():
        raise FileNotFoundError(f"Moon array not found: {moon_path}")

    earth = np.load(earth_path, mmap_mode="r", allow_pickle=False)
    moon = np.load(moon_path, mmap_mode="r", allow_pickle=False)

    if earth.ndim != 2 or moon.ndim != 2:
        raise ValueError("Earth and Moon arrays must both be two-dimensional.")

    return earth, moon, metadata


def _read_feature_chunk(
    grid: np.ndarray, row_start: int, row_stop: int, col_start: int, col_stop: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute elevation, slope and nodata-aware 3x3 roughness for one chunk."""
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

    left = padded[rows, slice(core_c0 - 1, core_c0 - 1 + col_stop - col_start)]
    right = padded[rows, slice(core_c0 + 1, core_c0 + 1 + col_stop - col_start)]
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

    source_values = np.where(finite, source, 0.0).astype(np.float32, copy=False)
    source_mask = finite.astype(np.float32, copy=False)
    value_sum = ndimage.uniform_filter(source_values, size=3, mode="constant", cval=0.0) * 9.0
    square_sum = ndimage.uniform_filter(source_values * source_values, size=3, mode="constant", cval=0.0) * 9.0
    count = ndimage.uniform_filter(source_mask, size=3, mode="constant", cval=0.0) * 9.0

    core = (slice(row_start - src_r0, row_stop - src_r0), slice(col_start - src_c0, col_stop - src_c0))
    core_count = count[core]
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = value_sum[core] / core_count
        variance = np.maximum(square_sum[core] / core_count - mean * mean, 0.0)
        roughness = np.sqrt(variance).astype(np.float32, copy=False)

    roughness[(core_count == 0) | ~center_valid] = np.nan
    return center.copy(), slope, roughness


def _compute_template_features(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    slope = np.empty(grid.shape, dtype=np.float32)
    roughness = np.empty(grid.shape, dtype=np.float32)
    for r0 in range(0, grid.shape[0], CHUNK_ROWS):
        r1 = min(r0 + CHUNK_ROWS, grid.shape[0])
        _, slope[r0:r1], roughness[r0:r1] = _read_feature_chunk(grid, r0, r1, 0, grid.shape[1])
    return slope, roughness


def _init_worker(
    earth_path: str,
    moon_path: str,
    bbox: tuple[int, int, int, int],
    scale_factor: int,
    chunk_rows: int,
) -> None:
    global _WORK_EARTH, _WORK_MOON, _WORK_MOON_SLOPE, _WORK_MOON_ROUGHNESS, _WORK_CHUNK_ROWS
    earth_map = np.load(earth_path, mmap_mode="r", allow_pickle=False)
    moon_map = np.load(moon_path, mmap_mode="r", allow_pickle=False)
    r0, r1, c0, c1 = bbox
    _WORK_EARTH = earth_map[r0:r1:scale_factor, c0:c1:scale_factor]
    _WORK_MOON = moon_map[::scale_factor, ::scale_factor]
    _WORK_CHUNK_ROWS = chunk_rows
    _WORK_MOON_SLOPE, _WORK_MOON_ROUGHNESS = _compute_template_features(_WORK_MOON)


def _score_candidate(row: int, col: int) -> tuple[int, int, tuple[float, float, float] | None]:
    assert _WORK_EARTH is not None and _WORK_MOON is not None
    assert _WORK_MOON_SLOPE is not None and _WORK_MOON_ROUGHNESS is not None
    h, w = _WORK_MOON.shape
    sums = np.zeros(3, dtype=np.float64)
    counts = np.zeros(3, dtype=np.int64)
    for offset in range(0, h, _WORK_CHUNK_ROWS):
        stop = min(offset + _WORK_CHUNK_ROWS, h)
        earth_values, earth_slope, earth_rough = _read_feature_chunk(
            _WORK_EARTH, row + offset, row + stop, col, col + w
        )
        moon_values = np.asarray(_WORK_MOON[offset:stop, :], dtype=np.float32)
        moon_slope = _WORK_MOON_SLOPE[offset:stop, :]
        moon_rough = _WORK_MOON_ROUGHNESS[offset:stop, :]
        for i, (a, b) in enumerate(
            ((earth_values, moon_values), (earth_slope, moon_slope), (earth_rough, moon_rough))
        ):
            valid = np.isfinite(a) & np.isfinite(b)
            if valid.any():
                delta = a[valid].astype(np.float64) - b[valid].astype(np.float64)
                sums[i] += np.dot(delta, delta)
                counts[i] += delta.size
    if np.any(counts == 0):
        return row, col, None
    return row, col, tuple(np.sqrt(sums / counts).tolist())


def _score_candidate_task(candidate: tuple[int, int]) -> tuple[int, int, tuple[float, float, float] | None]:
    row, col = candidate
    return _score_candidate(row, col)


def _validate_weights(weights: list[tuple[float, float, float]]) -> None:
    if not weights:
        raise ValueError("At least one weight combination is required.")
    for triple in weights:
        if len(triple) != 3 or any(v < 0 for v in triple) or not np.isclose(sum(triple), 1.0):
            raise ValueError(f"Weights must be three non-negative values summing to one: {triple!r}")


def _sample_candidate_positions(
    rows: list[int], cols: list[int], max_candidates: int | None
) -> list[tuple[int, int]]:
    total = len(rows) * len(cols)
    if max_candidates is None or total <= max_candidates:
        indices = range(total)
    else:
        indices = np.unique(np.linspace(0, total - 1, max_candidates, dtype=np.int64))
    ncols = len(cols)
    return [(rows[int(i) // ncols], cols[int(i) % ncols]) for i in indices]


def _results_for_candidate(
    row: int,
    col: int,
    rmse: tuple[float, float, float],
    weights: list[tuple[float, float, float]],
    scale_factor: int,
    bbox: tuple[int, int, int, int],
) -> list[dict[str, Any]]:
    re, rs, rr = rmse
    r0, _, c0, _ = bbox
    original_row, original_col = r0 + row * scale_factor, c0 + col * scale_factor
    return [{
        "row": original_row,
        "col": original_col,
        "scale_factor": scale_factor,
        "score": float(we * re + ws * rs + wr * rr),
        "weights": {"elevation": we, "slope": ws, "roughness": wr},
        "rmse": {"elevation": re, "slope": rs, "roughness": rr},
    } for we, ws, wr in weights]


def search_weight_grid(
    earth_grid: np.ndarray,
    moon_grid: np.ndarray,
    stride: int = STRIDE,
    weights: list[tuple[float, float, float]] | None = None,
    top_n: int = TOP_N,
    chunk_rows: int = CHUNK_ROWS,
    bounding_box: tuple[int, int, int, int] | None = None,
    scale_factor: int = SCALE_FACTOR,
    workers: int = WORKERS,
    max_candidates: int | None = MAX_CANDIDATES,
) -> list[dict[str, Any]]:
    """Parallel, downsampled candidate search; keeps only the best top_n results."""
    if min(stride, top_n, chunk_rows, scale_factor, workers) <= 0:
        raise ValueError("stride, top_n, chunk_rows, scale_factor, and workers must be positive.")
    if max_candidates is not None and max_candidates <= 0:
        raise ValueError("max_candidates must be positive or None.")
    if earth_grid.ndim != 2 or moon_grid.ndim != 2:
        raise ValueError("Earth and Moon grids must be two-dimensional.")
    weights = WEIGHT_COMBINATIONS if weights is None else weights
    _validate_weights(weights)

    bbox = bounding_box or (0, earth_grid.shape[0], 0, earth_grid.shape[1])
    if len(bbox) != 4:
        raise ValueError("bounding_box must be (row_start, row_stop, col_start, col_stop).")
    r0, r1, c0, c1 = map(int, bbox)
    if not (0 <= r0 < r1 <= earth_grid.shape[0] and 0 <= c0 < c1 <= earth_grid.shape[1]):
        raise ValueError(f"Invalid bounding box {bbox} for Earth grid {earth_grid.shape}.")

    sampled_h = (r1 - r0 + scale_factor - 1) // scale_factor
    sampled_w = (c1 - c0 + scale_factor - 1) // scale_factor
    moon_h = (moon_grid.shape[0] + scale_factor - 1) // scale_factor
    moon_w = (moon_grid.shape[1] + scale_factor - 1) // scale_factor
    if sampled_h < moon_h or sampled_w < moon_w:
        raise ValueError("The selected Earth region is smaller than the downsampled Moon template.")
    row_positions = list(range(0, sampled_h - moon_h + 1, stride))
    col_positions = list(range(0, sampled_w - moon_w + 1, stride))
    if row_positions[-1] != sampled_h - moon_h:
        row_positions.append(sampled_h - moon_h)
    if col_positions[-1] != sampled_w - moon_w:
        col_positions.append(sampled_w - moon_w)
    candidates = _sample_candidate_positions(row_positions, col_positions, max_candidates)
    print(
        f"Quick search: {len(candidates):,} of {len(row_positions) * len(col_positions):,} "
        f"candidate windows, scale={scale_factor}, stride={stride}, workers={workers}",
        flush=True,
    )

    if not (isinstance(earth_grid, np.memmap) and isinstance(moon_grid, np.memmap)) and workers > 1:
        print("Arrays are not both memory-mapped; using one worker to avoid copying them.", flush=True)
        workers = 1

    best: list[dict[str, Any]] = []

    def consume(row: int, col: int, rmse: tuple[float, float, float] | None) -> None:
        if rmse is None:
            return
        best.extend(_results_for_candidate(row, col, rmse, weights, scale_factor, (r0, r1, c0, c1)))
        best.sort(key=lambda result: (result["score"], result["row"], result["col"]))
        del best[top_n:]

    if workers > 1:
        earth_path = str(Path(earth_grid.filename).resolve())
        moon_path = str(Path(moon_grid.filename).resolve())
        context = mp.get_context("spawn")
        pool = context.Pool(
            processes=workers,
            initializer=_init_worker,
            initargs=(earth_path, moon_path, (r0, r1, c0, c1), scale_factor, chunk_rows),
        )
        try:
            tasks = pool.imap_unordered(_score_candidate_task, candidates, chunksize=1)
            for i, (row, col, rmse) in enumerate(tasks, 1):
                consume(row, col, rmse)
                if i % 16 == 0 or i == len(candidates):
                    print(f"Scored {i}/{len(candidates)} candidates", flush=True)
        except BaseException:
            pool.terminate()
            raise
        else:
            pool.close()
        finally:
            pool.join()
    else:
        global _WORK_EARTH, _WORK_MOON, _WORK_MOON_SLOPE, _WORK_MOON_ROUGHNESS, _WORK_CHUNK_ROWS
        _WORK_EARTH = earth_grid[r0:r1:scale_factor, c0:c1:scale_factor]
        _WORK_MOON = moon_grid[::scale_factor, ::scale_factor]
        _WORK_CHUNK_ROWS = chunk_rows
        _WORK_MOON_SLOPE, _WORK_MOON_ROUGHNESS = _compute_template_features(_WORK_MOON)
        for i, (row, col) in enumerate(candidates, 1):
            _, _, rmse = _score_candidate(row, col)
            consume(row, col, rmse)
            if i % 16 == 0 or i == len(candidates):
                print(f"Scored {i}/{len(candidates)} candidates", flush=True)

    if not best:
        raise ValueError("No candidate windows had valid data for all metrics.")
    return best


def sliding_window_multimetric_search(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
    return search_weight_grid(*args, **kwargs)


def _earth_metadata_entry(metadata: dict[str, Any]) -> dict[str, Any]:
    earth_entries = metadata.get("earth", {})
    for entry in earth_entries.values():
        if isinstance(entry, dict) and "transform" in entry:
            return entry
    raise KeyError("Could not find an Earth metadata entry containing 'transform'.")


def convert_pixel_to_geographic(row: int, col: int, metadata_entry: dict[str, Any]) -> tuple[float, float]:
    transform = metadata_entry.get("transform")
    if transform is None or len(transform) < 6:
        raise ValueError("Metadata entry must contain a six-value 'transform' list.")
    x, y = Affine(*transform[:6]) * (col + 0.5, row + 0.5)
    return float(x), float(y)


def _plot_sample(grid: np.ndarray, max_dimension: int) -> np.ndarray:
    rstep = max(1, int(np.ceil(grid.shape[0] / max_dimension)))
    cstep = max(1, int(np.ceil(grid.shape[1] / max_dimension)))
    return np.asarray(grid[::rstep, ::cstep], dtype=np.float32)


def plot_terrain_analog(moon_grid: np.ndarray, earth_grid: np.ndarray, result: dict[str, Any], rank: int) -> str:
    scale = result.get("scale_factor", 1)
    row, col = result["row"], result["col"]
    h, w = moon_grid.shape
    earth_window = earth_grid[row:row + h * scale:scale, col:col + w * scale:scale]
    moon_sample = moon_grid[::scale, ::scale]
    if earth_window.shape != moon_sample.shape:
        raise ValueError("Scaled Earth match window does not match the Moon template.")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    axes[0].imshow(_plot_sample(moon_sample, PLOT_MAX_DIMENSION), cmap="terrain")
    axes[0].set_title(f"Moon template (scale 1/{scale})")
    axes[1].imshow(_plot_sample(earth_window, PLOT_MAX_DIMENSION), cmap="terrain")
    axes[1].set_title(f"Earth analog #{rank}: row={row}, col={col}\nScore={result['score']:.6f}")
    for axis in axes:
        axis.set_axis_off()
    output = OUTPUT_IMAGE_TEMPLATE.format(rank=rank)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    earth, moon, metadata = load_data_and_metadata()
    print(f"Earth grid {earth.shape}, dtype={earth.dtype}, memmap={isinstance(earth, np.memmap)}")
    print(f"Moon template {moon.shape}, scale={SCALE_FACTOR}")
    results = search_weight_grid(
        earth, moon, stride=STRIDE, weights=WEIGHT_COMBINATIONS, top_n=TOP_N,
        chunk_rows=CHUNK_ROWS, bounding_box=EARTH_BBOX,
        scale_factor=SCALE_FACTOR, workers=WORKERS, max_candidates=MAX_CANDIDATES,
    )
    metadata_entry = _earth_metadata_entry(metadata)
    for rank, result in enumerate(results, 1):
        result["rank"] = rank
        result["utm_x_m"], result["utm_y_m"] = convert_pixel_to_geographic(
            result["row"], result["col"], metadata_entry
        )
        result["plot_file"] = plot_terrain_analog(moon, earth, result, rank)
    with RESULTS_FILE.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    print(f"\nTop {len(results)} approximate matches:")
    for item in results:
        w = item["weights"]
        print(f"#{item['rank']} pixel=({item['row']},{item['col']}) score={item['score']:.6f} "
              f"scale=1/{item['scale_factor']} weights=({w['elevation']:.2f},{w['slope']:.2f},{w['roughness']:.2f}) "
              f"UTM=({item['utm_x_m']:.2f},{item['utm_y_m']:.2f}) m; plot={item['plot_file']}")
    print(f"Saved results to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
