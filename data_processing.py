#!/usr/bin/env python3
"""Preprocess DEM GeoTIFFs and find a lunar-to-Earth terrain analog.

Put Earth DEMs in data/raw_earth/ and lunar DEMs in data/raw_moon/.
Running this script creates normalized NumPy arrays, spatial metadata, and—
when both types of DEM are available—analog_comparison.png.

Dependencies:
    pip install numpy rasterio matplotlib
"""

import argparse
import json
import logging
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine, from_origin
from rasterio.warp import (
    Resampling,
    calculate_default_transform,
    reproject,
    transform as transform_coordinates,
)


RAW_MOON_DIR = os.path.join("data", "raw_moon")
RAW_EARTH_DIR = os.path.join("data", "raw_earth")
PROCESSED_DIR = os.path.join("data", "processed")
METADATA_PATH = os.path.join(PROCESSED_DIR, "spatial_metadata.json")
COMPARISON_PLOT = "analog_comparison.png"

DEFAULT_TARGET_CRS = "EPSG:32616"
DEFAULT_RESOLUTION_METERS = 30.0

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)


def create_directories():
    """Create the input and output directory scaffolding if it is missing."""
    for directory in (RAW_MOON_DIR, RAW_EARTH_DIR, PROCESSED_DIR):
        os.makedirs(directory, exist_ok=True)


def _require_metric_projected_crs(crs, label):
    """Raise an error unless the CRS is projected and uses meters."""
    if not crs.is_projected:
        raise ValueError(f"{label} must use a projected CRS; got {crs}.")

    unit_name, meters_per_unit = crs.linear_units_factor
    if not math.isclose(meters_per_unit, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"{label} must use meters; got {unit_name!r}, "
            f"or {meters_per_unit} meters per unit."
        )


def _read_dem(path):
    """Read a single-band GeoTIFF as float32, with nodata represented by NaN."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"GeoTIFF not found: {path}")

    with rasterio.open(path) as src:
        if src.count != 1:
            raise ValueError(
                f"Expected a single-band elevation raster, but {path} "
                f"contains {src.count} bands."
            )
        if src.crs is None:
            raise ValueError(f"GeoTIFF has no coordinate reference system: {path}")

        dem = src.read(1, masked=True).astype(np.float32).filled(np.nan)
        return dem, src.transform, src.bounds, src.crs


def normalize_elevation(dem_array):
    """Min-Max normalize finite elevations to [0, 1], leaving nodata as NaN.

    A constant-valued DEM is mapped to zero at valid cells.

    Args:
        dem_array: Two-dimensional elevation array.

    Returns:
        A float32 array with normalized values and NaN nodata cells.

    Raises:
        ValueError: If the array is not 2D or contains no finite values.
    """
    dem = np.asarray(dem_array, dtype=np.float32)
    if dem.ndim != 2:
        raise ValueError("DEM array must be two-dimensional.")

    valid = np.isfinite(dem)
    if not np.any(valid):
        raise ValueError("DEM contains no finite elevation values.")

    minimum = float(np.min(dem[valid]))
    maximum = float(np.max(dem[valid]))
    normalized = np.full(dem.shape, np.nan, dtype=np.float32)

    if maximum == minimum:
        normalized[valid] = 0.0
    else:
        normalized[valid] = (dem[valid] - minimum) / (maximum - minimum)

    return normalized


def _grid_metadata(crs, transform, width, height, resolution):
    """Create JSON-friendly metadata sufficient to reconstruct a raster grid."""
    return {
        "crs": crs.to_string(),
        "transform": list(transform)[:6],
        "width": int(width),
        "height": int(height),
        "resolution_meters": float(resolution),
    }


def reproject_to_metric_grid(
    input_tif_path,
    output_numpy_path,
    target_crs=DEFAULT_TARGET_CRS,
    target_resolution=DEFAULT_RESOLUTION_METERS,
):
    """Reproject an Earth GeoTIFF to a meter-based grid and save a .npy array.

    The output grid uses the requested resolution. Its array contains raw
    elevations; normalization is performed separately by the processing
    orchestration.

    Args:
        input_tif_path: Source Earth GeoTIFF path.
        output_numpy_path: Destination .npy path.
        target_crs: Projected, meter-based CRS for the output.
        target_resolution: Output pixel size in meters.

    Returns:
        Metadata dictionary describing the output grid.
    """
    if target_resolution <= 0:
        raise ValueError("target_resolution must be greater than zero.")
    if not os.path.isfile(input_tif_path):
        raise FileNotFoundError(f"GeoTIFF not found: {input_tif_path}")

    dst_crs = CRS.from_user_input(target_crs)
    _require_metric_projected_crs(dst_crs, "Target CRS")

    with rasterio.open(input_tif_path) as src:
        if src.count != 1:
            raise ValueError(f"Expected a single-band DEM: {input_tif_path}")
        if src.crs is None:
            raise ValueError(f"GeoTIFF has no CRS: {input_tif_path}")

        dst_transform, dst_width, dst_height = calculate_default_transform(
            src.crs,
            dst_crs,
            src.width,
            src.height,
            *src.bounds,
            resolution=(target_resolution, target_resolution),
        )

        source = src.read(1, masked=True).astype(np.float32).filled(np.nan)
        destination = np.full((dst_height, dst_width), np.nan, dtype=np.float32)

        reproject(
            source=source,
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=np.nan,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )

    os.makedirs(os.path.dirname(os.path.abspath(output_numpy_path)), exist_ok=True)
    np.save(output_numpy_path, destination)

    return _grid_metadata(
        dst_crs, dst_transform, dst_width, dst_height, target_resolution
    )


def resample_to_match_scale(moon_tif_path, target_resolution=30.0):
    """Resample a projected lunar DEM to square pixels of the given size.

    The input lunar CRS must be projected and use meters. The output preserves
    the source CRS and covers its bounds, potentially extending by less than
    one pixel to keep the requested resolution exact.

    Args:
        moon_tif_path: Source lunar GeoTIFF path.
        target_resolution: Output pixel size in meters.

    Returns:
        Tuple of (resampled elevation array, grid metadata).
    """
    if target_resolution <= 0:
        raise ValueError("target_resolution must be greater than zero.")

    source, src_transform, bounds, src_crs = _read_dem(moon_tif_path)
    _require_metric_projected_crs(src_crs, "Lunar DEM CRS")

    width = max(1, math.ceil((bounds.right - bounds.left) / target_resolution))
    height = max(1, math.ceil((bounds.top - bounds.bottom) / target_resolution))
    dst_transform = from_origin(
        bounds.left, bounds.top, target_resolution, target_resolution
    )
    destination = np.full((height, width), np.nan, dtype=np.float32)

    reproject(
        source=source,
        destination=destination,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=np.nan,
        dst_transform=dst_transform,
        dst_crs=src_crs,
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )

    return destination, _grid_metadata(
        src_crs, dst_transform, width, height, target_resolution
    )


def _gradient_component(dem, axis, spacing):
    """Compute a NaN-aware gradient component using available neighbors."""
    values = np.asarray(dem, dtype=np.float64)
    valid = np.isfinite(values)

    before = np.roll(values, 1, axis=axis)
    after = np.roll(values, -1, axis=axis)
    before_valid = np.roll(valid, 1, axis=axis)
    after_valid = np.roll(valid, -1, axis=axis)

    # np.roll wraps array edges; those wrapped neighbors are not valid.
    first_edge = [slice(None)] * values.ndim
    last_edge = [slice(None)] * values.ndim
    first_edge[axis] = 0
    last_edge[axis] = -1
    before_valid[tuple(first_edge)] = False
    after_valid[tuple(last_edge)] = False

    before_valid &= valid
    after_valid &= valid

    gradient = np.full(values.shape, np.nan, dtype=np.float64)
    both = before_valid & after_valid
    only_after = ~before_valid & after_valid
    only_before = before_valid & ~after_valid

    gradient[both] = (after[both] - before[both]) / (2.0 * spacing)
    gradient[only_after] = (after[only_after] - values[only_after]) / spacing
    gradient[only_before] = (values[only_before] - before[only_before]) / spacing
    return gradient


def _local_nan_std(dem, window_size):
    """Calculate local standard deviation while ignoring NaNs and edges."""
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("roughness_window must be a positive odd integer.")

    values = np.asarray(dem, dtype=np.float64)
    height, width = values.shape
    radius = window_size // 2
    padded = np.pad(values, radius, mode="constant", constant_values=np.nan)

    total = np.zeros(values.shape, dtype=np.float64)
    total_squared = np.zeros(values.shape, dtype=np.float64)
    count = np.zeros(values.shape, dtype=np.int32)

    for row_offset in range(window_size):
        for col_offset in range(window_size):
            neighbor = padded[
                row_offset : row_offset + height,
                col_offset : col_offset + width,
            ]
            valid = np.isfinite(neighbor)
            total[valid] += neighbor[valid]
            total_squared[valid] += neighbor[valid] ** 2
            count[valid] += 1

    roughness = np.full(values.shape, np.nan, dtype=np.float64)
    has_neighbors = count > 0
    mean = np.zeros(values.shape, dtype=np.float64)
    mean[has_neighbors] = total[has_neighbors] / count[has_neighbors]

    variance = np.zeros(values.shape, dtype=np.float64)
    variance[has_neighbors] = (
        total_squared[has_neighbors] / count[has_neighbors]
        - mean[has_neighbors] ** 2
    )
    roughness[has_neighbors] = np.sqrt(np.maximum(variance[has_neighbors], 0.0))
    roughness[~np.isfinite(values)] = np.nan

    return roughness.astype(np.float32)


def compute_terrain_features(
    dem_array,
    pixel_size_x=1.0,
    pixel_size_y=1.0,
    roughness_window=3,
):
    """Compute slope magnitude and local roughness for a 2D DEM.

    Slope is the magnitude of the elevation gradient. Roughness is the local
    elevation standard deviation. Both outputs leave invalid DEM cells as NaN.

    Args:
        dem_array: Two-dimensional elevation array.
        pixel_size_x: Horizontal pixel spacing in coordinate units.
        pixel_size_y: Vertical pixel spacing in coordinate units.
        roughness_window: Odd neighborhood size for local standard deviation.

    Returns:
        Dictionary containing ``slope`` and ``roughness`` float32 arrays.
    """
    dem = np.asarray(dem_array, dtype=np.float32)
    if dem.ndim != 2:
        raise ValueError("DEM array must be two-dimensional.")
    if pixel_size_x <= 0 or pixel_size_y <= 0:
        raise ValueError("Pixel sizes must be greater than zero.")

    dz_dx = _gradient_component(dem, axis=1, spacing=pixel_size_x)
    dz_dy = _gradient_component(dem, axis=0, spacing=pixel_size_y)

    return {
        "slope": np.hypot(dz_dx, dz_dy).astype(np.float32),
        "roughness": _local_nan_std(dem, roughness_window),
    }


def find_best_terrain_analog(
    template_dem,
    candidate_dem,
    candidate_metadata,
    weights=None,
    minimum_valid_fraction=0.8,
    roughness_window=3,
):
    """Find the best candidate window using elevation, slope, and roughness.

    Each feature's RMSE is scaled by the standard deviation of that feature in
    the template. The weighted feature RMSEs are then combined into one score.

    Args:
        template_dem: 2D lunar DEM array.
        candidate_dem: 2D Earth DEM array to search.
        candidate_metadata: Grid metadata containing CRS and affine transform.
        weights: Optional weights for elevation, slope, and roughness.
        minimum_valid_fraction: Minimum valid elevation overlap for a window.
        roughness_window: Odd neighborhood size for roughness.

    Returns:
        Dictionary containing the best window coordinates, score, and
        geographic center coordinates (latitude and longitude).
    """
    template = np.asarray(template_dem, dtype=np.float32)
    candidate = np.asarray(candidate_dem, dtype=np.float32)

    if template.ndim != 2 or candidate.ndim != 2:
        raise ValueError("Template and candidate DEMs must both be 2D.")
    if template.shape[0] > candidate.shape[0] or template.shape[1] > candidate.shape[1]:
        raise ValueError("Template dimensions must fit inside the candidate DEM.")
    if not 0 < minimum_valid_fraction <= 1:
        raise ValueError("minimum_valid_fraction must be in (0, 1].")

    weights = weights or {"elevation": 1.0, "slope": 1.0, "roughness": 1.0}
    names = ("elevation", "slope", "roughness")
    if any(weights.get(name, 0.0) < 0 for name in names):
        raise ValueError("Feature weights cannot be negative.")
    if not any(weights.get(name, 0.0) > 0 for name in names):
        raise ValueError("At least one feature weight must be positive.")

    try:
        candidate_crs = CRS.from_user_input(candidate_metadata["crs"])
        candidate_transform = Affine(*candidate_metadata["transform"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Candidate metadata must include a valid CRS and affine transform."
        ) from exc

    pixel_size_x = math.hypot(candidate_transform.a, candidate_transform.d)
    pixel_size_y = math.hypot(candidate_transform.b, candidate_transform.e)
    if pixel_size_x <= 0 or pixel_size_y <= 0:
        raise ValueError("Candidate transform has invalid pixel dimensions.")

    template_features = compute_terrain_features(
        template, pixel_size_x, pixel_size_y, roughness_window
    )
    candidate_features = compute_terrain_features(
        candidate, pixel_size_x, pixel_size_y, roughness_window
    )
    template_features["elevation"] = template
    candidate_features["elevation"] = candidate

    scales = {}
    for name in names:
        values = template_features[name]
        valid = np.isfinite(values)
        if not np.any(valid):
            raise ValueError(f"Template has no valid values for {name}.")
        standard_deviation = float(np.std(values[valid]))
        scales[name] = standard_deviation if standard_deviation > 1e-12 else 1.0

    template_height, template_width = template.shape
    valid_template_pixels = int(np.isfinite(template).sum())
    if valid_template_pixels == 0:
        raise ValueError("Template has no valid elevation pixels.")

    best = None
    last_row = candidate.shape[0] - template_height
    last_col = candidate.shape[1] - template_width

    for row in range(last_row + 1):
        for col in range(last_col + 1):
            elevation_window = candidate[
                row : row + template_height, col : col + template_width
            ]
            elevation_overlap = np.isfinite(template) & np.isfinite(elevation_window)
            if elevation_overlap.sum() < minimum_valid_fraction * valid_template_pixels:
                continue

            weighted_squared_error = 0.0
            included_weight = 0.0

            for name in names:
                weight = float(weights.get(name, 0.0))
                if weight <= 0:
                    continue

                feature_template = template_features[name]
                feature_candidate = candidate_features[name][
                    row : row + template_height,
                    col : col + template_width,
                ]
                valid = np.isfinite(feature_template) & np.isfinite(feature_candidate)
                if not np.any(valid):
                    continue

                difference = (
                    feature_template[valid] - feature_candidate[valid]
                ) / scales[name]
                weighted_squared_error += weight * float(np.mean(difference ** 2))
                included_weight += weight

            if included_weight == 0:
                continue

            score = math.sqrt(weighted_squared_error / included_weight)
            if best is not None and score >= best["score"]:
                continue

            # Convert the matched window's center pixel to WGS84 coordinates.
            center_row = row + (template_height - 1) / 2.0
            center_col = col + (template_width - 1) / 2.0
            x, y = rasterio.transform.xy(
                candidate_transform, center_row, center_col, offset="center"
            )
            longitude, latitude = transform_coordinates(
                candidate_crs,
                CRS.from_epsg(4326),
                [x],
                [y],
            )

            best = {
                "score": score,
                "row": row,
                "col": col,
                "height": template_height,
                "width": template_width,
                "center_row": center_row,
                "center_col": center_col,
                "x": x,
                "y": y,
                "latitude": latitude[0],
                "longitude": longitude[0],
            }

    if best is None:
        raise ValueError("No candidate window met the valid-data requirements.")
    return best


def save_analog_comparison(
    template_dem,
    candidate_dem,
    match,
    output_path=COMPARISON_PLOT,
):
    """Save a high-resolution side-by-side plot of the template and analog."""
    template = np.asarray(template_dem, dtype=np.float32)
    candidate = np.asarray(candidate_dem, dtype=np.float32)
    row, col = match["row"], match["col"]
    height, width = template.shape
    analog = candidate[row : row + height, col : col + width]

    if analog.shape != template.shape:
        raise ValueError("Matched analog window does not match template dimensions.")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    for axis, image, title in (
        (axes[0], template, "Lunar template"),
        (axes[1], analog, "Best Earth analog"),
    ):
        image_plot = axis.imshow(image, cmap="terrain", interpolation="nearest")
        axis.set_title(title)
        axis.set_xlabel("Column (pixels)")
        axis.set_ylabel("Row (pixels)")
        figure.colorbar(image_plot, ax=axis, label="Normalized elevation")

    figure.suptitle(
        f"Terrain analog comparison — RMSE {match['score']:.4f}; "
        f"latitude {match['latitude']:.5f}, "
        f"longitude {match['longitude']:.5f}"
    )
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    LOGGER.info("Saved comparison plot to %s", output_path)


def _find_tiffs(directory):
    """Return sorted GeoTIFF paths directly inside a directory."""
    if not os.path.isdir(directory):
        return []
    return sorted(
        os.path.join(directory, filename)
        for filename in os.listdir(directory)
        if filename.lower().endswith((".tif", ".tiff"))
        and os.path.isfile(os.path.join(directory, filename))
    )


def preprocess_dem_files(
    target_crs=DEFAULT_TARGET_CRS,
    target_resolution=DEFAULT_RESOLUTION_METERS,
):
    """Preprocess all raw Earth and lunar DEMs and write spatial metadata.

    Returns:
        Metadata dictionary keyed by dataset names under ``earth`` and ``moon``.
    """
    create_directories()
    metadata = {"earth": {}, "moon": {}}

    for input_path in _find_tiffs(RAW_EARTH_DIR):
        name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(PROCESSED_DIR, f"{name}_earth.npy")

        try:
            grid = reproject_to_metric_grid(
                input_path,
                output_path,
                target_crs=target_crs,
                target_resolution=target_resolution,
            )
            raw_dem = np.load(output_path)
            np.save(output_path, normalize_elevation(raw_dem))
            grid.update(
                {
                    "source_file": os.path.relpath(input_path),
                    "processed_file": os.path.relpath(output_path),
                    "normalization": "min-max [0, 1]; nodata is NaN",
                }
            )
            metadata["earth"][name] = grid
            LOGGER.info("Processed Earth DEM: %s", input_path)
        except (OSError, ValueError, rasterio.errors.RasterioError) as exc:
            LOGGER.error("Skipping Earth DEM %s: %s", input_path, exc)

    for input_path in _find_tiffs(RAW_MOON_DIR):
        name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(PROCESSED_DIR, f"{name}_moon.npy")

        try:
            raw_dem, grid = resample_to_match_scale(
                input_path, target_resolution=target_resolution
            )
            np.save(output_path, normalize_elevation(raw_dem))
            grid.update(
                {
                    "source_file": os.path.relpath(input_path),
                    "processed_file": os.path.relpath(output_path),
                    "normalization": "min-max [0, 1]; nodata is NaN",
                }
            )
            metadata["moon"][name] = grid
            LOGGER.info("Processed lunar DEM: %s", input_path)
        except (OSError, ValueError, rasterio.errors.RasterioError) as exc:
            LOGGER.error("Skipping lunar DEM %s: %s", input_path, exc)

    with open(METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    LOGGER.info("Wrote spatial metadata to %s", METADATA_PATH)

    return metadata


def run_analog_matching(
    metadata,
    moon_key=None,
    earth_key=None,
    output_plot=COMPARISON_PLOT,
    weights=None,
):
    """Match one lunar DEM to one Earth DEM and save the comparison plot."""
    if not metadata["moon"] or not metadata["earth"]:
        LOGGER.warning(
            "Analog matching requires at least one processed DEM of each type."
        )
        return None

    moon_key = moon_key or next(iter(metadata["moon"]))
    earth_key = earth_key or next(iter(metadata["earth"]))

    try:
        moon_entry = metadata["moon"][moon_key]
        earth_entry = metadata["earth"][earth_key]
    except KeyError as exc:
        raise ValueError(f"Unknown DEM key: {exc}") from exc

    moon_path = moon_entry["processed_file"]
    earth_path = earth_entry["processed_file"]
    if not os.path.isfile(moon_path):
        raise FileNotFoundError(f"Processed lunar DEM not found: {moon_path}")
    if not os.path.isfile(earth_path):
        raise FileNotFoundError(f"Processed Earth DEM not found: {earth_path}")

    moon_dem = np.load(moon_path)
    earth_dem = np.load(earth_path)
    match = find_best_terrain_analog(
        moon_dem,
        earth_dem,
        earth_entry,
        weights=weights,
    )
    save_analog_comparison(moon_dem, earth_dem, match, output_plot)

    LOGGER.info(
        "Best analog: row=%d, col=%d, latitude=%.6f, longitude=%.6f, score=%.5f",
        match["row"],
        match["col"],
        match["latitude"],
        match["longitude"],
        match["score"],
    )
    return match


def main():
    """Run preprocessing and, when possible, match the first available DEM pair."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-crs",
        default=DEFAULT_TARGET_CRS,
        help="Projected meter-based CRS for Earth DEMs (default: %(default)s).",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=DEFAULT_RESOLUTION_METERS,
        help="Pixel size in meters for both DEM types (default: %(default)s).",
    )
    parser.add_argument("--moon-key", help="Lunar DEM basename from metadata.")
    parser.add_argument("--earth-key", help="Earth DEM basename from metadata.")
    parser.add_argument(
        "--no-match",
        action="store_true",
        help="Preprocess DEMs and write metadata without analog matching.",
    )
    args = parser.parse_args()

    metadata = preprocess_dem_files(
        target_crs=args.target_crs,
        target_resolution=args.resolution,
    )

    if not args.no_match:
        run_analog_matching(
            metadata,
            moon_key=args.moon_key,
            earth_key=args.earth_key,
        )


if __name__ == "__main__":
    main()
