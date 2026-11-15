#!/usr/bin/env python3
"""Create an interactive web map of ranked Earth terrain analogs.

Reads ``top_terrain_analogs.json`` and ``data/processed/spatial_metadata.json``
from the repository root, converts the UTM result coordinates to WGS84, and
writes an interactive Folium map as ``terrain_analogs_map.html``.

Install the optional mapping dependency with:
    python -m pip install folium

Then run:
    python map_analogs.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import folium
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.warp import transform as transform_coordinates

ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS = ROOT / "top_terrain_analogs.json"
DEFAULT_METADATA = ROOT / "data" / "processed" / "spatial_metadata.json"
DEFAULT_OUTPUT = ROOT / "terrain_analogs_map.html"


def _earth_metadata_entry(metadata: dict[str, Any]) -> dict[str, Any]:
    earth_entries = metadata.get("earth", {})
    for entry in earth_entries.values():
        if isinstance(entry, dict) and "transform" in entry:
            return entry
    raise KeyError("No Earth metadata entry with a coordinate transform was found.")


def _get_lat_lon(
    result: dict[str, Any], earth_metadata: dict[str, Any]
) -> tuple[float, float]:
    """Return a result's WGS84 latitude/longitude from its UTM coordinates."""
    if "utm_x_m" in result and "utm_y_m" in result:
        x, y = float(result["utm_x_m"]), float(result["utm_y_m"])
    elif "row" in result and "col" in result:
        transform = earth_metadata.get("transform")
        if not isinstance(transform, (list, tuple)) or len(transform) < 6:
            raise ValueError("Earth metadata must include a six-value affine transform.")
        affine = Affine(*transform[:6])
        x, y = affine * (int(result["col"]) + 0.5, int(result["row"]) + 0.5)
    else:
        raise ValueError("A result needs UTM coordinates or row and col pixel indices.")

    source_crs = CRS.from_user_input(earth_metadata.get("crs", "EPSG:32616"))
    longitude, latitude = transform_coordinates(
        source_crs, CRS.from_epsg(4326), [x], [y]
    )
    return float(latitude[0]), float(longitude[0])


def _popup_html(
    result: dict[str, Any], latitude: float, longitude: float
) -> str:
    rank = result.get("rank", "?")
    score = result.get("score")
    score_text = f"{float(score):.6f}" if score is not None else "not available"
    rows = result.get("row", "not available")
    cols = result.get("col", "not available")
    weights = result.get("weights", {})
    rmse = result.get("rmse", {})

    details = [
        f"<b>Earth terrain analog #{rank}</b>",
        f"Score: {score_text} (lower is better)",
        f"Pixel: row {rows}, col {cols}",
        f"Latitude: {latitude:.6f}",
        f"Longitude: {longitude:.6f}",
    ]
    if weights:
        details.append(
            "Weights: " + ", ".join(
                f"{name} {float(value):.2f}" for name, value in weights.items()
            )
        )
    if rmse:
        details.append(
            "RMSE: " + ", ".join(
                f"{name} {float(value):.6f}" for name, value in rmse.items()
            )
        )
    return "<br>".join(details)


def create_map(
    results_path: Path = DEFAULT_RESULTS,
    metadata_path: Path = DEFAULT_METADATA,
    output_path: Path = DEFAULT_OUTPUT,
) -> Path:
    if not results_path.is_file():
        raise FileNotFoundError(f"Results JSON not found: {results_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Spatial metadata JSON not found: {metadata_path}")

    with results_path.open(encoding="utf-8") as file:
        results = json.load(file)
    with metadata_path.open(encoding="utf-8") as file:
        metadata = json.load(file)

    if not isinstance(results, list) or not results:
        raise ValueError("Results JSON must contain a non-empty list of matches.")

    earth_metadata = _earth_metadata_entry(metadata)
    points: list[tuple[dict[str, Any], float, float]] = []
    for result in results:
        latitude, longitude = _get_lat_lon(result, earth_metadata)
        points.append((result, latitude, longitude))

    center_lat = sum(point[1] for point in points) / len(points)
    center_lon = sum(point[2] for point in points) / len(points)
    terrain_map = folium.Map(
        location=(center_lat, center_lon),
        zoom_start=5,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    for result, latitude, longitude in points:
        rank = int(result.get("rank", len([p for p in points if p[0] is result]) + 1))
        score = float(result.get("score", 0.0))
        folium.Marker(
            location=(latitude, longitude),
            tooltip=f"Rank #{rank} — score {score:.6f}",
            popup=folium.Popup(_popup_html(result, latitude, longitude), max_width=350),
            icon=folium.DivIcon(
                html=(
                    '<div style="font-size: 13px; font-weight: bold; color: white; '
                    'background: #1769aa; border: 2px solid white; border-radius: 50%; '
                    'width: 26px; height: 26px; line-height: 24px; text-align: center; '
                    f'box-shadow: 0 1px 4px #555">{rank}</div>'
                )
            ),
        ).add_to(terrain_map)

    folium.LayerControl().add_to(terrain_map)
    if len(points) > 1:
        terrain_map.fit_bounds([(point[1], point[2]) for point in points], padding=(30, 30))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    terrain_map.save(str(output_path))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS,
                        help="Path to top_terrain_analogs.json")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA,
                        help="Path to spatial_metadata.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Output HTML map path")
    args = parser.parse_args()

    output_path = create_map(args.results, args.metadata, args.output)
    print(f"Saved interactive map to {output_path}")
    print("Open the HTML file in a web browser. An internet connection is needed for map tiles.")


if __name__ == "__main__":
    main()
