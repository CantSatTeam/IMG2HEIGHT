"""
Downloads a corresponding RGB image for an input DEM/DSM GeoTIFF using Google Earth Engine.
Example:
  python DEM2RGB.py sample/temp.tif
Outputs:
  sample/temp.jpg       (RGB image resized to DEM width/height)
  sample/temp_rgb.tif   (RGB TIFF resized to DEM width/height, not georeferenced)
Notes:
  - Requires GDAL command-line tools: gdalwarp
  - Requires Earth Engine auth + project set
"""

import sys
import zipfile
import subprocess
from pathlib import Path

import ee
import requests
import rasterio
import rasterio.features
import rasterio.warp
import numpy as np
from PIL import Image

SATELLITE_SR = "COPERNICUS/S2_SR_HARMONIZED"
RGB_BANDS = ["B4", "B3", "B2"]  # R, G, B

MINN = 0
MAXX = 3000

SCALE = 2  # meters per pixel for download (bigger = smaller file)

FILTER_START = "2023-06-01"
FILTER_END = "2023-09-01"
MAX_CLOUD_PCT = 30


def run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed:\n  {' '.join(cmd)}\n\nSTDERR:\n{proc.stderr}")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python DEM2RGB.py /path/to/dem.tif")
        sys.exit(2)

    dem_path = Path(sys.argv[1]).resolve()
    if not dem_path.exists():
        raise FileNotFoundError(dem_path)

    tempdir = Path("tempdir")
    tempdir.mkdir(exist_ok=True)

    temp_dem = tempdir / "temp.tif"
    if temp_dem.exists():
        temp_dem.unlink()

    run(["gdalwarp", "-ot", "Float32", "-q", str(dem_path), str(temp_dem)])

    # Extract footprint geometry
    with rasterio.open(temp_dem) as ds:
        mask = ds.dataset_mask()
        geom = None
        for g, _val in rasterio.features.shapes(mask, transform=ds.transform):
            geom = rasterio.warp.transform_geom(ds.crs, "EPSG:4326", g, precision=6)
            break
        if geom is None:
            raise RuntimeError("Could not extract geometry from DEM mask (no valid pixels?).")

    print("Geometry selected:")
    print(geom)

    ee.Initialize()
    ee_geom = ee.Geometry(geom)

    # IMPORTANT: use a smaller download region to avoid request-size limits
    download_geom = ee_geom.centroid().buffer(5000).bounds()  # 5km box
    print("Using download region (small bounds):")
    print(download_geom.getInfo())

    # Build collection over the small download geom
    coll = (ee.ImageCollection(SATELLITE_SR)
            .filterBounds(download_geom)
            .filterDate(FILTER_START, FILTER_END)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", MAX_CLOUD_PCT))
            .select(RGB_BANDS))

    if coll.size().getInfo() == 0:
        print("Warning: no images in date/cloud filter. Falling back to all dates (still small region).")
        coll = ee.ImageCollection(SATELLITE_SR).filterBounds(download_geom).select(RGB_BANDS)

    img = coll.median()

    # Download (ZIP if filePerBand=True, but don’t assume!)
    url = img.getDownloadURL({
        "region": download_geom,        # <<< FIX: was ee_geom
        "scale": SCALE,
        "crs": "EPSG:4326",
        "filePerBand": True,
        "format": "GEO_TIFF"
    })

    print("Downloading file:")
    print(url)

    out_bin = tempdir / "gee_rgb.bin"
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(out_bin, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    # If it’s a zip, extract; if it’s a tif, keep it
    extracted_tifs: list[Path] = []
    if zipfile.is_zipfile(out_bin):
        zip_path = tempdir / "gee_rgb.zip"
        if zip_path.exists():
            zip_path.unlink()
        out_bin.rename(zip_path)

        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(tempdir)

        zip_path.unlink(missing_ok=True)
        extracted_tifs = list(tempdir.glob("*.tif"))
    else:
        # Some EE responses can be a single tif; try treating it as such
        tif_path = tempdir / "gee_rgb.tif"
        if tif_path.exists():
            tif_path.unlink()
        out_bin.rename(tif_path)
        extracted_tifs = [tif_path]

        # After extracted_tifs is set
    if len(extracted_tifs) == 1:
        # Likely a single multiband GeoTIFF (or already 8-bit RGB)
        tif_path = extracted_tifs[0]
        with rasterio.open(tif_path) as src:
            arr = src.read()  # shape: (bands, H, W)

        # If it's 3 bands, assume it's already RGB or reflectance-scaled
        if arr.shape[0] >= 3:
            r = arr[0].astype(np.float32)
            g = arr[1].astype(np.float32)
            b = arr[2].astype(np.float32)

            # If data looks like reflectance (0..3000), scale to uint8
            def scale_u8(x):
                x = (x - MINN) / max(1.0, (MAXX - MINN))
                x = np.clip(x, 0.0, 1.0)
                return (x * 255.0).astype(np.uint8)

            # Heuristic: if max > 255, scale; else assume already 8-bit
            if max(r.max(), g.max(), b.max()) > 255:
                r8, g8, b8 = scale_u8(r), scale_u8(g), scale_u8(b)
            else:
                r8, g8, b8 = r.astype(np.uint8), g.astype(np.uint8), b.astype(np.uint8)

            rgb = Image.merge("RGB", (
                Image.fromarray(r8, mode="L"),
                Image.fromarray(g8, mode="L"),
                Image.fromarray(b8, mode="L"),
            ))
        else:
            raise RuntimeError(f"Single tif returned but has {arr.shape[0]} band(s), expected 3.")
    else:
        # Existing per-band logic
        def find_band(band: str) -> Path:
            for p in extracted_tifs:
                if band in p.name:
                    return p
            raise FileNotFoundError(
                f"Could not find extracted GeoTIFF for band {band}. Files: {[p.name for p in extracted_tifs]}"
            )

        b4 = find_band("B4")
        b3 = find_band("B3")
        b2 = find_band("B2")

        def to_uint8(p: Path) -> Image.Image:
            arr = np.array(Image.open(p)).astype(np.float32)
            arr = (arr - MINN) / max(1.0, (MAXX - MINN))
            arr = np.clip(arr, 0.0, 1.0)
            arr = (arr * 255.0).astype(np.uint8)
            return Image.fromarray(arr, mode="L")

        rgb = Image.merge("RGB", (to_uint8(b4), to_uint8(b3), to_uint8(b2)))


    # Resize to match DEM pixel dims
    with Image.open(dem_path) as dem_img:
        width, height = dem_img.size
    rgb = rgb.resize((int(width), int(height)), Image.Resampling.LANCZOS)

    out_jpg = dem_path.with_suffix(".jpg")
    out_tif = dem_path.with_name(dem_path.stem + "_rgb.tif")

    rgb.save(out_jpg, quality=95)
    rgb.save(out_tif)

    print(f"Saved:\n  {out_jpg}\n  {out_tif}")

    # Cleanup tempdir contents (optional)
    for p in tempdir.glob("*"):
        try:
            if p.is_file():
                p.unlink()
        except Exception:
            pass


if __name__ == "__main__":
    main()
