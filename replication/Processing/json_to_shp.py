"""
Convert COCO building masks (gt / pred JSON files) into one shapefile per subfolder.
Set the paths below and run in Spyder.

The JSONs have no coordinates, only pixel masks inside 800x800 chips.
If CHIP_DIR points to the chip GeoTIFFs (named like COUNTY41CHIP00036668.tif),
polygons are placed on the map. Otherwise they stay in pixel coordinates.
"""
from pathlib import Path
import json

import numpy as np
import geopandas as gpd
from rasterio.features import shapes
from rasterio.transform import Affine
import rasterio
from shapely.geometry import shape

INPUT_DIR = Path(r"C:\Users\Rafael Tiara\Downloads\Pred\Pred")
OUTPUT_DIR = Path(r"C:\Users\Rafael Tiara\Downloads\Pred\Pred_shp")
CHIP_DIR = None  # e.g. Path(r"C:\path\to\chip_tifs"); leave None if you don't have them
SKIP = {".ipynb_checkpoints", "__MACOSX"}


def rle_decode(seg):
    """Decode a COCO RLE (compressed string or list) into a boolean mask."""
    h, w = seg["size"]
    counts = seg["counts"]
    if isinstance(counts, str):
        cnts, p = [], 0
        while p < len(counts):
            x, k, more = 0, 0, 1
            while more:
                c = ord(counts[p]) - 48
                x |= (c & 0x1F) << (5 * k)
                more = c & 0x20
                p += 1
                k += 1
                if not more and (c & 0x10):
                    x |= -1 << (5 * k)
            if len(cnts) > 2:
                x += cnts[-2]
            cnts.append(x)
        counts = cnts
    flat = np.zeros(h * w, dtype=np.uint8)
    pos, val = 0, 0
    for n in counts:
        flat[pos:pos + n] = val
        pos += n
        val = 1 - val
    return flat.reshape((h, w), order="F").astype(bool)


_geo_cache = {}
def chip_georef(chip):
    """Return (transform, crs) for a chip, from its GeoTIFF if available."""
    if chip in _geo_cache:
        return _geo_cache[chip]
    out = (Affine(1, 0, 0, 0, -1, 0), None)  # pixel coords, y flipped so maps look upright
    if CHIP_DIR is not None:
        tif = next(Path(CHIP_DIR).rglob(f"{chip}.tif"), None)
        if tif is not None:
            with rasterio.open(tif) as src:
                out = (src.transform, src.crs)
    _geo_cache[chip] = out
    return out


def read_annotations(f):
    d = json.load(open(f))
    return d["annotations"] if isinstance(d, dict) else d


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for sub in sorted(p for p in INPUT_DIR.iterdir() if p.is_dir() and p.name not in SKIP):
    files = [f for f in sub.rglob("*.json") if not SKIP & set(f.parts)]
    rows, crs = [], None
    for f in files:
        try:
            anns = read_annotations(f)
        except Exception as e:
            print(f"FAIL  {f}: {e}")
            continue
        for a in anns:
            chip = a.get("image_id_str", str(a.get("image_id")))
            transform, chip_crs = chip_georef(chip)
            crs = crs or chip_crs
            mask = rle_decode(a["segmentation"])
            for geom, _ in shapes(mask.astype(np.uint8), mask=mask, transform=transform):
                rows.append({
                    "file": f.stem,
                    "chip": chip,
                    "score": a.get("score"),
                    "category": a.get("category_id"),
                    "area_px": a.get("area"),
                    "geometry": shape(geom),
                })
    if not rows:
        print(f"SKIP  {sub.name}: nothing found")
        continue

    gdf = gpd.GeoDataFrame(rows, crs=crs)
    gdf.to_file(OUTPUT_DIR / f"{sub.name}.shp", driver="ESRI Shapefile")
    print(f"OK    {sub.name}: {len(files)} files, {len(gdf)} polygons")