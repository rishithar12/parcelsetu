"""Lightweight aerial-image analysis. PROTOTYPE / IMAGE-DERIVED EVIDENCE.
This is CLASSICAL computer vision (HSV colour segmentation, morphological black-hat line extraction,
contour detection, Canny edges). It is NOT a deep-learning model; thresholds are tuned for the demo
imagery and will need re-tuning (or a real segmentation model) for real drone data."""
import cv2, numpy as np, geopandas as gpd
from shapely.geometry import Polygon

MODEL_LABEL = "Classical OpenCV pipeline: HSV segmentation + black-hat line extraction + contours (no deep-learning model)"


def load_image(data, name, work_crs, parcel_bounds, meta_extent=None):
    """Returns (bgr uint8, extent=(minx,miny,maxx,maxy) in work_crs, note)."""
    if name.lower().endswith((".tif", ".tiff")):
        from rasterio.io import MemoryFile
        from rasterio.warp import transform_bounds
        with MemoryFile(data) as mf, mf.open() as src:
            arr = src.read([1, 2, 3]) if src.count >= 3 else np.repeat(src.read(1)[None], 3, 0)
            b = src.bounds
            ext = transform_bounds(src.crs, work_crs, *b) if src.crs else tuple(b)
        if arr.dtype != np.uint8:
            arr = cv2.normalize(arr.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return np.ascontiguousarray(arr.transpose(1, 2, 0)[:, :, ::-1]), tuple(ext), "GeoTIFF: georeferenced extent used."
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode the aerial image.")
    if meta_extent:
        return bgr, tuple(meta_extent), "Demo image: extent from demo metadata (simulated)."
    x0, y0, x1, y1 = parcel_bounds
    return bgr, (x0 - 10, y0 - 10, x1 + 10, y1 + 10), "JPG/PNG has no georeference: ASSUMED to cover the parcel extent (+10 m). Alignment is approximate."


def segment(bgr, res_m):
    """Pure-OpenCV masks. res_m = metres per pixel."""
    hsv, gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV), cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    k = lambda n: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))
    clean = lambda m, n=5: cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, k(n)), cv2.MORPH_CLOSE, k(n))
    veg = clean(((H > 35) & (H < 85) & (S > 60)).astype(np.uint8))
    bld = clean((((H < 12) | (H > 170)) & (S > 90) & (V > 70)).astype(np.uint8), 7)
    road = clean(((S < 30) & (V > 70) & (V < 170)).astype(np.uint8), 9)
    # thin dark lines (fences/walls/hedges) via black-hat; suppress roofs and vegetation
    kn = max(5, int(round(2.5 / res_m)) | 1)
    bh = cv2.morphologyEx(cv2.GaussianBlur(gray, (3, 3), 0), cv2.MORPH_BLACKHAT, k(kn))
    veg_d = cv2.dilate(veg, k(7)) > 0
    cue = ((bh > 25) & ~veg_d & ~(cv2.dilate(bld, k(9)) > 0)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(cue)
    cue = np.isin(lab, [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= 25]).astype(np.uint8)
    return dict(veg=veg, bld=bld, road=road, cue=cue, veg_d=veg_d, edges=cv2.Canny(gray, 50, 150))


def _polys(mask, ext, sx, sy, min_m2, eps=2.0):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cs:
        c = cv2.approxPolyDP(c, eps, True)[:, 0, :]
        if len(c) >= 3:
            p = Polygon([(ext[0] + x * sx, ext[3] - y * sy) for x, y in c])
            if p.is_valid and p.area >= min_m2:
                out.append(p)
    return out


def analyze_image(bgr, extent, crs):
    h, w = bgr.shape[:2]
    sx, sy = (extent[2] - extent[0]) / w, (extent[3] - extent[1]) / h
    m = segment(bgr, (sx + sy) / 2)
    dist = lambda mask: cv2.distanceTransform((mask == 0).astype(np.uint8), cv2.DIST_L2, 3) * (sx + sy) / 2
    road_edge = cv2.morphologyEx(m["road"], cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    return dict(
        ext=extent, sx=sx, sy=sy, masks=m, dsm_confirmed=None,
        dist_cue=dist(m["cue"]), dist_road=dist(road_edge), veg_d=m["veg_d"],
        buildings=gpd.GeoDataFrame(
    {"kind": ["building"] * len(_polys(m["bld"], extent, sx, sy, 8.0))},
    geometry=_polys(m["bld"], extent, sx, sy, 8.0),
    crs=crs
),
roads=gpd.GeoDataFrame(
    {"kind": ["road"] * len(_polys(m["road"], extent, sx, sy, 50.0))},
    geometry=_polys(m["road"], extent, sx, sy, 50.0),
    crs=crs
),
vegetation=gpd.GeoDataFrame(
    {"kind": ["vegetation"] * len(_polys(m["veg"], extent, sx, sy, 5.0))},
    geometry=_polys(m["veg"], extent, sx, sy, 5.0),
    crs=crs
))


def apply_dsm(res, data):
    """Optional DSM: normalised height (DSM minus morphological ground estimate) confirms building candidates.
    ASSUMES the DSM covers the same extent as the image."""
    from rasterio.io import MemoryFile
    with MemoryFile(data) as mf, mf.open() as src:
        dsm = src.read(1).astype(np.float32)
    h, w = res["masks"]["bld"].shape
    dsm = cv2.resize(dsm, (w, h))
    ndsm = dsm - cv2.morphologyEx(dsm, cv2.MORPH_OPEN, np.ones((61, 61), np.uint8))
    res["dsm_confirmed"] = float(((ndsm > 2.0) & (res["masks"]["bld"] > 0)).sum() / max(1, (res["masks"]["bld"] > 0).sum()))
    return res
