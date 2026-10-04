"""GIS input handling (GeoJSON / zipped or loose Shapefile / GNSS CSV) and shared-edge editing."""
import io, os, zipfile
import numpy as np, pandas as pd, geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon

UPLOAD_DIR = "data/uploads"
ID_CANDIDATES = ["parcel_id", "parcelid", "plot_id", "plot_no", "survey_no", "khasra", "pid", "id"]


def load_parcels(files):
    """files: list of (filename, bytes). Returns (GeoDataFrame in metric CRS, note)."""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    paths = {}
    for name, data in files:
        p = os.path.join(UPLOAD_DIR, os.path.basename(name))
        open(p, "wb").write(data)
        paths[name.lower()] = p
    src = next((p for n, p in paths.items() if n.endswith(".zip")), None)
    if src:
        with zipfile.ZipFile(src) as z:
            if not any(n.lower().endswith(".shp") for n in z.namelist()):
                raise ValueError("The ZIP does not contain a .shp file.")
        src = "zip://" + os.path.abspath(src)
    else:
        src = next((p for n, p in paths.items() if n.endswith((".shp", ".geojson", ".json"))), None)
    if src is None:
        raise ValueError("Upload a GeoJSON, a .zip Shapefile, or .shp together with .shx/.dbf.")
    try:
        gdf = gpd.read_file(src)
    except Exception as e:
        raise ValueError(f"Could not read parcel file: {e}")
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].reset_index(drop=True)
    if gdf.empty or not gdf.geom_type.isin(["Polygon", "MultiPolygon"]).all():
        raise ValueError("Parcel layer must contain only Polygon/MultiPolygon features.")
    note = ""
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
        note = "No CRS in file; assumed EPSG:4326. "
    if gdf.crs.is_geographic:
        gdf = gdf.to_crs(gdf.estimate_utm_crs())            # metres are needed for distances
    cols = {c.lower(): c for c in gdf.columns}
    idcol = next((cols[c] for c in ID_CANDIDATES if c in cols), None)
    ids = gdf[idcol].astype(str) if idcol else pd.Series([f"P-{i + 1:03d}" for i in range(len(gdf))])
    if ids.duplicated().any():
        ids = ids + "_" + ids.groupby(ids).cumcount().astype(str)
    return gpd.GeoDataFrame({"parcel_id": ids.values}, geometry=gdf.geometry.values, crs=gdf.crs), note


def load_gnss(data, work_crs):
    try:
        df = pd.read_csv(io.BytesIO(data))
    except Exception as e:
        raise ValueError(f"Invalid GNSS CSV: {e}")
    low = {c.lower().strip(): c for c in df.columns}
    pick = lambda opts: next((low[o] for o in opts if o in low), None)
    la, lo, pr = pick(["latitude", "lat"]), pick(["longitude", "lon", "lng"]), pick(["parcel_id", "parcel", "reference"])
    if not (la and lo and pr):
        raise ValueError("GNSS CSV needs latitude, longitude and parcel/reference ID columns.")
    ref = low.get("ref", pr)
    out = gpd.GeoDataFrame({"ref": df[ref].astype(str), "parcel_id": df[pr].astype(str)},
                           geometry=gpd.points_from_xy(df[lo], df[la]), crs=4326)
    return out.to_crs(work_crs)


def move_boundary(parcels, boundaries, bid, dx=0.0, dy=0.0, tol=3.0):
    """Edit ONE shared edge: both end-vertices are moved (by dx,dy) to a single common position in
    EVERY parcel that references them, so neighbours stay consistent (dx=dy=0 just snaps the copies)."""
    b = boundaries.set_index("boundary_id").geometry[bid]
    ends = [np.array(b.coords[0]), np.array(b.coords[-1])]
    new = []
    for g in parcels.geometry:
        out = []
        for poly in (list(g.geoms) if g.geom_type == "MultiPolygon" else [g]):
            c = []
            for v in poly.exterior.coords:
                d = [np.linalg.norm(np.array(v[:2]) - e) for e in ends]
                k = int(np.argmin(d))
                c.append(tuple(ends[k] + (dx, dy)) if d[k] < tol else tuple(v[:2]))
            out.append(Polygon(c, [i.coords for i in poly.interiors]))
        new.append(out[0] if len(out) == 1 else MultiPolygon(out))
    res = parcels.copy()
    res["geometry"] = new
    return res
