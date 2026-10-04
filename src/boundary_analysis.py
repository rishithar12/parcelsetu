"""Split parcels into boundary segments and merge matching neighbour edges into ONE shared edge."""
import numpy as np, geopandas as gpd
from shapely.geometry import LineString


def _edges(parcels):
    out = []
    for pid, g in zip(parcels.parcel_id, parcels.geometry):
        for poly in (g.geoms if g.geom_type == "MultiPolygon" else [g]):
            c = [np.array(v[:2]) for v in poly.exterior.coords]
            out += [(pid, a, b) for a, b in zip(c[:-1], c[1:]) if np.linalg.norm(a - b) > 1e-6]
    return out


def build_boundaries(parcels, tol=3.0):
    """Edges of different parcels whose end points agree within `tol` metres are ONE shared boundary.
    Stored geometry = mean of both copies; deviation_m = how far the copies disagree (>0 means
    overlap/gap/inconsistency). Limitation: needs vertex-to-vertex edge correspondence."""
    E, used, rows = _edges(parcels), set(), []
    for i, (pa, a, b) in enumerate(E):
        if i in used:
            continue
        used.add(i)
        best, match = tol, None
        for j in range(i + 1, len(E)):
            pb, c, d = E[j]
            if j in used or pb == pa:
                continue
            d_same = max(np.linalg.norm(a - c), np.linalg.norm(b - d))
            d_flip = max(np.linalg.norm(a - d), np.linalg.norm(b - c))
            dd, (c2, d2) = (d_same, (c, d)) if d_same <= d_flip else (d_flip, (d, c))
            if dd < best:
                best, match = dd, (j, pb, c2, d2)
        if match:
            used.add(match[0])
            p0, p1, other, dev = (a + match[2]) / 2, (b + match[3]) / 2, match[1], best
        else:
            p0, p1, other, dev = a, b, None, 0.0
        rows.append({"boundary_id": f"B-{len(rows) + 1:03d}", "parcel_a": pa, "parcel_b": other,
                     "shared": other is not None, "deviation_m": dev, "length_m": float(np.linalg.norm(p1 - p0)),
                     "geometry": LineString([p0, p1])})
    return gpd.GeoDataFrame(rows, crs=parcels.crs)
