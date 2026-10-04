"""GIS topology validation with Shapely/GeoPandas (pure geometry; no imagery involved)."""
import pandas as pd
from shapely.ops import unary_union
from shapely.validation import explain_validity

COLS = ["issue_type", "parcel_ids", "severity", "detail", "boundary_id", "geometry"]


def validate(parcels, boundaries=None):
    rows = []
    bmap = {}
    if boundaries is not None:
        bmap = {frozenset((r.parcel_a, r.parcel_b)): r.boundary_id for r in boundaries.itertuples() if r.parcel_b}

    def add(t, pids, sev, detail, geom, bid=None):
        rows.append(dict(zip(COLS, [t, pids, sev, detail, bid, geom])))

    ids, geoms = list(parcels.parcel_id), list(parcels.geometry)
    for pid, g in zip(ids, geoms):
        if not g.is_valid:
            add("Invalid polygon", [pid], "High", explain_validity(g), g.representative_point())
        polys = list(g.geoms) if g.geom_type == "MultiPolygon" else [g]
        if len(polys) > 1:
            add("Disconnected geometry", [pid], "Medium", f"{len(polys)} separate parts", g.representative_point())
        if any(not p.exterior.is_simple for p in polys):
            add("Self-intersection", [pid], "High", "Boundary ring crosses itself", g.representative_point())
    valid = [(i, g) for i, g in zip(ids, geoms) if g.is_valid]
    vids, vg = [v[0] for v in valid], [v[1] for v in valid]
    sidx = parcels.sindex
    for i, g in enumerate(geoms):                                     # overlaps
        for j in sidx.query(g, predicate="intersects"):
            if j > i and g.is_valid and geoms[j].is_valid:
                inter = g.intersection(geoms[j])
                if inter.area > 0.25:
                    add("Parcel overlap", [ids[i], ids[j]], "High", f"Overlap area {inter.area:.1f} m²",
                        inter.representative_point(), bmap.get(frozenset((ids[i], ids[j]))))
    if vg:                                                            # gaps = what morphological closing fills
        union = unary_union(vg)
        gaps = union.buffer(2.0, join_style=2).buffer(-2.0, join_style=2).difference(union)
        for gp in (gaps.geoms if hasattr(gaps, "geoms") else [gaps]):
            if gp.geom_type == "Polygon" and 0.3 < gp.area < 0.1 * union.area:
                r = list(gp.minimum_rotated_rectangle.exterior.coords)
                longest = max(((r[k][0] - r[k + 1][0]) ** 2 + (r[k][1] - r[k + 1][1]) ** 2) ** .5 for k in range(4))
                nb = [p for p, g in zip(vids, vg) if g.distance(gp) < 0.25]
                add("Parcel gap", nb, "Medium", f"Unclaimed sliver {gp.area:.1f} m², ≈{gp.area / max(longest, 1e-6):.1f} m wide",
                    gp.representative_point(), bmap.get(frozenset(nb[:2])) if len(nb) >= 2 else None)
    if boundaries is not None:                                        # shared-edge inconsistency
        for r in boundaries[boundaries.shared & (boundaries.deviation_m > 0.25)].itertuples():
            add("Shared-edge inconsistency", [r.parcel_a, r.parcel_b], "Medium",
                f"The two parcel copies of the shared edge differ by {r.deviation_m:.2f} m",
                r.geometry.interpolate(0.5, normalized=True), r.boundary_id)
    return pd.DataFrame(rows, columns=COLS)
