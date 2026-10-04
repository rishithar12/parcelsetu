"""Synthetic urban demo. SIMULATED DATA: parcels (with injected defects), an aerial image rendered
from the TRUE layout, and GNSS 'pegs'. The recorded parcels deviate from the image on purpose."""
import json, os
import cv2, numpy as np, geopandas as gpd, pandas as pd
from shapely.geometry import Polygon, Point

CRS = "EPSG:32643"                      # UTM 43N
OX, OY = 781000.0, 1430000.0            # local -> UTM origin
W, H, RES = 120.0, 110.0, 0.25          # scene size (m) and image resolution (m/px)
COLS, ROWS, CW, CH, X0, Y0 = 5, 4, 20.0, 20.0, 10.0, 15.0


def _pid(i, j):
    return f"P-{101 + j * COLS + i}"


def generate_demo(out_dir="data/demo"):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(7)
    node = lambda i, j: np.array([X0 + i * CW, Y0 + j * CH])
    # Defects injected into the RECORDED parcels (image shows the true layout):
    shared_shift = {(2, 1): (2.4, 0), (2, 2): (2.4, 0), (4, 2): (-1.7, 0), (4, 3): (-1.7, 0)}  # boundary mismatch
    one_parcel = {(3, 0, 0): (-1.5, 0), (3, 0, 3): (-1.5, 0),   # overlap
                  (1, 3, 1): (-1.2, 0), (1, 3, 2): (-1.2, 0)}   # gap
    rows = []
    for j in range(ROWS):
        for i in range(COLS):
            pts = []
            for k, c in enumerate([(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]):
                p = node(*c) + shared_shift.get(c, (0, 0)) + one_parcel.get((i, j, k), (0, 0))
                pts.append((OX + p[0], OY + p[1]))
            rows.append({"parcel_id": _pid(i, j), "geometry": Polygon(pts)})
    parcels = gpd.GeoDataFrame(rows, crs=CRS)

    # ---- aerial image rendered from TRUE geometry (BGR) ----
    px = lambda x, y: (int(round(x / RES)), int(round((H - y) / RES)))
    img = np.full((int(H / RES), int(W / RES), 3), (120, 170, 190), np.uint8)       # tan ground
    cv2.rectangle(img, px(X0, Y0 + ROWS * CH), px(X0 + COLS * CW, Y0), (130, 182, 198), -1)
    for y0, y1 in ((0, 12), (98, 110)):                                              # roads
        cv2.rectangle(img, px(0, y1), px(W, y0), (112, 112, 116), -1)
    for j in range(ROWS + 1):                                                        # fences / walls
        for i in range(COLS):
            if (i, j) == (0, 3):
                continue                                                             # no fence under the building
            cv2.line(img, px(*node(i, j)), px(*node(i + 1, j)), (45, 45, 50), 2)
    for i in range(COLS + 1):
        for j in range(ROWS):
            cv2.line(img, px(*node(i, j)), px(*node(i, j + 1)), (45, 45, 50), 2)
    for j in range(ROWS):                                                            # buildings
        for i in range(COLS):
            if i == 0 and j in (2, 3):
                continue
            x, y = X0 + i * CW + rng.uniform(4, 6), Y0 + j * CH + rng.uniform(4, 6)
            cv2.rectangle(img, px(x, y + rng.uniform(7, 9)), px(x + rng.uniform(8, 10), y), (50, 60, 185), -1)
    cv2.rectangle(img, px(14, 80), px(24, 70), (50, 60, 185), -1)                    # building crossing y=75
    for cx, cy, r in ((80, 75, 6.0), (30, 95, 5.0)):                                 # vegetation over fences
        cv2.circle(img, px(cx, cy), int(r / RES), (50, 140, 60), -1)
    img = np.clip(img + rng.normal(0, 4, img.shape), 0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (3, 3), 0.6)

    # ---- GNSS pegs at TRUE lattice nodes (some missing) ----
    g = []
    for i in range(COLS + 1):
        for j in range(ROWS + 1):
            if (i, j) in ((2, 1), (2, 2)) or rng.random() < 0.6:
                p = node(i, j) + rng.normal(0, 0.2, 2)
                g.append({"ref": f"GNSS-{i}{j}", "parcel_id": _pid(min(i, COLS - 1), min(j, ROWS - 1)),
                          "geometry": Point(OX + p[0], OY + p[1])})
    gnss = gpd.GeoDataFrame(g, crs=CRS).to_crs(4326)
    gnss["lat"], gnss["lon"] = gnss.geometry.y, gnss.geometry.x

    paths = {k: os.path.join(out_dir, v) for k, v in
             dict(parcels="parcels.geojson", gnss="gnss.csv", image="aerial.png", meta="meta.json").items()}
    cv2.imwrite(paths["image"], img)
    parcels.to_crs(4326).to_file(paths["parcels"], driver="GeoJSON")
    pd.DataFrame(gnss.drop(columns="geometry")).to_csv(paths["gnss"], index=False)
    json.dump({"extent": [OX, OY, OX + W, OY + H], "crs": CRS}, open(paths["meta"], "w"))
    return paths
