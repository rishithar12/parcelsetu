import json
import pandas as pd
import geopandas as gpd

from . import demo_data
from . import image_analysis
from . import parcel_analysis
from . import boundary_analysis
from . import topology


def load_demo():
    paths = demo_data.generate_demo()

    parcels, note = parcel_analysis.load_parcels([
        ("parcels.geojson", open(paths["parcels"], "rb").read())
    ])

    gnss = parcel_analysis.load_gnss(
        open(paths["gnss"], "rb").read(),
        parcels.crs
    )

    meta = json.load(open(paths["meta"], "r"))

    img, extent, img_note = image_analysis.load_image(
        open(paths["image"], "rb").read(),
        "aerial.png",
        parcels.crs,
        parcels.total_bounds,
        meta["extent"]
    )

    return {
        "parcels": parcels,
        "gnss": gnss,
        "img": img,
        "crs": parcels.crs,
        "extent": extent,
        "note": "Demo data is simulated. " + note,
        "img_note": img_note,
        "status": {p: "Unreviewed" for p in parcels.parcel_id},
        "history": []
    }


def load_inputs(parcel_files, image_name, image_bytes,
                gnss_bytes=None, dsm_bytes=None):

    parcels, note = parcel_analysis.load_parcels(parcel_files)

    img, extent, img_note = image_analysis.load_image(
        image_bytes,
        image_name,
        parcels.crs,
        parcels.total_bounds
    )

    gnss = None
    if gnss_bytes:
        gnss = parcel_analysis.load_gnss(
            gnss_bytes,
            parcels.crs
        )

    return {
        "parcels": parcels,
        "gnss": gnss,
        "img": img,
        "crs": parcels.crs,
        "extent": extent,
        "note": note,
        "img_note": img_note,
        "dsm": dsm_bytes,
        "status": {p: "Unreviewed" for p in parcels.parcel_id},
        "history": []
    }


def run_features(D):
    D["feat"] = image_analysis.analyze_image(
        D["img"],
        D["extent"],
        D["crs"]
    )

    if D.get("dsm"):
        D["feat"] = image_analysis.apply_dsm(
            D["feat"],
            D["dsm"]
        )

    return D


def run_boundaries(D):
    bnd = boundary_analysis.build_boundaries(D["parcels"]).copy()

    scores = []
    levels = []
    image_labels = []
    gnss_states = []
    reasons = []
    mismatch = []
    sdx = []
    sdy = []

    for r in bnd.itertuples():
        dev = float(r.deviation_m)

        # Prototype confidence score based on boundary consistency
        score = max(0.0, min(100.0, 100.0 - dev * 20.0))

        if score >= 80:
            level = "High"
        elif score >= 50:
            level = "Medium"
        else:
            level = "Low"

        if dev <= 0.25:
            image_label = "Boundary consistent"
            reason = "Shared boundary copies are closely aligned."
        else:
            image_label = "Boundary mismatch"
            reason = f"Boundary copies differ by {dev:.2f} m."

        if D.get("gnss") is not None and not D["gnss"].empty:
            gnss_state = "Available"
        else:
            gnss_state = "Not available"

        scores.append(score)
        levels.append(level)
        image_labels.append(image_label)
        gnss_states.append(gnss_state)
        reasons.append(reason)
        mismatch.append(dev)
        sdx.append(dev)
        sdy.append(0.0)

    bnd["score"] = scores
    bnd["level"] = levels
    bnd["image_label"] = image_labels
    bnd["gnss_state"] = gnss_states
    bnd["reasons"] = reasons
    bnd["mismatch_m"] = mismatch
    bnd["sdx"] = sdx
    bnd["sdy"] = sdy

    D["bnd"] = bnd
    return D


def run_conflicts(D):
    raw = topology.validate(
        D["parcels"],
        D.get("bnd")
    )

    rows = []

    for r in raw.itertuples():
        pids = r.parcel_ids

        if not isinstance(pids, list):
            pids = list(pids)

        parcel_id = pids[0] if pids else "Unknown"

        if r.severity == "High":
            score = 85
            action = "Field verification recommended"
        elif r.severity == "Medium":
            score = 65
            action = "Review recommended"
        else:
            score = 40
            action = "Monitor"

        rows.append({
            "parcel_id": parcel_id,
            "parcel_ids": pids,
            "boundary_id": r.boundary_id,
            "type": r.issue_type,
            "severity": r.severity,
            "score": score,
            "explanation": r.detail,
            "action": action,
            "status": "Open",
            "geometry": r.geometry
        })

    D["conf"] = pd.DataFrame(rows, columns=[
        "parcel_id",
        "parcel_ids",
        "boundary_id",
        "type",
        "severity",
        "score",
        "explanation",
        "action",
        "status",
        "geometry"
    ])

    D["topo"] = raw

    return D


def run_priority(D):
    rows = []

    conf = D.get("conf", pd.DataFrame())

    for pid in D["parcels"].parcel_id:

        issues = conf[
            conf.parcel_ids.apply(
                lambda ids: pid in ids if isinstance(ids, list) else False
            )
        ]

        status = D["status"].get(pid, "Unreviewed")

        if status == "Verified":
            level = 3
            score = 10
            issue = "Verified by surveyor"

        elif len(issues) == 0:
            level = 3
            score = 0
            issue = "No major issue detected"

        elif "High" in issues.severity.values:
            level = 1
            score = min(100, 70 + 10 * len(issues))
            issue = issues.iloc[0].type

        else:
            level = 2
            score = min(100, 40 + 10 * len(issues))
            issue = issues.iloc[0].type

        rows.append({
            "parcel_id": pid,
            "priority_level": level,
            "priority_score": score,
            "main_issue": issue
        })

    D["prio"] = pd.DataFrame(rows)

    return D


def run_all(D):
    run_features(D)
    run_boundaries(D)
    run_conflicts(D)
    run_priority(D)

    return D


def open_sig(D, parcel_id):
    conf = D.get("conf")

    if conf is None or conf.empty:
        return set()

    rows = conf[
        conf.parcel_ids.apply(
            lambda ids: parcel_id in ids if isinstance(ids, list) else False
        )
    ]

    return set(rows.type.astype(str))


def edit_geometry(D, boundary_id, dx=0.0, dy=0.0):
    if "bnd" not in D:
        return

    D["parcels"] = parcel_analysis.move_boundary(
        D["parcels"],
        D["bnd"],
        boundary_id,
        dx,
        dy
    )

    run_all(D)