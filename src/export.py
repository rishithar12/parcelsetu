import io
import pandas as pd
import geopandas as gpd


def geojson_bytes(parcels, priority, status):
    out = parcels.copy()

    out["verification_status"] = out["parcel_id"].map(
        lambda p: status.get(p, "Unreviewed")
    )

    if priority is not None and not priority.empty:
        out = out.merge(
            priority[["parcel_id", "priority_level", "priority_score", "main_issue"]],
            on="parcel_id",
            how="left"
        )

    return out.to_crs(4326).to_json().encode("utf-8")


def csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


def gpkg_bytes(parcels, boundaries, priority, status):
    try:
        buffer = io.BytesIO()

        out_parcels = parcels.copy()
        out_parcels["verification_status"] = out_parcels["parcel_id"].map(
            lambda p: status.get(p, "Unreviewed")
        )

        if priority is not None and not priority.empty:
            out_parcels = out_parcels.merge(
                priority[
                    ["parcel_id", "priority_level",
                     "priority_score", "main_issue"]
                ],
                on="parcel_id",
                how="left"
            )

        out_parcels.to_file(
            buffer,
            driver="GPKG",
            layer="parcels"
        )

        return buffer.getvalue()

    except Exception:
        return None