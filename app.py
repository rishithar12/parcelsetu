"""ParcelSetu AI: AI-assisted cadastral boundary verification dashboard (Streamlit)."""
import datetime
import os
import cv2, folium, geopandas as gpd, pandas as pd, plotly.express as px, streamlit as st
from shapely.geometry import box
from streamlit_folium import st_folium
from src import pipeline as P, export as X
from src.image_analysis import MODEL_LABEL

st.set_page_config(page_title="ParcelSetu AI", layout="wide", page_icon="🗺️")
st.markdown("""<style>
.block-container{padding-top:1.2rem}
.card{background:#fff;border:1px solid #e3e8ef;border-radius:10px;padding:12px 14px;border-top:4px solid var(--c)}
.card b{font-size:1.7rem;display:block;color:#0f2742}.card span{font-size:.75rem;color:#5b6b7f;text-transform:uppercase;letter-spacing:.04em}
.banner{background:#eef4fb;border-left:4px solid #1f5fa8;padding:8px 12px;border-radius:4px;font-size:.85rem;color:#17375e}
h1,h2,h3{color:#0f2742}</style>""", unsafe_allow_html=True)
S = st.session_state
COL = {"High": "#2e9e5b", "Medium": "#f0a020", "Low": "#d64545"}
SEVCOL = {"High": "#d64545", "Medium": "#f0a020", "Low": "#6b7bd6"}
card = lambda label, val, c="#1f5fa8": f'<div class="card" style="--c:{c}"><b>{val}</b><span>{label}</span></div>'

st.title("ParcelSetu AI")
st.caption("AI-Assisted Cadastral Boundary Verification: a quality-assurance layer that helps surveyors find "
           "conflicting or uncertain boundaries and prioritise field verification. Not a replacement for NAKSHA.")

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Data")
    if st.button("Load Demo Dataset", type="primary", use_container_width=True):
        try:
            with st.spinner("Generating synthetic demo and running analysis…"):
                S.D = P.load_demo(); P.run_all(S.D); S.sel = None; S.sel_b = None
        except Exception as e:
            st.error(f"Demo failed: {e}")
    img_f = st.file_uploader("Aerial image (JPG/PNG/GeoTIFF)", ["jpg", "jpeg", "png", "tif", "tiff"])
    par_f = st.file_uploader("Parcels (GeoJSON / zipped Shapefile / .shp+.shx+.dbf)", ["geojson", "json", "zip", "shp", "shx", "dbf", "prj", "cpg"], accept_multiple_files=True)
    gn_f = st.file_uploader("GNSS CSV (latitude, longitude, parcel_id)", ["csv"])
    dsm_f = st.file_uploader("DSM/DTM GeoTIFF (optional)", ["tif", "tiff"])
    if st.button("Load uploaded data", use_container_width=True):
        if not (img_f and par_f):
            st.warning("Upload at least an aerial image and a parcel file.")
        else:
            try:
                S.D = P.load_inputs([(f.name, f.getvalue()) for f in par_f], img_f.name, img_f.getvalue(),
                                    gn_f.getvalue() if gn_f else None, dsm_f.getvalue() if dsm_f else None)
                P.run_all(S.D); S.sel = None; S.sel_b = None
            except Exception as e:
                st.error(f"Could not load data: {e}")
    st.header("Analysis")
    D = S.get("D")
    for label, fn, need in [("Run Feature Analysis", P.run_features, "img"), ("Run Boundary Verification", P.run_boundaries, "feat"),
                            ("Run Conflict Detection", P.run_conflicts, "bnd"), ("Run Priority Analysis", P.run_priority, "conf")]:
        if st.button(label, use_container_width=True, disabled=D is None):
            if need not in D:
                st.warning("Run the previous stage first.")
            else:
                fn(D); st.success("Done")
    st.markdown(f'<div class="banner"><b>Prototype / image-derived evidence</b><br>{MODEL_LABEL}</div>', unsafe_allow_html=True)

D = S.get("D")
if D is None:
    st.info("Click **Load Demo Dataset** in the sidebar to begin, or upload your own aerial image and parcel layer.")
    st.markdown("**Workflow:** Drone/aerial data → feature evidence → existing parcel records → evidence fusion → boundary verification → "
                "conflict detection → uncertainty → field-verification priority → surveyor review → validated GIS output.")
    st.stop()
if D["note"] or D["img_note"]:
    st.caption(f"ℹ️ {D['note']}{D['img_note']}")
if "prio" not in D:
    st.warning("Run all four analysis stages from the sidebar to populate the dashboard."); st.stop()

bnd, conf, prio, parcels = D["bnd"], D["conf"], D["prio"], D["parcels"]
openc = conf[conf.status == "Open"]
pids = list(parcels.parcel_id)
if S.get("sel") not in pids:
    S.sel = prio.parcel_id.iloc[0]

# ------------------------------------------------------------------ 1. summary
st.subheader("1 · Area summary")
n_ver = int((prio.priority_level <= 2).sum())
cs = st.columns(7)
for c, (l, v, k) in zip(cs, [("Parcels", len(parcels), "#1f5fa8"), ("Boundaries analysed", len(bnd), "#1f5fa8"),
                             ("High confidence", int((bnd.level == "High").sum()), COL["High"]),
                             ("Medium confidence", int((bnd.level == "Medium").sum()), COL["Medium"]),
                             ("Low confidence", int((bnd.level == "Low").sum()), COL["Low"]),
                             ("Open conflicts", len(openc), "#8e2f2f"), ("Need field verification", n_ver, "#8e2f2f")]):
    c.markdown(card(l, v, k), unsafe_allow_html=True)
st.caption("Boundary Evidence Score is prototype-level evidence support, NOT legal accuracy. Priority is a workflow aid, not an official survey priority.")

# ------------------------------------------------------------------ 2. map
st.subheader("2 · Interactive map")
layers = st.multiselect("Layers", ["Aerial image", "Parcels", "Boundary confidence", "Buildings", "Roads", "Conflicts", "GNSS"],
                        default=["Aerial image", "Parcels", "Boundary confidence", "Buildings", "Conflicts", "GNSS"])


def make_map():
    to = lambda g: g.to_crs(4326)
    par = to(parcels)
    b = par.total_bounds
    ctr = [(b[1] + b[3]) / 2, (b[0] + b[2]) / 2]
    if S.sel:
        c = par[par.parcel_id == S.sel].geometry.iloc[0].centroid
        ctr = [c.y, c.x]
    m = folium.Map(location=ctr, zoom_start=20 if S.sel else 19, max_zoom=22, tiles=None)
    folium.TileLayer("CartoDB positron", max_native_zoom=20, max_zoom=22, name="Basemap (needs internet)").add_to(m)
    ll = lambda g: [(y, x) for x, y in g.coords]
    if "Aerial image" in layers:
        w, s, e, n = to(gpd.GeoSeries([box(*D["extent"])], crs=D["crs"])).total_bounds
        os.makedirs("outputs", exist_ok=True)
        cv2.imwrite("outputs/overlay.png", D["img"])
        folium.raster_layers.ImageOverlay("outputs/overlay.png", bounds=[[s, w], [n, e]], opacity=0.95, name="Aerial image").add_to(m)
    lv = dict(zip(prio.parcel_id, prio.priority_level))
    if "Parcels" in layers:
        fg = folium.FeatureGroup("Parcels")
        for r in par.itertuples():
            for poly in (r.geometry.geoms if r.geometry.geom_type == "MultiPolygon" else [r.geometry]):
                folium.Polygon(ll(poly.exterior), color="#fff" if r.parcel_id != S.sel else "#00e5ff", weight=1 if r.parcel_id != S.sel else 4,
                               fill=True, fill_color={1: "#d64545", 2: "#f0a020", 3: "#2e9e5b"}[lv[r.parcel_id]],
                               fill_opacity=0.18, tooltip=f"Parcel {r.parcel_id}").add_to(fg)
            c = r.geometry.representative_point()
            folium.Marker([c.y, c.x], icon=folium.DivIcon(icon_size=(60, 14), icon_anchor=(30, 7), html=
                          f'<div style="font:600 11px sans-serif;color:#fff;text-shadow:0 0 3px #000">{r.parcel_id}</div>')).add_to(fg)
        fg.add_to(m)
    if "Boundary confidence" in layers:
        fg, bg = folium.FeatureGroup("Boundary confidence"), to(bnd)
        for r in bg.itertuples():
            pop = (f"<b>Boundary {r.boundary_id}</b><br>Parcels: {r.parcel_a} / {r.parcel_b or '—'}<br>Confidence: {r.score:.0f}% ({r.level})"
                   f"<br>Image evidence: {r.image_label}<br>GNSS: {r.gnss_state}<br>Reason: {r.reasons}<br>"
                   f"Status: {'Verified' if all(D['status'].get(p) == 'Verified' for p in (r.parcel_a, r.parcel_b) if p) else 'Needs verification' if r.level != 'High' else 'OK'}")
            folium.PolyLine(ll(r.geometry), color=COL[r.level], weight=6 if r.boundary_id == S.get("sel_b") else 4, opacity=0.95,
                            dash_array="6 6" if r.level == "Low" else None, tooltip=f"Boundary {r.boundary_id}",
                            popup=folium.Popup(pop, max_width=300)).add_to(fg)
        fg.add_to(m)
    for key, gdf, colr in (("Buildings", D["feat"]["buildings"], "#ff7043"), ("Roads", D["feat"]["roads"], "#455a64")):
        if key in layers and len(gdf):
            folium.GeoJson(to(gdf), name=key, style_function=lambda f, c=colr: dict(color=c, weight=1, fillOpacity=0.25)).add_to(m)
    if "Conflicts" in layers:
        fg = folium.FeatureGroup("Conflicts")
        for r in openc.itertuples():
            pt = to(gpd.GeoSeries([r.geometry], crs=D["crs"])).iloc[0]
            folium.CircleMarker([pt.y, pt.x], radius=7, color="#000", weight=1, fill=True, fill_color=SEVCOL[r.severity], fill_opacity=0.95,
                                popup=folium.Popup(f"<b>{r.type}</b> ({r.severity})<br>{r.parcel_id}<br>{r.explanation}", max_width=280)).add_to(fg)
        fg.add_to(m)
    if "GNSS" in layers and D["gnss"] is not None:
        fg = folium.FeatureGroup("GNSS")
        for r in to(D["gnss"]).itertuples():
            folium.CircleMarker([r.geometry.y, r.geometry.x], radius=3, color="#1565c0", fill=True, tooltip=f"{r.ref} → {r.parcel_id}").add_to(fg)
        fg.add_to(m)
    folium.LayerControl().add_to(m)
    return m


out = st_folium(make_map(), height=560, use_container_width=True, returned_objects=["last_object_clicked_tooltip"], key="map")
tip = ((out or {}).get("last_object_clicked_tooltip") or "").strip()
if tip and tip != S.get("last_tip"):          # clicking a parcel/boundary on the map selects it
    S.last_tip = tip
    if tip.startswith("Parcel "):
        S.sel, S.sel_b = tip[7:].strip(), None; st.rerun()
    elif tip.startswith("Boundary "):
        b = bnd[bnd.boundary_id == tip.split()[1]]
        if len(b):
            S.sel, S.sel_b = b.parcel_a.iloc[0], b.boundary_id.iloc[0]; st.rerun()
st.caption("Green/amber/red lines = high/medium/low boundary confidence (dashed = low). Parcel fill = verification priority. Click a parcel or boundary to inspect it.")

# ------------------------------------------------------------------ 3 + 4. conflicts & queue
c1, c2 = st.columns([3, 2])
with c1:
    st.subheader("3 · Conflict panel")
    show = conf.assign(Confidence=conf.score.round(0).astype("Int64").astype(str) + "%").rename(
        columns={"parcel_id": "Parcel", "type": "Conflict", "severity": "Severity", "action": "Action", "status": "Status"})
    st.dataframe(show[["Parcel", "Conflict", "Severity", "Confidence", "Action", "Status"]], hide_index=True, use_container_width=True, height=300)
with c2:
    st.subheader("4 · Field verification queue")
    q = prio[prio.priority_level <= 2].reset_index(drop=True)
    q.insert(0, "Rank", q.index + 1)
    ev = st.dataframe(q[["Rank", "parcel_id", "priority_level", "priority_score", "main_issue"]].rename(
        columns={"parcel_id": "Parcel", "priority_level": "Priority", "priority_score": "Score", "main_issue": "Main issue"}),
        hide_index=True, use_container_width=True, height=300, on_select="rerun", selection_mode="single-row", key="queue")
    rows = ev.selection.rows if ev is not None else []
    if rows and q.parcel_id.iloc[rows[0]] != S.get("last_q"):
        S.last_q = S.sel = q.parcel_id.iloc[rows[0]]; S.sel_b = None; st.rerun()
    st.caption("Priority 1 = immediate · 2 = review recommended · 3 = low (not listed).")

# ------------------------------------------------------------------ 5. details
st.subheader("5 · Boundary details")
S.sel = st.selectbox("Inspect parcel", pids, index=pids.index(S.sel))
pb = bnd[(bnd.parcel_a == S.sel) | (bnd.parcel_b == S.sel)].copy()
pb["Neighbour"] = pb.apply(lambda r: (r.parcel_b if r.parcel_a == S.sel else r.parcel_a) or "— (outer edge)", axis=1)
pb["Conflicts"] = pb.boundary_id.apply(lambda b: ", ".join(openc[openc.boundary_id == b].type.unique()) or "none")
pb["Confidence"] = pb.score.round(0).astype(int).astype(str) + "% " + pb.level
st.dataframe(pb[["boundary_id", "Neighbour", "Confidence", "image_label", "gnss_state", "Conflicts", "reasons"]].rename(
    columns={"boundary_id": "Boundary", "image_label": "Image evidence", "gnss_state": "GNSS", "reasons": "Evidence / reason"}),
    hide_index=True, use_container_width=True)
mine = openc[openc.parcel_ids.apply(lambda l: S.sel in l)]
for r in mine.itertuples():
    st.markdown(f"**{r.type}** · {r.severity} · evidence {r.score:.0f}%  \n{r.explanation}  \n*Recommended:* {r.action}")
st.caption("Shared edges are stored once: a boundary listed with a neighbour is a single edge used by both parcels.")

# ------------------------------------------------------------------ 6. surveyor edit
st.subheader("6 · Surveyor review & edit")
e1, e2 = st.columns(2)
with e1:
    opts = ["Unreviewed", "Verified", "Needs Field Survey"]
    new = st.radio("Parcel status", opts, index=opts.index(D["status"].get(S.sel, "Unreviewed")), horizontal=True)
    if st.button("Save status"):
        old, before = D["status"][S.sel], P.open_sig(D, S.sel)
        D["status"][S.sel] = new; P.run_priority(D)
        D["history"].append(dict(Time=datetime.datetime.now().strftime("%H:%M:%S"), Parcel=S.sel, Old=old, New=new, Action="Surveyor review"))
        D["last_edit"] = dict(parcel=S.sel, before=before, after=P.open_sig(D, S.sel)); st.rerun()
with e2:
    bsel = st.selectbox("Shared/boundary edge to edit", list(pb.boundary_id), index=list(pb.boundary_id).index(S.sel_b) if S.get("sel_b") in list(pb.boundary_id) else 0)
    brow = bnd[bnd.boundary_id == bsel].iloc[0]
    d1, d2 = st.columns(2)
    dx, dy = d1.number_input("Shift east (m)", value=0.0, step=0.1), d2.number_input("Shift north (m)", value=0.0, step=0.1)
    acts = {"Snap copies to one edge": (0.0, 0.0), "Apply manual shift": (dx, dy)}
    if brow.mismatch_m:
        acts[f"Apply image-suggested shift ({brow.sdx:+.1f}, {brow.sdy:+.1f} m)"] = (float(brow.sdx), float(brow.sdy))
    pick = st.selectbox("Geometry action", list(acts))
    if st.button("Apply geometry edit"):
        before = P.open_sig(D, S.sel)
        P.edit_geometry(D, bsel, *acts[pick])
        D["history"].append(dict(Time=datetime.datetime.now().strftime("%H:%M:%S"), Parcel=S.sel, Old=bsel, New="edited", Action=pick))
        D["last_edit"] = dict(parcel=S.sel, before=before, after=P.open_sig(D, S.sel)); st.rerun()
le = D.get("last_edit")
if le:
    b4, af = le["before"], le["after"]
    st.markdown(f"**Before / after for {le['parcel']}:** resolved {sorted(b4 - af) or 'none'} · remaining {sorted(b4 & af) or 'none'} · new {sorted(af - b4) or 'none'}")
if D["history"]:
    st.dataframe(pd.DataFrame(D["history"]), hide_index=True, use_container_width=True)
tp = D["topo"]
st.markdown(f"**Topology validation:** {len(tp)} issue(s)" + ("" if len(tp) else " ✅ none"))
if len(tp):
    st.dataframe(tp.assign(parcel_ids=tp.parcel_ids.apply(" / ".join))[["issue_type", "parcel_ids", "severity", "detail", "boundary_id"]], hide_index=True, use_container_width=True)

# ------------------------------------------------------------------ charts
st.subheader("Analytics")
a, b_, c_ = st.columns(3)
a.plotly_chart(px.histogram(bnd, x="score", nbins=10, title="Boundary evidence score", color_discrete_sequence=["#1f5fa8"]), use_container_width=True)
b_.plotly_chart(px.bar(openc.groupby("type").size().reset_index(name="n"), x="type", y="n", title="Open conflicts by type", color_discrete_sequence=["#d64545"]), use_container_width=True)
c_.plotly_chart(px.pie(prio, names="priority_level", title="Parcels by priority", color="priority_level",
                       color_discrete_map={1: "#d64545", 2: "#f0a020", 3: "#2e9e5b"}), use_container_width=True)

# ------------------------------------------------------------------ export
st.subheader("Export")
x = st.columns(5)
x[0].download_button("Corrected GeoJSON", X.geojson_bytes(parcels, prio, D["status"]), "corrected_parcels.geojson")
x[1].download_button("Conflict report CSV", X.csv_bytes(conf), "conflicts.csv", "text/csv")
x[2].download_button("Verification priority CSV", X.csv_bytes(prio), "priority.csv", "text/csv")
x[3].download_button("Boundary confidence CSV", X.csv_bytes(bnd), "boundary_confidence.csv", "text/csv")
gp = X.gpkg_bytes(parcels, bnd, prio, D["status"])
if gp:
    x[4].download_button("GeoPackage", gp, "parcelsetu.gpkg")
