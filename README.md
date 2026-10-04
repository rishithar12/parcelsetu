<<<<<<< HEAD
# parcelsetu
=======
# ParcelSetu AI: AI-Assisted Cadastral Boundary Verification

A QA layer that helps surveyors find conflicting/uncertain parcel boundaries and prioritise field verification.
It does **not** replace NAKSHA or the government cadastral workflow.

## Run
```
cd parcelsetu_ai
python -m venv venv
venv\Scripts\activate          # Windows   (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```
Open the sidebar and click **Load Demo Dataset**. Run tests with `pytest tests -q`.
If rasterio/geopandas fail to install on Windows, see SETUP_GUIDE.md (Conda route).

## What is real vs simulated
- **Computer vision (real, classical):** OpenCV HSV segmentation, black-hat fence-line extraction, contours. No deep-learning model.
- **GIS processing (real):** Shapely/GeoPandas topology (overlap, gap, invalid, self-intersection, shared-edge).
- **Demo data (simulated):** parcels, aerial image and GNSS are synthetically generated with injected defects.
- Evidence score = prototype evidence support, not legal accuracy. Priority = workflow aid, not official.

## Layout
`app.py` dashboard · `src/pipeline.py` stage glue · `image_analysis` · `boundary_analysis` (shared edges) · `evidence_scoring` ·
`conflict_detection` · `priority_engine` · `topology` · `export` · `demo_data` · `tests/test_basic.py`

## Limitations
Thresholds are tuned for demo imagery; JPG/PNG are assumed to cover the parcel extent; shared-edge matching needs
vertex-to-vertex edge correspondence; DSM is only used to confirm building candidates.
>>>>>>> fa03940 (Initial ParcelSetu AI prototype)
