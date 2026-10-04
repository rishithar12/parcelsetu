# ParcelSetu AI: Setup & Status Guide

*AI-assisted cadastral verification and quality-assurance layer. It helps surveyors find uncertain or conflicting
parcel boundaries and prioritise field verification. It is not a replacement for NAKSHA.*

## 1. What is happening right now (build status)

| Module | Status |
|---|---|
| `requirements.txt` | Written |
| `src/demo_data.py` (synthetic parcels, aerial image, GNSS) | Written |
| `src/parcel_analysis.py` (GeoJSON/Shapefile/CSV loading, shared-edge editing) | Written |
| `src/boundary_analysis.py` (segments, ONE shared edge per neighbour pair) | Written |
| `src/topology.py` (overlap, gap, invalid, self-intersection, shared-edge checks) | Written |
| `src/image_analysis.py` (OpenCV buildings/roads/vegetation/fence lines) | Written and tested here |
| `src/evidence_scoring.py`, `conflict_detection.py`, `priority_engine.py`, `export.py`, `pipeline.py` | Written (syntax-checked only) |
| `app.py`, `tests/test_basic.py`, `README.md` | Written (syntax-checked only) |

**Tested so far:** the image analysis was run on the generated demo image. It finds the fence line at its true
position, finds no support at the deliberately wrong recorded position, and finds weak support under the
vegetation and the building. The GIS and Streamlit parts could not be run in my sandbox (see section 2).

## 2. Packages: what was missing and why I could not install it

My sandbox has no internet access, so `pip install` fails here. It already had: **opencv, numpy, pandas, Pillow**.

Missing there (install them on YOUR machine; this is what `requirements.txt` does):
`streamlit, geopandas, shapely, rasterio, plotly, folium, streamlit-folium, pyogrio, pytest`

## 3. Extra data needed from you

**None.** Click **Load Demo Dataset** and the app generates parcels, an aerial image and GNSS points.
Optional real data for later:
- Aerial image: GeoTIFF is best (carries coordinates). JPG/PNG is assumed to cover the parcel area.
- Parcels: GeoJSON, or a Shapefile as a .zip.
- GNSS CSV columns: `latitude, longitude, parcel_id`.
- DSM: GeoTIFF covering the same area as the image (optional).

## 4. How to run on your computer

Requires Python 3.10 to 3.12. Windows PowerShell shown first, then macOS/Linux.

```
cd parcelsetu_ai
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
streamlit run app.py             
```

If `rasterio` or `geopandas` fails to install on Windows, use Conda instead:
```
conda create -n parcelsetu python=3.11 -y
conda activate parcelsetu
conda install -c conda-forge geopandas rasterio shapely -y
pip install streamlit folium streamlit-folium plotly opencv-python-headless pyogrio pytest
```

Quick check that the image part works now (no Streamlit needed):
```
python -c "from src.demo_data import generate_demo; print(generate_demo())"
```
This writes `data/demo/aerial.png` plus parcels, GNSS and metadata files.

## 5. Honest labelling used in the project
- Image evidence = classical computer vision (colour segmentation, morphology, contours). No deep-learning model.
- The evidence score is prototype-level support, not legal accuracy.
- Priority ranking is a workflow aid, not an official survey priority.
- Demo parcels, image and GNSS are simulated.
