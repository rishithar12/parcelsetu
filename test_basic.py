import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
from src import pipeline as P


@pytest.fixture(scope="module")
def D(tmp_path_factory):
    d = P.load_demo(str(tmp_path_factory.mktemp("demo")))
    P.run_all(d)
    return d


def types(D):
    return set(D["conf"][D["conf"].status == "Open"].type)


def test_demo_size(D):
    assert 15 <= len(D["parcels"]) <= 25


def test_expected_conflicts(D):
    t = types(D)
    for need in ("Parcel overlap", "Parcel gap", "Building crossing boundary", "Boundary mismatch", "GNSS conflict"):
        assert need in t, f"{need} not detected: {t}"


def test_scores_in_range(D):
    assert D["bnd"].score.between(0, 100).all()
    assert set(D["bnd"].level) <= {"High", "Medium", "Low"}


def test_shared_edges(D):
    assert D["bnd"].shared.sum() > 0 and D["prio"].priority_level.isin([1, 2, 3]).all()


def test_snap_resolves_overlap(D):
    ov = D["conf"][D["conf"].type == "Parcel overlap"].iloc[0]
    P.edit_geometry(D, ov.boundary_id)
    assert "Parcel overlap" not in types(D)


def test_verify_closes_evidence_conflicts(D):
    row = D["conf"][(D["conf"].kind == "evidence") & (D["conf"].status == "Open")].iloc[0]
    for p in row.parcel_ids:
        D["status"][p] = "Verified"
    P.run_priority(D)
    assert (D["conf"].loc[D["conf"].conflict_id == row.conflict_id, "status"] != "Open").all()
