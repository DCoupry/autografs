"""Smoke tests for the benchmark drivers in scripts/benchmarks.

The drivers are scripts, not package modules; they are imported here
via a path insertion so CI catches breakage of the library APIs they
lean on (deconstruct outcomes, fragment compatibility, verify_net).
"""

import os
import sys

import pytest

SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "scripts", "benchmarks")
FIXTURE_PATH = os.path.join(
    os.path.dirname(__file__), "data", "topologies_fixture.json"
)


@pytest.fixture(scope="module")
def mofgen():
    from autografs import Autografs

    return Autografs(topofile=FIXTURE_PATH)


@pytest.fixture(scope="module")
def roundtrip():
    sys.path.insert(0, SCRIPTS)
    try:
        import roundtrip as module
    finally:
        sys.path.pop(0)
    return module


@pytest.fixture(scope="module")
def embedding():
    sys.path.insert(0, SCRIPTS)
    try:
        import embedding as module
    finally:
        sys.path.pop(0)
    return module


@pytest.fixture(scope="module")
def mof5(mofgen):
    topology = mofgen.topologies["pcu"]
    mappings = {}
    for key in topology.mappings:
        conn = len(key.atoms.indices_from_symbol("X"))
        mappings[key] = {6: "Zn_mof5_octahedral", 2: "Benzene_linear"}[conn]
    return mofgen.build(topology, mappings=mappings, max_rmsd=0.5)


def test_roundtrip_closes_on_a_built_framework(mofgen, roundtrip, tmp_path):
    """build -> CIF -> deconstruct -> rebuild -> verify_net closes."""
    topology = mofgen.topologies["pcu"]
    mappings = {}
    for key in topology.mappings:
        conn = len(key.atoms.indices_from_symbol("X"))
        mappings[key] = {6: "Zn_mof5_octahedral", 2: "Benzene_linear"}[conn]
    mof = mofgen.build(topology, mappings=mappings, max_rmsd=0.5)
    mof.write_cif(tmp_path / "mof5.cif")

    payload = roundtrip.run([tmp_path / "mof5.cif"], mofgen, max_rmsd=0.5)

    assert payload["outcomes"] == {"closed": 1}
    record = payload["structures"]["mof5.cif"]
    assert record["net"] == ["pcu"]
    assert record["rebuilt_net"] == "pcu"


def test_roundtrip_gates_on_composition(mofgen, roundtrip, tmp_path):
    """#180: verify_net alone is not evidence the right material was
    built. A mixed-linker pcu framework deconstructs to two ditopic
    fragments, but candidate_mappings assigns one fragment per slot
    type - every rebuild is topologically pcu yet chemically wrong,
    and the composition gate must say so instead of scoring 'closed'."""
    topology = mofgen.topologies["pcu"]
    mappings = {}
    for key in topology.mappings:
        conn = len(key.atoms.indices_from_symbol("X"))
        mappings[key] = {6: "Zn_mof5_octahedral", 2: "Benzene_linear"}[conn]
    # one edge overridden: a 2:1 linker mix. Phenazine spans the same
    # 4.2 A as benzene, so the cubic cell serves both bonds and the
    # build is chemically clean - only the composition differs.
    mappings[3] = "Phenazine_linear"
    mixed = mofgen.build(topology, mappings=mappings, max_rmsd=0.5)
    mixed.write_cif(tmp_path / "mixed.cif")

    payload = roundtrip.run([tmp_path / "mixed.cif"], mofgen, max_rmsd=0.5)

    assert payload["outcomes"] == {"closed_wrong_composition": 1}


def test_roundtrip_reports_failures_as_data(mofgen, roundtrip, tmp_path):
    """A non-framework input lands in the failure taxonomy, not a raise."""
    from pymatgen.core.lattice import Lattice
    from pymatgen.core.structure import Structure

    Structure(Lattice.cubic(20.0), ["He"], [[0.5, 0.5, 0.5]]).to(
        filename=str(tmp_path / "guest.cif")
    )
    payload = roundtrip.run([tmp_path / "guest.cif"], mofgen, max_rmsd=0.5)
    assert payload["outcomes"] == {"deconstruction_failed": 1}
    assert "periodic" in payload["structures"]["guest.cif"]["error"]


def test_embedding_of_an_idealized_build_is_exact(mofgen, embedding, mof5, tmp_path):
    """A framework built ON the blueprint has, by construction, the
    blueprint's proportions - so both errors must read zero. This is
    the metric's calibration: anything else means the reduced lengths
    are not measuring what they claim to."""
    mof5.write_cif(tmp_path / "mof5.cif")

    payload = embedding.run([tmp_path / "mof5.cif"], mofgen, verbose=False)

    assert payload["outcomes"] == {"measured": 1}
    record = payload["structures"]["mof5.cif"]
    assert record["best_net"] == "pcu"
    assert record["size_error"] == pytest.approx(0.0, abs=1e-6)
    assert record["shape_error"] == pytest.approx(0.0, abs=1e-6)


def test_embedding_is_supercell_invariant(mofgen, embedding, mof5, tmp_path):
    """The reduced length divides by (volume per vertex), so a cell
    multiple leaves it alone - which is what lets a real crystal in an
    unrelated setting be compared against a blueprint at all."""
    mof5.supercell((2, 1, 1)).write_cif(tmp_path / "super.cif")

    payload = embedding.run([tmp_path / "super.cif"], mofgen, verbose=False)

    record = payload["structures"]["super.cif"]
    assert record["outcome"] == "measured"
    assert record["size_error"] == pytest.approx(0.0, abs=1e-6)
    entry = record["nets"][record["best_net"]]
    # twice the edges on the real side, same reduced length
    assert entry["n_edges_real"] == 2 * entry["n_edges_ideal"]
    assert entry["mean_lambda_real"] == pytest.approx(entry["mean_lambda_ideal"])


def test_embedding_predicts_the_rebuilt_cell(mofgen, embedding, mof5, tmp_path):
    """size_ratio ** 3 is a prediction of the built-to-real volume
    ratio; --rebuild is what checks it."""
    mof5.write_cif(tmp_path / "mof5.cif")

    payload = embedding.run(
        [tmp_path / "mof5.cif"], mofgen, rebuild=True, verbose=False
    )

    entry = payload["structures"]["mof5.cif"]["nets"]["pcu"]
    assert entry["rebuild"] is not None
    assert entry["rebuild"]["volume_ratio"] == pytest.approx(
        entry["predicted_volume_ratio"], abs=0.05
    )


def test_embedding_reports_failures_as_data(mofgen, embedding, tmp_path):
    """A non-framework input lands in the taxonomy, not a raise."""
    from pymatgen.core.lattice import Lattice
    from pymatgen.core.structure import Structure

    Structure(Lattice.cubic(20.0), ["He"], [[0.5, 0.5, 0.5]]).to(
        filename=str(tmp_path / "guest.cif")
    )
    payload = embedding.run([tmp_path / "guest.cif"], mofgen, verbose=False)
    assert payload["outcomes"] == {"deconstruction_failed": 1}
    assert payload["summary"]["n_measured"] == 0


def test_embedding_scores_only_what_the_pipeline_would_build(
    mofgen, embedding, mof5, tmp_path
):
    """A structure whose units no slot accepts is measured but kept out
    of the buildable population: its gap is a compatibility failure,
    not an embedding one."""
    mof5.write_cif(tmp_path / "mof5.cif")
    payload = embedding.run([tmp_path / "mof5.cif"], mofgen, verbose=False)

    record = payload["structures"]["mof5.cif"]
    assert record["buildable"] is True
    assert payload["summary"]["n_buildable"] == 1
    assert payload["summary"]["buildable"]["size_error"]["max"] == pytest.approx(
        0.0, abs=1e-6
    )
