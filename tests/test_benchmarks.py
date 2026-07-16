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
