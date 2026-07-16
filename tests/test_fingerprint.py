"""Tests for assembly fingerprints (autografs.fingerprint)."""

import dataclasses
import os

import pytest

from autografs import fingerprint
from autografs.deconstruct import match_fragment

FIXTURE_PATH = os.path.join(
    os.path.dirname(__file__), "data", "topologies_fixture.json"
)


@pytest.fixture(scope="module")
def mofgen():
    from autografs import Autografs

    return Autografs(topofile=FIXTURE_PATH)


@pytest.fixture(scope="module")
def mof5(mofgen):
    topology = mofgen.topologies["pcu"]
    mappings = {}
    for key in topology.mappings:
        conn = len(key.atoms.indices_from_symbol("X"))
        mappings[key] = {6: "Zn_mof5_octahedral", 2: "Benzene_linear"}[conn]
    return mofgen.build(topology, mappings=mappings, max_rmsd=0.5)


@pytest.fixture(scope="module")
def deconstructed(mofgen, mof5):
    return mofgen.deconstruct(mof5.structure)


class TestMatchFragment:
    def test_matches_identical_fragment(self, deconstructed):
        linker = deconstructed.fragments["linker_C6H4_2X"]
        name = match_fragment(deconstructed.fragments, linker.copy(), "linker_C6H4_2X")
        assert name == "linker_C6H4_2X"

    def test_read_only_and_none_on_miss(self, deconstructed):
        linker = deconstructed.fragments["linker_C6H4_2X"]
        empty: dict = {}
        assert match_fragment(empty, linker, "linker_C6H4_2X") is None
        assert empty == {}


class TestAssemblyFingerprint:
    def test_built_from_harvest_equals_experimental(self, mofgen, deconstructed):
        """The core property: a framework built from harvested blocks
        fingerprints equal to the experimental structure expressed in
        that harvest's vocabulary - 'realized' becomes a set lookup."""
        experimental = fingerprint.from_deconstruction(
            deconstructed, library=deconstructed.fragments
        )
        topology = mofgen.topologies["pcu"]
        mappings = {}
        for key in topology.mappings:
            conn = len(key.atoms.indices_from_symbol("X"))
            name = {6: "node_C6O13Zn4_6X", 2: "linker_C6H4_2X"}[conn]
            mappings[key] = deconstructed.fragments[name]
        rebuilt = mofgen.build(topology, mappings=mappings, max_rmsd=0.5)
        hypothetical = fingerprint.from_framework(rebuilt)

        assert experimental == hypothetical
        assert hash(experimental) == hash(hypothetical)
        assert experimental.is_buildable_vocabulary
        assert experimental in {hypothetical}  # the intended usage

    def test_supercells_fingerprint_identically(self, mofgen, mof5, deconstructed):
        doubled = mofgen.deconstruct(mof5.supercell((2, 1, 1)).structure)
        assert fingerprint.from_deconstruction(
            doubled
        ) == fingerprint.from_deconstruction(deconstructed)

    def test_unmatched_blocks_never_collide(self, deconstructed):
        foreign = fingerprint.from_deconstruction(deconstructed, library={})
        assert not foreign.is_buildable_vocabulary
        assert all(name.startswith("unmatched:") for name, _ in foreign.blocks)
        assert foreign != fingerprint.from_deconstruction(deconstructed)

    def test_str_is_readable(self, deconstructed):
        text = str(fingerprint.from_deconstruction(deconstructed))
        assert "pcu" in text
        assert "node_C6O13Zn4_6X" in text

    def test_rod_units_are_rejected(self, deconstructed):
        result = dataclasses.replace(deconstructed)
        result.rod_units = [object()]  # what the rod branch reports
        with pytest.raises(ValueError, match="[Rr]od"):
            fingerprint.from_deconstruction(result)
