"""Tests for batch SBU harvesting (autografs.harvest)."""

import os

import pytest

from autografs.harvest import harvest

FIXTURE_PATH = os.path.join(
    os.path.dirname(__file__), "data", "topologies_fixture.json"
)


@pytest.fixture(scope="module")
def mofgen():
    from autografs import Autografs

    return Autografs(topofile=FIXTURE_PATH)


def build(mofgen, topo_name, choices, **kwargs):
    topology = mofgen.topologies[topo_name]
    mappings = {
        key: choices[len(key.atoms.indices_from_symbol("X"))]
        for key in topology.mappings
    }
    return mofgen.build(topology, mappings=mappings, max_rmsd=0.5, **kwargs)


@pytest.fixture(scope="module")
def mof5(mofgen):
    return build(
        mofgen, "pcu", {6: "Zn_mof5_octahedral", 2: "Benzene_linear"}, refine_cell=True
    )


@pytest.fixture(scope="module")
def paddlewheel(mofgen):
    return build(mofgen, "sql", {4: "Zn_square_paddlewheel", 2: "Benzene_linear"})


@pytest.fixture(scope="module")
def cif_dir(tmp_path_factory, mof5, paddlewheel):
    directory = tmp_path_factory.mktemp("cifs")
    mof5.write_cif(directory / "mof5.cif")
    paddlewheel.write_cif(directory / "paddlewheel.cif")
    return directory


class TestHarvestDirectory:
    def test_processes_every_cif(self, mofgen, cif_dir):
        result = mofgen.harvest(cif_dir)
        assert result.n_processed == 2
        assert result.failures == {}

    def test_shared_fragment_deduplicated_with_provenance(self, mofgen, cif_dir):
        result = mofgen.harvest(cif_dir)
        # the benzene linker is in both structures: one fragment, two sources
        assert "linker_C6H4_2X" in result.fragments
        assert result.provenance["linker_C6H4_2X"] == ["mof5", "paddlewheel"]

    def test_distinct_nodes_kept_separate(self, mofgen, cif_dir):
        result = mofgen.harvest(cif_dir)
        nodes = [n for n, k in result.kinds.items() if k == "node"]
        assert set(nodes) == {"node_C6O13Zn4_6X", "node_C4O8Zn2_4X"}

    def test_per_source_nets(self, mofgen, cif_dir):
        result = mofgen.harvest(cif_dir)
        assert result.nets == {"mof5": ["pcu"], "paddlewheel": ["sql"]}

    def test_building_units_excludes_caps(self, mofgen, cif_dir):
        result = mofgen.harvest(cif_dir)
        assert set(result.building_units) == set(result.fragments)
        assert result.caps == {}


class TestHarvestRoundTrip:
    def test_write_xyz_is_library_loadable(self, mofgen, cif_dir, tmp_path):
        from autografs.utils import xyz_to_sbu

        result = mofgen.harvest(cif_dir)
        path = result.write_xyz(tmp_path / "harvested.xyz")
        loaded = xyz_to_sbu(str(path))
        # default write_xyz emits nodes + linkers
        assert set(loaded) == set(result.building_units)

    def test_write_xyz_kinds_filter(self, mofgen, cif_dir, tmp_path):
        from autografs.utils import xyz_to_sbu

        result = mofgen.harvest(cif_dir)
        path = result.write_xyz(tmp_path / "nodes.xyz", kinds=("node",))
        loaded = xyz_to_sbu(str(path))
        assert all(name.startswith("node_") for name in loaded)

    def test_harvested_library_rebuilds(self, mofgen, cif_dir, tmp_path):
        """A harvested library must build a framework end to end."""
        from autografs import Autografs

        result = mofgen.harvest(cif_dir)
        path = result.write_xyz(tmp_path / "harvested.xyz")
        rebuilt_gen = Autografs(topofile=FIXTURE_PATH, xyzfile=str(path))
        topology = rebuilt_gen.topologies["pcu"]
        mappings = {
            key: {6: "node_C6O13Zn4_6X", 2: "linker_C6H4_2X"}[
                len(key.atoms.indices_from_symbol("X"))
            ]
            for key in topology.mappings
        }
        framework = rebuilt_gen.build(
            topology, mappings=mappings, refine_cell=True, max_rmsd=0.5
        )
        framework.verify_net(topology)


class TestHarvestInputs:
    def test_iterable_of_structures(self, mofgen, mof5, paddlewheel):
        result = mofgen.harvest([mof5.structure, paddlewheel.structure])
        assert result.n_processed == 2
        assert set(result.nets) == {"structure_0", "structure_1"}

    def test_single_structure(self, mofgen, mof5):
        result = mofgen.harvest(mof5.structure)
        assert result.n_processed == 1
        assert "node_C6O13Zn4_6X" in result.fragments


class TestIterSources:
    """_iter_sources is pure path plumbing; no deconstruction runs."""

    @pytest.fixture()
    def cif_files(self, tmp_path):
        for stem in ("alpha", "beta"):
            (tmp_path / f"{stem}.cif").write_text("placeholder\n")
        (tmp_path / "notes.txt").write_text("ignored\n")
        return tmp_path

    def test_absolute_glob_pattern(self, cif_files):
        from autografs.harvest import _iter_sources

        pairs = _iter_sources(str(cif_files / "*.cif"))
        assert [label for label, _ in pairs] == ["alpha", "beta"]

    def test_relative_glob_pattern(self, cif_files, monkeypatch):
        from autografs.harvest import _iter_sources

        monkeypatch.chdir(cif_files)
        pairs = _iter_sources("*.cif")
        assert [label for label, _ in pairs] == ["alpha", "beta"]

    def test_glob_matching_nothing_is_empty(self, cif_files):
        from autografs.harvest import _iter_sources

        assert _iter_sources(str(cif_files / "*.xyz")) == []


class TestHarvestRobustness:
    def test_unreadable_file_recorded_not_raised(self, mofgen, mof5, tmp_path):
        mof5.write_cif(tmp_path / "good.cif")
        (tmp_path / "bad.cif").write_text("not a valid cif at all\n")
        result = mofgen.harvest(tmp_path)
        assert result.n_processed == 1
        assert "bad" in result.failures

    def test_metal_free_cof_is_harvested(self, mofgen, tmp_path):
        """A COF is metal-free but now deconstructs via branch points."""
        cof = build(mofgen, "hcb", {3: "Boroxine_triangle", 2: "Benzene_linear"})
        cof.write_cif(tmp_path / "cof.cif")
        result = mofgen.harvest(tmp_path)
        assert result.n_processed == 1
        assert "node_B3O3_3X" in result.building_units
        assert result.nets["cof"] == ["hcb"]

    def test_molecular_crystal_recorded_as_failure(self, mofgen, tmp_path):
        from pymatgen.core.lattice import Lattice
        from pymatgen.core.structure import Structure

        guest = Structure(Lattice.cubic(20.0), ["He"], [[0.5, 0.5, 0.5]])
        guest.to(filename=str(tmp_path / "guest.cif"))
        result = mofgen.harvest(tmp_path)
        assert result.n_processed == 0
        assert "DeconstructionError" in result.failures["guest"]

    def test_report_mentions_failures(self, mofgen, tmp_path):
        (tmp_path / "junk.cif").write_text("garbage\n")
        result = mofgen.harvest(tmp_path)
        assert "1 failed" in result.report()


def test_harvest_without_topologies_skips_nets(mof5):
    result = harvest(mof5.structure)
    assert result.nets == {}
    assert result.n_processed == 1
