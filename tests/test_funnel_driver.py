"""
Smoke tests for the multi-fidelity funnel driver (scripts/funnel).

The driver is a script, not a package module; it is imported via a
path insertion so CI catches breakage of the library APIs it leans
on. The heavy relaxation levels need the optional backends and are
exercised on HPC, not here: these tests drive the geometric level,
the selection/checkpoint/restart machinery, and the rank-stability
report on synthetic records.
"""

import json
import os
import sys

import pytest

SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "scripts", "funnel")
FIXTURE_PATH = os.path.join(
    os.path.dirname(__file__), "data", "topologies_fixture.json"
)


def _load():
    sys.path.insert(0, SCRIPTS)
    try:
        return __import__("funnel")
    finally:
        sys.path.pop(0)


funnel = _load()


@pytest.fixture(scope="module")
def candidates(tmp_path_factory):
    """Two saved frameworks: MOF-5 (pcu) and a boroxine COF layer."""
    from autografs import Autografs

    mofgen = Autografs(topofile=FIXTURE_PATH)
    inputs = tmp_path_factory.mktemp("inputs")

    pcu = mofgen.topologies["pcu"]
    mappings = {}
    for key in pcu.mappings:
        conn = len(key.atoms.indices_from_symbol("X"))
        mappings[key] = {6: "Zn_mof5_octahedral", 2: "Benzene_linear"}[conn]
    mofgen.build(pcu, mappings=mappings).save(inputs / "mof5.json.gz")

    hcb = mofgen.topologies["hcb"]
    mappings = {}
    for key in hcb.mappings:
        conn = len(key.atoms.indices_from_symbol("X"))
        mappings[key] = {3: "Boroxine_triangle", 2: "Benzene_linear"}[conn]
    mofgen.build(hcb, mappings=mappings, max_rmsd=0.5).save(inputs / "cof.json.gz")
    return inputs


class TestSelectTop:
    RECORDS = {
        "a": {"void_fraction": 0.8},
        "b": {"void_fraction": 0.5},
        "c": {"void_fraction": 0.9},
        "broken": {"error": "boom"},
    }

    def test_descending_default(self):
        assert funnel.select_top(self.RECORDS, "void_fraction", 2) == ["c", "a"]

    def test_ascending(self):
        chosen = funnel.select_top(self.RECORDS, "void_fraction", 2, ascending=True)
        assert chosen == ["b", "a"]

    def test_missing_key_dropped(self):
        assert "broken" not in funnel.select_top(self.RECORDS, "void_fraction", 4)

    def test_ties_break_by_name(self):
        records = {"b": {"x": 1.0}, "a": {"x": 1.0}}
        assert funnel.select_top(records, "x", 1) == ["a"]


class TestRunFunnel:
    def test_geometric_level_and_provenance(self, candidates, tmp_path):
        workdir = tmp_path / "funnel"
        summary = funnel.run_funnel(
            [str(candidates / "*.json.gz")],
            workdir,
            levels=["asbuilt"],
            keep=[],
            rank_by="void_fraction",
            spacing=1.0,
        )
        assert summary["history"][0]["evaluated"] == ["cof", "mof5"]
        for name in ("mof5", "cof"):
            props = json.loads((workdir / name / "asbuilt.props.json").read_text())
            assert 0.0 < props["void_fraction"] < 1.0
            assert props["density"] > 0.0
            assert (workdir / name / "asbuilt.json.gz").exists()
        assert (workdir / "funnel.json").exists()

    def test_selection_between_levels(self, candidates, tmp_path):
        workdir = tmp_path / "funnel"
        summary = funnel.run_funnel(
            [str(candidates / "*.json.gz")],
            workdir,
            levels=["asbuilt", "asbuilt"],
            keep=[1],
            rank_by="void_fraction",
            spacing=1.0,
        )
        first = summary["history"][0]
        assert len(first["selected"]) == 1
        records = {
            name: json.loads((workdir / name / "asbuilt.props.json").read_text())
            for name in first["evaluated"]
        }
        best = max(records, key=lambda n: records[n]["void_fraction"])
        assert first["selected"] == [best]
        assert summary["history"][1]["evaluated"] == [best]

    def test_restart_skips_finished_work(self, candidates, tmp_path):
        workdir = tmp_path / "funnel"
        args = dict(levels=["asbuilt"], keep=[], rank_by="void_fraction", spacing=1.0)
        funnel.run_funnel([str(candidates / "*.json.gz")], workdir, **args)
        stamp = (workdir / "mof5" / "asbuilt.props.json").stat().st_mtime_ns
        funnel.run_funnel([str(candidates / "*.json.gz")], workdir, **args)
        assert (workdir / "mof5" / "asbuilt.props.json").stat().st_mtime_ns == stamp

    def test_failures_recorded_not_fatal(self, candidates, tmp_path):
        broken = tmp_path / "broken.json.gz"
        broken.write_bytes(b"not a gzip")
        workdir = tmp_path / "funnel"
        summary = funnel.run_funnel(
            [str(candidates / "mof5.json.gz"), str(broken)],
            workdir,
            levels=["asbuilt"],
            keep=[],
            rank_by="void_fraction",
            spacing=1.0,
        )
        assert set(summary["history"][0]["evaluated"]) == {"mof5", "broken"}
        props = json.loads((workdir / "broken" / "asbuilt.props.json").read_text())
        assert "error" in props

    def test_bad_level_and_keep_validation(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown level"):
            funnel.run_funnel(["x.json"], tmp_path, ["nope"], [], rank_by="density")
        with pytest.raises(ValueError, match="keep"):
            funnel.run_funnel(
                ["x.json"], tmp_path, ["asbuilt", "uff4mof"], [], rank_by="density"
            )


class TestRankStability:
    def test_spearman_between_levels(self, tmp_path):
        levels = ["asbuilt", "uff4mof"]
        low = {"a": 1.0, "b": 2.0, "c": 3.0}
        # monotone for void_fraction (rho +1), reversed for density (-1)
        for name, value in low.items():
            candidate = tmp_path / name
            candidate.mkdir()
            (candidate / "asbuilt.props.json").write_text(
                json.dumps({"void_fraction": value, "density": value})
            )
            (candidate / "uff4mof.props.json").write_text(
                json.dumps({"void_fraction": value * 2.0, "density": -value})
            )
        (tmp_path / "funnel.json").write_text(json.dumps({"levels": levels}))
        table = funnel.rank_stability(tmp_path)
        assert table["void_fraction"]["asbuilt->uff4mof"] == pytest.approx(1.0)
        assert table["density"]["asbuilt->uff4mof"] == pytest.approx(-1.0)

    def test_too_few_candidates_ignored(self, tmp_path):
        (tmp_path / "only").mkdir()
        (tmp_path / "only" / "asbuilt.props.json").write_text(json.dumps({"x": 1.0}))
        (tmp_path / "only" / "uff4mof.props.json").write_text(json.dumps({"x": 2.0}))
        (tmp_path / "funnel.json").write_text(
            json.dumps({"levels": ["asbuilt", "uff4mof"]})
        )
        assert funnel.rank_stability(tmp_path) == {}
