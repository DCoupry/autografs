"""
Tests for the local-fetch machinery and the --use_iza pipeline.

Everything runs offline: HTTP is faked at the requests.Session level,
the IZA code list is shrunk via monkeypatching, and conversion runs on
the bundled tests/data/IZA-SOD.cif (the sodalite framework, 12 T + 24
O per cell — the smallest classic zeolite). The license gate's
interactive path is covered through the captured-stdin (non-tty)
branch.
"""

import os

import pytest

import autografs.fetch as fetch
from autografs.data.iza_codes import IZA_CODES, cif_filename

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
FIXTURE_PATH = os.path.join(DATA_DIR, "topologies_fixture.json")


class TestIzaCodes:
    def test_code_count(self):
        assert len(IZA_CODES) == 274
        assert len(set(IZA_CODES)) == 274

    def test_filename_strips_prefixes(self):
        assert cif_filename("FAU") == "FAU.cif"
        assert cif_filename("-CLO") == "CLO.cif"
        assert cif_filename("*BEA") == "BEA.cif"
        assert cif_filename("*-ITN") == "ITN.cif"


class _FakeResponse:
    def __init__(self, content=b"data", status=200):
        self.content = content
        self.status = status

    def raise_for_status(self):
        if self.status != 200:
            raise fetch.requests.HTTPError(f"status {self.status}")


class _FakeSession:
    """Stands in for requests.Session; records requested URLs."""

    calls: list[str] = []
    responses: dict[str, _FakeResponse] = {}

    def __init__(self):
        self.headers = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get(self, url, timeout=None):
        _FakeSession.calls.append(url)
        return _FakeSession.responses.get(url, _FakeResponse())


@pytest.fixture
def fake_http(monkeypatch):
    _FakeSession.calls = []
    _FakeSession.responses = {}
    monkeypatch.setattr(fetch.requests, "Session", _FakeSession)
    monkeypatch.setattr(fetch, "REQUEST_DELAY", 0.0)
    return _FakeSession


class TestFetchFiles:
    def test_downloads_and_caches(self, fake_http, tmp_path):
        urls = {"a.cif": "http://x/a.cif", "b.cif": "http://x/b.cif"}
        got = fetch.fetch_files(urls, tmp_path, delay=0.0)
        assert sorted(got) == ["a.cif", "b.cif"]
        assert (tmp_path / "a.cif").read_bytes() == b"data"
        assert len(fake_http.calls) == 2
        # second run: everything cached, no network at all
        fake_http.calls.clear()
        got = fetch.fetch_files(urls, tmp_path, delay=0.0)
        assert sorted(got) == ["a.cif", "b.cif"]
        assert fake_http.calls == []

    def test_resumes_partial_cache(self, fake_http, tmp_path):
        (tmp_path / "a.cif").write_bytes(b"already here")
        urls = {"a.cif": "http://x/a.cif", "b.cif": "http://x/b.cif"}
        fetch.fetch_files(urls, tmp_path, delay=0.0)
        assert fake_http.calls == ["http://x/b.cif"]
        # the cached file is untouched
        assert (tmp_path / "a.cif").read_bytes() == b"already here"

    def test_failures_skipped_not_raised(self, fake_http, tmp_path):
        fake_http.responses["http://x/bad.cif"] = _FakeResponse(status=404)
        urls = {"ok.cif": "http://x/ok.cif", "bad.cif": "http://x/bad.cif"}
        got = fetch.fetch_files(urls, tmp_path, delay=0.0)
        assert sorted(got) == ["ok.cif"]
        assert not (tmp_path / "bad.cif").exists()

    def test_no_leftover_tmp_files(self, fake_http, tmp_path):
        fetch.fetch_files({"a.cif": "http://x/a.cif"}, tmp_path, delay=0.0)
        assert list(tmp_path.glob("*.tmp")) == []


class TestLicenseGate:
    def test_accept_flag_passes(self, capsys):
        fetch.require_acceptance(fetch.IZA_SOURCE, accept=True)
        shown = capsys.readouterr().out
        assert "IZA-SC" in shown
        # Check against the source object's own homepage field (not a bare
        # hostname literal) so this isn't mistaken for URL-host sanitization
        # by static analysis - it's just confirming the printed notice
        # includes the source's homepage.
        assert fetch.IZA_SOURCE.homepage in shown

    def test_non_interactive_without_flag_exits(self):
        # pytest replaces stdin with a non-tty object
        with pytest.raises(SystemExit, match="accept-licenses"):
            fetch.require_acceptance(fetch.IZA_SOURCE, accept=False)


class TestFetchIzaCifs:
    def test_cached_run_touches_no_network(self, monkeypatch, tmp_path):
        monkeypatch.setattr(fetch, "IZA_CODES", ("SOD", "-CLO"))
        for code in ("SOD", "-CLO"):
            (tmp_path / cif_filename(code)).write_bytes(b"cif data")

        def boom(*args, **kwargs):
            raise AssertionError("network touched despite full cache")

        monkeypatch.setattr(fetch.requests, "Session", boom)
        got = fetch.fetch_iza_cifs(cache_dir=tmp_path, accept_licenses=True)
        # keys are the official codes, prefixes preserved
        assert sorted(got) == ["-CLO", "SOD"]
        assert got["-CLO"].name == "CLO.cif"


class TestIzaConversion:
    def test_sod_converts(self):
        from pathlib import Path

        from autografs.extract_topology import topologies_from_tetrahedral_cifs

        out = topologies_from_tetrahedral_cifs({"SOD": Path(DATA_DIR) / "IZA-SOD.cif"})
        topo = out["SOD"]
        # sodalite: 12 T slots + 24 bridging O edge centers, Im-3m
        assert len(topo) == 36
        assert topo.spacegroup_number == 229
        connectivities = sorted(
            len(slot.atoms.indices_from_symbol("X")) for slot in topo.slots
        )
        assert connectivities.count(4) == 12
        assert connectivities.count(2) == 24

    def test_sod_identifies_against_itself(self):
        # the converted entry is a well-formed topology: its own
        # quotient graph identifies it on the exact tier
        from pathlib import Path

        from autografs.extract_topology import topologies_from_tetrahedral_cifs
        from autografs.net import identify_net, topology_quotient_edges

        topo = topologies_from_tetrahedral_cifs(
            {"SOD": Path(DATA_DIR) / "IZA-SOD.cif"}
        )["SOD"]
        matches = identify_net(topology_quotient_edges(topo), {"SOD": topo})
        assert "SOD" in matches
        assert matches.tier == "exact"

    def test_bad_cif_skipped(self, tmp_path):
        from pathlib import Path

        from autografs.extract_topology import topologies_from_tetrahedral_cifs

        bad = tmp_path / "bad.cif"
        bad.write_text("this is not a cif")
        out = topologies_from_tetrahedral_cifs(
            {"BAD": bad, "SOD": Path(DATA_DIR) / "IZA-SOD.cif"}
        )
        assert sorted(out) == ["SOD"]


class TestAliasShadowing:
    def test_real_entry_wins_over_alias(self):
        from autografs.topology_io import load_topologies

        library = load_topologies(FIXTURE_PATH)
        # an alias that would shadow a real entry is dropped, and
        # resolution keeps returning the real entry
        attached = library.attach_aliases({"pcu": "dia"})
        assert attached == 0
        assert library["pcu"].name == "pcu"


class TestCliWiring:
    def test_use_iza_end_to_end_offline(self, monkeypatch, tmp_path):
        import shutil

        from autografs.cgd import main
        from autografs.topology_io import load_topologies

        monkeypatch.setattr(fetch, "IZA_CODES", ("SOD",))
        cache = tmp_path / "cache"
        cache.mkdir()
        shutil.copy(os.path.join(DATA_DIR, "IZA-SOD.cif"), cache / "SOD.cif")

        def boom(*args, **kwargs):
            raise AssertionError("network touched despite full cache")

        monkeypatch.setattr(fetch.requests, "Session", boom)
        out = tmp_path / "zeolites.json.gz"
        main(
            [
                "--use_iza",
                "--accept-licenses",
                "--cache-dir",
                str(cache),
                "-o",
                str(out),
            ]
        )
        library = load_topologies(out)
        assert "SOD" in library
        assert len(library["SOD"]) == 36
