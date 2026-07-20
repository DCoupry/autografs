"""
Smoke test for the rod-MOF scanner (scripts/rods/scan_rod_mofs.py).

The scanner is a research script, not a package module; it is imported
via a path insertion so CI catches breakage of the library APIs it
leans on. It runs over a tiny directory of synthetic CIFs (the pillar
and helical rod fixtures written out), so the whole scan/report/filter
path is exercised without any external data.
"""

import json
import os
import sys

import pytest
from pymatgen.io.cif import CifWriter

from .test_deconstruct import _helical_rod_structure, _rod_pillar_structure

SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "scripts", "rods")


def _load():
    sys.path.insert(0, SCRIPTS)
    try:
        return __import__("scan_rod_mofs")
    finally:
        sys.path.pop(0)


scan_rod_mofs = _load()


@pytest.fixture(scope="module")
def cif_dir(tmp_path_factory):
    """A directory holding the pillar (straight) and helical rod CIFs
    plus one non-rod molecular crystal."""
    directory = tmp_path_factory.mktemp("rod_cifs")
    CifWriter(_rod_pillar_structure()).write_file(directory / "pillar.cif")
    CifWriter(_helical_rod_structure()).write_file(directory / "helix.cif")
    return str(directory)


@pytest.fixture(scope="module")
def scanned(cif_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("scan") / "rods.jsonl"
    n_rod = scan_rod_mofs.scan_directory(cif_dir, str(out), n_jobs=1)
    return out, n_rod


class TestScan:
    def test_finds_both_rods(self, scanned):
        out, n_rod = scanned
        assert n_rod == 2
        records = [json.loads(line) for line in open(out) if line.strip()]
        by_file = {r["file"]: r for r in records}
        assert by_file["pillar.cif"]["n_rods"] == 1
        assert by_file["helix.cif"]["n_rods"] == 1

    def test_records_screw(self, scanned):
        out, _ = scanned
        records = {
            json.loads(line)["file"]: json.loads(line)
            for line in open(out)
            if line.strip()
        }
        pillar_rod = records["pillar.cif"]["rods"][0]
        helix_rod = records["helix.cif"]["rods"][0]
        assert pillar_rod["straight"] is True
        assert pillar_rod["screw_angle"] == pytest.approx(0.0, abs=1.0)
        assert helix_rod["straight"] is False
        assert abs(helix_rod["screw_angle"]) == pytest.approx(180.0, abs=1.0)


class TestPalette:
    def test_filter_off_by_default(self):
        assert scan_rod_mofs._resolve_palette(None) is None

    def test_common_palette(self):
        palette = scan_rod_mofs._resolve_palette("common")
        assert "Zn" in palette and "C" in palette
        assert "Gd" not in palette  # f-block excluded

    def test_explicit_palette(self):
        palette = scan_rod_mofs._resolve_palette("H C N O Zn")
        assert palette == {"H", "C", "N", "O", "Zn"}

    def test_report_default_reports_all(self, scanned, capsys):
        out, _ = scanned
        scan_rod_mofs.report(str(out), palette=None)
        printed = capsys.readouterr().out
        assert "all elements" in printed
        assert "2 with rods" in printed

    def test_report_palette_excludes(self, scanned, capsys):
        # a palette without Zn drops both -Zn-O- rods
        out, _ = scanned
        scan_rod_mofs.report(str(out), palette={"C", "H", "N", "O"})
        printed = capsys.readouterr().out
        assert "0 reported" in printed
        assert "Zn" in printed  # named among the excluded elements


class TestCli:
    def test_scan_then_report(self, cif_dir, tmp_path, capsys):
        out = tmp_path / "cli.jsonl"
        assert scan_rod_mofs.main(["scan", cif_dir, "-o", str(out)]) == 0
        assert scan_rod_mofs.main(["report", str(out)]) == 0
        assert "all elements" in capsys.readouterr().out
