"""
Tests for partial-charge assignment (autografs.charges).

The EQeq implementation is validated against the reference C++ code
of Wilmer et al.: tests/data/IRMOF-1.cif is the example structure
shipped with it, and tests/data/eqeq_irmof1_reference.json holds the
charges the compiled reference binary prints for it with default
parameters (lambda=1.2, hI0=-2.0, ewald, mR=mK=2). The reference
rounds to 3 decimals and redistributes the rounding residue over a
few atoms, so agreement is asserted to 2e-3.
"""

import json
import os

import numpy as np
import pytest

from autografs.charges import (
    CHARGE_METHODS,
    _element_parameters,
    _eqeq_solve,
    assign_charges,
    register_charge_method,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
FIXTURE_PATH = os.path.join(DATA_DIR, "topologies_fixture.json")


@pytest.fixture(scope="module")
def mof5():
    from autografs import Autografs

    mofgen = Autografs(topofile=FIXTURE_PATH)
    topology = mofgen.topologies["pcu"]
    mappings = {}
    for key in topology.mappings:
        conn = len(key.atoms.indices_from_symbol("X"))
        mappings[key] = "Zn_mof5_octahedral" if conn == 6 else "Benzene_linear"
    return mofgen.build(topology, mappings=mappings)


class TestElementParameters:
    def test_hydrogen_uses_effective_affinity(self):
        chi, hardness = _element_parameters(["H"], hydrogen_affinity=-2.0)
        assert chi[0] == pytest.approx(0.5 * (13.598 - 2.0))
        assert hardness[0] == pytest.approx(13.598 + 2.0)

    def test_zinc_expanded_around_charge_center(self):
        # Zn is centered at +2: chi and J come from IP2/IP3, shifted
        # back to zero charge by -center * J
        chi, hardness = _element_parameters(["Zn"], hydrogen_affinity=-2.0)
        ip2, ip3 = 17.964, 39.722
        assert hardness[0] == pytest.approx(ip3 - ip2)
        assert chi[0] == pytest.approx(0.5 * (ip3 + ip2) - 2.0 * (ip3 - ip2))

    def test_carbon_centered_at_zero(self):
        chi, hardness = _element_parameters(["C"], hydrogen_affinity=-2.0)
        ea, ip1 = 1.26212, 11.26
        assert chi[0] == pytest.approx(0.5 * (ip1 + ea))
        assert hardness[0] == pytest.approx(ip1 - ea)

    def test_unknown_element_raises(self):
        with pytest.raises(ValueError, match="No EQeq ionization data"):
            _element_parameters(["Uuo"], hydrogen_affinity=-2.0)


def _read_irmof1() -> tuple[list[str], np.ndarray, np.ndarray]:
    """Parse the reference IRMOF-1.cif (its loops are too loose for
    pymatgen's CIF parser, but the format is perfectly regular)."""
    import re

    with open(os.path.join(DATA_DIR, "IRMOF-1.cif")) as handle:
        text = handle.read()
    lengths = [
        float(re.search(rf"_cell_length_{axis}\s+([\d.]+)", text).group(1))
        for axis in "abc"
    ]
    cell = np.diag(lengths)
    symbols, frac = [], []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 5 and re.fullmatch(r"[A-Za-z]{1,2}\d+", parts[0]):
            symbols.append(parts[1])
            frac.append([float(x) for x in parts[2:5]])
    return symbols, np.array(frac) @ cell, cell


class TestEqeqAgainstReference:
    def test_irmof1_matches_reference_binary(self):
        symbols, coords, cell = _read_irmof1()
        with open(os.path.join(DATA_DIR, "eqeq_irmof1_reference.json")) as handle:
            reference = np.array(json.load(handle))
        charges = _eqeq_solve(
            symbols,
            coords,
            cell,
            scaling=1.2,
            hydrogen_affinity=-2.0,
            eta=50.0,
            m_real=2,
            m_reciprocal=2,
            total_charge=0.0,
        )
        assert charges.shape == reference.shape
        assert charges.sum() == pytest.approx(0.0, abs=1e-9)
        # the reference rounds to 3 decimals and redistributes the
        # residue over a few atoms; 2e-3 absorbs exactly that
        np.testing.assert_allclose(charges, reference, atol=2e-3)


class TestAssignCharges:
    def test_mof5_eqeq_end_to_end(self, mof5):
        charged = mof5.assign_charges("eqeq")
        charges = charged.charges
        assert charges is not None
        assert charges.shape == (len(mof5),)
        # neutral cell, exactly
        assert charges.sum() == pytest.approx(0.0, abs=1e-9)
        # built MOF-5 is IRMOF-1: Zn should land near the reference
        # +1.21, the acidic carboxylate carbon positive, oxygens negative
        symbols = np.array(charged.symbols)
        zinc = charges[symbols == "Zn"]
        np.testing.assert_allclose(zinc, 1.21, atol=0.15)
        assert charges[symbols == "O"].max() < 0.0
        # the input framework is untouched
        assert mof5.charges is None
        assert charged.graph.graph["charge_method"] == "eqeq"

    def test_unknown_method_raises(self, mof5):
        with pytest.raises(ValueError, match="Unknown charge method"):
            assign_charges(mof5, method="no-such-scheme")

    def test_custom_method_registers(self, mof5):
        name = "test-zeros"

        @register_charge_method(name)
        def zeros(framework):
            return np.zeros(len(framework))

        try:
            charged = mof5.assign_charges(name)
            assert charged.charges is not None
            np.testing.assert_allclose(charged.charges, 0.0)
        finally:
            del CHARGE_METHODS[name]

    def test_wrong_length_raises(self, mof5):
        name = "test-short"
        CHARGE_METHODS[name] = lambda framework: np.zeros(3)
        try:
            with pytest.raises(ValueError, match="returned"):
                assign_charges(mof5, method=name)
        finally:
            del CHARGE_METHODS[name]


class TestChargePersistence:
    @pytest.fixture(scope="class")
    def charged(self, mof5):
        return mof5.assign_charges("eqeq")

    def test_save_load_roundtrip(self, charged, tmp_path):
        from autografs.framework import Framework

        path = charged.save(tmp_path / "charged.json.gz")
        loaded = Framework.load(path)
        np.testing.assert_allclose(loaded.charges, charged.charges)
        assert loaded.graph.graph["charge_method"] == "eqeq"

    def test_uncharged_roundtrip_stays_uncharged(self, mof5, tmp_path):
        from autografs.framework import Framework

        path = mof5.save(tmp_path / "plain.json.gz")
        assert Framework.load(path).charges is None

    def test_charges_survive_editing(self, charged):
        doubled = charged.supercell((2, 1, 1))
        assert doubled.charges is not None
        assert doubled.charges.shape == (2 * len(charged),)
        # replicas carry the original atom's charge
        np.testing.assert_allclose(
            np.sort(doubled.charges)[::2], np.sort(charged.charges)
        )


class TestChargeExports:
    @pytest.fixture(scope="class")
    def charged(self, mof5):
        return mof5.assign_charges("eqeq")

    def test_cif_gains_charge_column(self, charged, tmp_path):
        path = charged.write_cif(tmp_path / "charged.cif")
        text = path.read_text()
        assert "_atom_site_charge" in text
        # every site row carries a value; spot-check the count by
        # re-reading the loop
        lines = text.splitlines()
        start = lines.index("  _atom_site_charge") + 1
        rows = [ln for ln in lines[start:] if ln.strip()]
        assert len(rows) == len(charged)
        # pymatgen can still read the file
        from pymatgen.core import Structure

        assert len(Structure.from_file(path)) == len(charged)

    def test_uncharged_cif_unchanged(self, mof5, tmp_path):
        path = mof5.write_cif(tmp_path / "plain.cif")
        assert "_atom_site_charge" not in path.read_text()

    def test_ase_initial_charges(self, charged):
        atoms = charged.to_ase()
        np.testing.assert_allclose(atoms.get_initial_charges(), charged.charges)

    def test_structure_site_property(self, charged):
        assert "charge" in charged.structure.site_properties
        np.testing.assert_allclose(
            charged.structure.site_properties["charge"], charged.charges
        )

    def test_gulp_charges_and_electrostatics(self, charged, mof5):
        gulp = charged.to_gulp(write_to_file=False)
        assert "noelectrostatics" not in gulp.splitlines()[0]
        plain = mof5.to_gulp(write_to_file=False)
        assert "noelectrostatics" in plain.splitlines()[0]
        core_lines = [ln for ln in gulp.splitlines() if " core " in ln]
        assert len(core_lines[0].split()) == 6
