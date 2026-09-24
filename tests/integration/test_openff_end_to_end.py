"""End-to-end integration test for the OpenFF (SMIRNOFF) force field path.

Runs the full three-stage pipeline including moltemplate on a small PEO melt
plus a water molecule. Skipped when the optional openff-toolkit is missing.
"""

import json
from pathlib import Path

import pytest

from AutoPoly import Molecule, Polymer, System, generate

openff = pytest.importorskip("openff.toolkit", reason="openff-toolkit not installed")

PEO_SEQUENCE = ["CCO[*]", "[*]CCO[*]", "[*]CCO"]


def _data_counts(data_text):
    counts = {}
    for line in data_text.splitlines()[2:20]:
        parts = line.split()
        if len(parts) == 2 and parts[0].lstrip("-").isdigit():
            counts[parts[1]] = int(parts[0])
    return counts


@pytest.mark.integration
class TestOpenFFEndToEnd:
    def test_peo_melt_with_water(self, tmp_path):
        system = System(out=str(tmp_path / "out"))
        poly = Polymer(chain_num=1, sequence=PEO_SEQUENCE, tacticity="atactic")
        water = Molecule(Count=5, Smiles="O", Name="water")
        generate(
            system, "peo_openff", [poly, water],
            force_field="openff",
            typer_options={"charge_method": "gasteiger"},
            box_size=30.0,
            rng_seed=42,
        )

        project = tmp_path / "out" / "peo_openff"
        data_path = project / "system.data"
        settings_path = project / "system.in.settings"
        init_path = project / "system.in.init"
        assert data_path.is_file() and settings_path.is_file()

        counts = _data_counts(data_path.read_text())
        n_molecules = 1 + 5  # one chain + five waters
        # Acyclic system: every molecule is a tree -> bonds = atoms - molecules
        assert counts["bonds"] == counts["atoms"] - n_molecules
        assert counts["angles"] > 0
        assert counts["dihedrals"] > 0
        assert counts["impropers"] == 0  # PEO/water have no trigonal centers

        settings = settings_path.read_text()
        assert " fourier " in settings  # SMIRNOFF proper torsions
        assert "lj/charmm/coul/long" in settings  # Sage uses a switching window
        init = init_path.read_text()
        assert "special_bonds" in init and "0.833333" in init

        # Charges neutral per molecule
        charges = {}
        atoms_block = data_path.read_text().split("Atoms  # full")[1].split("Bonds")[0]
        for line in atoms_block.strip().splitlines():
            p = line.split()
            if len(p) >= 4:
                charges[int(p[1])] = charges.get(int(p[1]), 0.0) + float(p[3])
        for mol_id, total in charges.items():
            assert abs(total) < 1e-3, f"molecule {mol_id} not neutral: {total}"

        # units.json round-trips and references the generated openff.lt
        manifest = json.loads(
            (project / "build" / "openff" / "units.json").read_text()
        )
        assert manifest["force_field"] == "openff"
        assert "openff.lt" in manifest["monomer_files"]

    def test_same_topology_as_gaff2(self, tmp_path):
        """OpenFF term counts must equal the exact graph-theoretic enumeration
        of the chain's bond graph (every length-2/3 path). GAFF2 is shown for
        atoms/bonds only: its By-Type tables silently drop angles lacking a
        parameter, so angle/dihedral counts legitimately differ."""
        counts = {}
        for ff, kwargs in (("gaff2", {}),
                           ("openff", {"typer_options": {"charge_method": "gasteiger"}})):
            system = System(out=str(tmp_path / f"out_{ff}"))
            poly = Polymer(chain_num=1, sequence=PEO_SEQUENCE, tacticity="atactic")
            generate(
                system, f"peo_{ff}", [poly],
                force_field=ff, box_size=30.0, rng_seed=42, **kwargs,
            )
            data = (tmp_path / f"out_{ff}" / f"peo_{ff}" / "system.data").read_text()
            counts[ff] = _data_counts(data)
        for key in ("atoms", "bonds", "impropers"):
            assert counts["openff"][key] == counts["gaff2"][key], (
                f"{key}: openff={counts['openff'][key]} "
                f"gaff2={counts['gaff2'][key]}"
            )

        # Graph-theoretic counts from the typed chain graph
        from collections import defaultdict
        from rdkit import Chem
        geom = json.loads(
            (tmp_path / "out_openff" / "peo_openff" / "geometry" / "geometry.json")
            .read_text()
        )
        mol = Chem.MolFromSmiles(
            geom["chain_graphs"]["model_0"]["smiles_mapped"], sanitize=False
        )
        mol.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(mol)
        deg = defaultdict(int)
        for b in mol.GetBonds():
            deg[b.GetBeginAtomIdx()] += 1
            deg[b.GetEndAtomIdx()] += 1
        graph_angles = sum(d * (d - 1) // 2 for d in deg.values())
        graph_dihedrals = sum(
            (deg[b.GetBeginAtomIdx()] - 1) * (deg[b.GetEndAtomIdx()] - 1)
            for b in mol.GetBonds()
        )
        assert counts["openff"]["atoms"] == mol.GetNumAtoms()
        assert counts["openff"]["bonds"] == mol.GetNumBonds()
        assert counts["openff"]["angles"] == graph_angles
        assert counts["openff"]["dihedrals"] == graph_dihedrals
