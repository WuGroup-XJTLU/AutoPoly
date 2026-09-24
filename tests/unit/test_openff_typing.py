"""Unit tests for the optional OpenFF (SMIRNOFF) force field backend.

The OpenFF packages are optional dependencies: tests that need them are
skipped when they are not installed, and the missing-dependency error path is
only exercised when they are absent.
"""

import importlib.util
import json
from pathlib import Path

import pytest

from AutoPoly.core.conf import FORCE_FIELD_DESCRIPTIONS, FORCE_FIELD_REGISTRY
from AutoPoly.core.exceptions import GenerationError, ValidationError
from AutoPoly.core.system import System
from AutoPoly.models.molecule import Molecule
from AutoPoly.models.polymer import Polymer
from AutoPoly.pipeline.geometry import GeometryBuilder, GeometryConfig
from AutoPoly.pipeline.typing import UnitTyper

PE_SEQUENCE = ["CC[*]", "[*]CC[*]", "[*]CC"]
PEO_SEQUENCE = ["CCO[*]", "[*]CCO[*]", "[*]CCO"]
# Branch wildcard: both connection atoms sit on bonded backbone carbons
PMMA_SEQUENCE = [
    "CC(C)(C(=O)OC)[*]",
    "[*]CC([*])(C)C(=O)OC",
    "[*]CC(C)(C(=O)OC)",
]

def _has_openff() -> bool:
    # find_spec("openff.toolkit") imports the parent package; guard it.
    if importlib.util.find_spec("openff") is None:
        return False
    return importlib.util.find_spec("openff.toolkit") is not None


HAS_OPENFF = _has_openff()
needs_openff = pytest.mark.skipif(not HAS_OPENFF, reason="openff-toolkit not installed")


def _build_geometry(tmp_path, models=None, name="sys"):
    system = System(out=str(tmp_path / "out"))
    if models is None:
        models = [Polymer(chain_num=1, sequence=PEO_SEQUENCE, tacticity="atactic")]
    config = GeometryConfig(use_mc_chain_growth=False)
    return GeometryBuilder(system, name, config).build(models)


def _lt_sections(lt_path):
    """Map write/write_once section names -> list of content lines."""
    sections = {}
    current = None
    for line in Path(lt_path).read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith('write("') or stripped.startswith("write('"):
            current = stripped
            sections[current] = []
            continue
        if stripped.startswith('write_once("'):
            current = stripped
            sections[current] = []
            continue
        if current and stripped == "}":
            current = None
            continue
        if current and stripped:
            sections[current].append(stripped)
    return sections


class TestRegistry:
    """openff is a registered force field (no openff packages needed)."""

    def test_registry_entry(self):
        entry = FORCE_FIELD_REGISTRY["openff"]
        assert entry["lt_file"] == "openff.lt"
        assert entry["inherits"] == "OPENFF"
        assert entry["fdefn"] is None
        assert "openff" in FORCE_FIELD_DESCRIPTIONS

    def test_typer_options_rejected_for_smarts_force_fields(self, tmp_path):
        geom = _build_geometry(tmp_path)
        with pytest.raises(ValidationError, match="typer_options"):
            UnitTyper(geom.dir, "gaff2", typer_options={"charge_method": "nagl"})

    def test_ring_topology_rejected(self):
        if not HAS_OPENFF:
            pytest.skip("ring guard lives behind the OpenFFTyper import")
        # Ring sequences type fine with the SMARTS force fields but openff
        # must fail loudly (linear-capped chain graph limitation).
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            system = System(out=str(Path(td) / "out"))
            ring_seq = ["CC[*]", "[*]CC[*]", "[*]CC"]
            poly = Polymer(chain_num=1, sequence=ring_seq, topology="ring")
            geom = GeometryBuilder(
                system, "ring", GeometryConfig(use_mc_chain_growth=False)
            ).build([poly])
            with pytest.raises(ValidationError, match="ring"):
                UnitTyper(geom.dir, "openff")

    def test_smarts_typer_no_silent_fallback_for_openff(self):
        """SMARTSTyper must raise instead of silently OPLS-typing 'openff'."""
        from AutoPoly.monomers.monomer_generator import AtomTypingError, SMARTSTyper
        with pytest.raises(AtomTypingError, match="openff"):
            SMARTSTyper("openff", verbose=False)


class TestMissingDependency:
    """Clean error when the OpenFF packages are not installed."""

    @pytest.mark.skipif(HAS_OPENFF, reason="openff-toolkit IS installed")
    def test_generation_error_with_install_hint(self, tmp_path):
        geom = _build_geometry(tmp_path)
        with pytest.raises(GenerationError, match="openff-toolkit"):
            UnitTyper(geom.dir, "openff")


@needs_openff
class TestOpenFFTyper:
    """OpenFFTyper parameterization (gasteiger charges: fast, no torch)."""

    def _typer(self, **kwargs):
        from AutoPoly.forcefields.openff_typing import OpenFFTyper
        kwargs.setdefault("charge_method", "gasteiger")
        return OpenFFTyper(**kwargs)

    def test_parameterize_molecule(self):
        from rdkit import Chem
        typer = self._typer()
        rdmol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        pm = typer.parameterize(rdmol)
        assert pm.n_atoms == rdmol.GetNumAtoms()
        assert all(v for v in pm.vdw_ids)
        assert len(pm.charges) == pm.n_atoms
        assert abs(sum(pm.charges) - pm.formal_charge) < 1e-6
        # ethanol: 8 bonds, 13 angles, 12 dihedrals, 0 impropers
        assert len(pm.bonds) == 8
        assert len(pm.angles) == 13
        assert len(pm.propers) == 12
        assert pm.impropers == []

    def test_improper_trefoil_and_cvff_coeff(self):
        """Formaldehyde: one SMIRNOFF improper (n=2, phase=180) -> cvff K=k/3."""
        from rdkit import Chem
        typer = self._typer()
        rdmol = Chem.AddHs(Chem.MolFromSmiles("C=O"))
        pm = typer.parameterize(rdmol)
        assert len(pm.impropers) == 1
        i, j, k, l, pid = pm.impropers[0]
        # SMIRNOFF order: central atom (the carbonyl C) is second
        assert rdmol.GetAtomWithIdx(j).GetSymbol() == "C"
        assert all(
            rdmol.GetBondBetweenAtoms(j, x) is not None for x in (i, k, l)
        )
        # Coeff registered as k/idivf (=k/3) with d=-1, n=2
        K, d, n = typer._impropers[pid]
        assert (d, n) == (-1, 2)
        assert K == pytest.approx(10.1416678 / 3.0, rel=0.05)

    def test_write_force_field_lt(self, tmp_path):
        from rdkit import Chem
        typer = self._typer()
        typer.parameterize(Chem.AddHs(Chem.MolFromSmiles("CCO")))
        out = tmp_path / "openff.lt"
        typer.write_force_field_lt(out)
        text = out.read_text()
        assert "OPENFF {" in text
        assert 'write_once("In Init")' in text
        assert "pair_modify     mix arithmetic" in text
        # scale14 read from the handlers (Sage: 0.5 / 0.8333)
        assert "special_bonds   lj 0.0 0.0 0.500000 coul 0.0 0.0 0.833333" in text
        assert "pair_coeff @atom:" in text
        assert "bond_coeff @bond:" in text and " harmonic " in text
        assert "dihedral_coeff @dihedral:" in text and " fourier " in text
        # No By-Type inference sections anywhere
        for section in ("Bonds By Type", "Angles By Type",
                        "Dihedrals By Type", "Impropers By Type"):
            assert section not in text

    def test_nagl_charges_assigned(self):
        """NAGL backend (skipped without openff-nagl / its models)."""
        pytest.importorskip("openff.nagl")
        from rdkit import Chem
        from AutoPoly.forcefields.openff_typing import OpenFFTyper
        typer = OpenFFTyper(charge_method="nagl")
        rdmol = Chem.AddHs(Chem.MolFromSmiles("CO"))
        pm = typer.parameterize(rdmol)
        charges = pm.charges
        o_charge = charges[[a.GetIdx() for a in rdmol.GetAtoms()
                            if a.GetSymbol() == "O"][0]]
        assert o_charge < -0.3  # AM1-BCC-like methanol oxygen
        assert abs(sum(charges)) < 1e-6
        assert typer.nagl_model_used is not None


@needs_openff
class TestOpenFFTypingPipeline:
    """Stage-2 typing with force_field='openff' (explicit-term .lt files)."""

    def test_variant_files_have_explicit_terms(self, tmp_path):
        geom = _build_geometry(tmp_path)
        library = UnitTyper(
            geom.dir, "openff", typer_options={"charge_method": "gasteiger"}
        ).type()
        build_dir = Path(geom.dir).parent / "build" / "openff"

        assert (build_dir / "openff.lt").is_file()
        assert "openff.lt" in library.monomer_files
        library.validate(build_dir)

        # Middle variant: typed atoms + explicit bonds/angles/dihedrals
        sections = _lt_sections(build_dir / "monomer_0_1i.lt")
        atoms = sections['write("Data Atoms") {']
        assert all("@atom:of_" in line for line in atoms)
        assert 'write("Data Bonds") {' in sections
        assert 'write("Data Angles") {' in sections
        assert 'write("Data Dihedrals") {' in sections
        assert any("@bond:ofb_" in line for line in sections['write("Data Bonds") {'])
        assert any("@angle:ofa_" in line for line in sections['write("Data Angles") {'])
        # No untyped bond list anywhere in the build dir
        for lt in build_dir.glob("*.lt"):
            assert "Data Bond List" not in lt.read_text()

    def test_boundary_terms_in_poly_lt(self, tmp_path):
        geom = _build_geometry(tmp_path)
        UnitTyper(
            geom.dir, "openff", typer_options={"charge_method": "gasteiger"}
        ).type()
        build_dir = Path(geom.dir).parent / "build" / "openff"
        poly = (build_dir / "poly_1.lt").read_text()
        assert "poly_1 inherits OPENFF {" in poly
        assert 'import "openff.lt"' in poly
        # 3-monomer chain: 2 boundary bonds with cross-monomer refs
        sections = _lt_sections(build_dir / "poly_1.lt")
        bonds = sections['write("Data Bonds") {']
        assert len(bonds) == 2
        assert all("$atom:monomer[" in line for line in bonds)
        assert all("@bond:ofb_" in line for line in bonds)
        # Boundary-spanning angles and dihedrals exist too
        assert 'write("Data Angles") {' in sections
        assert 'write("Data Dihedrals") {' in sections

    def test_branch_wildcard_monomer(self, tmp_path):
        """PMMA's middle monomer has both connections on bonded carbons —
        the inter-monomer bond detection must not confuse the intra-monomer
        connection-connection bond with a boundary bond."""
        models = [Polymer(chain_num=1, sequence=PMMA_SEQUENCE, tacticity="atactic")]
        geom = _build_geometry(tmp_path, models=models, name="pmma")
        UnitTyper(
            geom.dir, "openff", typer_options={"charge_method": "gasteiger"}
        ).type()
        build_dir = Path(geom.dir).parent / "build" / "openff"
        sections = _lt_sections(build_dir / "poly_1.lt")
        assert len(sections['write("Data Bonds") {']) == 2  # exactly 2 boundaries
        # Ester C=O impropers: trefoil-expanded (3 cvff entries per match)
        imp_lines = []
        for lt in build_dir.glob("*.lt"):
            imp_lines += [
                line for line in lt.read_text().splitlines()
                if line.strip().startswith("$improper:")
            ]
        n_matches = sum(1 for line in imp_lines)  # trefoil-expanded count
        assert n_matches % 3 == 0 and n_matches >= 9

    def test_composition_charge_neutrality(self, tmp_path):
        """Written chain composition must sum to the formal charge even though
        variant charges are shared across occurrences."""
        geom = _build_geometry(tmp_path)
        UnitTyper(
            geom.dir, "openff", typer_options={"charge_method": "gasteiger"}
        ).type()
        build_dir = Path(geom.dir).parent / "build" / "openff"
        geom_data = json.loads((Path(geom.dir) / "geometry.json").read_text())
        total = 0.0
        for chain in geom_data["chains"]:
            for placement in chain["placements"]:
                lt = build_dir / f"{placement['variant']}.lt"
                assert lt.is_file(), f"missing {lt}"
                for line in _lt_sections(lt)['write("Data Atoms") {']:
                    total += float(line.split()[3])
        assert abs(total) < 1e-4

    def test_molecule_explicit_terms(self, tmp_path):
        models = [
            Polymer(chain_num=1, sequence=PE_SEQUENCE, tacticity="atactic"),
            Molecule(Count=3, Smiles="O", Name="water"),
        ]
        geom = _build_geometry(tmp_path, models=models, name="sol")
        library = UnitTyper(
            geom.dir, "openff", typer_options={"charge_method": "gasteiger"}
        ).type()
        build_dir = Path(geom.dir).parent / "build" / "openff"
        water = _lt_sections(build_dir / "water.lt")
        atoms = water['write("Data Atoms") {']
        assert len(atoms) == 3
        o_line = [line for line in atoms if line.startswith("$atom:O")][0]
        assert float(o_line.split()[3]) < 0  # O carries the negative charge
        assert 'write("Data Bonds") {' in water
        assert 'write("Data Angles") {' in water
        assert any(u.id == "water" and u.count == 3 for u in library.units)

    def test_variant_split_on_parameter_conflict(self, tmp_path):
        """When two occurrences of one variant get different SMIRNOFF
        parameters, the typing stage must write separate files and remap
        placements (never silently share). Exercises the grouping machinery
        with a synthetic conflict on one of two middle occurrences."""
        from types import SimpleNamespace
        from AutoPoly.pipeline.typing import mol_from_mapped_smiles

        # DOP=4: two middle occurrences of the same deduplicated variant
        models = [Polymer(chain_num=1,
                          sequence=["CCO[*]"] + ["[*]CCO[*]"] * 2 + ["[*]CCO"],
                          tacticity="atactic")]
        geom = _build_geometry(tmp_path, models=models, name="split")
        typer = UnitTyper(
            geom.dir, "openff", typer_options={"charge_method": "gasteiger"}
        )
        typer.output_dir.mkdir(parents=True, exist_ok=True)  # done by type()

        model_id = "model_0"
        graph = typer.geometry["chain_graphs"][model_id]
        chain_mol = mol_from_mapped_smiles(graph["smiles_mapped"])
        pm_real = typer.typer.parameterize(chain_mol)
        decomp = typer._openff_decompose(model_id, chain_mol, pm_real)

        # Middle occurrences are at positions 1 and 2 of one variant
        middle = [o for o in decomp["occurrences"] if o["pos"] in (1, 2)]
        assert len(middle) == 2
        assert middle[0]["base_name"] == middle[1]["base_name"]

        # Synthetic conflict: retype one atom of occurrence 2 only
        pm_fake = SimpleNamespace(
            n_atoms=pm_real.n_atoms,
            vdw_ids=list(pm_real.vdw_ids),
            charges=list(pm_real.charges),
            bonds=pm_real.bonds, angles=pm_real.angles,
            propers=pm_real.propers, impropers=pm_real.impropers,
            formal_charge=pm_real.formal_charge,
        )
        inv2 = {lt: ci for ci, lt in middle[1]["map"].items()}
        pm_fake.vdw_ids[inv2[0]] = "n_conflict"

        typer._openff_resolve_variant_groups(decomp, pm_fake)
        typer._openff_neutralize_composition(decomp, pm_fake)
        written = typer._openff_write_variants(decomp, pm_fake)

        base = middle[0]["base_name"]
        resolved = decomp["resolved"]
        # Two distinct files for the two middle positions
        assert resolved[1] != resolved[2]
        assert {resolved[1], resolved[2]} == {base, f"{base}_p2"}
        assert f"{base}.lt" in written
        assert f"{base}_p2.lt" in written
        assert f"{base}_T1.lt" in written
        assert f"{base}_p2_T1.lt" in written
        # Both files exist and differ in their Data Atoms
        build_dir = Path(geom.dir).parent / "build" / "openff"
        a = (build_dir / f"{base}.lt").read_text()
        b = (build_dir / f"{base}_p2.lt").read_text()
        assert "@atom:of_n_conflict" in b
        assert "@atom:of_n_conflict" not in a

