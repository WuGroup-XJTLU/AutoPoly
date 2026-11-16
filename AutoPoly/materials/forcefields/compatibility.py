#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Organic-inorganic force-field compatibility checks and cross terms.

Three responsibilities:

1. Host (organic) force-field style introspection: read the pair/bond
   style, mixing rule, special_bonds and kspace settings from the
   moltemplate force-field .lt the film/matrix is typed with
   (host_styles()).  Inorganic .lt classes must carry the host's hybrid
   sub-style keywords, and class2 / many-body hosts are refused.

2. Explicit cross interactions: after moltemplate has numbered the atom
   types, write an explicit ``pair_coeff i j`` for every pair involving an
   inorganic type, computed with the declared combining rule of the
   inorganic adapter (or the host rule when the adapter declares
   "host").  The resulting system no longer depends on ``pair_modify
   mix`` for any organic-inorganic pair (write_explicit_cross_terms()).

3. A machine-readable record of every decision for the force-field
   compatibility report.

Created on 2026-11-13
@author: zwu
"""
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ...core.exceptions import ValidationError

#: Host pair styles (or hybrid LJ sub-styles) whose 12-6 LJ coefficients
#: AutoPoly can combine explicitly with inorganic 12-6 parameters.
LJ_12_6_STYLES = ("lj/cut/coul/long", "lj/charmm/coul/long",
                  "lj/cut/coul/cut", "lj/charmm/coul/charmm",
                  "lj/cut/coul/long/soft", "lj/cut")


@dataclass
class HostStyles:
    """Style settings of the organic host force field."""
    force_field: str
    pair_style: str = ""
    bond_style: str = ""
    pair_modify: str = ""
    special_bonds: str = ""
    kspace_style: str = ""

    @property
    def is_class2(self) -> bool:
        return "class2" in self.pair_style or "class2" in self.bond_style

    @property
    def is_hybrid_pair(self) -> bool:
        return self.pair_style.split()[1:2] in (["hybrid"],
                                                 ["hybrid/overlay"])

    @property
    def pair_substyle(self) -> Optional[str]:
        """LJ sub-style for hybrid pair styles (None for plain styles)."""
        if not self.is_hybrid_pair:
            return None
        for token in self.pair_style.split()[2:]:
            if token.startswith("lj/"):
                return token
        return None

    @property
    def bond_substyle(self) -> Optional[str]:
        """'harmonic' if the host uses a hybrid bond style."""
        return "harmonic" if "hybrid" in self.bond_style else None

    @property
    def lj_style(self) -> Optional[str]:
        """The 12-6 LJ pair (sub-)style in use, or None."""
        tokens = self.pair_style.split()[1:]
        for token in tokens:
            if token in LJ_12_6_STYLES:
                return token
        return None

    @property
    def mixing_rule(self) -> str:
        """'geometric', 'arithmetic', 'sixthpower' (LAMMPS default geometric)."""
        m = re.search(r"\bmix\s+(\w+)", self.pair_modify)
        return m.group(1) if m else "geometric"

    def to_dict(self) -> Dict[str, object]:
        return {
            "force_field": self.force_field,
            "pair_style": self.pair_style,
            "bond_style": self.bond_style,
            "pair_modify": self.pair_modify,
            "special_bonds": self.special_bonds,
            "kspace_style": self.kspace_style,
            "pair_substyle": self.pair_substyle,
            "bond_substyle": self.bond_substyle,
            "mixing_rule": self.mixing_rule,
        }


#: openff.lt (generated per system) always declares these styles.
_OPENFF_DEFAULT = dict(
    pair_style="pair_style hybrid lj/charmm/coul/long 9.0 10.0",
    bond_style="bond_style hybrid harmonic",
    pair_modify="pair_modify mix arithmetic",
    special_bonds="special_bonds lj/coul 0.0 0.0 0.5",
    kspace_style="kspace_style pppm 0.0001",
)


def host_styles(force_field: str, ff_lt: Optional[Path]) -> HostStyles:
    """
    Parse the host force-field style lines from its moltemplate .lt.

    Args:
        force_field: Host force-field name.
        ff_lt: Path of the force-field .lt (None = not available, only
               allowed for "openff", whose generated file always uses the
               same styles).
    """
    if ff_lt is None or not Path(ff_lt).is_file():
        if force_field == "openff":
            return HostStyles(force_field, **_OPENFF_DEFAULT)
        raise ValidationError(
            f"Force-field file for '{force_field}' not found: {ff_lt}"
        )
    found: Dict[str, str] = {}
    _collect_styles(Path(ff_lt), found, depth=0)
    return HostStyles(force_field, **found)


_STYLE_KEYS = ("pair_style", "bond_style", "pair_modify", "special_bonds",
               "kspace_style")


def _collect_styles(path: Path, found: Dict[str, str], depth: int) -> None:
    """Style lines of an .lt file, following its ``import`` statements
    (e.g. loplsaa.lt imports oplsaa2024.lt); first definition wins."""
    imports = []
    for line in path.read_text().splitlines():
        ls = line.split("#", 1)[0].strip()
        if not ls:
            continue
        m = re.match(r'^import\s+"?([^"\s]+)"?', ls)
        if m:
            imports.append(m.group(1))
            continue
        for key in _STYLE_KEYS:
            if key not in found and re.match(rf"^{key}\b", ls):
                found[key] = " ".join(ls.split())
    if depth < 3:
        for name in imports:
            dep = path.parent / name
            if dep.is_file():
                _collect_styles(dep, found, depth + 1)


# ----------------------------------------------------------------------
# Combining rules
# ----------------------------------------------------------------------
def combine(rule: str, a: Tuple[float, float],
            b: Tuple[float, float]) -> Tuple[float, float]:
    """(epsilon, sigma) cross parameters for a 12-6 LJ pair."""
    ea, sa = a
    eb, sb = b
    eps = math.sqrt(max(ea, 0.0) * max(eb, 0.0))
    if rule == "arithmetic":
        sig = 0.5 * (sa + sb)
    elif rule == "geometric":
        sig = math.sqrt(max(sa, 0.0) * max(sb, 0.0))
    else:
        raise ValidationError(
            f"Unsupported combining rule '{rule}' for explicit cross terms "
            "(use 'arithmetic' or 'geometric')"
        )
    return eps, sig


def resolve_rule(rule: str, host: HostStyles) -> str:
    """Concrete rule for a declared adapter rule ("host" -> host rule)."""
    if rule == "host":
        host_rule = host.mixing_rule
        if host_rule not in ("arithmetic", "geometric"):
            raise ValidationError(
                f"Host force field '{host.force_field}' uses mixing rule "
                f"'{host_rule}', which cannot be combined with 12-6 "
                "inorganic parameters"
            )
        return host_rule
    return rule


# ----------------------------------------------------------------------
# moltemplate output parsing
# ----------------------------------------------------------------------
def read_type_assignments(moltemplate_dir: Path) -> Dict[int, Tuple[str, str]]:
    """
    Atom type id -> (namespace path, type name) from
    output_ttree/ttree_assignments.txt (e.g. 6 -> ("QuartzSlab",
    "i15_sc4"), 1 -> ("GAFF", "c3")).
    """
    path = _ttree_file(moltemplate_dir)
    out: Dict[int, Tuple[str, str]] = {}
    for line in path.read_text().splitlines():
        m = re.match(r"^@/atom:(\S+)\s+(\d+)", line)
        if m:
            full = m.group(1)
            ns, _, name = full.rpartition("/")
            out[int(m.group(2))] = (ns, name)
    return out


def read_instance_assignments(moltemplate_dir: Path):
    """
    (mol ids, atom ids) from ttree_assignments.txt:

    - mols: instance path -> molecule id (``$/mol:np_1`` -> 2),
    - atoms: instance path -> sorted atom ids of that instance subtree.
    """
    path = _ttree_file(moltemplate_dir)
    mols: Dict[str, int] = {}
    atoms: Dict[str, List[int]] = {}
    for line in path.read_text().splitlines():
        m = re.match(r"^\$/mol:(\S+)\s+(\d+)", line)
        if m:
            mols[m.group(1)] = int(m.group(2))
            continue
        m = re.match(r"^\$/atom:(\S+)\s+(\d+)", line)
        if m:
            instance = m.group(1).split("/", 1)[0]
            atoms.setdefault(instance, []).append(int(m.group(2)))
    for key in atoms:
        atoms[key].sort()
    return mols, atoms


def _ttree_file(moltemplate_dir: Path) -> Path:
    d = Path(moltemplate_dir)
    for cand in (d / "output_ttree" / "ttree_assignments.txt",
                 d / "output" / "output_ttree" / "ttree_assignments.txt"):
        if cand.is_file():
            return cand
    raise ValidationError(
        f"ttree_assignments.txt not found under {d} (run moltemplate first)"
    )


def read_self_pair_coeffs(settings: Path) -> Dict[int, Tuple[float, float]]:
    """type id -> (epsilon, sigma) from ``pair_coeff i i ...`` lines."""
    out: Dict[int, Tuple[float, float]] = {}
    for line in Path(settings).read_text().splitlines():
        parts = line.split("#", 1)[0].split()
        if len(parts) < 5 or parts[0] != "pair_coeff":
            continue
        if not (parts[1].isdigit() and parts[2].isdigit()):
            continue
        i, j = int(parts[1]), int(parts[2])
        if i != j:
            continue
        nums = []
        for tok in parts[3:]:
            try:
                nums.append(float(tok))
            except ValueError:
                if nums:
                    break
                continue  # sub-style keyword
        if len(nums) >= 2:
            out[i] = (nums[0], nums[1])
    return out


# ----------------------------------------------------------------------
# Explicit cross terms
# ----------------------------------------------------------------------
@dataclass
class InorganicClassPolicy:
    """Cross-interaction policy of one inorganic moltemplate class."""
    class_name: str
    adapter: str
    native_rule: str          # adapter's declared rule (may be "host")
    organic_rule: str         # rule for pairs with organic types
    qualification: Optional[str] = None


@dataclass
class CrossTermReport:
    """What write_explicit_cross_terms() did."""
    n_pairs: int = 0
    rules: Dict[str, str] = field(default_factory=dict)
    lines: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {"n_pairs": self.n_pairs, "rules": dict(self.rules)}


def write_explicit_cross_terms(
    moltemplate_dir: Path,
    policies: Dict[str, InorganicClassPolicy],
    host: HostStyles,
    settings_file: str = "system.in.settings",
) -> CrossTermReport:
    """
    Append explicit pair_coeff lines for every pair involving an
    inorganic type to system.in.settings.

    Rules:
    - inorganic x organic: the class's ``organic_rule`` ("host" resolves
      to the host mixing rule),
    - inorganic x inorganic of the same adapter: the adapter's native
      rule ("host" resolves to the host rule),
    - inorganic x inorganic of different adapters: refused (no validated
      policy).

    Raises:
        ValidationError: Missing self coefficients, unsupported host pair
            style, or an unqualified inorganic-inorganic pair.
    """
    d = Path(moltemplate_dir)
    report = CrossTermReport()
    if not policies:
        return report
    lj = host.lj_style
    if lj is None or host.is_class2:
        raise ValidationError(
            f"Host force field '{host.force_field}' pair style "
            f"'{host.pair_style}' is not a supported 12-6 LJ style for "
            "inorganic cross interactions"
        )
    types = read_type_assignments(d)
    selfc = read_self_pair_coeffs(d / settings_file)
    inorganic = {t: ns for t, (ns, _) in types.items() if ns in policies}
    unmatched = sorted(set(policies) - set(inorganic.values()))
    if unmatched:
        raise ValidationError(
            f"No atom types found in class namespace(s) {unmatched}: "
            "explicit cross interactions cannot be written for them"
        )
    missing = [t for t in types if t not in selfc]
    if missing:
        names = [f"{types[t][0]}/{types[t][1]}" for t in missing[:5]]
        raise ValidationError(
            f"No pair_coeff i i found for atom type(s) {names}; cannot "
            "build explicit cross interactions"
        )
    sub = f" {host.pair_substyle}" if host.is_hybrid_pair else ""
    lines = [
        "",
        "# --- AutoPoly: explicit organic-inorganic cross interactions ---",
    ]
    for cls, pol in sorted(policies.items()):
        organic = resolve_rule(pol.organic_rule, host)
        native = resolve_rule(pol.native_rule, host)
        report.rules[cls] = f"{pol.adapter}: {organic} rule with organic " \
                            f"types, {native} rule within the adapter"
        lines.append(
            f"# {cls} ({pol.adapter}): with organic types -> {organic} "
            f"rule; within adapter -> {native} rule"
            + (f"; qualification {pol.qualification}"
               if pol.qualification else "")
        )
    ids = sorted(types)
    for a_idx, i in enumerate(ids):
        for j in ids[a_idx:]:
            if i == j:
                continue
            ci, cj = inorganic.get(i), inorganic.get(j)
            if ci is None and cj is None:
                continue
            if ci is not None and cj is not None:
                pi, pj = policies[ci], policies[cj]
                if pi.adapter != pj.adapter:
                    raise ValidationError(
                        f"No validated cross-interaction policy between "
                        f"inorganic adapters '{pi.adapter}' ({ci}) and "
                        f"'{pj.adapter}' ({cj})"
                    )
                rule = resolve_rule(pi.native_rule, host)
            else:
                pol = policies[ci if ci is not None else str(cj)]
                rule = resolve_rule(pol.organic_rule, host)
            eps, sig = combine(rule, selfc[i], selfc[j])
            lines.append(
                f"pair_coeff {i} {j}{sub} {eps:.6g} {sig:.6g}"
                f"  # {types[i][0]}/{types[i][1]} - "
                f"{types[j][0]}/{types[j][1]} ({rule})"
            )
            report.n_pairs += 1
    lines.append("")
    with open(d / settings_file, "a") as f:
        f.write("\n".join(lines))
    report.lines = lines
    return report
