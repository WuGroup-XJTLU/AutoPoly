#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LAMMPS smoke test of a generated system.

Turns the generated in.run into a short run (bounded minimization, a few
hundred MD steps, no dumps), executes LAMMPS, and checks that

- LAMMPS exits cleanly (no ERROR, no lost atoms/bonds),
- every thermo energy is finite, the temperature stays bounded,
- the expected number of rigid bodies was created (nanocomposites).

Used by the qualification tests (tests/test_nanocomposite_lammps_smoke.py)
and available to users:

    >>> from AutoPoly.pipeline.smoke import run_smoke
    >>> report = run_smoke("runs/ps_silica", steps=200)

Created on 2026-11-13
@author: zwu
"""
import math
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from ..core.exceptions import WorkflowError


def lammps_executable(candidates=("lmp", "lmp_serial", "lmp_mpi")) -> Optional[str]:
    """First LAMMPS binary found on PATH (None if none)."""
    for name in candidates:
        path = shutil.which(name)
        if path:
            return path
    return None


@dataclass
class SmokeReport:
    """Outcome of a smoke run."""
    ok: bool
    thermo: List[Dict[str, float]] = field(default_factory=list)
    rigid_bodies: Optional[int] = None
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    log: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {"ok": self.ok, "thermo_last": self.thermo[-1] if self.thermo
                else None, "rigid_bodies": self.rigid_bodies,
                "warnings": self.warnings[:20], "errors": self.errors}


def smoke_input(run_script: str, steps: int = 200,
                min_iterations: int = 200) -> str:
    """Shorten a generated in.run: bounded minimize, short run, no dumps."""
    out = []
    for line in run_script.splitlines():
        s = line.strip()
        if s.startswith("dump") or s.startswith("undump"):
            continue
        if s.startswith("minimize"):
            line = f"minimize        1.0e-4 1.0e-6 {min_iterations} " \
                   f"{10 * min_iterations}"
        elif s.startswith("run "):
            line = f"run             {steps}"
        elif s.startswith("thermo ") and not s.startswith("thermo_"):
            line = f"thermo          {max(1, steps // 4)}"
        out.append(line)
    return "\n".join(out) + "\n"


_THERMO_HEAD = re.compile(r"^\s*Step\s")


def parse_log(text: str) -> List[Dict[str, float]]:
    """All thermo rows of a LAMMPS log."""
    rows: List[Dict[str, float]] = []
    lines = text.splitlines()
    k = 0
    while k < len(lines):
        if _THERMO_HEAD.match(lines[k]):
            keys = lines[k].split()
            k += 1
            while k < len(lines):
                parts = lines[k].split()
                if len(parts) != len(keys):
                    break
                try:
                    rows.append({key: float(v) for key, v in zip(keys, parts)})
                except ValueError:
                    break
                k += 1
        k += 1
    return rows


def run_smoke(project_dir, steps: int = 200, lmp: Optional[str] = None,
              expected_rigid_bodies: Optional[int] = None,
              max_temperature: float = 5000.0,
              timeout: float = 1800.0,
              script: str = "in.run") -> SmokeReport:
    """
    Run the shortened ``script`` (default in.run; in.polymer for the
    bead-spring backends) of ``project_dir`` with LAMMPS.

    Raises:
        WorkflowError: No LAMMPS executable or no run script.
    """
    project = Path(project_dir)
    lmp = lmp or lammps_executable()
    if lmp is None:
        raise WorkflowError("No LAMMPS executable (lmp) found on PATH")
    run_in = project / script
    if not run_in.is_file():
        raise WorkflowError(f"No {script} in {project}")
    (project / "in.smoke").write_text(smoke_input(run_in.read_text(), steps))
    proc = subprocess.run(
        [lmp, "-nb", "-in", "in.smoke", "-log", "log.smoke",
         "-screen", "none"],
        cwd=project, capture_output=True, text=True, timeout=timeout,
    )
    log_path = project / "log.smoke"
    log = log_path.read_text() if log_path.is_file() else proc.stdout
    report = SmokeReport(ok=True, log=log)
    report.thermo = parse_log(log)
    for line in log.splitlines():
        if line.startswith("ERROR"):
            report.errors.append(line)
        elif line.startswith("WARNING"):
            report.warnings.append(line)
        m = re.match(r"^\s*(\d+) rigid bodies with (\d+) atoms", line)
        if m:
            report.rigid_bodies = int(m.group(1))
    if proc.returncode != 0 and not report.errors:
        report.errors.append(
            f"LAMMPS exited with code {proc.returncode}: "
            f"{(proc.stderr or proc.stdout)[-500:]}")
    if any("Lost atoms" in w or "lost" in w.lower() and "bond" in w.lower()
           for w in report.warnings + report.errors):
        report.errors.append("atoms or bonds were lost")
    if not report.thermo:
        report.errors.append("no thermo output")
    for row in report.thermo:
        for key, value in row.items():
            if not math.isfinite(value):
                report.errors.append(f"non-finite {key} = {value}")
                break
    temps = [row.get("Temp", row.get("c_tmatrix", 0.0)) for row in
             report.thermo[-5:]]
    if temps and max(temps) > max_temperature:
        report.errors.append(f"temperature blew up ({max(temps):.0f} K)")
    if expected_rigid_bodies is not None and \
            report.rigid_bodies != expected_rigid_bodies:
        report.errors.append(
            f"expected {expected_rigid_bodies} rigid bodies, LAMMPS created "
            f"{report.rigid_bodies}")
    report.ok = not report.errors
    return report
