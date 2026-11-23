"""Lightweight CPU unit tests for PBC metrics, MD diagnostics, and configs."""

import importlib.util
import unittest
from pathlib import Path

import numpy as np
import yaml
from ase import Atoms

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_script_module(rel_path: str, module_name: str):
    path = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    mod = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class TestBenchmarkMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.eval_mod = _load_script_module("scripts/03_evaluate_rmsd.py", "eval_rmsd")
        cls.md_run_mod = _load_script_module("scripts/07_run_md_benchmark.py", "run_md")
        cls.md_ana_mod = _load_script_module("scripts/08_analyze_md.py", "analyze_md")

    def test_pbc_rmsd_wrapping_and_translation(self) -> None:
        ref = Atoms(
            symbols=["Sb", "Se", "Se"],
            positions=[[0.2, 5.0, 5.0], [5.0, 5.0, 5.0], [9.8, 5.0, 5.0]],
            cell=[10.0, 10.0, 10.0],
            pbc=True,
        )
        # Rigidly translate by +0.5 A and wrap across periodic boundary
        pred_wrapped = ref.copy()
        pred_wrapped.translate([0.5, 0.0, 0.0])
        pred_wrapped.wrap()

        rmsd_zero = self.eval_mod.compute_pbc_rmsd(pred_wrapped, ref)
        self.assertAlmostEqual(rmsd_zero, 0.0, places=7)

        # Displace one atom across the boundary by 0.3 A (relative to the rest)
        pred_disp = ref.copy()
        pos = pred_disp.get_positions()
        pos[0, 0] = 9.9  # 0.3 A across x=0 boundary from 0.2
        pred_disp.set_positions(pos)
        rmsd_disp = self.eval_mod.compute_pbc_rmsd(pred_disp, ref)
        self.assertGreater(rmsd_disp, 0.1)
        self.assertLess(rmsd_disp, 0.2)

    def test_pbc_rmsd_local_cutoff(self) -> None:
        ref = Atoms(
            symbols=["Sb", "Se"],
            positions=[[5.0, 5.0, 5.0], [10.0, 10.0, 10.0]],
            cell=[20.0, 20.0, 20.0],
            pbc=True,
        )
        pred = ref.copy()
        pos = pred.get_positions()
        pos[1, 0] += 1.0  # Far atom shifted by 1.0 A
        pred.set_positions(pos)

        # Defect center at [0.25, 0.25, 0.25] -> [5, 5, 5]; cutoff 3.0 A only sees atom 0 (which did not move relative to ref)
        # Note: compute_pbc_rmsd removes mean shift first, so local_cutoff isolates the local subset magnitude
        local_rmsd = self.eval_mod.compute_pbc_rmsd(
            pred, ref, center_frac=[0.25, 0.25, 0.25], local_cutoff=3.0
        )
        global_rmsd = self.eval_mod.compute_pbc_rmsd(pred, ref)
        self.assertEqual(local_rmsd, global_rmsd)

    def test_unwrapped_msd_across_pbc(self) -> None:
        f0 = Atoms(symbols=["Sb"], positions=[[9.5, 5.0, 5.0]], cell=[10.0, 10.0, 10.0], pbc=True)
        f1 = Atoms(symbols=["Sb"], positions=[[0.5, 5.0, 5.0]], cell=[10.0, 10.0, 10.0], pbc=True)
        msd = self.md_ana_mod.compute_unwrapped_msd([f0, f1])
        self.assertAlmostEqual(float(msd[0]), 0.0, places=7)
        # 9.5 -> 0.5 in 10 A box is +1.0 A displacement -> MSD = 1.0 A^2
        self.assertAlmostEqual(float(msd[1]), 1.0, places=6)

    def test_pair_coordination_and_rdf(self) -> None:
        atoms = Atoms(
            symbols=["Sb", "Se"],
            positions=[[0.0, 0.0, 0.0], [2.6, 0.0, 0.0]],
            cell=[15.0, 15.0, 15.0],
            pbc=True,
        )
        cn = self.md_ana_mod.compute_pair_coordination([atoms], {("Sb", "Se"): 3.4})
        self.assertAlmostEqual(cn["Sb-Se"][0], 1.0, places=7)
        r_centers, g_r = self.md_ana_mod.compute_all_atom_rdf([atoms], r_max=8.0, n_bins=80)
        self.assertEqual(len(r_centers), len(g_r))
        self.assertGreater(float(np.max(g_r)), 0.0)

    def test_sha256_atoms_and_sanitize_name(self) -> None:
        atoms = Atoms(symbols=["Sb", "Se"], positions=[[0.0, 0.0, 0.0], [2.6, 0.0, 0.0]], cell=[10, 10, 10], pbc=True)
        h1 = self.md_run_mod.sha256_atoms(atoms)
        h2 = self.md_run_mod.sha256_atoms(atoms.copy())
        self.assertEqual(h1, h2)
        self.assertEqual(len(h1), 64)
        self.assertEqual(self.md_run_mod.sanitize_name("MACE-POLAR-1-M (FT Stage2)"), "MACE-POLAR-1-M__FT_Stage2")

    def test_configs_valid_yaml(self) -> None:
        for cfg_name in ("config/config.yaml", "config/config_full.yaml", "config/config_md.yaml"):
            cfg_path = REPO_ROOT / cfg_name
            self.assertTrue(cfg_path.exists(), f"Missing config: {cfg_name}")
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            self.assertIn("dataset", data)
            self.assertIn("models", data)


if __name__ == "__main__":
    unittest.main()
