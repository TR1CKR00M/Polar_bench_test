#!/usr/bin/env python3
"""
06_finetune_polar.py
====================
Launches `mace.cli.run_train` to fine-tune `MACE-POLAR-1-M` (`polar-1-m`) on the
prepared charged-defect extxyz dataset, while applying the runtime compatibility
shims required for `mace-torch 0.3.16` + `graph_longrange 0.4.4`.

Usage:
    python scripts/06_finetune_polar.py \
        --train-file data/finetune/v_sb1_pbe/train.xyz \
        --valid-file data/finetune/v_sb1_pbe/valid.xyz \
        --name polar_1m_v_sb1_pbe \
        --max-num-epochs 50 \
        --batch-size 4 \
        --device cpu
"""

import argparse
import inspect
import os
import sys
from pathlib import Path

# PyTorch 2.6+ defaults torch.load(..., weights_only=True), which breaks
# pickled MACE/e3nn/graph_longrange checkpoints unless explicitly allowed.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")


def apply_graph_longrange_compat_shims(pbc_handling: str = "pbc") -> None:
    """
    Patch graph_longrange 0.4.4 and unpickled MACE-POLAR-1 checkpoints so that
    mace_run_train runs cleanly under mace-torch 0.3.16.
    """
    from graph_longrange.energy import GTOElectrostaticEnergy
    from graph_longrange.features import GTOElectrostaticFeatures
    import mace.calculators.foundations_models as fm
    import mace.cli.run_train as rt

    if "force_pbc_evaluator" not in inspect.signature(GTOElectrostaticFeatures.precompute_geometry).parameters:
        _orig_precompute = GTOElectrostaticFeatures.precompute_geometry

        def _patched_precompute(self, *args, force_pbc_evaluator=None, **kwargs):
            return _orig_precompute(self, *args, **kwargs)

        GTOElectrostaticFeatures.precompute_geometry = _patched_precompute

    if "pbc" not in inspect.signature(GTOElectrostaticFeatures.forward_dynamic).parameters:
        _orig_forward_dynamic = GTOElectrostaticFeatures.forward_dynamic

        def _patched_forward_dynamic(self, *args, pbc=None, **kwargs):
            if "source_feats" in kwargs and hasattr(kwargs["source_feats"], "dim"):
                sf = kwargs["source_feats"]
                if sf.dim() == 3 and sf.size(1) == 1:
                    kwargs["source_feats"] = sf.squeeze(1)
            elif len(args) >= 2 and hasattr(args[1], "dim"):
                sf = args[1]
                if sf.dim() == 3 and sf.size(1) == 1:
                    args = (args[0], sf.squeeze(1), *args[2:])
            return _orig_forward_dynamic(self, *args, **kwargs)

        GTOElectrostaticFeatures.forward_dynamic = _patched_forward_dynamic

    if "force_pbc_evaluator" not in inspect.signature(GTOElectrostaticEnergy.forward).parameters:
        _orig_energy_forward = GTOElectrostaticEnergy.forward

        def _patched_energy_forward(self, *args, force_pbc_evaluator=None, **kwargs):
            return _orig_energy_forward(self, *args, **kwargs)

        GTOElectrostaticEnergy.forward = _patched_energy_forward

    _orig_mace_polar = fm.mace_polar

    def _patched_mace_polar(*args, **kwargs):
        res = _orig_mace_polar(*args, **kwargs)
        models = [res] if kwargs.get("return_raw_model", False) else getattr(res, "models", [])
        for m in models:
            for sub in m.modules():
                if hasattr(sub, "set_pbc_handling"):
                    sub.set_pbc_handling(pbc_handling)
                if hasattr(sub, "features_irreps") and not hasattr(sub, "features_dim"):
                    sub.features_dim = sub.features_irreps.dim
        return res

    fm.mace_polar = _patched_mace_polar
    rt.mace_polar = _patched_mace_polar

    import torch

    _orig_tensor_deepcopy = torch.Tensor.__deepcopy__

    def _safe_tensor_deepcopy(self, memo):
        if not self.is_leaf:
            return self.detach().clone()
        return _orig_tensor_deepcopy(self, memo)

    torch.Tensor.__deepcopy__ = _safe_tensor_deepcopy


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune MACE-POLAR-1-M with compatibility shims.")
    parser.add_argument("--name", type=str, default="polar_1m_v_sb1_pbe", help="Run & checkpoint name")
    parser.add_argument(
        "--foundation-model",
        type=str,
        default="polar-1-m",
        help="PolarMACE foundation model name (polar-1-m or polar-1-l)",
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default="data/finetune/v_sb1_pbe/train.xyz",
        help="Path to training extxyz",
    )
    parser.add_argument(
        "--valid-file",
        type=str,
        default="data/finetune/v_sb1_pbe/valid.xyz",
        help="Path to validation extxyz",
    )
    parser.add_argument("--work-dir", type=str, default="model/polar_1m_finetuned", help="Output directory")
    parser.add_argument("--max-num-epochs", type=int, default=50, help="Maximum training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    parser.add_argument("--valid-batch-size", type=int, default=4, help="Validation batch size")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--energy-weight", type=float, default=1.0, help="Energy loss weight")
    parser.add_argument("--forces-weight", type=float, default=100.0, help="Forces loss weight")
    parser.add_argument("--stress-weight", type=float, default=0.0, help="Stress loss weight")
    parser.add_argument("--e0s", type=str, default="average", help="E0s mode ('average' or JSON/dict)")
    parser.add_argument("--device", type=str, default="cpu", help="Device ('cpu' or 'cuda')")
    parser.add_argument("--default-dtype", type=str, default="float64", help="Floating point precision")
    parser.add_argument("--pbc-handling", type=str, default="pbc", help="PolarMACE PBC evaluator mode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print constructed mace_run_train CLI arguments without starting training.",
    )
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    train_args = [
        "mace_run_train",
        f"--name={args.name}",
        "--model=PolarMACE",
        f"--foundation_model={args.foundation_model}",
        f"--train_file={args.train_file}",
        f"--valid_file={args.valid_file}",
        f"--work_dir={work_dir}",
        f"--model_dir={work_dir}",
        f"--log_dir={work_dir / 'logs'}",
        f"--checkpoints_dir={work_dir / 'checkpoints'}",
        f"--results_dir={work_dir / 'results'}",
        "--energy_key=REF_energy",
        "--forces_key=REF_forces",
        "--stress_key=REF_stress",
        "--total_charge_key=total_charge",
        "--total_spin_key=total_spin",
        f"--E0s={args.e0s}",
        "--loss=weighted",
        f"--energy_weight={args.energy_weight}",
        f"--forces_weight={args.forces_weight}",
        f"--stress_weight={args.stress_weight}",
        f"--lr={args.lr}",
        "--force_mh_ft_lr=True",
        "--ema",
        "--ema_decay=0.99",
        "--amsgrad",
        f"--batch_size={args.batch_size}",
        f"--valid_batch_size={args.valid_batch_size}",
        f"--max_num_epochs={args.max_num_epochs}",
        "--eval_interval=1",
        "--error_table=PerAtomMAE",
        "--scaling=rms_forces_scaling",
        f"--default_dtype={args.default_dtype}",
        f"--device={args.device}",
        f"--seed={args.seed}",
        "--save_cpu",
        "--restart_latest",
    ]

    if args.dry_run:
        print("[DRY-RUN] Equivalent mace_run_train arguments:")
        for item in train_args:
            print(f"  {item}")
        return

    apply_graph_longrange_compat_shims(pbc_handling=args.pbc_handling)
    import mace.cli.run_train as rt

    sys.argv = train_args
    rt.main()


if __name__ == "__main__":
    main()
