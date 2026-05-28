"""
Smoke + validation tests for the MedSAM-conditioning extension to SwinFIRUNet.

Run from the repo root (one level above `Swin_pfgm/`):
    cd /mnt/nvme/lloyd/Swin-pfgm
    python -m Swin_pfgm.tests.test_medsam_conditioning

What this test verifies
-----------------------
1. **Construction**: `SwinFIRUNet(use_foundation_cond=True)` builds without error
   and exposes the 4 new submodules (`cond_global_proj`, `bottleneck_xattn`,
   `bottleneck_xattn_gate`, `context_proj`).
2. **Identity at init**: With `cond_*=None`, the foundation-cond model
   produces the SAME output as a non-cond model (same seed). Confirms the
   new path is dormant by default.
3. **Identity-when-zero-init**: With `cond_*` provided but gates/projections
   at zero-init, the output is STILL identical to the non-cond model.
   Confirms zero-init makes the addition exactly zero.
4. **Activation when gates flipped**: After manually setting the gate and
   global projection weights to small non-zero values, the output CHANGES.
   Confirms the conditioning paths actually wire into the network.
5. **Checkpoint compatibility (smoke)**: A state_dict from the non-cond model
   loads into the cond model with `strict=False`; only cond-related keys are
   missing; nothing is unexpected.

This is a self-contained CPU test. It uses tiny tensors (4x4 spatial) and a
miniature SwinFIRUNet config so it runs in a few seconds.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

# Make `Swin_pfgm` package importable when running as a module or script.
_HERE = Path(__file__).resolve().parent
_PKG = _HERE.parent  # Swin_pfgm/
_ROOT = _PKG.parent  # repo root
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


from Swin_pfgm.models.network import SwinFIRUNet  # noqa: E402


# Minimal config: 64×64 spatial input, base_channels=16, 3 stages.
# window_size=8 means input must be divisible by 8 * 2^(num_stages-1) = 32.
# 64 satisfies this.
_TINY_KWARGS = dict(
    in_channels=2,
    out_channels=1,
    base_channels=16,
    channel_mults=(1, 2, 4),
    num_res_blocks=1,
    radius_emb_dim=64,
    attention_resolutions=(4,),
    num_heads=4,
    window_size=8,
    mlp_ratio=2.0,
    num_swin_blocks=1,
    num_sfb_blocks=1,
    num_bottleneck_blocks=2,
    use_sfb=False,
    dropout=0.0,
    drop_path_rate=0.0,
    gradient_checkpoint=False,
    attention_type="window",
    deformable_kwargs=None,
)


def _make_models(seed=0):
    torch.manual_seed(seed)
    base = SwinFIRUNet(**_TINY_KWARGS, use_foundation_cond=False)
    torch.manual_seed(seed)
    cond = SwinFIRUNet(**_TINY_KWARGS, use_foundation_cond=True)
    base.eval()
    cond.eval()
    return base, cond


def _fake_inputs(B=2, H=64, W=64, token_dim=256, grid=64):
    torch.manual_seed(123)
    x = torch.randn(B, 1, H, W)
    mri = torch.randn(B, 1, H, W)
    r = torch.rand(B) * 0.5
    cond_tokens = torch.randn(B, grid, grid, token_dim)
    cond_global = torch.randn(B, token_dim)
    return x, mri, r, cond_tokens, cond_global


def test_construction():
    _, cond = _make_models()
    assert cond.use_foundation_cond is True
    assert cond.cond_global_proj is not None
    assert cond.bottleneck_xattn is not None
    assert cond.bottleneck_xattn_gate is not None
    assert cond.context_proj is not None
    # Verify the gate is exactly zero at init.
    assert torch.all(cond.bottleneck_xattn_gate == 0.0)
    # Verify the final Linear of cond_global_proj is exactly zero at init.
    final_lin = cond.cond_global_proj[-1]
    assert torch.all(final_lin.weight == 0.0)
    assert torch.all(final_lin.bias == 0.0)
    # context_proj is identity-init.
    assert torch.allclose(cond.context_proj.weight, torch.eye(cond.foundation_token_dim))
    assert torch.all(cond.context_proj.bias == 0.0)
    print("[ok] test_construction")


def test_identity_when_cond_none():
    base, cond = _make_models(seed=42)
    x, mri, r, _, _ = _fake_inputs()
    with torch.no_grad():
        y_base = base(x, mri, r)
        y_cond = cond(x, mri, r)  # no cond passed
    assert torch.allclose(y_base, y_cond, atol=1e-6), (
        f"max abs diff = {(y_base - y_cond).abs().max().item():.3e}"
    )
    print("[ok] test_identity_when_cond_none")


def test_identity_when_zero_init_with_cond():
    base, cond = _make_models(seed=42)
    x, mri, r, ct, cg = _fake_inputs()
    with torch.no_grad():
        y_base = base(x, mri, r)
        y_cond = cond(x, mri, r, cond_tokens=ct, cond_global=cg)
    assert torch.allclose(y_base, y_cond, atol=1e-6), (
        f"max abs diff = {(y_base - y_cond).abs().max().item():.3e}"
    )
    print("[ok] test_identity_when_zero_init_with_cond")


def test_paths_change_output_when_active():
    base, cond = _make_models(seed=42)
    x, mri, r, ct, cg = _fake_inputs()
    # Activate the global path: set the final Linear of cond_global_proj to
    # small random weights.
    with torch.no_grad():
        torch.nn.init.normal_(cond.cond_global_proj[-1].weight, std=0.1)
        # Activate the spatial path: set gate to a non-trivial value.
        cond.bottleneck_xattn_gate.fill_(0.5)
        # Also break the to_out zero-init in CrossAttention so the spatial
        # path produces a non-zero contribution.
        torch.nn.init.normal_(cond.bottleneck_xattn.to_out[0].weight, std=0.05)
    with torch.no_grad():
        y_base = base(x, mri, r)
        y_cond = cond(x, mri, r, cond_tokens=ct, cond_global=cg)
    diff = (y_base - y_cond).abs().max().item()
    assert diff > 1e-3, (
        f"expected non-trivial output change after activating cond paths; "
        f"got max diff = {diff:.3e}"
    )
    print(f"[ok] test_paths_change_output_when_active (max diff = {diff:.3e})")


def test_checkpoint_compatibility():
    base, cond = _make_models(seed=0)
    state = base.state_dict()
    missing, unexpected = cond.load_state_dict(state, strict=False)
    assert len(unexpected) == 0, f"unexpected keys: {unexpected[:5]}"
    # All missing keys must be cond-related.
    cond_prefixes = ("cond_global_proj", "bottleneck_xattn", "context_proj")
    bad = [
        k for k in missing
        if not k.startswith(cond_prefixes) and k != "bottleneck_xattn_gate"
    ]
    assert not bad, f"non-cond missing keys after partial load: {bad[:5]}"
    # Now confirm the cond model with loaded weights and cond=None still
    # matches the base model's forward output.
    x, mri, r, _, _ = _fake_inputs()
    cond.eval()
    base.eval()
    with torch.no_grad():
        y_base = base(x, mri, r)
        y_cond = cond(x, mri, r)
    assert torch.allclose(y_base, y_cond, atol=1e-6)
    print(f"[ok] test_checkpoint_compatibility ({len(missing)} cond-only keys missing)")


def main():
    test_construction()
    test_identity_when_cond_none()
    test_identity_when_zero_init_with_cond()
    test_paths_change_output_when_active()
    test_checkpoint_compatibility()
    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
