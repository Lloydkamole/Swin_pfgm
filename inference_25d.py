"""Generate a single 2.5D patch with matching configuration and safetensors weights."""

import argparse
import json
from pathlib import Path

import torch
import torchmetrics
from safetensors.torch import load_file

from examples.synthetic_demo import save_comparison
from loader_25d import MonaiNifti25DDataset, get_monai_nifti_transforms_25d
from train_accelerate_25d import build_model, compute_masked_metrics, edm_sample, load_config, mask_generated_ct


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Plain model.safetensors; sharded FSDP files unsupported")
    parser.add_argument("--mr", type=Path, required=True)
    parser.add_argument("--ct", type=Path, help="Optional aligned reference CT for center-slice metrics")
    parser.add_argument("--mask", type=Path)
    parser.add_argument("--patch-index", type=int, default=0)
    parser.add_argument("--steps", type=int, help="Override configured sampler steps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("inference_output_25d"))
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this environment")
    if args.patch_index < 0 or (args.steps is not None and args.steps < 2):
        parser.error("patch-index must be nonnegative; sampling requires at least two steps")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    cfg = load_config(args.config)
    use_body_mask = cfg["use_body_mask"] or args.mask is not None
    entry = {"mr": str(args.mr), "ct": str(args.ct or args.mr)}
    if args.mask:
        entry["mask"] = str(args.mask)
    dataset = MonaiNifti25DDataset([entry], get_monai_nifti_transforms_25d("val", use_body_mask),
                                  target_size=cfg["image_size"], num_slices=cfg["num_slices"],
                                  stride=cfg["slice_stride"], use_body_mask=use_body_mask, mode="val")
    if args.patch_index >= len(dataset):
        parser.error(f"patch-index must be less than {len(dataset)}")
    patch = dataset[args.patch_index]
    mr = patch["mr"].unsqueeze(0).to(args.device)
    mask = patch.get("mask")
    mask = mask.unsqueeze(0).to(args.device) if mask is not None else None
    model = build_model(cfg).to(args.device).eval()
    model.load_state_dict(load_file(str(args.checkpoint), device=args.device), strict=True)
    kwargs = {key: cfg[key] for key in ["sigma_min", "sigma_max", "sigma_data", "rho", "S_churn",
              "S_min", "S_max", "S_noise", "cfg_scale", "pfgmpp", "D", "use_cfg_plus_plus",
              "cfg_rescale_factor", "use_dynamic_cfg", "cfg_scale_min", "cfg_scale_max"]}
    # A supplied mask can black out output even if mask conditioning was disabled in training.
    sampling_mask = mask if cfg["use_body_mask"] else None
    prediction = edm_sample(model, mr, num_steps=args.steps or cfg["num_sample_steps"], mask=sampling_mask, **kwargs)
    prediction = mask_generated_ct(prediction, mask)
    if not torch.isfinite(prediction).all():
        raise RuntimeError("Sampler produced nonfinite values")
    center = cfg["num_slices"] // 2
    panels = [("MRI input", mr[0, 0, center]), ("Predicted CT", prediction[0, 0, center])]
    report = {"seed": args.seed, "patch_index": args.patch_index, "shape": list(prediction.shape),
              "background": "-1 outside the supplied body mask; 0 after mapping to [0, 1]",
              "weight_type": "raw checkpoint weights; EMA weights not restored", "metrics": {}}
    if args.ct:
        ct = patch["ct"].unsqueeze(0).to(args.device)
        panels.insert(1, ("CT reference", ct[0, 0, center]))
        report["metrics"] = compute_masked_metrics(
            (prediction[:, :, center].clamp(-1, 1) + 1) / 2, (ct[:, :, center] + 1) / 2,
            mask=mask[:, :, center] if mask is not None else None,
            psnr_metric=torchmetrics.image.PeakSignalNoiseRatio(data_range=1.0).to(args.device),
            ssim_metric=torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0).to(args.device),
            mae_metric=torchmetrics.MeanAbsoluteError().to(args.device))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Patch tensor only: no claim to reconstruct a full-volume NIfTI affine.
    from safetensors.torch import save_file
    save_file({"prediction": prediction.detach().cpu().contiguous()}, str(args.output_dir / "prediction.safetensors"))
    save_comparison(args.output_dir / "comparison.png", panels, "2.5D patch inference: center slice; checkpoint weights")
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
