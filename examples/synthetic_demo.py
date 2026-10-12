"""Exercise the real 2.5D loader, loss, backward pass and sampler on synthetic data."""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from safetensors.torch import save_file

from loader_25d import MonaiNifti25DDataset, get_monai_nifti_transforms_25d
from train_accelerate_25d import build_model, edm_loss, edm_sample, load_config, save_config


def make_dataset(root, patients=4, size=32, depth=8):
    """Generate aligned analytic phantoms in [H, W, D], already in [-1, 1]."""
    root = Path(root)
    x, y, z = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size),
                          np.linspace(-1, 1, depth), indexing="ij")
    entries = []
    for index in range(patients):
        folder = root / f"phantom_{index:03d}"
        folder.mkdir(parents=True, exist_ok=True)
        radius = ((x - index * 0.01) / 0.72) ** 2 + (y / 0.86) ** 2 + z ** 2 * 0.08
        body = radius < 1
        shell = body & (radius > 0.78)
        inclusion = (x + 0.20) ** 2 + (y - 0.10) ** 2 < 0.09
        mr = np.where(body, 0.25 + 0.15 * np.cos(x * 5), -1.0)
        mr[shell] = -0.45
        mr[inclusion & body] = 0.65
        ct = np.where(body, -0.25 + 0.08 * np.sin(y * 4), -1.0)
        ct[shell] = 0.85
        ct[inclusion & body] = 0.10
        for name, volume in [("mr", mr), ("ct", ct), ("mask", body)]:
            nib.save(nib.Nifti1Image(volume.astype(np.float32), np.eye(4)), folder / f"{name}.nii.gz")
        entries.append({"id": folder.name, **{k: str(folder / f"{k}.nii.gz") for k in ["mr", "ct", "mask"]}})
    return entries


def save_comparison(path, panels, title):
    """Use one fixed [-1, 1] grayscale window for every panel, without per-image scaling."""
    tile, margin = 224, 16
    width = len(panels) * (tile + margin) + margin
    image = Image.new("RGB", (width, tile + 108), "#f7f7f7")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    draw.text((margin, 12), title, fill="#111111", font=font)
    for index, (label, array) in enumerate(panels):
        if isinstance(array, torch.Tensor):
            array = array.detach().float().cpu().numpy()
        pixels = np.rint((np.clip(np.squeeze(array), -1, 1) + 1) * 127.5).astype(np.uint8)
        panel = Image.fromarray(pixels).convert("RGB").resize((tile, tile), Image.Resampling.NEAREST)
        left = margin + index * (tile + margin)
        draw.text((left, 42), label, fill="#111111", font=font)
        image.paste(panel, (left, 68))
    draw.text((margin, tile + 80), "All panels: fixed [-1, 1] display; no clinical interpretation.", fill="#333333", font=font)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("examples/generated"))
    parser.add_argument("--preview", type=Path, help="Optional second copy of the synthetic preview")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    np.random.seed(42)
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs/smoke_25d.json")
    entries = make_dataset(args.output_dir / "data")
    dataset = MonaiNifti25DDataset(entries, get_monai_nifti_transforms_25d("val", True),
                                  target_size=cfg["image_size"], num_slices=cfg["num_slices"],
                                  stride=cfg["slice_stride"], use_body_mask=True, mode="val")
    patch = dataset[0]
    mr, ct, mask = (patch[key].unsqueeze(0) for key in ["mr", "ct", "mask"])
    assert mr.shape == ct.shape == mask.shape == (1, 1, 4, 32, 32)
    assert torch.isfinite(mr).all() and torch.isfinite(ct).all()
    model = build_model(cfg).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    loss, _ = edm_loss(model, ct, mr, torch.tensor([0.5]), sigma_data=cfg["sigma_data"],
                       pfgmpp=True, D=cfg["D"], body_mask=mask)
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    optimizer.step()
    optimizer.zero_grad()
    model.eval()
    prediction = edm_sample(model, mr, num_steps=2, sigma_min=0.002, sigma_max=1.0,
                            sigma_data=cfg["sigma_data"], pfgmpp=True, D=cfg["D"], mask=mask)
    assert prediction.shape == ct.shape and torch.isfinite(prediction).all()
    panels = [("Synthetic MRI input", mr[0, 0, 2]), ("Synthetic CT reference", ct[0, 0, 2]),
              ("Pipeline output (untrained)", prediction[0, 0, 2])]
    save_comparison(args.output_dir / "comparison.png", panels,
                    "Synthetic pipeline demonstration: one optimization step; no pretrained weights")
    if args.preview:
        save_comparison(args.preview, panels,
                        "Synthetic pipeline demonstration: one optimization step; no pretrained weights")
    save_file(model.state_dict(), str(args.output_dir / "model.safetensors"))
    save_config(cfg, args.output_dir / "config.json")
    report = {"purpose": "Pipeline check, not trained synthesis results", "seed": 42,
              "shape": list(prediction.shape), "loss": float(loss.detach()),
              "finite_gradients": True, "finite_prediction": True, "optimization_steps": 1,
              "sampling_steps": 2, "sampling_sigma_max": 1.0}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Comparison: {args.output_dir / 'comparison.png'}")


if __name__ == "__main__":
    main()
