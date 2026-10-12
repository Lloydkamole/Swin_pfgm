# A small, runnable 2.5D example

Run all commands from the repository root after installation. No patient data or pretrained weights are required.

```bash
python -m examples.synthetic_demo
```

The script creates four normalized, aligned analytic phantoms, loads 4 × 32 × 32 patches through the actual NIfTI loader, computes the PFGM++ training loss, checks finite gradients, takes one optimizer step, and samples with a tiny 3D spatial-frequency/deformable-attention model on CPU. Inspect `examples/generated/report.json` and `examples/generated/comparison.png`.

The comparison labels the input, synthetic CT reference, and **untrained pipeline output**. All panels use a fixed `[-1, 1]` window; the generated CT background is -1 outside its body mask and therefore displays as black. This proves the code path executes; one optimization step does not produce meaningful trained CT synthesis. The demonstration uses two sampler steps and sigma_max 1 for an inexpensive check, rather than the full training sampler's settings.

To exercise the complete trainer, including patient splitting, EMA, validation, TensorBoard and best/final checkpoint saves:

```bash
OMP_NUM_THREADS=2 python train_accelerate_25d.py \
  --config configs/smoke_25d.json \
  --data-dir examples/generated/data \
  --output-dir examples/generated/training --cpu
```

This is one epoch over three synthetic training patients and one synthetic validation patient. Synthetic scores are execution checks and must not be reported as medical-imaging results.

To test strict checkpoint loading, patch inference and center-slice evaluation with the tiny checkpoint created by the first command:

```bash
python inference_25d.py \
  --config examples/generated/config.json \
  --checkpoint examples/generated/model.safetensors \
  --mr examples/generated/data/phantom_000/mr.nii.gz \
  --ct examples/generated/data/phantom_000/ct.nii.gz \
  --mask examples/generated/data/phantom_000/mask.nii.gz \
  --steps 2 --device cpu --output-dir examples/generated/inference
```

The inference command uses the configuration's sampler settings (sigma_max 80 and guidance settings), so its output differs from the demo's sigma_max-1 illustration. It writes a patch tensor, a fixed-window comparison, and metrics. Use an appropriately trained checkpoint and its matching configuration, correctly preprocessed MRI, and an optional reference CT for real inference.

Focused configuration checks:

```bash
python -m tools.check_25d
```

Generated data, outputs and weights are ignored by Git. To regenerate the shareable synthetic preview deliberately:

```bash
python -m examples.synthetic_demo --preview examples/assets/synthetic_pipeline.png
```
