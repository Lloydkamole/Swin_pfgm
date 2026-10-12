# Validation performed during cleanup

## Execution checks

The following commands completed successfully in the existing `edm` environment, from the repository root:

```bash
python -m examples.synthetic_demo --preview examples/assets/synthetic_pipeline.png
python -m tools.check_25d
OMP_NUM_THREADS=2 python train_accelerate_25d.py \
  --config configs/smoke_25d.json --data-dir examples/generated/data \
  --output-dir examples/generated/training --cpu
python inference_25d.py \
  --config examples/generated/config.json \
  --checkpoint examples/generated/model.safetensors \
  --mr examples/generated/data/phantom_000/mr.nii.gz \
  --ct examples/generated/data/phantom_000/ct.nii.gz \
  --mask examples/generated/data/phantom_000/mask.nii.gz \
  --steps 2 --device cpu --output-dir examples/generated/inference
python -m tools.summarize_runs_25d \
  --runs-dir outputs_accelerate_25d --output-dir docs --plot
```

The synthetic example checked paired tensor shapes `[1,1,4,32,32]`, finite inputs, finite gradients after the actual PFGM++ backward pass, and finite sampler output of the expected shape. Its single-update loss was 0.44548917; this is an execution check, not a research result. The complete trainer processed six training patches from three phantoms and two validation patches from one phantom, ran training-time EMA sampling and validation, logged images/metrics, and saved `best` and `final` checkpoints. Resolved configuration and split JSON files were written.

Patch inference loaded a plain safetensors checkpoint strictly and wrote predictions, a fixed-window comparison and center-slice metrics. Configuration checks verified independent copies, JSON round-tripping including integer stage keys and an infinite sampler limit, and rejection of misspelled keys. The run report inspected 15 local run directories and found 10 with validation-generation scores. The best-PSNR event agrees with an independent raw-event extraction.

The synthetic quick start, configuration-print command and focused checks were also repeated from an isolated copy containing only the 31 public candidate files, confirming that private/ignored model variants are not needed. Both synthetic runs produced the same recorded loss. Python source syntax, local Markdown links and Git diff whitespace checks passed. All 32 files removed from the index were verified byte-for-byte unchanged on disk.

The showcase epoch-165 checkpoint header was compared with an instantiated full model: all 409 state tensors have matching keys and shapes. No historical full-model prediction was generated.

## Environment

| Dependency | Observed version |
|---|---|
| Python | 3.9 |
| PyTorch | 2.8.0+cu128 |
| torchvision | 0.13.1 |
| Accelerate | 1.10.1 |
| MONAI | 1.5.1 |
| nibabel | 5.3.2 |
| TorchMetrics | 1.4.0.post0 |
| TensorBoard | 2.19.0 |
| NumPy | 1.24.3 |
| SciPy | 1.10.1 |
| Pillow | 11.3.0 |
| safetensors | 0.7.0 |
| Matplotlib | 3.9.2 |
| PyYAML | 6.0.2 |
| tqdm | 4.67.1 |

The available environment's torchvision image extension emitted a missing CUDA-library warning; its `utils.make_grid` worked for the executed CPU checks. MONAI emitted an orientation-default deprecation warning; PyTorch warned that pinned memory is ineffective without an accelerator. These warnings did not prevent the checks. The README specifies the officially matched PyTorch 2.8.0 / torchvision 0.23.0 pair for new installations.

## Limits of verification

A fresh dependency installation, Python 3.10/3.11 execution, full-model GPU training/inference, bf16 GPU numerical behavior, multi-GPU DDP, FSDP, distributed EMA, long training, and evaluation on the full real-data cohort were not run. NVIDIA-SMI could not communicate with the driver and PyTorch reported no CUDA devices. CPU smoke checks do not establish those paths' correctness or performance.

No dataset was downloaded, no full real-data training was started, and no assets were uploaded or pushed. Historical metrics remain sample-level logged observations with the limitations in [results.md](results.md).
