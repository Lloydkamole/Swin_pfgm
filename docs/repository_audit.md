# Repository audit and cleanup

## Public scope and preservation

The public source consists of the 2.5D trainer, its NIfTI loader, six model modules, configuration presets, patch inference, synthetic examples, run-reporting utilities and documentation. `models/__init__.py` now exposes the 2.5D API without importing historical 2D/magnitude-preserving modules. Its preceding contents are preserved locally as `models/__init__.py.bak`.

The preceding README is preserved locally as `README.md.bak`. Notebooks, alternate trainers, profiling/diagnostic scripts, presentation assets, research writing, auxiliary loaders, unused model variants, old tests, all training output directories and checkpoints remain on disk. They are excluded by the explicit public-file allowlist in `.gitignore`. Removing already tracked analysis files from the Git index is necessary because `.gitignore` alone does not stop tracking them. No working files are deleted by that operation.

No commits, pushes, uploads, remote changes or history rewriting are part of this cleanup. The Git remote remains `https://github.com/Lloydkamole/Swin_pfgm.git` on branch `main`.

## Secret and sensitive-material checks

Eight reachable commits and 81 reachable file blobs were inspected for recognizable private-key blocks, AWS access-key IDs, GitHub token prefixes, OpenAI key prefixes and quoted credential assignments. No matches were found. A similar pattern scan covered 84 current project text files and found no matches. These are targeted pattern checks, not a guarantee that all confidential information is absent.

No tracked NIfTI/DICOM files or checkpoint files were found in the reachable history. However, two historical notebooks contain embedded PNG outputs:

- `sampling_steps_analysis.ipynb`
- `visualize_loader.ipynb`

Their historical blobs are approximately 4.25 MB and 3.78 MB. A historical `sampling_steps_comparison.png` is approximately 3.51 MB. Source files/notebooks also contain workstation paths. Git author metadata includes personal identifiers. Embedded outputs may include medical data; the author should assess whether every historical image falls within the stated public-data permission, especially any restricted SynthRAD2025 Center D data.

**Ignoring or untracking these files does not erase existing commits or copies already on GitHub.** No history has been rewritten. If historical material must be removed, decide that separately and obtain approval before rewriting shared history.

## Attribution and licensing

The author confirmed original implementation authorship and identified SynthRAD2023/2025 as the sample source. Published method and dataset credits are recorded in [ATTRIBUTION.md](../ATTRIBUTION.md). The original “Adapted from SwinFIR” header is preserved rather than removed.

The preceding README declared MIT but no project license text was found in reachable history. No new license is assumed. The author should confirm the project license and add a `LICENSE` file. Source-derived dataset images carry separate data terms. SynthRAD2025 Center D has restricted terms; the current prepared pool has no Center D prefix, but historical case identities were not saved.

## Scientific behavior and reproducibility

The model layers, loss, perturbation/prior functions, metric formulas, sampler dynamics, guidance calculations and data transforms are retained. At the author's request, final generated outputs now set masked-out voxels to -1 (0 after mapping to [0,1]); this does not change foreground predictions or training calculations. Configuration/launch changes remove the embedded workstation data path, allow JSON overrides, reject unknown keys, fail explicitly on unsupported resume, and save new runs' resolved configurations and patient split manifests. New inference wraps the existing model, loader, sampler and metric helpers.

Remaining gaps are documented rather than replaced with assumptions: historical preprocessing/CT inverse scaling, dataset membership and source revision; sampled rather than full-dataset generation metrics; absent historical EMA checkpoint state; no exact checkpoint for the highest-PSNR event; unverified FSDP and full-model GPU execution; patch-only inference; loader error fallbacks and pre-orientation slice indexing; and gradient monitoring occurring after gradients have been zeroed in the existing training loop.

`mask_outside_value` and some logging/configuration entries are retained historical settings; their presence should not be taken as evidence of additional implemented behavior. The option named CFG++ implements standard-deviation rescaling, not verified paper-level equivalence.
