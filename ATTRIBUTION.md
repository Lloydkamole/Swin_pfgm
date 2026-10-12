# Attribution and licensing notes

## Implementation authorship

[Lloydkamole](https://github.com/Lloydkamole) confirmed that the project's implementation is their own work, with no collaborators or copied implementation identified. This is an authorship statement, not a claim that the underlying scientific methods were invented in this repository.

The original `models/swin_attention.py` header says “Adapted from SwinFIR with modifications for 2D medical imaging.” That wording is preserved. The author reports original implementation of the adaptations. If any upstream source snippets are subsequently identified, preserve their copyright/license notices and identify the exact source revision.

## Scientific methods

- [SwinFIR](https://github.com/Zdafeng/SwinFIR): Dafeng Zhang et al., *SwinFIR: Revisiting the SwinIR with Fast Fourier Convolution and Improved Training for Image Super-Resolution*. Spatial-frequency design reference; this repository adapts the approach to a conditional 3D U-Net.
- [Swin Transformer](https://arxiv.org/abs/2103.14030): Ze Liu et al., shifted-window attention reference.
- [PFGM++](https://arxiv.org/abs/2302.04265): Yilun Xu et al., *Unlocking the Potential of Physics-Inspired Generative Models*. Augmented-dimension perturbation and prior reference.
- [EDM](https://arxiv.org/abs/2206.00364): Tero Karras et al., *Elucidating the Design Space of Diffusion-Based Generative Models*. Preconditioning, noise schedule and sampling reference.
- [Deformable Attention Transformer](https://arxiv.org/abs/2309.01430): attention design reference; `deformable_attention_3d.py` provides the project's 3D adaptation.
- [CFG++](https://arxiv.org/abs/2406.08070): guidance reference from the historical project documentation. The actual `apply_cfg_plus_plus` helper implements guided-prediction standard-deviation rescaling, so paper-level equivalence is not claimed.

No pretrained foundation encoder or upstream trained weights are required by the public 2.5D trainer. The retained local MedSAM experiments are outside this repository's public scope.

## Dataset and image attribution

The author identifies the sample source as **SynthRAD2023 and SynthRAD2025** and explicitly authorized including sample images. Credit belongs to the SynthRAD dataset creators, data contributors and challenge organizers:

- Adrian Thummerer et al., [*SynthRAD2023 Grand Challenge dataset: Generating synthetic CT for radiotherapy*](https://doi.org/10.1002/mp.16529). See the [challenge rules](https://synthrad2023.grand-challenge.org/Rules/) and [dataset collection](https://doi.org/10.5281/zenodo.7260705).
- Adrian Thummerer et al., [*SynthRAD2025 Grand Challenge dataset: Generating synthetic CTs for radiotherapy from head to abdomen*](https://doi.org/10.1002/mp.17981). See the [official data page](https://synthrad2025.grand-challenge.org/data/) and [dataset collection](https://doi.org/10.5281/zenodo.14918089).

SynthRAD2023 data and SynthRAD2025 centers A/B/C/E data are distributed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). **SynthRAD2025 Center D has separate, challenge-limited terms and must not be assumed shareable under CC BY-NC.** Consult the actual source terms for the files you use; the code license does not override dataset restrictions.

The currently configured prepared pool has folder prefixes consistent with 163 SynthRAD2023 Task-1 brain cases (centers A/B/C) and 141 SynthRAD2025 Task-1 head-and-neck cases (centers A/C). These counts describe the current local pool, not a verified historical split. No Center D prefix was found in this pool. Historical image logs lack case identifiers, so the exact edition/center membership of individual showcase tiles cannot be reconstructed.

`docs/assets/synthrad_validation.png` is derived from an existing TensorBoard validation grid. Changes: labels, event selection, explanatory caption and attribution added; original independent per-image normalization retained. No raw NIfTI data, patient IDs or trained weights are distributed. Its event provenance is recorded in [showcase_provenance.json](docs/showcase_provenance.json). Dataset terms continue to apply to source-derived image panels.

`examples/assets/synthetic_pipeline.png` is generated entirely by the included analytic phantom script, without patient images or pretrained weights. It is a pipeline demonstration, not a model performance example.

## Project license status

The preceding README stated “MIT,” but no license text was present in the accessible repository or its history. No license has been invented or applied to third-party material during this cleanup. Confirm the intended project license and add a corresponding `LICENSE` file before release. Preserve any subsequently identified upstream notices and keep data-derived assets under their own terms.
