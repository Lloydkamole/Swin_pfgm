"""Read local TensorBoard logs and write a sanitized, sample-level run comparison."""

import argparse
import ast
import csv
import json
import math
import struct
from pathlib import Path

import yaml
from tensorboard.compat.proto.event_pb2 import Event


def iter_events(path):
    """Stream complete TFRecord events without retaining image payloads.

    This local reporting utility validates record lengths, not TFRecord CRCs.
    A partial trailing record from an interrupted run is skipped.
    """
    with Path(path).open("rb") as stream:
        while True:
            header = stream.read(12)
            if len(header) != 12:
                return
            length = struct.unpack("<Q", header[:8])[0]
            if length > 64 * 1024 * 1024:
                raise ValueError(f"Unexpected event length in {Path(path).name}")
            payload = stream.read(length)
            crc = stream.read(4)
            if len(payload) != length or len(crc) != 4:
                return
            yield Event.FromString(payload)


def sanitize_config(config):
    clean = {}
    for key, value in config.items():
        if key in {"data_dir", "output_dir", "resume"}:
            continue
        if isinstance(value, str):
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                pass
        if isinstance(value, float) and not math.isfinite(value):
            value = "Infinity" if value > 0 else "-Infinity"
        clean[key] = value
    return clean


def summarize(root):
    rows = []
    for run in sorted(Path(root).iterdir()):
        if not run.is_dir():
            continue
        scalars = {}
        image_steps = set()
        for path in sorted(run.rglob("events.out.tfevents.*")):
            for event in iter_events(path):
                for value in event.summary.value:
                    if value.HasField("simple_value") and value.tag in {
                        "val_gen/psnr", "val_gen/ssim", "val_gen/mae", "val/loss", "system/gpu_memory_MB"
                    } and math.isfinite(value.simple_value):
                        by_step = scalars.setdefault(value.tag, {})
                        old = by_step.get(event.step)
                        if old is None or event.wall_time >= old[0]:
                            by_step[event.step] = (event.wall_time, float(value.simple_value))
                    if value.HasField("image") and value.tag == "Val_Gen/MRI_Pred_GT_CenterSlice":
                        image_steps.add(event.step)
        config_paths = sorted(run.rglob("hparams.yml"))
        config = sanitize_config(yaml.safe_load(config_paths[-1].read_text()) or {}) if config_paths else {}
        checkpoints = sorted(p.name for p in (run / "checkpoints").glob("*") if p.is_dir())

        def best(tag, maximize):
            values = scalars.get(tag, {})
            if not values:
                return None
            step = (max if maximize else min)(values, key=lambda i: values[i][1])
            return {"step": step, "epoch": step + 1, "value": values[step][1]}

        psnr = best("val_gen/psnr", True)
        selected = None
        if psnr:
            step = psnr["step"]
            selected = {"step": step, "epoch": step + 1,
                        **{tag.split("/")[-1]: values[step][1] for tag, values in scalars.items()
                           if tag.startswith("val_gen/") and step in values},
                        "logged_comparison_available": step in image_steps,
                        "exact_epoch_checkpoint": f"epoch_{step + 1:04d}" in checkpoints}
        rows.append({"run": run.name, "validation_sample_count": len(scalars.get("val_gen/psnr", {})),
                     "at_best_psnr": selected, "best_ssim": best("val_gen/ssim", True),
                     "best_mae": best("val_gen/mae", False), "best_validation_loss": best("val/loss", False),
                     "peak_allocated_gpu_mib": best("system/gpu_memory_MB", True),
                     "configuration": config, "checkpoints": checkpoints})
    return rows


def write_reports(rows, output_dir, plot=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"scope": "Best observed validation-sample center-slice metrics, not whole-dataset evaluation",
              "selection": "Maximum val_gen/psnr; SSIM and MAE reported at the same event step",
              "epoch_convention": "Logged steps are zero-based; displayed epochs are step + 1",
              "runs": rows}
    (output_dir / "run_summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    scored = sorted((r for r in rows if r["at_best_psnr"]), key=lambda r: r["at_best_psnr"]["psnr"], reverse=True)
    columns = ["run", "epoch", "psnr_db", "ssim", "mae_normalized", "validation_sample_count", "exact_epoch_checkpoint"]
    with (output_dir / "run_scores.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in scored:
            point = row["at_best_psnr"]
            writer.writerow({"run": row["run"], "epoch": point["epoch"], "psnr_db": point["psnr"],
                             "ssim": point.get("ssim"), "mae_normalized": point.get("mae"),
                             "validation_sample_count": row["validation_sample_count"],
                             "exact_epoch_checkpoint": point["exact_epoch_checkpoint"]})
    if plot and scored:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
        labels = [f"{r['run'][:8]} {r['run'][9:15]} / epoch {r['at_best_psnr']['epoch']}" for r in scored]
        ax.barh(labels[::-1], [r["at_best_psnr"]["psnr"] for r in scored][::-1], color="#347b9c")
        ax.set_xlabel("Best observed validation-sample PSNR (dB); higher is better")
        ax.set_title("Exploratory run comparison: sampled center slices\nChanging samples; configuration and checkpoint caveats apply", fontsize=11)
        ax.set_xlim(0, 40)
        ax.grid(axis="x", alpha=0.2)
        (output_dir / "assets").mkdir(exist_ok=True)
        fig.savefig(output_dir / "assets/run_comparison.png", dpi=150)
        plt.close(fig)
    return scored


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("docs"))
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    rows = summarize(args.runs_dir)
    scored = write_reports(rows, args.output_dir, args.plot)
    print(f"Scanned {len(rows)} runs; {len(scored)} have validation-generation scores")
    if scored:
        print(json.dumps({"run": scored[0]["run"], **scored[0]["at_best_psnr"]}, indent=2))


if __name__ == "__main__":
    main()
