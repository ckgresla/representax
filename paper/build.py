"""Freeze measured evidence, then render paper figures without training dependencies."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics as stats
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEEDS = (7, 42, 773)
PANEL_SEEDS = (*SEEDS, 1234, 2026)
COLORS = {
    "sky": "#3f78b5",
    "rose": "#c85f65",
    "mint": "#4f8a67",
    "periwinkle": "#8278b8",
    "apricot": "#cc7a3a",
    "slate": "#68717c",
}
LABELS = {
    "dense-retrieval": "Dense retrieval",
    "semantic-similarity-mpnet-base": "Similarity / MPNet",
    "semantic-similarity-bert-base": "Similarity / BERT",
    "pair-classification-mpnet-base": "Classification / MPNet",
    "pair-classification-bert-base": "Classification / BERT",
    "cross-encoder": "Cross-encoder",
    "late-interaction": "Late interaction [L]",
    "outcome-reward": "Outcome reward",
    "process-reward": "Process reward",
    "image-text": "Image-text [C]",
    "audio-text": "Audio-text",
    "video-text": "Video-text",
    "v-jepa": "V-JEPA 2.1 [C]",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def describe(values):
    values = list(values)
    require(bool(values) and all(math.isfinite(v) for v in values), "Invalid values")
    return {
        "mean": stats.mean(values),
        "sd": stats.stdev(values) if len(values) > 1 else 0,
        "values": values,
    }


def warm_rows(rows):
    return [
        r
        for r in rows
        if r.get("event") == "training_step"
        and "perf/step_seconds" in r.get("metrics", {})
        and "perf/compilation_and_first_step_seconds" not in r["metrics"]
        and not r["metrics"].get("perf/excluded_from_steady_state", False)
    ][-20:]


def freeze(root, output):
    require(
        not output.exists(), f"Evidence already frozen: {output}; use a new output path"
    )
    manifest = {}

    def read(path, jsonl=False):
        path = Path(path)
        manifest[str(path)] = {"sha256": digest(path), "bytes": path.stat().st_size}
        content = path.read_text()
        return (
            [json.loads(line) for line in content.splitlines() if line.strip()]
            if jsonl
            else json.loads(content)
        )

    panels = {}
    for platform in ("gpu-rtx4090", "tpu-v5e-16", "gpu-rtx4090-torchinductor"):
        base = root / "10-cross-accelerator-framework-comparison" / platform
        source = read(base / "results.json")
        records = []
        for record in source["runs"]:
            path = Path(record["result_directory"])
            if not path.exists() and path.name == "representax":
                path = path.with_name("representax-local")
            metrics = read(path / "metrics.jsonl", jsonl=True)
            require(
                "sha256:" + digest(path / "metrics.jsonl") == record["metrics_sha256"],
                f"Historical metric hash mismatch: {path}",
            )
            summary = read(path / "summary.json")
            require(
                "sha256:" + digest(path / "summary.json") == record["summary_sha256"],
                f"Historical summary hash mismatch: {path}",
            )
            run = read(path / "run.json")
            require(run["status"] == "completed", f"Incomplete run: {path}")
            measured = warm_rows(metrics)
            require(
                len(measured) == record["measured_updates"] >= 15, f"Warm count: {path}"
            )
            seconds = sum(r["metrics"]["perf/step_seconds"] for r in measured)
            rate = sum(r["metrics"]["perf/examples"] for r in measured) / seconds
            require(
                math.isclose(rate, record["examples_per_second"], rel_tol=1e-9),
                f"Rate mismatch: {path}",
            )
            # Preserve metric records, including startup and first-use costs.
            records.append(
                {
                    **record,
                    "metrics": metrics,
                    "run": run,
                    "contract": summary.get("contract"),
                    "resolved_directory": str(path),
                }
            )
        expected_recipes = (
            {"dense-retrieval"} if platform.endswith("torchinductor") else set(LABELS)
        )
        expected_frameworks = {r["framework"] for r in records}
        require(
            expected_frameworks <= {"representax", "reference"}, "Unexpected framework"
        )
        if not platform.endswith("torchinductor"):
            require(
                expected_frameworks == {"representax", "reference"}, "Missing framework"
            )
        keys = [(r["recipe"], r["framework"], r["seed"]) for r in records]
        expected = {
            (r, f, s)
            for r in expected_recipes
            for f in expected_frameworks
            for s in PANEL_SEEDS
        }
        require(
            len(keys) == len(set(keys)) and set(keys) == expected,
            f"Incomplete panel: {platform}",
        )
        panels[platform] = {"aggregates": source["aggregates"], "runs": records}

    learning = {}
    dense = read(root / "11-dense-retrieval-convergence/summary.json")
    require({int(s) for s in dense["runs"]} == set(SEEDS), "Dense seed coverage")

    def dense_metric(stage):
        return [
            dense["runs"][str(s)][stage]["valid/NanoMSMARCO/cosine_ndcg@10"]
            for s in SEEDS
        ]

    learning["Dense / NanoMSMARCO"] = {
        "initial": describe(dense_metric("initial_evaluation")),
        "final": describe(dense_metric("final_evaluation")),
    }
    transfer = {}
    for name in ("trec-dl-2019", "natural-questions"):
        transfer[name] = describe(
            dense["transfer_evaluations"][str(s)][name]["metrics"][
                f"valid/{name}/cosine_ndcg@10"
            ]
            for s in SEEDS
        )
    clip = read(root / "13-image-text-convergence/summary.json")
    require({int(s) for s in clip["runs"]} == set(SEEDS), "CLIP seed coverage")
    for title, key in (
        ("CLIP / text-to-image", "text_to_image_ndcg@10"),
        ("CLIP / image-to-text", "image_to_text_ndcg@10"),
    ):
        learning[title] = {
            stage: {
                "mean": clip["results"][key][stage]["mean"],
                "sd": clip["results"][key][stage]["sample_standard_deviation"],
            }
            for stage in ("initial", "final")
        }
    late = [
        read(
            root
            / "12-late-interaction-convergence/hard-negatives/runs"
            / f"seed-{s}/report.json"
        )
        for s in SEEDS
    ]
    learning["Late interaction / NanoMSMARCO"] = {
        stage: describe(
            r[f"{stage}_evaluation"]["metrics"]["valid/nanobeir-msmarco/maxsim_ndcg@10"]
            for r in late
        )
        for stage in ("initial", "final")
    }
    omni = read(root / "14-text-to-any-modality/summary.json")
    require(
        omni["completed_runs"] == 9 and omni["skipped_or_nonfinite_updates"] == 0,
        "Omni incomplete",
    )

    base = root / "15-modernbert-mlm-scaling/lambda-a100-20260911"
    scaling = read(base / "summary.json")
    require(
        scaling["status"] == "completed" and len(scaling["rows"]) == 12,
        "Scaling incomplete",
    )
    originals = {}
    for row in scaling["rows"]:
        run = read(Path(row["result"]))
        phase = run["phases"][0]
        obs = phase["observations"]
        require(
            len(obs) == 21 and all(o["finite"] and not o["skipped"] for o in obs),
            "Scaling updates",
        )
        require(
            run["configuration"]["tokens_per_update"] == 131072
            and phase["global_batch"] == 256,
            "Scaling work budget",
        )
        require(
            math.isclose(
                131072 * 20 / sum(o["seconds"] for o in obs[1:]),
                row["tokens_per_second"],
                rel_tol=1e-9,
            ),
            "Scaling rate",
        )
        originals[(row["seed"], row["gpus"])] = run
    require(
        set(originals) == {(s, g) for s in SEEDS for g in (1, 2, 4, 8)}, "Scaling grid"
    )
    for seed in SEEDS:
        baseline = originals[seed, 1]
        for gpus in (2, 4, 8):
            candidate = originals[seed, gpus]
            require(
                baseline["configuration"] == candidate["configuration"],
                "Scaling configuration mismatch",
            )
            require(
                {
                    k: v
                    for k, v in baseline["source_sha256"].items()
                    if k.startswith("src/")
                }
                == {
                    k: v
                    for k, v in candidate["source_sha256"].items()
                    if k.startswith("src/")
                },
                "Scaling library source mismatch",
            )
            for a, b in zip(
                baseline["phases"][0]["observations"],
                candidate["phases"][0]["observations"],
                strict=True,
            ):
                require(
                    all(
                        a[k] == b[k]
                        for k in (
                            "tokens_sha256",
                            "corrupted_sha256",
                            "labels_sha256",
                            "positions_sha256",
                            "supervised_tokens",
                        )
                    ),
                    "Scaling data mismatch",
                )
    for relative in (
        "experiments/uv.lock",
        "experiments/pyproject.toml",
        "experiments/10-cross-accelerator-framework-comparison/data-manifest.json",
        "experiments/10-cross-accelerator-framework-comparison/model-manifest.json",
    ):
        path = HERE.parent / relative
        manifest[str(path)] = {
            "sha256": digest(path),
            "bytes": path.stat().st_size,
            "role": "current reproduction files, not historical run environment",
        }
    for path in (
        HERE / "build.py",
        HERE / "assets/InterVariable.ttf",
        HERE / "assets/Inter-LICENSE.txt",
    ):
        manifest[str(path)] = {"sha256": digest(path), "bytes": path.stat().st_size}
    evidence = {
        "panels": panels,
        "learning": learning,
        "transfer_final_only": transfer,
        "dense_contract": dense["contract"],
        "clip_contract": clip["contract"],
        "omni": omni,
        "scaling": scaling,
        "scaling_raw": {f"{s}-{g}": r for (s, g), r in originals.items()},
        "sources": manifest,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    print(f"Frozen {len(manifest)} source files into {output}")


def paired_rates(panel, recipe):
    records = [r for r in panel["runs"] if r["recipe"] == recipe]
    groups = {
        f: {r["seed"]: r["examples_per_second"] for r in records if r["framework"] == f}
        for f in ("representax", "reference")
    }
    require(set(groups["representax"]) == set(groups["reference"]), "Unpaired seeds")
    return [
        groups["representax"][s] / groups["reference"][s]
        for s in sorted(groups["reference"])
    ]


def render(evidence, destination):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import font_manager

    font_manager.fontManager.addfont(HERE / "assets/InterVariable.ttf")
    font = font_manager.FontProperties(
        fname=HERE / "assets/InterVariable.ttf"
    ).get_name()
    plt.rcParams.update(
        {
            "font.family": font,
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "normal",
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 180,
        }
    )
    destination.mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        for ext in ("png", "pdf"):
            fig.savefig(
                destination / f"{name}.{ext}", bbox_inches="tight", facecolor="white"
            )
        plt.close(fig)

    recipes = list(LABELS)
    fig, axes = plt.subplots(1, 2, figsize=(12, 7.5), sharey=True, layout="constrained")
    tables = [
        "# Measured Framework Throughput",
        "",
        "Ratio of median per-seed examples/s. Dots in the figure are paired seed "
        "ratios, not confidence intervals.",
        "",
        "| Recipe | GPU native | GPU reference | GPU ratio | "
        "TPU native | TPU reference | TPU ratio |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    lookups = {}
    for ax, platform, title, color in zip(
        axes,
        ("gpu-rtx4090", "tpu-v5e-16"),
        ("RTX 4090 / single GPU", "v5e / full 16-chip slice"),
        (COLORS["sky"], COLORS["mint"]),
        strict=True,
    ):
        panel = evidence["panels"][platform]
        lookup = {r["recipe"]: r for r in panel["aggregates"]}
        lookups[platform] = lookup
        for i, recipe in enumerate(recipes):
            ratios = paired_rates(panel, recipe)
            ax.scatter(
                ratios,
                i + np.linspace(-0.11, 0.11, len(ratios)),
                color=color,
                alpha=0.5,
                s=22,
            )
            ax.scatter(
                lookup[recipe]["representax_to_reference_ratio"],
                i,
                color=color,
                marker="D",
                s=42,
            )
        ax.axvline(1, color=COLORS["slate"], lw=1, ls="--")
        ax.set(
            xscale="log",
            xlim=(0.35, 14),
            title=title,
            xlabel="Representax / reference throughput",
        )
        ax.set_xticks([0.5, 1, 2, 4, 8], ["0.5x", "1x", "2x", "4x", "8x"])
        ax.grid(axis="x", alpha=0.15)
    axes[0].set_yticks(range(len(recipes)), [LABELS[r] for r in recipes])
    axes[0].invert_yaxis()
    fig.suptitle("Framework comparisons: five seeds per recipe", fontsize=15)
    fig.supxlabel(
        "[L] Historical late-interaction limitation. "
        "[C] TPU cold-cache order effect. See captions.md.",
        fontsize=9,
    )
    save(fig, "framework-throughput")
    for recipe in recipes:
        values = []
        for platform in ("gpu-rtx4090", "tpu-v5e-16"):
            row = lookups[platform][recipe]
            values.extend(
                [
                    row["frameworks"][f]["median_examples_per_second"]
                    for f in ("representax", "reference")
                ]
            )
            values.append(row["representax_to_reference_ratio"])
        tables.append(
            "| "
            + LABELS[recipe]
            + " | "
            + " | ".join(f"{v:.3f}" for v in values)
            + " |"
        )
    compiled = evidence["panels"]["gpu-rtx4090-torchinductor"]
    reference = stats.median(
        r["examples_per_second"]
        for r in compiled["runs"]
        if r["framework"] == "reference"
    )
    native = lookups["gpu-rtx4090"]["dense-retrieval"]["frameworks"]["representax"][
        "median_examples_per_second"
    ]
    tables += [
        "",
        f"Dense Inductor control: {reference:.3f} examples/s; "
        f"native / compiled = {native / reference:.3f}x.",
        "",
        "GPU and TPU allocations and negative-pool semantics differ; the scatter "
        "compares relative framework rates, not equal-cost hardware performance.",
    ]

    fig, (ax, legend) = plt.subplots(
        1,
        2,
        figsize=(11, 6),
        gridspec_kw={"width_ratios": [1.3, 1]},
        layout="constrained",
    )
    for i, recipe in enumerate(recipes, 1):
        x = lookups["gpu-rtx4090"][recipe]["representax_to_reference_ratio"]
        y = lookups["tpu-v5e-16"][recipe]["representax_to_reference_ratio"]
        color = COLORS["slate"] if recipe == "late-interaction" else COLORS["sky"]
        ax.scatter(x, y, s=45, c=color, alpha=0.88, zorder=3)
        offset = {2: (-12, -16), 4: (12, 9), 3: (-12, -12), 5: (12, 9)}
        ax.annotate(
            str(i),
            (x, y),
            xytext=offset.get(i, (10, 8)),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
        legend.text(0, 1 - (i - 1) * 0.071, f"{i:02d}  {LABELS[recipe]}", va="top")
    legend.axis("off")
    ax.set(
        xscale="log",
        yscale="log",
        xlim=(0.5, 13),
        ylim=(0.6, 14),
        xlabel="GPU: Representax / reference",
        ylabel="TPU: Representax / reference",
    )
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_ticks([1, 2, 4, 8], ["1x", "2x", "4x", "8x"])
    ax.axvline(1, ls="--", color=COLORS["slate"])
    ax.axhline(1, ls="--", color=COLORS["slate"])
    fig.suptitle("Backend-dependent advantages, not a universal speedup", fontsize=14)
    save(fig, "cross-accelerator")

    fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
    for i, row in enumerate(evidence["learning"].values()):
        initial, final = row["initial"]["mean"], row["final"]["mean"]
        color = COLORS["mint"] if final > initial else COLORS["rose"]
        ax.plot([initial, final], [i, i], color=color, lw=3)
        ax.scatter(
            initial, i, facecolors="white", edgecolors=COLORS["slate"], s=55, zorder=3
        )
        ax.errorbar(final, i, xerr=row["final"]["sd"], fmt="o", color=color, capsize=4)
        ax.text(
            max(initial, final) + 0.022,
            i,
            f"{initial:.3f} to {final:.3f}",
            va="center",
            fontsize=9,
        )
    ax.set_yticks(range(len(evidence["learning"])), list(evidence["learning"]))
    ax.invert_yaxis()
    ax.set(
        xlim=(0, 1.03),
        xlabel="Held-out nDCG@10",
        title="Fixed-budget learning: initial to final",
    )
    ax.grid(axis="x", alpha=0.15)
    fig.supxlabel(
        "Open circle: initial. Filled circle: three-seed final mean; "
        "whiskers: sample SD. Not a plateau claim.",
        fontsize=9,
    )
    save(fig, "held-out-learning")

    fig, axes = plt.subplots(1, 4, figsize=(13, 4.1), sharey=True, layout="constrained")
    arms = list(evidence["omni"]["groups"])
    for ax, dataset, title in zip(
        axes,
        ("flickr30k", "audiocaps", "msrvtt", "nanomsmarco"),
        (
            "Image / Flickr30k",
            "Audio / AudioCaps",
            "Video / MSR-VTT",
            "Text / NanoMSMARCO",
        ),
        strict=True,
    ):
        for i, (arm, color) in enumerate(
            zip(arms, (COLORS["sky"], COLORS["mint"], COLORS["rose"]), strict=True)
        ):
            result = evidence["omni"]["groups"][arm]["quality"][
                f"valid/{dataset}/cosine_ndcg@10"
            ]
            ax.plot(
                [0, 1],
                [result["initial"]["mean"], result["final"]["mean"]],
                color=color,
                alpha=0.7,
            )
            ax.scatter(
                0, result["initial"]["mean"], facecolors="white", edgecolors=color, s=35
            )
            ax.errorbar(
                1 + (i - 1) * 0.02,
                result["final"]["mean"],
                yerr=result["final"]["sample_standard_deviation"],
                fmt="o",
                color=color,
                label=arm,
                capsize=3,
            )
        ax.set(
            title=title,
            xlim=(-0.15, 1.15),
            ylim=(0.3, 0.83),
            xticks=[0, 1],
            xticklabels=["Initial", "Final"],
        )
        ax.grid(axis="y", alpha=0.15)
    axes[0].set_ylabel("nDCG@10")
    axes[1].legend(fontsize=8, loc="upper left")
    fig.suptitle(
        "Text-anchored multimodal adaptation: gains and text retention", fontsize=14
    )
    fig.supxlabel(
        "Three seeds per arm; sample SD. Each arm uses its own initial scores. "
        "Mixtures differ across arms.",
        fontsize=9,
    )
    save(fig, "multimodal-adaptation")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), layout="constrained")
    scaling = evidence["scaling"]
    devices = [1, 2, 4, 8]
    means = [r["mean_tokens_per_second"] / 1000 for r in scaling["aggregate"]]
    sds = [r["stdev_tokens_per_second"] / 1000 for r in scaling["aggregate"]]
    axes[0].plot(
        devices,
        [means[0] * g for g in devices],
        "--",
        color=COLORS["slate"],
        label="Ideal linear",
    )
    axes[0].errorbar(
        devices,
        means,
        yerr=sds,
        fmt="o-",
        color=COLORS["sky"],
        capsize=4,
        label="Measured",
    )
    axes[0].set(ylabel="Thousand input tokens/s", title="505M ModernBERT MLM")
    axes[0].legend()
    for seed in SEEDS:
        rows = sorted(
            (r for r in scaling["rows"] if r["seed"] == seed), key=lambda r: r["gpus"]
        )
        axes[1].plot(
            devices,
            [100 * r["efficiency"] for r in rows],
            "o-",
            alpha=0.5,
            color=COLORS["mint"],
        )
    axes[1].set(
        ylabel="Parallel efficiency (%)", ylim=(75, 103), title="Fixed scientific batch"
    )
    for row in scaling["aggregate"][1:]:
        axes[1].annotate(
            f"{row['mean_paired_speedup']:.2f}x",
            (row["gpus"], 100 * row["mean_paired_speedup"] / row["gpus"]),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
        )
    for ax in axes:
        ax.set(xlabel="A100 SXM4 GPUs", xticks=devices)
        ax.grid(alpha=0.15)
    fig.supxlabel(
        "131,072 tokens/update; length 512; local microbatch 16; "
        "replicated DDP; 3 seeds x 20 warm updates.",
        fontsize=9,
    )
    save(fig, "strong-scaling")
    tables += [
        "",
        "## Held-Out Learning",
        "",
        "Three-seed mean and sample SD; nDCG@10.",
        "",
        "| Recipe / evaluation | Initial | Final mean | Final SD |",
        "|---|---:|---:|---:|",
    ]
    for name, row in evidence["learning"].items():
        tables.append(
            f"| {name} | {row['initial']['mean']:.4f} | "
            f"{row['final']['mean']:.4f} | {row['final']['sd']:.4f} |"
        )
    for name, row in evidence["transfer_final_only"].items():
        tables.append(
            f"| Dense / {name} | unmeasured | {row['mean']:.4f} | {row['sd']:.4f} |"
        )
    tables += [
        "",
        "## Multimodal Adaptation",
        "",
        "Each arm uses its own initial evaluation; mean +/- sample SD.",
        "",
        "| Strategy | Evaluation | Initial | Final | Delta |",
        "|---|---|---:|---:|---:|",
    ]
    for arm, group in evidence["omni"]["groups"].items():
        for key, values in group["quality"].items():
            if key.endswith("cosine_ndcg@10"):
                fields = [
                    f"{values[s]['mean']:.4f} +/- "
                    f"{values[s]['sample_standard_deviation']:.4f}"
                    for s in ("initial", "final", "delta")
                ]
                tables.append(
                    f"| {arm} | {key.split('/')[1]} | " + " | ".join(fields) + " |"
                )
    tables += [
        "",
        "## A100 Strong Scaling",
        "",
        "| GPUs | Tokens/s mean | Sample SD | Paired speedup | Efficiency |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in scaling["aggregate"]:
        tables.append(
            f"| {row['gpus']} | {row['mean_tokens_per_second']:.1f} | "
            f"{row['stdev_tokens_per_second']:.1f} | "
            f"{row['mean_paired_speedup']:.3f}x | "
            f"{100 * row['mean_paired_speedup'] / row['gpus']:.1f}% |"
        )
    tables += [
        "",
        "## Recorded First-Use Costs",
        "",
        "Sum of `perf/compilation_and_first_step_seconds` records per run. "
        "This includes first execution and may include cache loading; "
        "it is not necessarily cold compilation. Missing is not zero. "
        "Ranges span the five seeds, and are not confidence intervals.",
        "",
        "| Panel | Recipe | Framework | Recorded runs | Seconds min / median / max |",
        "|---|---|---|---:|---:|",
    ]
    for platform, panel in evidence["panels"].items():
        for recipe in dict.fromkeys(r["recipe"] for r in panel["runs"]):
            for framework in ("representax", "reference"):
                costs = []
                for run in panel["runs"]:
                    if (run["recipe"], run["framework"]) != (recipe, framework):
                        continue
                    fields = [
                        m["metrics"]["perf/compilation_and_first_step_seconds"]
                        for m in run["metrics"]
                        if "perf/compilation_and_first_step_seconds"
                        in m.get("metrics", {})
                    ]
                    if fields:
                        costs.append(sum(fields))
                value = (
                    f"{min(costs):.2f} / {stats.median(costs):.2f} / {max(costs):.2f}"
                    if costs
                    else "not recorded in this field"
                )
                tables.append(
                    f"| {platform} | {recipe} | {framework} | {len(costs)} | {value} |"
                )
    (destination.parent / "results.md").write_text("\n".join(tables) + "\n")
    print(f"Rendered five figure pairs into {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    capture = sub.add_parser(
        "freeze", help="Read and validate original artifacts; refuse overwrite"
    )
    capture.add_argument("--root", type=Path, default=Path("/raid/representax-paper"))
    capture.add_argument("--output", type=Path, default=HERE / "evidence.json")
    plot = sub.add_parser(
        "render", help="Rebuild figures from frozen evidence, no /raid needed"
    )
    plot.add_argument("--evidence", type=Path, default=HERE / "evidence.json")
    plot.add_argument("--output", type=Path, default=HERE / "figures")
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args.root, args.output)
    else:
        render(json.loads(args.evidence.read_text()), args.output)


if __name__ == "__main__":
    main()
