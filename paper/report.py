"""Build the manuscript's tables and figures from immutable, local evidence."""

from __future__ import annotations

import json
import statistics as stats
from pathlib import Path

from build import COLORS, LABELS, PANEL_SEEDS, SEEDS, paired_rates, warm_rows

HERE = Path(__file__).resolve().parent
STRATEGIES = ("connectors", "connectors-lora", "full")
STRATEGY_NAMES = ("Connectors", "Connectors + LoRA", "Full fine-tuning")
DATASETS = ("flickr30k", "audiocaps", "msrvtt", "nanomsmarco")
DATASET_NAMES = ("Flickr30k", "AudioCaps", "MSR-VTT", "NanoMSMARCO")
NAMES = {
    **LABELS,
    "outcome-reward": "Outcome reward [O]",
    "process-reward": "Process reward [P,I]",
    "audio-text": "Audio-text [M]",
    "video-text": "Video-text [M]",
    "v-jepa": "V-JEPA 2.1 [C,G]",
}
UNMATCHED_RECIPES = frozenset({
    "late-interaction", "outcome-reward", "process-reward", "audio-text", "video-text",
})


def load():
    return json.loads((HERE / "evidence.json").read_text())


def comparison_matched(evidence, platform, recipe):
    short = {"gpu-rtx4090": "gpu", "tpu-v5e-16": "tpu"}[platform]
    return recipe not in UNMATCHED_RECIPES or (
        f"fairness-20260916-{short}-{recipe}" in evidence.get("corrections", {})
    )


def measured_rows(run):
    excluded = set(run.get("analysis_excluded_iterations", ()))
    return [row for row in warm_rows(run["metrics"]) if row["iteration"] not in excluded]


def summary(values):
    values = list(values)
    return stats.mean(values), stats.stdev(values) if len(values) > 1 else 0.0


def pm(mean, sd, digits=4):
    return rf"${mean:.{digits}f} \pm {sd:.{digits}f}$"


def startup_diagnostics(run):
    """Keep setup, native first-use events, and reference intervals distinct."""
    metrics = run["metrics"]
    first_use = [
        {"iteration": row["iteration"],
         "seconds": row["metrics"]["perf/compilation_and_first_step_seconds"]}
        for row in metrics
        if "perf/compilation_and_first_step_seconds" in row.get("metrics", {})
    ]
    setup = [row["metrics"]["perf/startup_seconds"] for row in metrics
             if "perf/startup_seconds" in row.get("metrics", {})]
    # Reference timing includes input wait; native first-use timing does not.
    reference_steps = {
        row["iteration"]: row["metrics"]["perf/step_seconds"]
        for row in metrics
        if run["framework"] == "reference"
        and row.get("event") == "training_step"
        and "perf/step_seconds" in row.get("metrics", {})
    }
    return {
        "seed": run["seed"],
        "source_directory": run["resolved_directory"],
        "metrics_sha256": run["metrics_sha256"],
        "initial_setup_seconds": setup[0] if setup else None,
        "setup_intervals_seconds": setup,
        "native_first_use_events": first_use,
        "native_first_use_total_seconds": (
            sum(row["seconds"] for row in first_use) if first_use else None
        ),
        "reference_step_1_seconds": reference_steps.get(1),
        "reference_step_2_seconds": reference_steps.get(2),
    }


def startup_cells(native, reference):
    def median(values):
        present = [value for value in values if value is not None]
        return f"{stats.median(present):,.1f}" if present else "---"

    counts = [len(row["native_first_use_events"]) for row in native
              if row["native_first_use_events"]]
    events = (str(counts[0]) if len(set(counts)) == 1 else
              f"{min(counts)}--{max(counts)}") if counts else "---"
    return [
        median(row["native_first_use_total_seconds"] for row in native),
        events,
        median(row["reference_step_1_seconds"] for row in reference),
        median(row["reference_step_2_seconds"] for row in reference),
    ]


def table(name, caption, columns, headers, rows, *, tabcolsep=4):
    body = "\n".join(" & ".join(row) + r" \\" for row in rows)
    text = (
        "#+begin_export latex\n\\begin{table}[tbp]\n\\centering\\small\n"
        + rf"\setlength{{\tabcolsep}}{{{tabcolsep}pt}}" + "\n"
        + rf"\caption{{{caption}}}\label{{tab:{name}}}" + "\n"
        + rf"\begin{{tabular}}{{{columns}}}\toprule" + "\n"
        + " & ".join(headers) + r" \\ \midrule" + "\n"
        + body + "\n\\bottomrule\\end{tabular}\n\\end{table}\n#+end_export\n"
    )
    (HERE / "tables" / f"{name}.org").write_text(text)


def throughput_cells(native, reference, *, matched=True):
    """Compare measured medians within one workload and hardware configuration."""
    if not matched:
        return [f"{native:,.2f}", f"{reference:,.2f}", "---"]
    rates = (native, reference)
    best = max(rates)
    cells = []
    for rate in rates:
        value = f"{rate:,.2f}"
        cells.append(rf"\textbf{{{value}}}" if rate == best else value)
    cells.append(f"{native / reference:.3f}")
    return cells


def framework_overview(e):
    def median_rate(runs, recipe, framework):
        selected = [r for r in runs
                    if r["recipe"] == recipe and r["framework"] == framework]
        assert sorted(r["seed"] for r in selected) == sorted(PANEL_SEEDS)
        return stats.median(r["examples_per_second"] for r in selected)

    rows = []
    for platform, heading in (
        ("gpu-rtx4090", "Single RTX 4090; reference: PyTorch eager"),
        ("tpu-v5e-16", "16-chip v5e slice; reference: PyTorch/XLA"),
    ):
        separator = r"\midrule" if rows else ""
        rows.append([separator + rf"\multicolumn{{4}}{{l}}{{\textit{{{heading}}}}}"])
        runs = e["panels"][platform]["runs"]
        for recipe in LABELS:
            native = median_rate(runs, recipe, "representax")
            reference = median_rate(runs, recipe, "reference")
            rows.append([NAMES[recipe], *throughput_cells(
                native, reference, matched=comparison_matched(e, platform, recipe)
            )])
    table(
        "framework-throughput",
        "Warm training throughput: median examples/s across five seeds. "
        "Bold marks the higher of the two reported rates within each workload and hardware panel, "
        "not statistical significance. $R$ denotes Representax; ratios divide "
        "its median rate by the reference median. GPU references use eager "
        "execution; the dense TorchInductor control is in "
        r"Appendix \ref{sec:compiled-reference}. "
        "GPU and TPU allocations/batches differ. Winner styling and ratios are shown "
        "only for validated matched protocols; historical discrepancies and replacements "
        "are documented in the text. Qualifications [L], [O], [I], [M], [P], [C], [G] are defined in the text; "
        "per-seed variation is shown in the appendix.",
        "lrrr",
        ["Workload", r"\shortstack{Representax\\ex/s}",
         r"\shortstack{Reference\\ex/s}", r"$R/\mathrm{Ref.}$"],
        rows,
        tabcolsep=6,
    )


def learning_tables(e):
    labels = {
        "Dense / NanoMSMARCO": ("Dense", "NanoMSMARCO"),
        "CLIP / text-to-image": ("CLIP", r"Flickr30k text $\to$ image"),
        "CLIP / image-to-text": ("CLIP", r"Flickr30k image $\to$ text"),
        "Late interaction / NanoMSMARCO": ("Late interaction", "NanoMSMARCO"),
    }
    rows = []
    for key, row in e["learning"].items():
        if key.startswith("Late"):
            continue
        a, b = row["initial"], row["final"]
        rows.append([*labels[key], f'{a["mean"]:.4f}', pm(b["mean"], b["sd"])])
    for name, row in e["transfer_final_only"].items():
        initial = e["transfer_initial"][name]["metrics"][f"valid/{name}/cosine_ndcg@10"]
        rows.append(["Dense", {"trec-dl-2019": "TREC DL 2019", "natural-questions": "Natural Questions"}[name], f"{initial:.4f}", pm(row["mean"], row["sd"])])
    table("learning", "Held-out nDCG@10 before and after native adaptation. Final scores are mean and sample SD across three seeds. The pretrained transfer baseline is shared across seeds; initial and final evaluations use the same complete corpora.", "llrr", ["Model", "Evaluation", "Initial", "Final"], rows)
    rows = []
    for stage in ("initial", "final"):
        for strategy, label in zip(STRATEGIES, STRATEGY_NAMES):
            quality = e["omni"]["groups"][strategy]["quality"]
            values = [quality[f"valid/{d}/cosine_ndcg@10"][stage] for d in DATASETS]
            rows.append([label + (" (initial)" if stage == "initial" else " (final)"), *[pm(v["mean"], v["sample_standard_deviation"]) if stage == "final" else f'{v["mean"]:.4f}' for v in values]])
    table("omni", "Text-anchored multimodal adaptation: nDCG@10, mean and sample SD over three seeds. Each strategy is compared with its own initial model. Learning rates and source mixtures differ across strategies.", "lrrrr", ["Strategy", *DATASET_NAMES], rows)
    rows = []
    for strategy, label in zip(STRATEGIES, STRATEGY_NAMES):
        q = e["omni"]["groups"][strategy]["quality"]
        for d, name in zip(DATASETS, DATASET_NAMES):
            vals = [q[f"valid/{d}/cosine_recall@{k}"]["final"] for k in (1, 5, 10)]
            rows.append([label, name, *[pm(v["mean"], v["sample_standard_deviation"]) for v in vals]])
    table("omni-recall", "Final text-to-media or text-to-text recall, mean and sample SD across three seeds. These supplement nDCG; no task-averaged score is formed.", "llrrr", ["Strategy", "Dataset", "R@1", "R@5", "R@10"], rows)


def framework_tables(e):
    framework_overview(e)
    audit = {}
    for platform, short in (("gpu-rtx4090", "gpu"), ("tpu-v5e-16", "tpu")):
        panel = e["panels"][platform]
        rows, startup = [], []
        for recipe in LABELS:
            values, batches = [], []
            startup_by_framework = {}
            audit[f"{short}/{recipe}"] = {}
            for framework in ("representax", "reference"):
                runs = sorted([r for r in panel["runs"] if r["recipe"] == recipe and r["framework"] == framework], key=lambda r: r["seed"])
                assert [r["seed"] for r in runs] == sorted(PANEL_SEEDS)
                rates, step_rates, first_use, cvs, diagnostics = [], [], [], [], []
                for run in runs:
                    warm = [r["metrics"] for r in measured_rows(run)]
                    seconds = [r["perf/step_seconds"] for r in warm]
                    counts = {r["perf/examples"] for r in warm}
                    assert len(counts) == 1
                    batches.append(counts.pop())
                    rates.append(sum(r["perf/examples"] for r in warm) / sum(seconds))
                    step_rates.append(len(warm) / sum(seconds))
                    cvs.append(stats.stdev(seconds) / stats.mean(seconds))
                    diagnostic = startup_diagnostics(run)
                    diagnostics.append(diagnostic)
                    first_use.append(diagnostic["native_first_use_total_seconds"])
                values.extend([f"{stats.median(rates):,.2f}", f"{stats.median(step_rates):.3f}"])
                startup_by_framework[framework] = diagnostics
                audit[f"{short}/{recipe}"][framework] = {"step_time_cv": cvs, "first_use_seconds": first_use, "examples_per_second": rates, "startup_diagnostics": diagnostics}
            assert len(set(batches)) == 1, (recipe, batches)
            ratio = stats.median(audit[f"{short}/{recipe}"]["representax"]["examples_per_second"]) / stats.median(audit[f"{short}/{recipe}"]["reference"]["examples_per_second"])
            rows.append([NAMES[recipe], str(int(batches[0])), *values,
                         f"{ratio:.3f}" if comparison_matched(e, platform, recipe) else "---"])
            startup.append([NAMES[recipe], *startup_cells(startup_by_framework["representax"], startup_by_framework["reference"])])
        if short == "gpu":
            rates = [r["examples_per_second"] for r in e["panels"]["gpu-rtx4090-torchinductor"]["runs"]]
            native = audit["gpu/dense-retrieval"]["representax"]["examples_per_second"]
            rows.append(["Dense / Inductor", "2048", f"{stats.median(native):.2f}", f"{stats.median(native)/2048:.3f}", f"{stats.median(rates):.2f}", f"{stats.median(rates)/2048:.3f}", f"{stats.median(native)/stats.median(rates):.3f}"])
            diagnostics = [startup_diagnostics(run) for run in sorted(e["panels"]["gpu-rtx4090-torchinductor"]["runs"], key=lambda r: r["seed"])]
            startup.append(["Dense / Inductor", *startup_cells([], diagnostics)])
            audit["gpu/dense-retrieval-torchinductor"] = {"reference": {"startup_diagnostics": diagnostics}}
        title = "one RTX 4090" if short == "gpu" else "the full 16-chip v5e slice"
        table(short + "-rates", f"Absolute warm training rates on {title}. Columns show median per-seed examples/s (ex/s) and optimizer steps/s (st/s); ratio is native/reference examples/s. Batch is global examples per update. Ratios are shown only for validated matched protocols; historical discrepancies and replacements are documented in the text. Qualifications are defined in the text.", "lrrrrrr", ["Recipe", "Batch", "Native ex/s", "st/s", "Ref. ex/s", "st/s", "Ratio"], rows)
        table(short + "-startup", f"Recorded early-step diagnostics on {title}, median seconds over five seeds. Native first-use is the sum of dispatch-to-completion intervals across the reported number of events, including new shapes and resumed execution. Reference columns are complete first and second optimizer-step intervals, including input wait. They are different timing boundaries, not a compilation-speed comparison; neither isolates pure compilation or full startup. Cache states vary. A dash means unrecorded, not zero, or no separate native Inductor run.", "lrrrr", ["Recipe", "Native first-use (s)", "Events", "Ref. step 1 (s)", "Ref. step 2 (s)"], startup)
    (HERE / "analysis.json").write_text(json.dumps(audit, indent=2) + "\n")


def scaling_table(e):
    rows = []
    for n in (1, 2, 4, 8):
        raw = [r for r in e["scaling"]["rows"] if r["gpus"] == n]
        tok = summary(r["tokens_per_second"] for r in raw)
        sec = summary(r["step_seconds"] for r in raw)
        speed = stats.mean(r["speedup"] for r in raw)
        mem = max(r["compiled_bytes_per_device"] for r in raw) / 2**30
        rows.append([str(n), str(16 // n), pm(*tok, 0), pm(*sec, 3), f"{speed:.3f}", f"{100*speed/n:.1f}\\%", f"{mem:.2f}"])
    table("scaling", "Strong scaling of a fixed 131,072-token update. Means and sample SD use three seeds. Speedup is the mean within-seed ratio; efficiency divides it by GPU count. Memory is the maximum compiled executable footprint per device, not the allocator's reserved pool.", "rrrrrrr", ["GPUs", "Accum.", "Tokens/s", "Seconds/update", "Speedup", "Efficiency", "GiB"], rows)


def figures(e):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import font_manager
    from matplotlib.patches import Rectangle
    font_manager.fontManager.addfont(HERE / "assets/InterVariable.ttf")
    font = font_manager.FontProperties(fname=HERE / "assets/InterVariable.ttf").get_name()
    plt.rcParams.update({"font.family": font, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none"})

    def save(fig, name):
        for ext in ("pdf", "png"):
            fig.savefig(HERE / "figures" / f"{name}.{ext}", bbox_inches="tight", facecolor="white", dpi=180, metadata={"CreationDate": None} if ext == "pdf" else None)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.1, 3.5))
    ax.set(xlim=(0, 10), ylim=(0, 5.0))
    ax.axis("off")
    def block(x, y, w, h, title, lines, color):
        ax.add_patch(Rectangle((x, y), w, h, facecolor="white", edgecolor=color, lw=1.2))
        ax.plot([x, x+w], [y+h, y+h], color=color, lw=3)
        ax.text(x+.13, y+h-.24, title, fontsize=10, va="top")
        ax.text(x+.13, y+h-.70, lines, fontsize=7.8, va="top", linespacing=1.45)
    block(.05, 2.02, 2.1, 1.90, "Data", "Source sampling\nDecode + map\nProcess + prefetch", COLORS["mint"])
    block(2.62, 2.02, 2.1, 1.90, "Model", "Modality towers\nConnectors + LoRA\nTrainable selection", COLORS["sky"])
    block(5.19, 2.02, 2.1, 1.90, "Task", "Representations\nLoss + modifiers\nMetrics", COLORS["periwinkle"])
    block(7.76, 2.02, 2.1, 1.90, "Update", "Gradients\nClip + optimize\nTarget update", COLORS["apricot"])
    for start in (2.15, 4.72, 7.29):
        ax.annotate("", xy=(start+.46, 3.1), xytext=(start+.04, 3.1), arrowprops={"arrowstyle": "->", "color": COLORS["slate"]})
    ax.text(.05, 4.48, "ONE CONFIGURATION: scientific choices + execution choices", fontsize=10)
    ax.text(.05, 1.54, "Shared execution", color=COLORS["slate"])
    ax.text(.05, 1.22, "GradCache / accumulation    ·    precision / sharding    ·    checkpoint / resume", fontsize=9)
    ax.plot([.05, 9.86], [.88, .88], color="#d7dbe0", lw=1)
    ax.text(.05, .46, "Evaluation", color=COLORS["rose"])
    ax.text(1.8, .46, "Model outputs → batch evaluation → corpus accumulation → metrics", fontsize=9)
    save(fig, "architecture")

    recipes = [r for r in LABELS if r != "late-interaction"] + ["late-interaction"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.9), sharey=True, layout="constrained")
    for ax, platform, title, color in zip(axes, ("gpu-rtx4090", "tpu-v5e-16"), ("Single RTX 4090", "16-chip v5e slice"), (COLORS["sky"], COLORS["mint"])):
        panel = e["panels"][platform]
        lookup = {r["recipe"]: r["representax_to_reference_ratio"] for r in panel["aggregates"]}
        for i, recipe in enumerate(recipes):
            if not comparison_matched(e, platform, recipe):
                ax.text(.52, i, "Pairing under correction", va="center",
                        fontsize=7, color=COLORS["slate"])
                continue
            v = paired_rates(panel, recipe)
            ax.scatter(v, i+np.linspace(-.13, .13, len(v)), s=12, color=color, alpha=.45, edgecolors="none")
            ax.scatter(lookup[recipe], i, marker="D", s=28, color=color, zorder=3)
        ax.axvline(1, ls="--", lw=.9, color=COLORS["slate"])
        ax.axhspan(11.55, 12.45, color="#f0f1f2", zorder=-1)
        ax.set(xscale="log", xlim=(.45, 16), title=title)
        ax.set_xticks([.5, 1, 2, 4, 8, 16], ["0.5", "1", "2", "4", "8", "16"])
        ax.grid(axis="x", alpha=.15)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(range(len(recipes)), [NAMES[r] for r in recipes])
    axes[0].invert_yaxis()
    # The compiled dense control uses the same native runs, not a new native trial.
    compiled = {r["seed"]: r["examples_per_second"] for r in e["panels"]["gpu-rtx4090-torchinductor"]["runs"]}
    native = {r["seed"]: r["examples_per_second"] for r in e["panels"]["gpu-rtx4090"]["runs"] if r["recipe"] == "dense-retrieval" and r["framework"] == "representax"}
    ratio = stats.median(native.values()) / stats.median(compiled.values())
    axes[0].scatter(ratio, -.36, marker="s", s=22, color=COLORS["rose"], zorder=4)
    axes[0].annotate("Inductor", xy=(ratio, -.36), xytext=(1.65, -.36), va="center", fontsize=7.5, color=COLORS["rose"])
    fig.supxlabel("Representax / reference examples per second (log scale)", fontsize=9)
    save(fig, "framework-throughput")

    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.65), sharey=True, layout="constrained")
    for ax, dataset, label in zip(axes, DATASETS, DATASET_NAMES):
        for i, strategy in enumerate(STRATEGIES):
            values = []
            for seed in SEEDS:
                hist = e["omni"]["runs"][f"{strategy}/seed-{seed}"]["evaluation_history"]
                key = f"valid/{dataset}/cosine_ndcg@10"
                values.append(hist[-1]["metrics"][key] - hist[0]["metrics"][key])
            mean, sd = summary(values)
            color = (COLORS["sky"], COLORS["mint"], COLORS["rose"])[i]
            ax.scatter(np.arange(3)*.12 + i-.12, values, s=18, alpha=.5, color=color)
            ax.errorbar(i, mean, yerr=sd, fmt="D", color=color, capsize=3, ms=4)
        ax.axhline(0, lw=.9, color=COLORS["slate"], ls="--")
        ax.set(title=label, xticks=[0, 1, 2], xticklabels=["C", "C+L", "Full"], ylim=(-.075, .205), xlim=(-.5, 2.5))
        ax.grid(axis="y", alpha=.12)
    axes[0].set_ylabel("Change in nDCG@10")
    save(fig, "multimodal-adaptation")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), layout="constrained")
    counts = (1, 2, 4, 8)
    for seed in SEEDS:
        rows = sorted([r for r in e["scaling"]["rows"] if r["seed"] == seed], key=lambda r: r["gpus"])
        axes[0].plot(counts, [r["speedup"] for r in rows], color=COLORS["sky"], alpha=.35, marker="o", ms=3)
        axes[1].scatter(counts, [r["efficiency"]*100 for r in rows], color=COLORS["mint"], alpha=.45, s=20)
    means = [stats.mean(r["speedup"] for r in e["scaling"]["rows"] if r["gpus"] == n) for n in counts]
    axes[0].plot(counts, counts, color=COLORS["slate"], ls="--", lw=1, label="Ideal")
    axes[0].plot(counts, means, color=COLORS["sky"], marker="D", ms=4, label="Measured")
    axes[0].set(ylabel="Speedup over one GPU", ylim=(0, 8.5), yticks=[0, 2, 4, 6, 8])
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(counts, [100*s/n for n, s in zip(counts, means)], color=COLORS["mint"], marker="D", ms=4)
    axes[1].axhline(100, color=COLORS["slate"], ls="--", lw=1)
    axes[1].set(ylabel="Parallel efficiency (%)", ylim=(0, 105), yticks=[0, 25, 50, 75, 100])
    for ax in axes:
        ax.set(xticks=counts, xlabel="A100 SXM4 GPUs", xlim=(.6, 8.4))
        ax.grid(alpha=.13)
    save(fig, "strong-scaling")

    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.55), layout="constrained")
    for ax, (name, row) in zip(axes, e["learning"].items()):
        a, b = row["initial"], row["final"]
        color = COLORS["rose"] if name.startswith("Late") else COLORS["sky"]
        ax.plot([0, 1], [a["mean"], b["mean"]], color=color, lw=1)
        ax.scatter([0], [a["mean"]], color=color, facecolors="white", zorder=3)
        ax.errorbar(1, b["mean"], yerr=b["sd"], fmt="o", color=color, capsize=3)
        title = {"Dense / NanoMSMARCO": "Dense\nNanoMSMARCO", "CLIP / text-to-image": "CLIP\nText → image", "CLIP / image-to-text": "CLIP\nImage → text", "Late interaction / NanoMSMARCO": "Late interaction\nNanoMSMARCO"}[name]
        ax.set(title=title, xticks=[0, 1], xticklabels=["Initial", "Final"], ylim=(0, 1), xlim=(-.25, 1.25))
        ax.grid(axis="y", alpha=.15)
    axes[0].set_ylabel("nDCG@10")
    save(fig, "held-out-learning")


def main():
    e = load()
    (HERE / "tables").mkdir(exist_ok=True)
    (HERE / "figures").mkdir(exist_ok=True)
    framework_tables(e)
    learning_tables(e)
    scaling_table(e)
    figures(e)
    print("Generated nine tables, five figures, and per-run timing diagnostics.")


if __name__ == "__main__":
    main()
