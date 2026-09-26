"""Build the manuscript's tables and figures from immutable, local evidence."""

from __future__ import annotations

import json
import math
import statistics as stats
from pathlib import Path

from build import COLORS, LABELS, PANEL_SEEDS, SEEDS, paired_rates, warm_rows

HERE = Path(__file__).resolve().parent
STRATEGIES = ("connectors", "connectors-lora", "full")
STRATEGY_NAMES = ("Connectors", "Connectors + LoRA", "Full fine-tuning")
DATASETS = ("flickr30k", "audiocaps", "msrvtt", "nanomsmarco")
DATASET_NAMES = ("Flickr30k", "AudioCaps", "MSR-VTT", "NanoMSMARCO")
DESIGN_NAMES = {
    "native_coarse": "Representax / coarse buckets",
    "native_fine": "Representax / fine buckets",
    "eager_native": "ST eager / dynamic padding",
    "eager_fixed": "ST eager / fixed padding",
    "inductor_native": "ST Inductor / dynamic padding",
    "inductor_fixed": "ST Inductor / fixed padding",
    "native_rematerialized": "Representax / rematerialized",
    "native_optimized": "Representax / optimized, cold",
    "native_cached": "Representax / optimized, cached",
}
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


def load_design():
    return json.loads((HERE / "design-evidence.json").read_text())


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


def table(name, caption, columns, headers, rows, *, tabcolsep=4, arraystretch=None,
          placement="tbp"):
    body = "\n".join(" & ".join(row) + r" \\" for row in rows)
    text = (
        f"#+begin_export latex\n\\begin{{table}}[{placement}]\n\\centering\\small\n"
        + rf"\setlength{{\tabcolsep}}{{{tabcolsep}pt}}" + "\n"
        + (rf"\renewcommand{{\arraystretch}}{{{arraystretch}}}" + "\n"
           if arraystretch is not None else "")
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
        "Warm throughput: median examples/s across five seeds. Bold identifies "
        "the higher reported median within a matched pair, not significance. "
        "$R$ denotes Representax; ratios divide its median by the reference's. "
        "GPU references use eager execution; the dense TorchInductor control "
        r"is in Appendix \ref{sec:compiled-reference}. "
        "Ratios compare frameworks within a hardware panel, not GPU and TPU "
        "performance across different allocations and batches. Lettered protocol "
        r"qualifications and per-seed variation appear in Appendix \ref{sec:paired-methods}.",
        "lrrr",
        ["Workload", r"\shortstack{Representax\\ex/s}",
         r"\shortstack{Reference\\ex/s}", r"$R/\mathrm{Ref.}$"],
        rows,
        tabcolsep=10,
        arraystretch=1.12,
        placement="H",
    )


def training_losses(run):
    """Read optimizer-update losses, retaining first-use and checkpoint intervals."""
    updates = [row for row in run["metrics"] if row.get("event") == "training_step"]
    points = [(row["iteration"], row["metrics"]["train/loss"])
              for row in updates if "train/loss" in row.get("metrics", {})]
    # The corrected GPU outcome reference stores losses beside its timing rows.
    if not points and run.get("loss_history"):
        points = [(row["iteration"], loss)
                  for row, loss in zip(updates, run["loss_history"], strict=True)]
    if not points or len(points) != len(updates):
        raise ValueError("Incomplete optimizer-update loss history")
    steps = [step for step, _ in points]
    if steps != sorted(set(steps)):
        raise ValueError("Loss updates must be unique and ordered")
    if not all(math.isfinite(loss) for _, loss in points):
        raise ValueError("Non-finite training loss")
    return points


def loss_figures(e):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import font_manager
    from matplotlib.lines import Line2D

    font_path = HERE / "assets/InterVariable.ttf"
    font_manager.fontManager.addfont(font_path)
    font = font_manager.FontProperties(fname=font_path).get_name()
    styles = (("representax", "Representax", COLORS["sky"], "-"),
              ("reference", "Reference", COLORS["rose"], "--"))
    with plt.rc_context({
        "font.family": font, "font.size": 9, "axes.titlesize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#d7dbe0", "axes.linewidth": .6,
        "grid.color": "#d7dbe0", "grid.linewidth": .5,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "pdf.fonttype": 42,
    }):
        for platform, name in (("gpu-rtx4090", "loss-gpu"), ("tpu-v5e-16", "loss-tpu")):
            runs = e["panels"][platform]["runs"]
            panels = [(recipe, LABELS[recipe], runs) for recipe in LABELS]
            if platform == "gpu-rtx4090":
                panels.append(("dense-retrieval", "Dense / Inductor reference",
                               [r for r in runs if r["framework"] == "representax"]
                               + e["panels"]["gpu-rtx4090-torchinductor"]["runs"]))
            fig, axes = plt.subplots(5, 3, figsize=(7.2, 8.2), layout="constrained")
            fig.get_layout_engine().set(rect=(0, 0, 1, .955))
            for ax, (recipe, title, source) in zip(axes.flat, panels):
                for framework, _, color, linestyle in styles:
                    selected = [r for r in source
                                if r["recipe"] == recipe and r["framework"] == framework]
                    assert sorted(r["seed"] for r in selected) == sorted(PANEL_SEEDS)
                    histories = [training_losses(run) for run in selected]
                    steps = [step for step, _ in histories[0]]
                    assert all([step for step, _ in h] == steps for h in histories)
                    values = np.array([[loss for _, loss in h] for h in histories])
                    for values_for_run in values:
                        ax.plot(steps, values_for_run, color=color, linestyle=linestyle,
                                alpha=.25, linewidth=.6)
                    ax.plot(steps, np.median(values, axis=0), color=color,
                            linestyle=linestyle, linewidth=1.4)
                ax.set(title=title, xlim=(1, 22), xticks=[1, 11, 22])
                ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))
                ax.grid(axis="y")
                ax.tick_params(length=2)
            for ax in list(axes.flat)[len(panels):]:
                ax.set_visible(False)
            fig.legend([Line2D([], [], color=c, linestyle=s, linewidth=1.4)
                        for _, _, c, s in styles], [label for _, label, _, _ in styles],
                       loc="upper center", ncol=2, frameon=False)
            fig.supxlabel("Optimizer update", fontsize=10)
            fig.supylabel("Training loss", fontsize=10)
            for ext in ("pdf", "png"):
                fig.savefig(HERE / "figures" / f"{name}.{ext}", bbox_inches="tight",
                            facecolor="white", dpi=180,
                            metadata={"CreationDate": None} if ext == "pdf" else None)
            plt.close(fig)


def learning_tables(e):
    labels = {
        "Dense / NanoMSMARCO": ("Dense", "NanoMSMARCO"),
        "CLIP / text-to-image": ("CLIP", r"Flickr30k text $\to$ image"),
        "CLIP / image-to-text": ("CLIP", r"Flickr30k image $\to$ text"),
        "Late interaction / NanoMSMARCO": ("Late interaction", "NanoMSMARCO"),
    }
    def learning_cells(initial, final, sd):
        # Sample SD as a superscript keeps the column narrow; gain is final minus initial.
        return [f"{initial:.4f}", rf"${final:.4f}^{{\pm {sd:.4f}}}$", f"{final - initial:+.4f}"]

    rows = []
    for key, row in e["learning"].items():
        if key.startswith("Late"):
            continue
        a, b = row["initial"], row["final"]
        rows.append([*labels[key], *learning_cells(a["mean"], b["mean"], b["sd"])])
    for name, row in e["transfer_final_only"].items():
        initial = e["transfer_initial"][name]["metrics"][f"valid/{name}/cosine_ndcg@10"]
        rows.append(["Dense", {"trec-dl-2019": "TREC DL 2019", "natural-questions": "Natural Questions"}[name], *learning_cells(initial, row["mean"], row["sd"])])
    table("learning", "Held-out nDCG@10 before and after native adaptation. Final scores are means across three seeds, with the sample SD as a superscript; gain is final minus initial. The pretrained transfer baseline is shared across seeds; initial and final evaluations use the same complete corpora.", "llrrr", ["Model", "Evaluation", "Initial", "Final", "Gain"], rows, placement="H")
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


def design_table(e):
    rows = []
    for key, label in DESIGN_NAMES.items():
        r = e["rows"][key]["report"]
        startup = r["compilation_and_first_use_seconds"]
        rows.append([
            label, str(r["steady_state_step_count"]),
            f'{r["steady_state_examples_per_second"]:.2f}',
            "---" if startup is None else f"{startup:.2f}",
        ])
    table(
        "design-diagnostics",
        "Single-seed ModernBERT execution diagnostics. Each process runs 30 "
        "updates; the recorded warm subset contains 13--15 updates. Rates use "
        "those subsets, not equal-length wall-clock trials. ST denotes Sentence "
        "Transformers. First-use includes compilation or cache loading plus "
        "execution; a dash means unrecorded, not zero. The optimized rows combine "
        "several changes and are not a single-factor ablation.",
        "lrrr", ["Configuration", "Warm updates", "Examples/s", "First-use (s)"],
        rows,
    )


def figures(e):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import font_manager
    font_manager.fontManager.addfont(HERE / "assets/InterVariable.ttf")
    font = font_manager.FontProperties(fname=HERE / "assets/InterVariable.ttf").get_name()
    plt.rcParams.update({
        "font.family": font, "font.size": 10.5, "axes.titlesize": 10.5,
        "axes.labelsize": 10.5, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
        "text.color": "#24272b", "axes.labelcolor": "#24272b",
        "axes.edgecolor": "#d7dbe0", "axes.linewidth": .7,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": COLORS["slate"], "ytick.color": COLORS["slate"],
        "grid.color": "#d7dbe0", "grid.linewidth": .6,
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })

    def save(fig, name):
        for ext in ("pdf", "png"):
            fig.savefig(HERE / "figures" / f"{name}.{ext}", bbox_inches="tight", facecolor="white", dpi=180, metadata={"CreationDate": None} if ext == "pdf" else None)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 2.95))
    ax.set(xlim=(0, 10), ylim=(0, 4.45))
    ax.axis("off")
    ax.text(.05, 4.18, "Scientific choices")
    for x, title, lines, color in (
        (.05, "Data", "Sources + mixtures\nProcessing", COLORS["mint"]),
        (2.62, "Model", "Encoders + adapters\nConnectors", COLORS["sky"]),
        (5.19, "Task", "Learning objective\nLoss modifiers", COLORS["periwinkle"]),
        (7.76, "Update", "Optimizer + schedule\nTarget transitions", COLORS["apricot"]),
    ):
        ax.plot([x, x + 2.10], [3.84, 3.84], color=color, lw=2.5)
        ax.text(x, 3.55, title, va="top")
        ax.text(x, 2.99, lines, fontsize=9.5, va="top", linespacing=1.5)
    for start in (2.15, 4.72, 7.29):
        ax.annotate("", xy=(start + .40, 3.38), xytext=(start, 3.38),
                    arrowprops={"arrowstyle": "->", "color": COLORS["slate"], "lw": .9})
    ax.plot([.05, 9.86], [1.98, 1.98], color="#d7dbe0", lw=.8)
    ax.text(.05, 1.57, "Shared execution")
    ax.text(3.05, 1.57, "Chunking / precision / sharding / checkpoint + resume", fontsize=9.5)
    ax.plot([.05, 9.86], [1.10, 1.10], color="#d7dbe0", lw=.8)
    ax.text(.05, .66, "Evaluation", color=COLORS["rose"])
    ax.text(3.05, .66, "Model + processor → corpus accumulation → metrics", fontsize=9.5)
    save(fig, "architecture")

    recipes = list(LABELS)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 5.0), sharey=True, layout="constrained")
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
        ax.set(xscale="log", xlim=(.45, 16), title=title)
        ax.set_xticks([.5, 1, 2, 4, 8, 16], ["0.5", "1", "2", "4", "8", "16"])
        ax.grid(axis="x")
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
    axes[0].set_yticks(range(len(recipes)), [NAMES[r] for r in recipes])
    axes[0].invert_yaxis()
    # The compiled dense control uses the same native runs, not a new native trial.
    compiled = {r["seed"]: r["examples_per_second"] for r in e["panels"]["gpu-rtx4090-torchinductor"]["runs"]}
    native = {r["seed"]: r["examples_per_second"] for r in e["panels"]["gpu-rtx4090"]["runs"] if r["recipe"] == "dense-retrieval" and r["framework"] == "representax"}
    ratio = stats.median(native.values()) / stats.median(compiled.values())
    axes[0].scatter(ratio, -.36, marker="s", s=22, color=COLORS["rose"], zorder=4)
    axes[0].annotate("Inductor", xy=(ratio, -.36), xytext=(1.65, -.36), va="center", fontsize=7.5, color=COLORS["rose"])
    fig.supxlabel("Representax / reference examples per second (log scale)", fontsize=10.5)
    save(fig, "framework-throughput")

    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.45), sharey=True, layout="constrained")
    directions = ("Text → image", "Text → audio", "Text → video", "Text → text")
    for ax, dataset, label, direction in zip(axes, DATASETS, DATASET_NAMES, directions):
        for i, strategy in enumerate(STRATEGIES):
            values = []
            for seed in SEEDS:
                hist = e["omni"]["runs"][f"{strategy}/seed-{seed}"]["evaluation_history"]
                key = f"valid/{dataset}/cosine_ndcg@10"
                values.append(hist[-1]["metrics"][key] - hist[0]["metrics"][key])
            mean, sd = summary(values)
            color = (COLORS["sky"], COLORS["mint"], COLORS["rose"])[i]
            ax.scatter(values, i + np.linspace(-.14, .14, len(values)),
                       s=18, alpha=.55, color=color, edgecolors="none")
            ax.errorbar(mean, i, xerr=sd, fmt="D", color=color, capsize=3, ms=4.5)
        ax.axvline(0, lw=.9, color=COLORS["slate"], ls="--")
        ax.set(title=f"{direction}\n{label}", ylim=(2.48, -.48), xlim=(-.1, .22))
        ax.set_xticks([-.1, 0, .1, .2], ["−0.1", "0", "+0.1", "+0.2"])
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.grid(axis="x")
    axes[0].set_yticks(range(3), ["Connectors", "+ LoRA", "Full tuning"])
    fig.supxlabel("Change in nDCG@10 from each recipe's initial checkpoint", fontsize=10.5)
    save(fig, "multimodal-adaptation")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), layout="constrained")
    counts = (1, 2, 4, 8)
    for seed in SEEDS:
        rows = sorted([r for r in e["scaling"]["rows"] if r["seed"] == seed], key=lambda r: r["gpus"])
        axes[0].plot(counts, [r["speedup"] for r in rows], color=COLORS["sky"], alpha=.35, marker="o", ms=3)
        axes[1].scatter(counts, [r["efficiency"]*100 for r in rows], color=COLORS["mint"], alpha=.45, s=20)
    means = [stats.mean(r["speedup"] for r in e["scaling"]["rows"] if r["gpus"] == n) for n in counts]
    axes[0].plot(counts, counts, color=COLORS["slate"], ls="--", lw=1)
    axes[0].plot(counts, means, color=COLORS["sky"], marker="D", ms=4)
    axes[0].set(ylabel="Speedup over one GPU", ylim=(0, 8.5), yticks=[0, 2, 4, 6, 8])
    axes[0].text(4.7, 7.1, "Ideal", color=COLORS["slate"])
    axes[0].annotate(f"{means[-1]:.2f}×", (8, means[-1]), xytext=(-2, -18),
                     textcoords="offset points", ha="right", color=COLORS["sky"])
    axes[1].plot(counts, [100*s/n for n, s in zip(counts, means)], color=COLORS["mint"], marker="D", ms=4)
    axes[1].axhline(100, color=COLORS["slate"], ls="--", lw=1)
    axes[1].set(ylabel="Parallel efficiency (%)", ylim=(0, 105), yticks=[0, 25, 50, 75, 100])
    axes[1].annotate(f"{100 * means[-1] / 8:.1f}%", (8, 100 * means[-1] / 8),
                     xytext=(-2, -18), textcoords="offset points", ha="right",
                     color=COLORS["mint"])
    for ax in axes:
        ax.set(xticks=counts, xlabel="A100 SXM4 GPUs", xlim=(.6, 8.4))
        ax.grid(axis="y")
    save(fig, "strong-scaling")

    fig, ax = plt.subplots(figsize=(7.2, 2.9), layout="constrained")
    for i, (name, row) in enumerate(e["learning"].items()):
        a, b = row["initial"], row["final"]
        color = COLORS["rose"] if name.startswith("Late") else COLORS["sky"]
        ax.plot([a["mean"], b["mean"]], [i, i], color=color, lw=1.3)
        ax.scatter(a["mean"], i, facecolors="white", edgecolors=COLORS["slate"], zorder=4, s=24)
        ax.errorbar(b["mean"], i, xerr=b["sd"], fmt="D", color=color, capsize=3, ms=4)
        ax.text(1.08, i, f'{a["mean"]:.4f}', va="center", ha="center", fontsize=9.5)
        ax.text(1.32, i, f'{b["mean"]:.4f}', va="center", ha="center", fontsize=9.5, color=color)
    ax.text(1.08, -.60, "Initial", ha="center", fontsize=9.5)
    ax.text(1.32, -.60, "Final", ha="center", fontsize=9.5)
    ax.set(xlim=(0, 1.45), ylim=(3.6, -.90), xlabel="nDCG@10",
           xticks=np.linspace(0, 1, 6), yticks=range(4),
           yticklabels=["Dense / NanoMSMARCO", "CLIP / text → image",
                        "CLIP / image → text", "Late / NanoMSMARCO"])
    ax.spines["bottom"].set_bounds(0, 1)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x")
    save(fig, "held-out-learning")

    design = load_design()["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), layout="constrained",
                             gridspec_kw={"width_ratios": [1.2, 1]})
    keys = ("native_coarse", "native_fine", "eager_fixed", "eager_native",
            "inductor_fixed", "inductor_native")
    for i, key in enumerate(keys):
        value = design[key]["report"]["steady_state_examples_per_second"]
        color = COLORS[("sky", "sky", "mint", "mint", "periwinkle", "periwinkle")[i]]
        axes[0].plot([0, value], [i, i], color=color, alpha=.35, lw=1.4)
        axes[0].scatter(value, i, color=color, marker="D" if i % 2 else "o", s=25)
        axes[0].annotate(f"{value:.0f}", (value, i), xytext=(5, 0),
                         textcoords="offset points", va="center", fontsize=9)
    axes[0].set(title="Padding and warm throughput", xlim=(0, 810), ylim=(5.6, -.6),
                xlabel="Examples/s", xticks=[0, 200, 400, 600, 800], yticks=range(6),
                yticklabels=["Native / coarse", "Native / fine", "Eager / fixed",
                             "Eager / dynamic", "Inductor / fixed", "Inductor / dynamic"])
    for i, key in enumerate(("native_optimized", "native_cached")):
        r = design[key]["report"]
        seconds = r["compilation_and_first_use_seconds"]
        color = COLORS["sky" if i == 0 else "mint"]
        axes[1].plot([0, seconds], [i, i], color=color, alpha=.35, lw=1.4)
        axes[1].scatter(seconds, i, color=color, s=30, marker="D")
        axes[1].annotate(f"{seconds:.1f} s", (seconds, i), xytext=(0, 13),
                         textcoords="offset points", ha="center", fontsize=9.5, color=color)
        axes[1].text(0, i + .25, f'{r["steady_state_examples_per_second"]:.1f} examples/s warm',
                     fontsize=9, color=COLORS["slate"])
    axes[1].set(title="Persistent-cache replay", xlim=(-5, 350), ylim=(1.6, -.65),
                xlabel="Compilation + first execution (s)", xticks=[0, 100, 200, 300],
                yticks=[0, 1], yticklabels=["Cold", "Cached"])
    for ax in axes:
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x")
    save(fig, "design-diagnostics")


def main():
    e = load()
    (HERE / "tables").mkdir(exist_ok=True)
    (HERE / "figures").mkdir(exist_ok=True)
    framework_tables(e)
    learning_tables(e)
    scaling_table(e)
    design_table(load_design())
    figures(e)
    loss_figures(e)
    print("Generated ten tables, eight figures, and per-run timing diagnostics.")


if __name__ == "__main__":
    main()
