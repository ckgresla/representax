# Paper Working Set

- `paper.org`: canonical editable manuscript, including the approved abstract.
- `references.bib`: primary toolkit, method, model and dataset bibliography.
- `export.el`, `preamble.tex`, `Makefile`: Org -> LaTeX -> PDF and source archive.
- `abstract-variants.md`: three original voices and the author's blend decision.
- `scaling-context.md`: published scaling context and comparison boundaries.
- `figure-plan.md`: historical plan for the visual revisions.
- `capabilities.md`: capability/evidence boundaries and recorded reference pins.
- `evidence.json`: frozen compact evidence, raw metric records and source hashes.
- `update_process_reward.py`: audited, one-time replacement of five mismatched
  GPU references, retaining the original evidence in correction history.
- `update_fairness.py`: complete paired-panel replacements after the wider audit,
  verifying all ten runs and preserving superseded evidence and analysis exclusions.
- `update_transfer.py`: hash-checked full-corpus initial dense evaluations,
  retaining the original trained-seed results unchanged.
- `methods.json`: hash-checked historical settings and environments for 265 paired
  runs, plus longer-run configurations and prepared-data manifests.
- `report.py`: current renderer for nine numeric tables and five PDF/PNG figures.
- `tables/`, `figures/`, `analysis.json`: generated paper assets and timing diagnostics.
- `results.md`, `captions.md`: historical first-pass summaries; the Org manuscript
  and `report.py` own the current presentation.
- `review-notes.md`: author-review handoff and qualifications found during writing.
- `fairness-audit.md`: full paired-configuration audit, reproductions and repair matrix.

No training code or original measured artifacts are changed by this directory. The
roadmap and remaining submission tasks stay in `../todo.org`.

## Manuscript Build

Edit `paper.org`, not exported LaTeX or a separate Markdown abstract. The old
`abstract.md` and `draft.md` are superseded; their history remains in Git.
The approved abstract is preserved verbatim, with only source line wrapping.
Author notes tagged `noexport` stay out of both outputs. The complete body has no
outline placeholders. The author confirmed Chris Kerwell Gresla, Independent
Researcher. Final prose, empirical interpretation and disclosure still require
author review before submission.

On Debian/Ubuntu, install the system typesetting tools once:

```bash
sudo apt-get install --no-install-recommends emacs-nox make latexmk texlive-latex-extra texlive-fonts-recommended
```

From the repository root:

```bash
make -C paper pdf       # build/preprint/paper.pdf, named-author draft
make -C paper review    # build/review/paper.pdf, anonymous ICLR layout
make -C paper arxiv     # build/representax-arxiv.tar.gz, self-contained TeX sources
experiments/.venv/bin/python -m unittest discover -s paper -p 'test_*.py'
```

Paths above are relative to `paper/`. `make -C paper tex` exports LaTeX without
requiring a TeX installation. Each export also writes `abstract.txt` beside
`paper.tex` for the OpenReview form. Emacs runs with `-Q`, and Babel evaluation
is disabled: no personal editor configuration or executable notebook blocks.
The separate example test executes the named Python block from `paper.org`
in the installed Representax environment and checks finite loss and nonzero,
finite gradients. It neither downloads weights/data nor runs a training job.

The unmodified official ICLR 2027 style is vendored with provenance in
`vendor/README.md`. The preprint uses its named-author layout but replaces the
acceptance banner with `Preprint. Work in progress.` The review mode does not
enable the accepted-paper setting. It still needs a human anonymity and
page-limit audit; producing a review PDF does not mean a submission was made.

arXiv compiles the exported LaTeX, not Org. The source archive contains only
`paper.tex`, `paper.bbl`, `references.bib`, `preamble.tex`, the two ICLR style
files, and the five referenced PDF figures. It contains neither the manuscript
PDF nor evidence JSON, checkpoint data, notes, or absolute workspace paths.
After extraction it builds with `latexmk -pdf paper.tex`, without Emacs, Python,
the repository, or `/raid`. Bibliography processing is standard BibTeX/natbib.
See [arXiv's TeX instructions](https://info.arxiv.org/help/submit_tex.html) and
[ICLR's author guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines).

There is no mandatory arXiv visual template; this gives the preprint and planned
ICLR submission a shared layout. No upload or submission is performed by any
build command. Final author approval, public artifact publication and venue checks
remain before publication.

Rebuild figures without checkpoints, model downloads or access to `/raid`:

```bash
experiments/.venv/bin/python paper/report.py
```

The renderer requires Matplotlib and NumPy. Inter is vendored under its OFL
license in `assets/`; source: `https://github.com/rsms/inter`,
`docs/font-files/InterVariable.ttf`, downloaded 2026-09-11. The font bytes are
hashed in the evidence manifest. Colors use the author's named light palette.

The main framework comparison is `tables/framework-throughput.org`: stacked
GPU/TPU panels show absolute median examples/s and reference-relative ratios.
GPU references in this table use eager execution. The dedicated compiled-GPU
appendix subsection and `tables/gpu-rates.org` retain the five-seed dense
TorchInductor control. Bold marks the higher of the two reported medians
within each workload/hardware group, not statistical significance. The
per-seed ratio plot remains in the appendix so variability is not hidden.
Rows with unresolved objective, precision or initialization discrepancies retain
absolute rates but have no winner styling or ratio. See `fairness-audit.md`;
historical raw evidence is not silently deleted or replaced.

The evidence was captured with:

```bash
experiments/.venv/bin/python paper/build.py freeze
```

`freeze` refuses to overwrite an existing snapshot. It verifies historical
paired metric/summary hashes and recomputes rates from complete step intervals;
it also checks the A100 per-step token/mask hashes and library-source equality.
The snapshot is not a checkpoint archive. It preserves sufficient data for these
figures, while the original `/raid/representax-paper` run trees retain larger
logs, environments, source patches and model artifacts. The current lock is
identified separately from historical run environments. No schema versioning
or new experiment dispatcher is introduced.

The GPU process-reward correction is applied explicitly with
`experiments/.venv/bin/python paper/update_process_reward.py`. It verifies the
new source/data/metric hashes, observed 256-token execution shapes, batch order
against the five retained native runs and twenty warm intervals per seed.
It replaces only those five reference cells and their aggregate, preserving
the originals under `corrections`; it refuses to apply the correction twice.
The corrected TRL median is 16.9460 examples/s (arithmetic rate ratio 3.3181).
The subsequent audit found a separate scalar-head initialization issue. Both
frameworks have now been rerun with a shared scalar head: medians are 55.1760
and 16.7875 examples/s (3.2867x). This replacement was promoted using
`python paper/update_fairness.py --platform gpu --recipe process-reward`.
The original 9.005x ratio compared different padded shapes and is invalid.
The intervening padding-only panel and original panel both remain in correction
history. All five affected GPU panels are now replaced. TPU process reward is
also replaced, as is TPU outcome reward; other TPU panels remain pending validation/completion. See the
live status at the top of `fairness-audit.md`.

The paired-panel importer requires 22 finite updates and matching data manifests,
checks stored hashes and clean source provenance, and requires a final replica
hash for TPU references. A TPU-derived GPU launcher's empty native outer log is
left untouched; the importer instead records and hashes `run/metrics.jsonl`.
Corrected GPU timing excludes updates 1, 2 and 12 in both frameworks, leaving
19 identical intervals: the last exclusion removes reference checkpoint work
included in the following step timer. The importer requires matching measured
update indices within every seed pair.

The historical-methods snapshot was captured separately with
`experiments/.venv/bin/python paper/collect_methods.py`. It refuses replacement
and verifies saved summary/manifest hashes. It retains source-artifact
hashes, per-run environments and actual executed configurations. It is not part
of an ordinary rebuild, and no current configuration is substituted for an old
run. Following each promoted correction, its `collect()` function is rerun to
record both frameworks' actual settings and environments. A regression check
matches each methods record's commit and data manifest to the active evidence.
Files in the
evidence inventories retain absolute source paths for local
audit; those paths are not emitted in the PDF or LaTeX source archive.

## Claim-to-Evidence Map

| Claim | Source | Limit |
|---|---|---|
| Workload-dependent framework throughput | Exp. 10 per-run metrics and summaries, including the five corrected GPU process-reward references in `evidence.json` | Short windows; local negatives on TPU; cache, historical late interaction, short-input process reward and small-batch GPU V-JEPA qualifications |
| Useful dense learning | Exp. 11 summary, evaluation history and full-corpus initial transfer reports | One shared pretrained baseline; three trained seeds; finite-budget adaptation |
| Useful image-text learning | Exp. 13 summary | COCO fine-tuning, not CLIP pretraining from scratch |
| Multimodal adaptation tradeoffs | Exp. 14 summary with per-arm evaluation history | Different mixtures; text-to-any, not any-to-any |
| Negative late-interaction result | Exp. 12 hard-negative per-seed reports | Small held-out panel; cause unresolved |
| 3.81x / 6.87x strong scaling | Exp. 15 A100 summary and 12 raw results | One workload; DDP; short window; not FSDP capacity |

Figure readiness is not submission readiness. Venue-format/anonymity checks,
author review and public artifact publication
remain before submission. Nothing here is uploaded or submitted automatically.
