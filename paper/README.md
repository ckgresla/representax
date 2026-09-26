# Paper Working Set

- `paper.org`: canonical editable manuscript, including the approved abstract.
- `references.bib`: primary toolkit, method, model and dataset bibliography.
- `export.el`, `preamble.tex`, `Makefile`: Org -> LaTeX -> PDF and source archive.
- `requirements-render.txt`: pinned NumPy/Matplotlib renderer dependencies,
  independent of the training environments.
- `submission-checklist.md`, `check_submission.py`: current venue/release checklist
  and mechanical checks run by every review-PDF build.
- `abstract-variants.md`: three original voices and the author's blend decision.
- `scaling-context.md`: published scaling context and comparison boundaries.
- `figure-plan.md`: historical plan for the visual revisions.
- `revision-plan.md`: scope and progress of the current interior/figure revision.
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
- `design-evidence.json`, `collect_design.py`: selected existing Experiment 09
  diagnostics, with source hashes and measurement windows; no new training.
- `report.py`: current renderer for ten numeric tables and eight PDF/PNG figures.
- `tables/`, `figures/`, `analysis.json`: generated paper assets and timing diagnostics.
- `results.md`, `captions.md`: historical first-pass summaries; the Org manuscript
  and `report.py` own the current presentation.
- `review-notes.md`: author-review handoff and qualifications found during writing.
- `fairness-audit.md`: full paired-configuration audit, reproductions and repair matrix.

No training code or original measured artifacts are changed by this directory. The
roadmap and remaining submission tasks stay in `../todo.org`.

## Training and Audit Closeout

The agreed training and rerun campaigns are complete as of 2026-09-17. All 100
corrected runs are included in the 265 active paired-run records. Native
adaptation, full-corpus initial transfer evaluations and twelve A100 scaling
runs are also present. No additional training is required for the current
manuscript scope; an FSDP capacity demonstration remains future work.

The manuscript uses eight current figures and fifteen tables: ten generated
numeric tables plus five methods/capability/research-map tables. `report.py` reproduces the
current assets without changes. The older `cross-accelerator` figure and
`results.md`, `captions.md`, and `figure-plan.md` are historical working material,
not current paper results or release inputs. Use `paper.org` and its referenced
tables/figures for review.

Both TPU allocations and their queues are deleted; the retained asset bucket
is intentional and has a post-preprint cleanup item in `../todo.org`. Closing
the training work does not delete checkpoints, logs, manifests or run history.
Final author review, venue-specific checks, reviewed artifact publication and
submission remain open.

The editorial revisions address the author's first 17 PDF comments and the
next 21 introduction comments. They add the learning-signal research map in
the appendices, develop the multimodal motivation, and replace undated software
citations with upstream recommended references. See `review-notes.md` for
the dated review history. The September 23 interior pass preserves the approved
abstract, introduction, and conclusion while following the nine-section outline.
The September 25 pass preserves the now-approved Sections 2 and 3 as well,
revises Sections 4--8, and adds the selected logo to the named preprint. The
anonymous submission omits both the logo and its explanatory footnote. The
approved Section 4 revision has now been landed with workload bullets, the
original absolute-throughput table, and appendix loss trajectories. Design
Analysis is now Appendix F, and the standalone capabilities/limitations section
is removed: qualifications accompany the experiments, capability scope remains
in Appendix G, and reproduction details are in the statement and Appendix E.
Section 5 now gives dense retrieval, CLIP alignment, and multimodal adaptation
separate scientific treatments; the fourth workload family's late-interaction
regression and follow-up questions are in Appendix B.2. The expanded draft has
eleven main-text pages (34 total) in the unmodified official style, so the
nine-page submission guard currently fails. The author has deferred length
reduction until the content is settled. Final review of Sections 5--6 remains
open; the conclusion is now Section 7.

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
sudo apt-get install --no-install-recommends emacs-nox make latexmk texlive-latex-extra texlive-fonts-recommended poppler-utils
```

From the repository root:

```bash
make -C paper figures pdf review arxiv
make -C paper check
```

The figure target additionally requires `uv`; it runs the renderer in an
isolated Python 3.13 environment with the versions in `requirements-render.txt`.
The first invocation may download Python and renderer dependencies. The renderer
itself uses local evidence only, without model loading, data downloads, or `/raid`.
Without `figures`, the PDF targets use the checked-in figures directly.
`make check` runs the source/evidence checks on CPU; the executable API example
is skipped unless JAX, Equinox, and Representax are installed. For the full
suite including that example, use the prepared experiment environment:

```bash
experiments/.venv/bin/python -m unittest discover -s paper -p 'test_*.py'
```

Outputs are `paper/build/preprint/paper.pdf`, `paper/build/review/paper.pdf`, and
`paper/build/representax-arxiv.tar.gz`. Review checks require each figure to share
a page with its labeled discussion, as well as the nine-page and identity checks.

`make -C paper tex` exports LaTeX without
requiring a TeX installation. Each export also writes `abstract.txt` beside
`paper.tex` for the OpenReview form. Emacs runs with `-Q`, and Babel evaluation
is disabled: no personal editor configuration or executable notebook blocks.
The separate example test executes the named Python block from `paper.org`
in the installed Representax environment and checks finite loss and nonzero,
finite gradients. It neither downloads weights/data nor runs a training job.

The unmodified official ICLR 2027 style is vendored with provenance in
`vendor/README.md`. The preprint uses its named-author layout but replaces the
acceptance banner with `Preprint. Work in progress.` The review mode does not
enable the accepted-paper setting. Each review build checks the nine-page
main-text limit, official style and known identity leaks, including metadata and
links. A human layout/anonymity audit is still required; producing a review PDF
does not mean a submission was made. See [the checklist](submission-checklist.md).

arXiv compiles the exported LaTeX, not Org. The source archive contains only
`paper.tex`, `paper.bbl`, `references.bib`, `preamble.tex`, the two ICLR style
files, the eight referenced PDF figures, and the selected vector logo. It contains neither the manuscript
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
make -C paper figures
```

The renderer requires Matplotlib and NumPy. Inter is vendored under its OFL
license in `assets/`; source: `https://github.com/rsms/inter`,
`docs/font-files/InterVariable.ttf`, downloaded 2026-09-11. The font bytes are
hashed in the evidence manifest. Colors use the author's named light palette.
Run the renderer to completion before exporting either PDF so an export cannot
copy a figure while it is being written. The eight current figures are the research
interfaces, paired throughput, multimodal adaptation, strong scaling, held-out
learning, design diagnostics, and the GPU/TPU training-loss panels. Individual seeds remain visible; whiskers
denote sample SD, not confidence intervals. Detailed narrative tables use
ragged-right columns to avoid excessive inter-word spacing.

`design-evidence.json` is a separate, small capture of nine existing Experiment
09 reports. It neither replaces nor changes the main frozen evidence. Its
single-seed 30-update diagnostics are explicitly separated from the five-seed
framework panel and from quality evidence. Capturing another snapshot requires
an explicit new `--output` path; the collector refuses to overwrite an existing
file. Ordinary report generation uses the checked-in snapshot, not `/raid`.

The main framework comparison is `tables/framework-throughput.org`: stacked
GPU/TPU panels show absolute median examples/s and reference-relative ratios.
GPU references in this table use eager execution. The dedicated compiled-GPU
appendix subsection and `tables/gpu-rates.org` retain the five-seed dense
TorchInductor control. Bold marks the higher of the two reported medians
within each workload/hardware group, not statistical significance. The
per-seed ratio plot remains in the appendix so variability is not hidden.
The appendix loss plots show every recorded optimizer update for the five seeded
runs, with thin individual traces and thick pointwise medians, without smoothing
or normalization. They include timing-excluded updates and reuse the native dense
runs for the additional compiled-reference panel. The main table retains all 26
absolute-rate rows and uses wider columns and taller rows for readability.
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
history. All five affected panels on both GPU and TPU are now replaced: 100
corrected runs. The final TPU video panel reaches 88.6636 versus 7.3792
examples/s (12.0153x). The last three references used a replacement 16-chip v5e
allocation with unchanged source, data and locked training environments; its
location is recorded in the collection receipts and manuscript. Both allocations
have been deleted. See `fairness-audit.md` for validation and shutdown records.

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
| Workload-dependent framework throughput | Exp. 10 per-run metrics and summaries, including all 100 corrected GPU/TPU runs in `evidence.json` | Short windows; declared local/global negative pools; cache, late-interaction evaluation, short-input process reward and small-batch GPU V-JEPA qualifications |
| Useful dense learning | Exp. 11 summary, evaluation history and full-corpus initial transfer reports | One shared pretrained baseline; three trained seeds; finite-budget adaptation |
| Useful image-text learning | Exp. 13 summary | COCO fine-tuning, not CLIP pretraining from scratch |
| Multimodal adaptation tradeoffs | Exp. 14 summary with per-arm evaluation history | Different mixtures; text-to-any, not any-to-any |
| Negative late-interaction result | Exp. 12 hard-negative per-seed reports | Small held-out panel; cause unresolved |
| 3.81x / 6.87x strong scaling | Exp. 15 A100 summary and 12 raw results | One workload; DDP; short window; not FSDP capacity |
| Padding, replay, and cache costs | Nine Exp. 09 reports captured in `design-evidence.json` | One seed, 30 updates, 13--15 warm intervals; duplicate-heavy data; cumulative optimization is not a single-factor ablation |

Figure readiness is not submission readiness. Venue-format/anonymity checks,
author review and public artifact publication
remain before submission. Nothing here is uploaded or submitted automatically.
