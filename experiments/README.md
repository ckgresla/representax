# Paper experiments

Each numbered directory is one serious, reproducible paper experiment. Its
flat `run.py` defines the scientific commands for native training, pinned
reference training, shared Representax evaluation, and result aggregation.
The shared `pyproject.toml` and lock define named groups for every native and
reference runtime. `setup.sh` creates the main `.venv`, the isolated
`.venv-late-interaction`, and the exact LeJEPA, stable-pretraining, and V-JEPA
source checkouts. PyLate is isolated because version 1.6.0 requires Sentence
Transformers 5.3.0 while the other paper comparisons use 5.6.1. Flat shell
scripts inside numbered directories schedule their commands for a particular
machine. Durable outputs use the same number and name under
`/raid/representax-paper/`.

Readiness checks and acceptance sweeps live separately in
`experiments/preflights/`; they are not serious experiment entrypoints.

## 01 Dense model training

Create both locked experiment environments with `uv`. The machine needs an
NVIDIA driver version 525 or newer, but it does not need Conda or a system CUDA
toolkit. JAX and PyTorch use pinned CUDA 12 wheels.
Representax itself is installed editably from the parent checkout, so this
setup does not require a published Representax wheel.

```bash
experiments/setup.sh
```

Run one paired seed directly:

```bash
python experiments/01-dense-model-training/run.py run --seed 7 --gpus 0 1
```

Or let the machine-level shell script schedule all three seeds. One GPU pair
runs serially, two pairs run in two waves, and three pairs run concurrently.

```bash
experiments/01-dense-model-training/run.sh -g 0,1,2,3,4,5
```

Set `REPRESENTAX_EXPERIMENT_ENV` to relocate the local virtual environment and
`REPRESENTAX_LATE_INTERACTION_ENV` to relocate the PyLate environment. Set
`REPRESENTAX_REFERENCE_ROOT` to relocate source checkouts and
`REPRESENTAX_PAPER_ROOT` to relocate durable experiment roots.

`aggregate` reads exactly
`runs/seed-{7,42,773}/summary.json` beneath the selected artifact root and
writes `three-seed-summary.json` beside `runs/`. It does not launch training.

The frozen defaults are MPNet, MS MARCO, exact GradCache at global batch 2,048,
seeds 7/42/773, and shared NanoMSMARCO evaluation. Override the artifact root,
checkpoint, or prepared data path only with the corresponding global options.

## 01.1 Optimized dense model training

This systems ablation preserves experiment 01's scientific contract while
tuning exact GradCache chunking and loader execution independently for both
frameworks. It also tests `torch.compile` for Sentence Transformers. Run the
bounded sweep on one GPU per framework, then run the selected 256-update pair:

```bash
experiments/.venv/bin/python experiments/01.1-optimized-dense-model-training/run.py tune --gpus 0 1
experiments/.venv/bin/python experiments/01.1-optimized-dense-model-training/run.py run --gpus 0 1
```

The sweep, selected settings, full logs, exported models, and shared evaluation
are written beneath `/raid/representax-paper/01.1-optimized-dense-model-training/`.
