# Preflights and acceptance checks

These modules establish lifecycle readiness, compatibility, and bounded systems
behavior before a workload becomes a numbered serious experiment. They may
launch native and reference workers, but they are not the paper's public
experiment commands. Serious runs live in the numbered directories directly
under `experiments/`.

Create the shared locked environments and pinned source checkouts:

```bash
experiments/setup.sh
```

Native, Sentence Transformers, TRL, LeJEPA, stable-pretraining, and V-JEPA use
`experiments/.venv`. PyLate uses `experiments/.venv-late-interaction` because
its pinned release requires an older Sentence Transformers and Transformers
stack. Both environments come from `experiments/pyproject.toml` and the same
`experiments/uv.lock`.

Every Representax training job enters through `representax.train.run_job`.
Reference frameworks are launched as isolated Python subprocesses so paired
runs can pin devices, revisions, logs, and output directories without shell
scripts.

Run the pinned compatibility panel on two GPUs:

```bash
python -m experiments.preflights.compatibility sweep \
  --gpus 0 1 \
  --cache-directory /raid/.cache/huggingface \
  --output-root /raid/representax/paper-compatibility-v1
```

Each model directory contains the complete run, `worker.log`, and `result.json`.
The sweep root contains the aggregate `summary.json` used to publish the table.

## Free Colab TPU acceptance

Open
[colab_tpu.ipynb](https://colab.research.google.com/github/ckgresla/representax/blob/codex/colab-tpu-acceptance/experiments/preflights/colab_tpu.ipynb),
select the current `TPU v2` runtime, leave the prefilled revision or replace it
with an immutable commit, and run every cell. The notebook installs the repository with its `tpu` extra and
invokes one repository-owned command:

```bash
python -m experiments.preflights.tpu \
  --output /content/representax-tpu-acceptance \
  --steps 20
```

The matrix covers FP32 and BF16 training, gradient accumulation, both GradCache
schedules, DDP, FSDP, midpoint checkpoint/resume, evaluation, inference export,
exact reload, and `logging.accelerator` through LibTPU. It emits a single archive
containing `summary.json` and every canonical run artifact. Free Colab permits
interactive notebook execution; it must not be converted into an SSH or remote
worker service.
