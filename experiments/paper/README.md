# Paper experiments

Install one environment for native and reference runs:

```bash
python -m pip install -e ".[config,hf,test,performance,wandb]" \
  --group static --group parity
```

Every Representax training job enters through `representax.train.run_job`.
Reference frameworks are launched as isolated Python subprocesses so paired
runs can pin devices, revisions, logs, and output directories without shell
scripts.

Run the pinned compatibility panel on two GPUs:

```bash
python -m experiments.paper.compatibility sweep \
  --gpus 0 1 \
  --cache-directory /raid/.cache/huggingface \
  --output-root /raid/representax/paper-compatibility-v1
```

Each model directory contains the complete run, `worker.log`, and `result.json`.
The sweep root contains the aggregate `summary.json` used to publish the table.
