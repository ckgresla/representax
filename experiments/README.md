# Paper experiments

Each numbered directory is one reproducible paper experiment. Its `run.py` is
the public command for native training, pinned reference training, shared
Representax evaluation, and result aggregation. Durable outputs use the same
number and name under `/raid/representax-paper/`.

## 01 Dense model training

Run one paired seed:

```bash
python experiments/01-dense-model-training/run.py pair --seed 7 --gpus 0 1
```

Run all three paired seeds concurrently on six GPUs and aggregate them:

```bash
python experiments/01-dense-model-training/run.py campaign \
  --gpus 0 1 2 3 4 5
```

The frozen defaults are MPNet, MS MARCO, exact GradCache at global batch 2,048,
seeds 7/42/773, and shared NanoMSMARCO evaluation. Override the artifact root,
checkpoint, or prepared data path only with the corresponding global options.
