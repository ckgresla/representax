# TPU experiment environments

`setup.sh` installs the host `ffmpeg` tools used for video preparation and
creates three isolated, locked environments:

- `.venv-jax`: Python 3.13 with the local Representax checkout and JAX TPU runtime.
- `.venv-torch-xla`: system Python 3.10 with Sentence Transformers, TRL,
  Meta V-JEPA dependencies, the CPU PyTorch 2.9 wheel, and PyTorch/XLA 2.9.
- `.venv-torch-xla-late`: the same PyTorch/XLA base with PyLate's required
  Sentence Transformers release. It is separate because those releases conflict.

PyTorch/XLA's pinned `libtpu` wheel is unavailable for Python 3.13. Its wheel
also requires the system Python shared library, so the TPU VM's Python 3.10 is
used rather than a managed Python build. The CPU PyTorch wheel avoids installing
unused CUDA libraries.

```bash
./experiments/tpu/setup.sh
```

Run each multi-host command concurrently on every TPU VM worker. JAX discovers
the Cloud TPU coordinator and process indices when `jax.distributed.initialize`
runs. PyTorch/XLA performs the corresponding setup through PJRT and
`torch_xla.launch`.

```bash
experiments/tpu/.venv-jax/bin/python \
  -m experiments.preflights.tpu_multihost \
  topology --distributed --scope both

experiments/tpu/.venv-jax/bin/python \
  -m experiments.preflights.tpu_multihost jax-dense

PJRT_DEVICE=TPU experiments/tpu/.venv-torch-xla/bin/python \
  -m experiments.preflights.tpu_multihost torch-dense

PJRT_DEVICE=TPU experiments/tpu/.venv-torch-xla/bin/python \
  -m experiments.preflights.tpu_multihost sentence-transformers-dense \
  --output /tmp/representax-sentence-transformers-tpu
```

The final command runs the official `SentenceTransformerTrainer` with a pinned
tiny BERT checkpoint and `MultipleNegativesRankingLoss` on the entire TPU slice.
It is an integration canary, not a paper throughput measurement.
