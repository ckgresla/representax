# TPU experiment environments

`setup.sh` creates two isolated, locked environments:

- `.venv-jax`: Python 3.13 with the local Representax checkout and JAX TPU runtime.
- `.venv-torch-xla`: system Python 3.10 with Sentence Transformers, the CPU
  PyTorch 2.9 wheel, and PyTorch/XLA 2.9.

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
```
