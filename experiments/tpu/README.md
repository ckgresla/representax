# TPU experiment environments

`setup.sh` creates two isolated, locked environments:

- `.venv-jax`: Python 3.13 with the local Representax checkout and JAX TPU runtime.
- `.venv-torch-xla`: Python 3.12 with Sentence Transformers, PyTorch 2.9,
  and PyTorch/XLA 2.9.

PyTorch/XLA's pinned `libtpu` wheel is unavailable for Python 3.13, so these
runtimes cannot share one environment.

```bash
./experiments/tpu/setup.sh
```

Run each multi-host command concurrently on every TPU VM worker. JAX discovers
the Cloud TPU coordinator and process indices when `jax.distributed.initialize`
runs. PyTorch/XLA performs the corresponding setup through PJRT and
`torch_xla.launch`.

```bash
experiments/tpu/.venv-jax/bin/python \
  experiments/preflights/tpu_multihost.py topology --distributed

experiments/tpu/.venv-jax/bin/python \
  experiments/preflights/tpu_multihost.py jax-dense

PJRT_DEVICE=TPU experiments/tpu/.venv-torch-xla/bin/python \
  experiments/preflights/tpu_multihost.py torch-dense
```
