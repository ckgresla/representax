# Contributing

Representax is alpha. Before adding a new model family or task, open an
issue describing its upstream reference, data contract, and intended parity
tests.

Install the lightweight development environment with:

```bash
python -m pip install -e ".[test]"
pytest
```

Model integrations must include deterministic forward parity. Trainable model
integrations must additionally cover gradients and one optimizer update using
the optional parity environment:

```bash
python -m pip install -e ".[test]" --group parity
pytest -m parity
```

Distributed and performance evidence lives in dedicated test lanes. Generated
profiles, checkpoints, benchmark results, and downloaded datasets are not
committed to the repository.

## Imports

Use `import representax as rpx` only when demonstrating the cohesive public
facade. Library code, tests, and focused examples should import names directly
from their owning module, for example:

```python
from representax.train import build_train_step, make_train_state, run_training
```

Do not use the shorter `rx` alias; it is already conventional in other domains.

## Commits

Use one lowercase word, a colon, and one concise one-line summary:

```text
init: bootstrap Representax
models: port ModernVBERT encoder
tests: add upstream gradient parity
```
