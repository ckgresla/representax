# Contributing

Representax is pre-alpha. Before adding a new model family or task, open an
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
python -m pip install -e ".[test,parity]"
pytest -m parity
```

Distributed and performance evidence lives in dedicated test lanes. Generated
profiles, checkpoints, benchmark results, and downloaded datasets are not
committed to the repository.

## Commits

Use one lowercase word, a colon, and one concise one-line summary:

```text
init: bootstrap Representax
models: port ModernVBERT encoder
tests: add upstream gradient parity
```
