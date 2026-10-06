# Third-party software and artifacts

Representax depends on separately distributed open-source packages including
JAX, Equinox, Optax, and Orbax. Optional integrations use Grain, Hydra-Zen,
Hugging Face libraries, PyTorch, and Sentence Transformers. Their respective
distributions and licenses govern those packages; none are vendored here.

Model checkpoints and datasets are not part of Representax. Users are
responsible for the licenses and terms attached to artifacts selected by their
recipes.

The checked-in Hugging Face architecture catalog contains identifiers
mechanically extracted from Hugging Face Transformers 5.3.0, distributed under
the Apache License 2.0. It does not contain upstream model implementation code.
The native ModernVBERT implementation is tested against Transformers 5.3.0 as
an optional development-time oracle; no checkpoint or upstream source file is
included in the Representax distribution.
