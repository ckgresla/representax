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

The optional native Jina v5 text integration is tested against
`jinaai/jina-embeddings-v5-omni-small-retrieval` at revision
`12949877f0092093f366c6450340011320152a05`. Representax contains an independent
Equinox implementation of the executed text architecture and does not include
the checkpoint, its remote Python modules, or any model weights. The referenced
checkpoint is separately distributed under CC BY-NC 4.0; that non-commercial
artifact license governs users who choose to download and use those weights.
