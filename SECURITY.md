# Security policy

Please use GitHub private vulnerability reporting once the public repository is
available. Until then, report security issues privately to the maintainers
rather than opening a public issue.

Representax does not execute remote model or dataset code by default. Model
artifacts should use safetensors, and recipes should pin immutable upstream
revisions when reproducibility matters. Enabling third-party Python mappers or
remote code is an explicit trust decision made by the caller.
