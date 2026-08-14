# Releasing Representax

Representax publishes from GitHub Actions to PyPI through OpenID Connect. No
long-lived PyPI token belongs in GitHub or a developer environment.

## Project identity

- Distribution: `representax`
- Repository owner: `ckgresla`
- Repository: `representax`
- Maintainer: Chris Kerwell Gresla `<ckgresla@gmail.com>`
- Workflow: `release.yml`
- GitHub environment: `pypi`

The GitHub environment permits only `v*` tags and requires approval from
`ckgresla` before its OIDC publish job runs.

Before the first release, add a pending GitHub publisher from the PyPI account's
Publishing page with exactly those values and environment `pypi`. A pending
publisher does not reserve the name; the first successful publication does.

## Release procedure

1. Confirm the source tree, provenance notice, tests, and compatibility matrix.
2. Build with `python -m build` and require `twine check --strict dist/*`.
3. Install the exact wheel into fresh CPU and GPU environments and run
   `scripts/smoke_install.py` in each.
4. Commit and push the release-ready tree.
5. Create the signed tag matching `v<project.version>` and push it.
6. Publish a GitHub release for that tag.
7. Approve the protected `pypi` deployment after inspecting the build artifact.
8. Read the version, metadata, files, and install behavior back from PyPI.

The workflow refuses a release whose tag does not exactly match the version in
`pyproject.toml`. PyPI does not permit replacing a published file, so a failed
or incorrect release is corrected with a new version rather than overwriting it.
