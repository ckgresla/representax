# Releasing Representax

Representax publishes from GitHub Actions to PyPI through OpenID Connect. No
long-lived PyPI token belongs in GitHub or a developer environment.

GitHub Packages is not part of the release architecture. PyPI is the permanent
Python package registry; GitHub Releases provide the human-facing release
record; and the short-lived GitHub Actions artifact carries the exact wheel and
source distribution between the build, publish, and release jobs.

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
6. Inspect the workflow's exact build artifact and approve the protected `pypi`
   deployment.
7. After PyPI succeeds, let the workflow create the GitHub Release for that tag
   and attach the same wheel and source distribution.
8. Read the version, metadata, files, and install behavior back from PyPI and
   the GitHub Release.

Only a pushed `v*` tag invokes the release workflow; ordinary pushes to `main`
run CI without publishing. The workflow refuses a tag whose version does not
exactly match `pyproject.toml`. PyPI does not permit replacing a published file,
so a failed or incorrect release is corrected with a new version rather than
overwriting it.
