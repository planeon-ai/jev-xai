# Contributing to jev-xai

## Public API and versioning

The public API is every name exported in `jev_xai.__all__`. The project follows semantic versioning.

- `0.x` releases may break the Python API. Breaking changes ship with a deprecation notice in `CHANGELOG.md` when a compatible path exists.
- `1.0` ships when the decision-record schema freezes.
- Record-schema compatibility is versioned separately from the code API via `schema_version` on every record. Readers accept older `1.x` records. A breaking schema change bumps the major version and ships a migration.

## Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,cli,stats]"
pytest
ruff check .
ruff format --check .
mypy
```

## Pull requests

Open a PR against `main`. Include tests for behavior changes. If you change the record schema, regenerate `schemas/` and bump `schema_version`.

## Conduct

See `CODE_OF_CONDUCT.md`.
