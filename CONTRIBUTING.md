# Contributing to ChunkKit

Thank you for helping make chunking pipelines more portable and measurable.

## Development setup

ChunkKit requires Python 3.12 or newer.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -e '.[dev,server]'
pytest
ruff check src tests
mypy src/chunkkit
```

Core changes must not introduce a mandatory cloud SDK, database client, hosted model, or telemetry.
Put optional integrations behind extras and use `chunkkit.errors.missing_extra` for actionable errors.

## Pull requests

1. Open an issue or RFC before a breaking API change or large new integration.
2. Add tests and sanitized fixtures. Never commit customer data, API tokens, or recorded credentials.
3. Update the changelog and relevant documentation.
4. Keep public types framework-neutral and JSON serializable.
5. Sign commits with `git commit -s` to certify the [Developer Certificate of Origin](https://developercertificate.org/).

New plugins should include a capability manifest and pass the public protocol contract tests. Live
SaaS tests must be opt-in and must skip cleanly without credentials.

## Compatibility

ChunkKit follows semantic versioning. See [docs/compatibility.md](docs/compatibility.md). Deprecations
must warn for at least one minor release before removal, except when immediate removal is required to
fix a security vulnerability.
