# Security Policy

## Supported versions

Until 1.0, only the latest minor release receives security fixes. After 1.0, the current major and
the immediately previous minor release will receive fixes for at least six months.

## Reporting

Use GitHub private vulnerability reporting. Do not open a public issue for an undisclosed
vulnerability. Include affected versions, reproduction steps, impact, and suggested mitigation.
Maintainers should acknowledge reports within five business days and coordinate disclosure.

## Security boundaries

- Core ChunkKit performs no network calls or telemetry.
- Connector credentials are passed directly or resolved through `SecretResolver`; pipeline specs
  must contain references, not secret values.
- Protected records with incomplete ACLs fail closed unless an explicit experimental override is
  enabled.
- The API derives tenant identity from an authenticated token and rejects cross-tenant chunks.
- The built-in `dev` token and local stores are demonstration defaults and are not safe for a shared
  production deployment.
- Plugins execute Python code with the host process's authority. Server operators must install and
  allowlist only reviewed packages.

See [docs/security-model.md](docs/security-model.md) for deployment guidance.
