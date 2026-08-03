# Security Model

ChunkKit treats source content, metadata, and model prompts as untrusted.

## ACLs

Every document and chunk has an `Acl`. Restricted content must have a complete source-derived policy
or an administrator mapping. Incomplete restricted ACLs raise `IncompleteAclError`. Retrieval uses
verified caller principals and tenant identity; clients must not translate arbitrary request fields
into principals.

## Connectors

Use least-privilege API scopes. Store only `env://`, vault, or equivalent secret references in
configuration. Bound response sizes, attachment recursion, retries, and crawl targets. Custom URL
connectors must block loopback, link-local, metadata-service, and private ranges unless explicitly
configured for a trusted network.

## Service

The included API-key verifier demonstrates tenant derivation and isolation. Shared deployments must
configure `CHUNKKIT_API_KEYS` or replace authentication with an OIDC/JWT plugin, place the API behind
TLS, use production stores, set quotas, and centralize audit logs. The fallback `dev` key is only for
localhost.

## LLM-assisted stages

They are disabled by default. Implementations must delimit source text, disable tools for judge
calls, validate structured output, cap cost, and record provider/model/data behavior.
