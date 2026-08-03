# Plugin SDK

Create a standalone project with:

```bash
chunkkit plugin scaffold example --group chunkers
```

A plugin exports a class or factory through a standard Python entry point:

```toml
[project.entry-points."chunkkit.chunkers"]
example = "chunkkit_example:ExamplePlugin"
```

The object must implement the matching protocol from `chunkkit.protocols`. Publish a
`PluginManifest` beside the implementation so users can inspect version, capabilities,
configuration schema, network access, and data behavior.

Supported groups are `chunkers`, `connectors`, `parsers`, `tokenizers`, `models`, `vectorstores`,
`evaluators`, and `storage`. Server processes accept an explicit `PluginManager` allowlist and never
install packages at runtime.
