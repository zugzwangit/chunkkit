# Compatibility Policy

- ChunkKit uses Semantic Versioning.
- Before 1.0, minor releases may change experimental APIs with migration notes.
- Stable schemas and protocols use explicit version fields.
- A deprecated stable API remains for at least one minor release and emits a warning.
- Patch releases do not change deterministic chunk output unless fixing a documented correctness or
  security issue; such changes are called out in release notes.
- Plugins declare the ChunkKit API version they support. Incompatible plugins fail during discovery,
  not midway through a pipeline run.
