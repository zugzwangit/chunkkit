"""Public exception hierarchy for ChunkKit."""


class ChunkKitError(Exception):
    """Base class for actionable ChunkKit errors."""


class ConfigurationError(ChunkKitError):
    """Raised when a pipeline or plugin configuration is invalid."""


class TokenBudgetError(ChunkKitError):
    """Raised when content cannot satisfy a declared token budget."""


class IncompleteAclError(ChunkKitError):
    """Raised when protected content does not have an enforceable ACL."""


class MissingExtraError(ChunkKitError, ImportError):
    """Raised when an optional feature is imported without its extra."""


class PluginError(ChunkKitError):
    """Raised when a plugin is invalid or cannot be loaded."""


def missing_extra(feature: str, extra: str, package: str | None = None) -> MissingExtraError:
    detail = f" Optional dependency: {package}." if package else ""
    return MissingExtraError(
        f"{feature} requires the '{extra}' extra. Install it with "
        f"`pip install 'chunkkit[{extra}]'`.{detail}"
    )
