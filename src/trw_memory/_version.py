"""Package version — derived from pyproject.toml via importlib.metadata."""

from importlib.metadata import version as _pkg_version

__version__: str = _pkg_version("trw-memory")
