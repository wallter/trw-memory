"""Namespace validation and path mapping — backward-compatibility shim.

This module re-exports from :mod:`trw_memory.namespaces` so that existing
imports (``from trw_memory.namespace import validate_namespace``) continue
to work.  The canonical implementations live in:

- :mod:`trw_memory.namespaces.validation` — ``validate_namespace``
- :mod:`trw_memory.namespaces.path_mapping` — ``namespace_to_path``
"""

from trw_memory.namespaces.path_mapping import namespace_to_path
from trw_memory.namespaces.validation import validate_namespace

__all__ = ["namespace_to_path", "validate_namespace"]
