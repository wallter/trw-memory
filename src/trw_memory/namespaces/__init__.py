"""Namespace package — validation, path mapping, and management.

Re-exports the core namespace functions for convenient access:

    from trw_memory.namespaces import validate_namespace, namespace_to_path
"""

from __future__ import annotations

from trw_memory.namespaces.identity import (
    ProjectIdentity,
    canonical_project_root,
    project_slug,
    resolve_project_identity,
    resolve_project_namespace,
)
from trw_memory.namespaces.manager import NamespaceManager
from trw_memory.namespaces.path_mapping import namespace_to_path
from trw_memory.namespaces.validation import validate_namespace

__all__ = [
    "NamespaceManager",
    "ProjectIdentity",
    "canonical_project_root",
    "namespace_to_path",
    "project_slug",
    "resolve_project_identity",
    "resolve_project_namespace",
    "validate_namespace",
]
