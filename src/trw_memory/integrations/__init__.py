"""Shared sync backend bridge used by trw-memory internals.

The VSCode adapter (``vscode.py``), the adapter factory (``factory.py``),
and their ``BackendOwnerMixin`` helper (``_mixin.py``) were removed as
unused surface — no production caller anywhere in the monorepo. See
CHANGELOG.md [Unreleased] Breaking.

``_backend.py`` remains: it backs ``create_backend_from_config``,
``discover_namespace_backends``, ``resolve_backend_location`` and friends,
which are used throughout ``trw_memory`` proper (client, cli, tools/,
lifecycle/tiers/, graph.py).
"""

from __future__ import annotations
