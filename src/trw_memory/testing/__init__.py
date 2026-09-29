"""Test support for suites that start trw-memory daemons; not runtime API.

Nothing in ``trw_memory`` or ``trw_mcp`` imports this package at runtime. It ships
in the wheel only because trw-mcp's test suite cannot import trw-memory's
``tests/`` tree, and the daemon reaper must exist once for both suites.
"""
