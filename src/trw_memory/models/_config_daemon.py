"""Loopback memory-daemon settings -- PRD-CORE-253 FR03.

Three tunables, and one deliberate absence. There is no ``bind_host`` field:
the daemon's bind address is the module constant
:data:`trw_memory.daemon.LOOPBACK_HOST`, because a configurable host turns one
typo into a network-reachable memory store carrying a bearer token. The port
is tunable and the host is not (PRD-CORE-253 NFR03, OQ-2).
"""

from __future__ import annotations

from pydantic import AliasChoices, BaseModel, Field

__all__ = ["_DaemonConfigMixin"]


class _DaemonConfigMixin(BaseModel):
    memory_single_store_path: str = Field(
        default="",
        validation_alias=AliasChoices("memory_single_store_path", "single_store_path"),
        description=(
            "Absolute path to the ONE SQLite file every namespace lands in (PRD-CORE-253 "
            "FR01). When set, it replaces the per-namespace 'base / <namespace_dir> / "
            "memory.db' join for every backend built from this config, which is what makes "
            "'one memory.db per user account' a fact rather than a claim. Safe because "
            "PRD-CORE-245 FR01 keys a row on (namespace, id), so one file holds every "
            "namespace without collision. The loopback daemon sets this to "
            "<user_memory_dir>/memory.db for its whole process; empty keeps the "
            "per-namespace layout that pre-daemon consumers still read, which FR02's "
            "migration retires. Not a behaviour switch -- it is a store LOCATION, in the "
            "same sense storage_path is."
        ),
    )
    memory_daemon_port: int = Field(
        default=0,
        ge=0,
        le=65535,
        validation_alias=AliasChoices("memory_daemon_port", "daemon_port"),
        description=(
            "TCP port the loopback memory daemon binds on 127.0.0.1. 0 (the default) asks "
            "the operating system for an ephemeral port, which removes the port-collision "
            "class between two user accounts on one box; clients read the assigned port "
            "from the daemon discovery file rather than assuming one."
        ),
    )
    memory_daemon_idle_shutdown_seconds: int = Field(
        default=1800,
        ge=60,
        validation_alias=AliasChoices(
            "memory_daemon_idle_shutdown_seconds",
            "daemon_idle_shutdown_seconds",
        ),
        description=(
            "Seconds without a served request after which the daemon exits, removes its "
            "discovery file and releases its lock. Bounds how long an idle daemon holds "
            "an open SQLite connection and a loaded embedder. The floor of 60 keeps an "
            "unattended daemon from thrashing start/stop; an operator running a "
            "short-lived daemon passes --idle-shutdown-seconds explicitly instead."
        ),
    )
    memory_daemon_startup_timeout_seconds: float = Field(
        default=10.0,
        gt=0.0,
        le=60.0,
        validation_alias=AliasChoices(
            "memory_daemon_startup_timeout_seconds",
            "daemon_startup_timeout_seconds",
        ),
        description=(
            "Client deadline, in seconds, waiting for an auto-started daemon's discovery "
            "file to appear. Exceeding it is a fail-closed error naming the discovery "
            "path and the start command, never a silent fallback to a local store."
        ),
    )
