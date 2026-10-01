"""Test-only synchronous maintain runner (deletion-wave replacement for ``memory_maintain_impl``).

``trw_memory.tools.maintain.memory_maintain_impl`` had zero production callers: the served
``memory_maintain`` MCP tool goes through ``tools._maintain_sweep.serve_maintain``'s lane-job
orchestration instead (the backend and its config resolved per job from the daemon's write lane, not
from an injected ``backend``/``config`` pair), so ``memory_maintain_impl`` was a synchronous-call
convenience only ever exercised by tests. Deleted in the trw-memory deletion wave (PRD-QUAL-145);
this helper keeps the same one-call-does-everything shape for the two test files that need it to
drive decay, consolidation, verification and the WAL checkpoint against an injected backend and
config, without the daemon's lane and pool machinery, calling the SAME production pass functions
(``maintain._run_decay``/``_run_consolidation``/``_run_checkpoint``/``_maintain_sweep.begin`` etc.)
``serve_maintain`` calls, just sequenced synchronously rather than as separate lane jobs.
"""

from __future__ import annotations

from trw_memory.models.config import MemoryConfig
from trw_memory.security.rbac import Permission
from trw_memory.storage.interface import StorageBackend
from trw_memory.tools import _maintain_sweep as sweep
from trw_memory.tools import maintain
from trw_memory.tools.entry import refused_namespace


def run_maintain_sync(
    namespace: str,
    *,
    backend: StorageBackend,
    config: MemoryConfig | None = None,
    verify_seconds: float | None = None,
) -> dict[str, object]:
    """Run decay, consolidation, verification and a WAL checkpoint against *backend*, in one call.

    See the module docstring: this is a test harness, not the served path (that resolves its own
    backend and config per lane job — see ``_maintain_sweep.serve_maintain``).
    """
    cfg = config or MemoryConfig()
    if refused := refused_namespace(namespace, Permission.WRITE, "maintain", cfg):
        return refused
    run = sweep.begin(namespace, backend, cfg)
    run.passes["decay"] = maintain._run_decay(backend, namespace, cfg)
    run.passes["consolidation"] = maintain._run_consolidation(namespace, backend, cfg)
    sweep.verify_slice(run, backend, verify_seconds)
    run.passes["security_maintenance"] = maintain._run_security_maintenance()
    run.passes["snapshot"] = maintain._run_snapshot(backend, cfg)
    run.passes["wal_checkpoint"] = maintain._run_checkpoint(backend)
    return sweep.finish(run, backend)
