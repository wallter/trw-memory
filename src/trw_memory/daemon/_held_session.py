"""The one MCP session a ``DaemonClient`` holds across calls (W27), never closed under a call.

A session is one ``initialize`` for many calls; a per-call session cost four HTTP
requests and a ``tools/list`` for every call. Each held session counts the calls
using it. Every path that stops using one (a transport failure, a restarted daemon,
a loop change, the owner retiring the client) only RELEASES it: it is handed to no
new call and closes when its last in-flight call returns (release-verify RES-01).
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastmcp import Client


@dataclasses.dataclass(eq=False)
class HeldSession:
    """One held session: its loop, the daemon it talks to, the open client and what closes it."""

    loop: asyncio.AbstractEventLoop
    key: tuple[str, int, str]
    client: Client[Any]
    stack: contextlib.AsyncExitStack
    active: int = 0
    released: bool = False
    closed: bool = False
    # Strong ref to the primed generator from ``_watch_loop_shutdown``: kept alive so the
    # owning loop's weak asyncgen registry still finds it at shutdown (B71-09 below).
    _loop_watcher: AsyncIterator[None] | None = None


class HeldSessions:
    """The held session of one client, and the accounting that keeps a close from landing under a call."""

    def __init__(self) -> None:
        self.held: HeldSession | None = None
        self.retired = False
        self._opening: tuple[asyncio.AbstractEventLoop, asyncio.Lock] | None = None

    async def acquire(self, key: tuple[str, int, str], open_client: Callable[[], Client[Any]]) -> HeldSession:
        """The session to the daemon *key* names on this loop, opened if needed, counted as in use.

        The caller must :meth:`done_with` it. A different loop or daemon gets a fresh one.
        """
        loop = asyncio.get_running_loop()
        if self._opening is None or self._opening[0] is not loop:
            self._opening = (loop, asyncio.Lock())
        async with self._opening[1]:
            held = self.held
            if held is None or held.loop is not loop or held.key != key:
                await self.drop()
                stack = contextlib.AsyncExitStack()
                client: Client[Any] = await stack.enter_async_context(open_client())
                held = HeldSession(loop, key, client, stack)
                # retire() may have run while this opened: then serve this one call and close it.
                held.released = self.retired
                if not self.retired:
                    self.held = held
                    # Primed past its yield so the loop's asyncgen finalizer hook tracks it
                    # (B71-09): nothing else ever runs this held session's close on loop end.
                    watcher = _watch_loop_shutdown(held)
                    await watcher.__anext__()
                    held._loop_watcher = watcher
            held.active += 1
            return held

    async def done_with(self, held: HeldSession) -> None:
        held.active -= 1
        if held.released and not held.active:
            await _close(held)

    async def release(self, held: HeldSession) -> None:
        """A call on *held* failed in transport: no retry may reuse it."""
        if self.held is held:
            await self.drop()

    async def drop(self) -> None:
        """Stop handing out the held session; it closes once no call is using it."""
        held, self.held = self.held, None
        if held is not None:
            held.released = True
            if not held.active:
                await _close(held)

    async def retire(self) -> None:
        self.retired = True
        await self.drop()


async def _close(held: HeldSession) -> None:
    if held.closed:
        return
    if held.loop is asyncio.get_running_loop():
        held.closed = True
        with contextlib.suppress(Exception):  # justified: closing a broken session must not mask the call's error
            await held.stack.aclose()


async def _watch_loop_shutdown(held: HeldSession) -> AsyncIterator[None]:
    """One-shot generator whose ``finally`` runs when *held*'s loop ends (B71-09 / PRD-CORE-331 FR09).

    ``asyncio.run()`` calls ``loop.shutdown_asyncgens()`` before closing the
    loop, ``aclose()``-ing every async generator the loop still tracks -- the
    one hook that fires ON the owning loop while it is still alive, even when
    no explicit ``close()``/``retire()`` ever ran. A no-op once ``_close`` has
    already run through any other path (``held.closed`` guards it either way).
    """
    try:
        yield
    finally:
        await _close(held)
