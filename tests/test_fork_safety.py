"""A forked child and the SQLite connections it inherits from the parent's OTHER threads (B71-133 (0), (c)).

The child's fork handler must never call into SQLite for a connection another
thread may have been using at the fork: that thread's connection mutex is
inherited locked, by a thread that does not exist in the child, so the driver's
close waits forever. Nor may the child use such a connection, whatever its
``check_same_thread`` setting. Each case runs in a subprocess with a deadline:
a hang is a failure, never a stuck test run.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(not hasattr(os, "fork"), reason="os.fork")

_ENV = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}

#: argv: <db> <busy|txn>. busy: another thread is inside a long query (holding the connection's mutex) at
#: the fork. txn: another thread's rollback-journal connection holds an open write transaction at the fork.
#: Both connections are check_same_thread=False, so the driver's thread check guards nothing. The child
#: tries a write through the connection, a cursor and a bound method made before the fork, then exits
#: NORMALLY (interpreter teardown included). Prints the child's refusals and the rows it managed to write.
_SCRIPT = r"""
import os, sqlite3, sys, threading, time
from trw_memory._live_stores import connect_registered
from trw_memory.exceptions import StorageError
db, mode = sys.argv[1:3]
conn = connect_registered(db, sqlite3, db, check_same_thread=False, isolation_level=None)
conn.execute("PRAGMA journal_mode=" + ("wal" if mode == "busy" else "delete"))
conn.execute("CREATE TABLE IF NOT EXISTS t(x)")
cur, ex = conn.cursor(), conn.execute
inside, finish = threading.Event(), threading.Event()

def other_thread():
    if mode == "busy":
        inside.set()
        try:
            conn.execute("WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT count(*) FROM c")
        except sqlite3.OperationalError:
            pass  # interrupted by the parent once the child is done
    else:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("INSERT INTO t VALUES ('parent')")
        inside.set()
        finish.wait()
        conn.execute("COMMIT")

worker = threading.Thread(target=other_thread, daemon=True)
worker.start()
inside.wait()
time.sleep(0.3)  # busy: well inside sqlite3_step, the connection's mutex held
pid = os.fork()
if pid == 0:
    insert = "INSERT INTO t VALUES ('child')"
    refused = 0
    for use in (lambda: conn.execute(insert), lambda: cur.execute(insert), lambda: ex(insert), lambda: conn.commit()):
        try:
            use()
        except (sqlite3.ProgrammingError, StorageError):
            refused += 1
        except Exception:
            pass
    print("child", refused, flush=True)
    sys.exit(0)
deadline = time.monotonic() + 15
while time.monotonic() < deadline:
    done, status = os.waitpid(pid, os.WNOHANG)
    if done:
        break
    time.sleep(0.05)
else:
    os.kill(pid, 9)
    os.waitpid(pid, 0)
    print("hung", flush=True)
    os._exit(0)
conn.interrupt()
finish.set()
worker.join(10)
rows = sqlite3.connect(db).execute("SELECT count(*) FILTER (WHERE x = 'child'), count(*) FROM t").fetchone()
print("exit", os.waitstatus_to_exitcode(status), "rows", *rows, flush=True)
os._exit(0)
"""


@pytest.mark.parametrize(
    ("mode", "rows"),
    [
        ("busy", "0 0"),  # (0): the child neither hangs nor gets to use the connection
        ("txn", "0 1"),  # (c): no write through another thread's check_same_thread=False connection
    ],
)
def test_a_child_forked_while_another_thread_uses_a_connection_neither_hangs_nor_uses_it(
    tmp_path: Path, mode: str, rows: str
) -> None:
    db = tmp_path / "store.db"
    run = subprocess.run(
        [sys.executable, "-c", _SCRIPT, str(db), mode], env=_ENV, capture_output=True, text=True, timeout=90
    )
    assert run.stdout.split("\n")[:2] == ["child 4", f"exit 0 rows {rows}"], run.stdout + run.stderr


#: argv: <db>. A cursor made with an explicit driver factory, and its bound execute, both retained across the
#: fork. The connection is check_same_thread=False, so the child keeps it open (quarantined, never closed):
#: only the owner guard stands between these and a write. Prints the child's refusals, then the child's rows.
_FACTORY_SCRIPT = r"""
import os, sqlite3, sys
from trw_memory._live_stores import connect_registered
from trw_memory.exceptions import StorageError
db = sys.argv[1]
conn = connect_registered(db, sqlite3, db, check_same_thread=False, isolation_level=None)
conn.execute("CREATE TABLE IF NOT EXISTS t(x)")
cur = conn.cursor(factory=sqlite3.Cursor)
ex = cur.execute
pid = os.fork()
if pid == 0:
    refused = 0
    for use in (lambda: cur.execute("INSERT INTO t VALUES ('child')"), lambda: ex("INSERT INTO t VALUES ('child')")):
        try:
            use()
        except StorageError:
            refused += 1
    print("child", refused, flush=True)
    os._exit(0)
os.waitpid(pid, 0)
print("rows", conn.execute("SELECT count(*) FROM t WHERE x = 'child'").fetchone()[0], flush=True)
os._exit(0)
"""


def test_a_cursor_from_a_custom_factory_is_refused_in_a_forked_child(tmp_path: Path) -> None:
    """sol r1 P1-b: ``conn.cursor(factory=sqlite3.Cursor)`` returned an unguarded driver cursor, whose retained
    bound ``execute`` wrote through the inherited connection in the child."""
    db = tmp_path / "store.db"
    run = subprocess.run(
        [sys.executable, "-c", _FACTORY_SCRIPT, str(db)], env=_ENV, capture_output=True, text=True, timeout=90
    )
    assert run.stdout.split("\n")[:2] == ["child 2", "rows 0"], run.stdout + run.stderr


def test_a_cursor_factory_that_cannot_be_guarded_is_refused(tmp_path: Path) -> None:
    import sqlite3

    from trw_memory._live_stores import connect_registered
    from trw_memory.exceptions import StorageError

    db = tmp_path / "store.db"
    conn = connect_registered(db, sqlite3, str(db))
    try:
        with pytest.raises(StorageError, match="cursor factory"):
            conn.cursor(factory=lambda connection: sqlite3.Cursor(connection))
        assert isinstance(conn.cursor(), sqlite3.Cursor)
    finally:
        conn.close()
