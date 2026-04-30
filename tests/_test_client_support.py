from __future__ import annotations

import sqlite3


class _RecordingSQLCipherConnection:
    def __init__(self, conn: sqlite3.Connection, statements: list[str]) -> None:
        object.__setattr__(self, "_conn", conn)
        object.__setattr__(self, "_statements", statements)

    def __getattr__(self, name: str) -> object:
        return getattr(self._conn, name)

    def __setattr__(self, name: str, value: object) -> None:
        setattr(self._conn, name, value)

    def execute(self, sql: str, *args: object) -> sqlite3.Cursor:
        self._statements.append(sql)
        return self._conn.execute(sql, *args)


class _RecordingSQLCipherDBAPI:
    Error = sqlite3.Error
    DatabaseError = sqlite3.DatabaseError

    def __init__(self, statements: list[str]) -> None:
        self._statements = statements

    def connect(self, database: str, **kwargs: object) -> _RecordingSQLCipherConnection:
        conn = sqlite3.connect(database, **kwargs)
        return _RecordingSQLCipherConnection(conn, self._statements)
