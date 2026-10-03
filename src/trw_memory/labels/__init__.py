"""Confidentiality labels for memory rows (PRD-SEC-023, phase 0).

Responsibility: decide, in ONE place, how sensitive a stored row is and whether a given surface or sink may see it. Nothing else in TRW
compares levels or reads the ``trw_label`` stamp; the five chokepoints (recall egress, write, platform row egress, session egress,
repository and derived rows) call this package.

Interface: :class:`Level`, :class:`Surface`, :class:`Sink`, :class:`LabelPolicy` (``current``, ``source``, ``label_of``, ``admit``),
:class:`Admission` and :class:`SessionMark`.

Invariants:
* A row's label is the MAXIMUM of the ``team`` floor, the namespace rule (``user:<name>`` other than ``user:local`` is ``personal``),
  every matching rule of the user's labels.yaml, and the row's own stamp. Rules and stamps can only raise a label.
* An unknown stamp, a failure while labelling, or an invalid policy file all resolve toward ``sensitive`` / ``strict`` (fail closed).
* ``sensitive`` exceeds every surface in phase 0: no agent sees it.
* The policy file is read from the USER base directory only (``$TRW_USER_DIR``, else ``$XDG_DATA_HOME/trw``, else ``~/.trw``), never from a
  project checkout, which may be hostile.
* Logs and messages carry level names, counts, the file path and an error class: never rules, tags, category names or content.

Knobs (``<user base>/labels.yaml``, optional; absent means today's behaviour): ``version``, ``auto_surface_max``, ``agent_max`` and up to
200 ``rules`` (``namespace`` glob and/or ``tags_any``, each raising to ``personal`` or ``sensitive``).
"""

from __future__ import annotations

from trw_memory.labels._levels import Level, Sink, Surface
from trw_memory.labels._mark import SessionMark
from trw_memory.labels._policy import Admission, LabelPolicy

__all__ = ["Admission", "LabelPolicy", "Level", "SessionMark", "Sink", "Surface"]
