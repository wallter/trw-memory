"""EngMem: a task-level benchmark for engineering memory.

LOCOMO and LongMemEval measure recall over personal conversation. trw-memory's
job is different: carry learnings, incidents, conventions and decisions across
engineering sessions so the next session starts from accumulated context. This
package holds the public half of the benchmark for that job -- the record
schema, the time-travel replay engine, the scorers and the retrieval arms.

The extractor that mines a real repository's git history, and every gold label
derived from it, are proprietary and live in ``trw-eval``. Nothing here may
depend on them: this package must run on a synthetic or third-party event stream
with no TRW data present.
"""

from .arms import Bm25Arm, GrepArm, RecencyArm, TrwFrameworkArm, TrwFtsFirstArm, TrwHybridArm
from .replay import leakage_check, replay_and_score
from .schema import Event, Query, freeze, holdout, read_events, read_queries, write_jsonl
from .score import Scored, aggregate, label_precision, paired_mcnemar, pairwise_order_accuracy, render

__all__ = [
    "Bm25Arm",
    "Event",
    "GrepArm",
    "Query",
    "RecencyArm",
    "Scored",
    "TrwFrameworkArm",
    "TrwFtsFirstArm",
    "TrwHybridArm",
    "aggregate",
    "freeze",
    "holdout",
    "label_precision",
    "leakage_check",
    "paired_mcnemar",
    "pairwise_order_accuracy",
    "read_events",
    "read_queries",
    "render",
    "replay_and_score",
    "write_jsonl",
]
