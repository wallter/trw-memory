"""Synthetic dataset generator for trw-memory benchmarks.

Produces deterministic corpora of MemoryEntry objects at configurable sizes
(100, 1000, 10000) with diverse, realistic content drawn from 8 topic domains.
Also generates query sets with ground-truth expected matches for quality benchmarks.
"""

from __future__ import annotations

import hashlib
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

from trw_memory.models.memory import MemoryEntry

# ---------------------------------------------------------------------------
# Domain vocabulary for realistic content generation
# ---------------------------------------------------------------------------

DOMAINS: list[tuple[str, list[str]]] = [
    ("python", ["pydantic", "fastapi", "asyncio", "typing", "pytest", "mypy"]),
    ("javascript", ["react", "typescript", "node", "vitest", "zod", "nextjs"]),
    ("database", ["sqlite", "postgres", "redis", "migration", "indexing", "query"]),
    ("devops", ["docker", "ci-cd", "kubernetes", "monitoring", "deployment", "terraform"]),
    ("security", ["auth", "encryption", "rbac", "jwt", "cors", "xss"]),
    ("testing", ["unit", "integration", "coverage", "mocking", "fixtures", "tdd"]),
    ("architecture", ["microservices", "monolith", "event-driven", "cqrs", "ddd", "clean"]),
    ("ai-ml", ["embeddings", "llm", "rag", "fine-tuning", "prompting", "agents"]),
]

CONTENT_TEMPLATES: list[str] = [
    "{topic} requires {subtopic} configuration for production use",
    "When using {topic}, always validate {subtopic} parameters",
    "{subtopic} in {topic} has a known gotcha: check default values",
    "Best practice: use {subtopic} with {topic} for better performance",
    "{topic} {subtopic} integration needs explicit error handling",
    "Fix: {topic} {subtopic} fails silently when misconfigured",
    "{subtopic} patterns in {topic} should follow SOLID principles",
    "Warning: {topic} {subtopic} has breaking changes in latest version",
]

DETAIL_TEMPLATES: list[str] = [
    "Extended analysis of {topic} {subtopic}: verify config before deployment.",
    "When {subtopic} is used with {topic}, ensure error boundaries are in place.",
    "The {topic} ecosystem recommends {subtopic} for reliability and safety.",
    "Document all {subtopic} settings in {topic} projects for team visibility.",
]


def generate_corpus(size: int, seed: int = 42) -> list[MemoryEntry]:
    """Generate a deterministic synthetic corpus of MemoryEntry objects.

    Args:
        size: Number of entries to generate (e.g. 100, 1000, 10000).
        seed: Random seed for reproducibility.

    Returns:
        List of MemoryEntry objects with diverse, realistic content.
    """
    rng = random.Random(seed)
    entries: list[MemoryEntry] = []
    base_time = datetime(2025, 1, 1, tzinfo=timezone.utc)

    for i in range(size):
        domain_name, subtopics = rng.choice(DOMAINS)
        subtopic = rng.choice(subtopics)
        template = rng.choice(CONTENT_TEMPLATES)
        content = template.format(topic=domain_name, subtopic=subtopic)

        detail_template = rng.choice(DETAIL_TEMPLATES)
        detail = detail_template.format(topic=domain_name, subtopic=subtopic)

        # Deterministic ID from seed + index
        entry_id = f"M-{hashlib.md5(f'{seed}-{i}'.encode()).hexdigest()[:8]}"

        importance = round(rng.uniform(0.1, 1.0), 2)
        tags: list[str] = [domain_name, subtopic]
        if rng.random() > 0.5:
            tags.append(rng.choice(["gotcha", "best-practice", "fix", "warning"]))

        created = base_time + timedelta(hours=i * 2)

        entry = MemoryEntry(
            id=entry_id,
            content=content,
            detail=detail,
            tags=tags,
            importance=importance,
            namespace="benchmark",
            created_at=created,
            updated_at=created,
            source="agent",
            recurrence=rng.randint(1, 5),
        )
        entries.append(entry)

    return entries


def generate_query_set(
    corpus: list[MemoryEntry], num_queries: int = 50, seed: int = 42
) -> list[dict[str, object]]:
    """Generate search queries with expected results for quality benchmarks.

    Each query is derived from a random entry's content, and expected_ids
    contains IDs of entries whose content contains all query words.

    Args:
        corpus: Source corpus to derive queries from.
        num_queries: Number of queries to generate.
        seed: Random seed for reproducibility.

    Returns:
        List of dicts with keys: query, expected_ids, expected_tags.
    """
    rng = random.Random(seed)
    queries: list[dict[str, object]] = []

    for _ in range(num_queries):
        entry = rng.choice(corpus)
        words = entry.content.split()
        query_words = rng.sample(words, min(3, len(words)))
        query = " ".join(query_words)

        # Find expected matches (entries containing all query words)
        expected: list[str] = [
            e.id
            for e in corpus
            if all(w.lower() in e.content.lower() for w in query_words)
        ]

        queries.append({
            "query": query,
            "expected_ids": expected[:20],  # Cap at top 20
            "expected_tags": list(set(entry.tags)),
        })

    return queries


def create_golden_set(output_path: Path) -> None:
    """Generate and write the 50-entry golden set fixture to JSON.

    Each entry includes content, tags, importance, and a set of queries
    with ground-truth relevance judgments for quality benchmarks.

    Args:
        output_path: Path to write the JSON file.
    """
    golden_entries: list[dict[str, object]] = []

    # 50 hand-crafted entries across 8 domains with relevance judgments
    _entries_data: list[tuple[str, str, list[str], float, list[tuple[str, bool]]]] = [
        # (id, content, tags, importance, [(query, relevant)])
        # --- python domain (7 entries) ---
        ("golden-001", "Pydantic v2 requires model_config = ConfigDict(strict=True) for type enforcement", ["python", "pydantic"], 0.9,
         [("pydantic strict mode", True), ("database indexing", False), ("pydantic ConfigDict", True)]),
        ("golden-002", "FastAPI dependency injection uses Depends() for request-scoped services", ["python", "fastapi"], 0.85,
         [("fastapi dependency injection", True), ("react hooks", False), ("Depends service", True)]),
        ("golden-003", "Python asyncio event loop must not be nested without nest_asyncio", ["python", "asyncio"], 0.8,
         [("asyncio event loop nested", True), ("docker compose", False), ("python async", True)]),
        ("golden-004", "mypy --strict catches implicit Any and missing return types", ["python", "mypy"], 0.75,
         [("mypy strict mode", True), ("javascript types", False), ("type checking python", True)]),
        ("golden-005", "pytest fixtures with autouse=True run for every test in scope", ["python", "pytest"], 0.8,
         [("pytest autouse fixture", True), ("vitest setup", False), ("test fixtures", True)]),
        ("golden-006", "Python typing.Protocol enables structural subtyping without inheritance", ["python", "typing"], 0.7,
         [("typing Protocol structural", True), ("kubernetes deployment", False), ("python duck typing", True)]),
        ("golden-007", "Pydantic Field(alias='name') requires populate_by_name=True in model_config", ["python", "pydantic"], 0.85,
         [("pydantic alias field", True), ("redis caching", False), ("populate_by_name", True)]),
        # --- javascript domain (7 entries) ---
        ("golden-008", "React useEffect cleanup function runs before component unmount", ["javascript", "react"], 0.8,
         [("react useEffect cleanup", True), ("python decorators", False), ("component lifecycle", True)]),
        ("golden-009", "TypeScript discriminated unions need a literal type tag field", ["javascript", "typescript"], 0.85,
         [("typescript discriminated union", True), ("sql migrations", False), ("tagged union pattern", True)]),
        ("golden-010", "Node.js streams handle backpressure via highWaterMark setting", ["javascript", "node"], 0.7,
         [("node streams backpressure", True), ("docker volumes", False), ("highWaterMark", True)]),
        ("golden-011", "Vitest uses vi.fn() for mocking functions in unit tests", ["javascript", "vitest"], 0.75,
         [("vitest mock function", True), ("pytest mock", False), ("vi.fn unit test", True)]),
        ("golden-012", "Zod schema validation integrates with React Hook Form via resolver", ["javascript", "zod"], 0.8,
         [("zod react hook form", True), ("pydantic validation", False), ("schema validation frontend", True)]),
        ("golden-013", "Next.js App Router uses server components by default for performance", ["javascript", "nextjs"], 0.85,
         [("nextjs server components", True), ("fastapi routing", False), ("app router default", True)]),
        ("golden-014", "TypeScript satisfies operator validates type without widening", ["javascript", "typescript"], 0.7,
         [("typescript satisfies operator", True), ("python assert", False), ("type narrowing", True)]),
        # --- database domain (6 entries) ---
        ("golden-015", "SQLite WAL mode improves concurrent read performance significantly", ["database", "sqlite"], 0.9,
         [("sqlite WAL mode", True), ("react rendering", False), ("concurrent reads", True)]),
        ("golden-016", "PostgreSQL EXPLAIN ANALYZE shows actual execution time per plan node", ["database", "postgres"], 0.85,
         [("postgres explain analyze", True), ("python profiling", False), ("query execution plan", True)]),
        ("golden-017", "Redis SET with NX and EX flags implements distributed locks", ["database", "redis"], 0.8,
         [("redis distributed lock", True), ("file system lock", False), ("SET NX EX", True)]),
        ("golden-018", "Database migration scripts must be idempotent for safe re-runs", ["database", "migration"], 0.9,
         [("idempotent migration", True), ("docker rebuild", False), ("database schema migration", True)]),
        ("golden-019", "Composite indexes in SQL follow leftmost prefix rule for queries", ["database", "indexing"], 0.85,
         [("composite index leftmost", True), ("python list index", False), ("sql index optimization", True)]),
        ("golden-020", "SQL query plans degrade when statistics are stale after bulk loads", ["database", "query"], 0.75,
         [("query plan statistics", True), ("javascript bundling", False), ("sql performance bulk", True)]),
        # --- devops domain (6 entries) ---
        ("golden-021", "Docker multi-stage builds reduce final image size by separating build deps", ["devops", "docker"], 0.85,
         [("docker multi-stage build", True), ("python packaging", False), ("image size reduction", True)]),
        ("golden-022", "CI/CD pipeline caching of node_modules speeds up JavaScript builds", ["devops", "ci-cd"], 0.8,
         [("ci cache node_modules", True), ("python virtualenv", False), ("pipeline build speed", True)]),
        ("golden-023", "Kubernetes liveness probes restart pods that fail health checks", ["devops", "kubernetes"], 0.9,
         [("kubernetes liveness probe", True), ("pytest health", False), ("pod restart health", True)]),
        ("golden-024", "Prometheus metrics collection requires instrumentation at application level", ["devops", "monitoring"], 0.75,
         [("prometheus metrics instrumentation", True), ("python logging", False), ("application monitoring", True)]),
        ("golden-025", "Blue-green deployment eliminates downtime by switching traffic atomically", ["devops", "deployment"], 0.85,
         [("blue-green deployment", True), ("database rollback", False), ("zero downtime deploy", True)]),
        ("golden-026", "Terraform state must be stored remotely for team collaboration", ["devops", "terraform"], 0.8,
         [("terraform remote state", True), ("git branching", False), ("infrastructure state", True)]),
        # --- security domain (6 entries) ---
        ("golden-027", "JWT tokens should use short expiry with refresh token rotation", ["security", "jwt"], 0.9,
         [("jwt refresh token rotation", True), ("database session", False), ("token expiry security", True)]),
        ("golden-028", "bcrypt with cost factor 12 provides adequate password hashing security", ["security", "encryption"], 0.85,
         [("bcrypt cost factor", True), ("base64 encoding", False), ("password hashing", True)]),
        ("golden-029", "RBAC policies should deny by default and grant explicit permissions", ["security", "rbac"], 0.9,
         [("rbac deny default", True), ("css display none", False), ("role permissions", True)]),
        ("golden-030", "OAuth2 PKCE flow prevents authorization code interception attacks", ["security", "auth"], 0.85,
         [("oauth2 pkce flow", True), ("api key header", False), ("authorization code security", True)]),
        ("golden-031", "CORS preflight OPTIONS requests must include Access-Control-Allow-Methods", ["security", "cors"], 0.8,
         [("cors preflight options", True), ("http GET request", False), ("access control headers", True)]),
        ("golden-032", "XSS prevention requires output encoding at every insertion point", ["security", "xss"], 0.9,
         [("xss output encoding", True), ("input validation", False), ("cross-site scripting prevention", True)]),
        # --- testing domain (6 entries) ---
        ("golden-033", "Unit tests should not depend on external services or file system state", ["testing", "unit"], 0.85,
         [("unit test isolation", True), ("integration database", False), ("external dependency test", True)]),
        ("golden-034", "Integration tests need database fixtures reset between test runs", ["testing", "integration"], 0.8,
         [("integration test fixture reset", True), ("unit mock", False), ("database test cleanup", True)]),
        ("golden-035", "Code coverage below 80% indicates untested critical paths", ["testing", "coverage"], 0.75,
         [("coverage threshold critical", True), ("performance benchmark", False), ("test coverage path", True)]),
        ("golden-036", "Mocking should target the module where the name is looked up, not defined", ["testing", "mocking"], 0.9,
         [("mock module lookup", True), ("dependency injection", False), ("monkeypatch import", True)]),
        ("golden-037", "Test fixtures should be minimal and create only what the test requires", ["testing", "fixtures"], 0.8,
         [("minimal test fixture", True), ("factory pattern", False), ("fixture scope", True)]),
        ("golden-038", "TDD red-green-refactor cycle catches regressions before they ship", ["testing", "tdd"], 0.85,
         [("tdd red green refactor", True), ("code review", False), ("test driven development", True)]),
        # --- architecture domain (6 entries) ---
        ("golden-039", "Microservices need distributed tracing to debug cross-service failures", ["architecture", "microservices"], 0.85,
         [("microservices distributed tracing", True), ("monolith logging", False), ("cross-service debug", True)]),
        ("golden-040", "Monolith-first approach reduces complexity until scale demands splitting", ["architecture", "monolith"], 0.8,
         [("monolith first approach", True), ("kubernetes scaling", False), ("complexity reduction", True)]),
        ("golden-041", "Event-driven architecture decouples producers from consumers via message bus", ["architecture", "event-driven"], 0.9,
         [("event driven message bus", True), ("synchronous API", False), ("producer consumer decouple", True)]),
        ("golden-042", "CQRS separates read models from write models for scalability", ["architecture", "cqrs"], 0.85,
         [("cqrs read write model", True), ("crud endpoint", False), ("query command separation", True)]),
        ("golden-043", "Domain-driven design bounded contexts prevent model pollution across teams", ["architecture", "ddd"], 0.9,
         [("ddd bounded context", True), ("database schema", False), ("domain model boundary", True)]),
        ("golden-044", "Clean architecture dependency rule: outer layers depend on inner layers only", ["architecture", "clean"], 0.85,
         [("clean architecture dependency", True), ("npm dependency", False), ("layer dependency rule", True)]),
        # --- ai-ml domain (6 entries) ---
        ("golden-045", "Embedding models must normalize vectors for cosine similarity to work correctly", ["ai-ml", "embeddings"], 0.9,
         [("embedding normalize cosine", True), ("sql index", False), ("vector similarity search", True)]),
        ("golden-046", "LLM temperature 0 produces deterministic outputs for reproducible pipelines", ["ai-ml", "llm"], 0.85,
         [("llm temperature deterministic", True), ("random seed", False), ("reproducible generation", True)]),
        ("golden-047", "RAG retrieval quality depends on chunk size and overlap parameters", ["ai-ml", "rag"], 0.9,
         [("rag chunk size overlap", True), ("database pagination", False), ("retrieval augmented generation", True)]),
        ("golden-048", "Fine-tuning on fewer than 100 examples risks catastrophic forgetting", ["ai-ml", "fine-tuning"], 0.8,
         [("fine-tuning catastrophic forgetting", True), ("transfer learning", False), ("few-shot training", True)]),
        ("golden-049", "System prompts should front-load critical instructions for attention", ["ai-ml", "prompting"], 0.85,
         [("system prompt instructions", True), ("user input parsing", False), ("prompt engineering attention", True)]),
        ("golden-050", "Agent tool-use loops need max-iteration guards to prevent infinite chains", ["ai-ml", "agents"], 0.9,
         [("agent tool loop guard", True), ("event loop python", False), ("infinite chain prevention", True)]),
    ]

    for eid, content, tags, importance, query_pairs in _entries_data:
        queries_list: list[dict[str, object]] = [
            {"query": q, "relevant": r} for q, r in query_pairs
        ]
        golden_entries.append({
            "id": eid,
            "content": content,
            "tags": tags,
            "importance": importance,
            "queries": queries_list,
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"entries": golden_entries}, indent=2) + "\n",
        encoding="utf-8",
    )


def create_dedup_set(output_path: Path) -> None:
    """Generate and write the 30-pair dedup accuracy set to JSON.

    Each pair has entry_a, entry_b, and an expected_duplicate boolean.
    Pairs alternate between true duplicates (paraphrases) and
    clearly distinct entries.

    Args:
        output_path: Path to write the JSON file.
    """
    pairs: list[dict[str, object]] = [
        # --- True duplicates (paraphrases) ---
        {"id": "dedup-001",
         "entry_a": {"content": "Pydantic v2 uses ConfigDict instead of class Config", "tags": ["python", "pydantic"]},
         "entry_b": {"content": "In Pydantic version 2, ConfigDict replaces the old class Config pattern", "tags": ["python", "pydantic"]},
         "expected_duplicate": True},
        {"id": "dedup-002",
         "entry_a": {"content": "Docker compose requires version 3.8 for GPU support", "tags": ["devops", "docker"]},
         "entry_b": {"content": "SQLite WAL mode improves write concurrency", "tags": ["database", "sqlite"]},
         "expected_duplicate": False},
        {"id": "dedup-003",
         "entry_a": {"content": "FastAPI dependency injection uses Depends for services", "tags": ["python", "fastapi"]},
         "entry_b": {"content": "In FastAPI, service dependencies are injected via the Depends function", "tags": ["python", "fastapi"]},
         "expected_duplicate": True},
        {"id": "dedup-004",
         "entry_a": {"content": "React useState hook triggers re-render on state change", "tags": ["javascript", "react"]},
         "entry_b": {"content": "Kubernetes pod affinity rules control scheduling placement", "tags": ["devops", "kubernetes"]},
         "expected_duplicate": False},
        {"id": "dedup-005",
         "entry_a": {"content": "pytest fixtures provide reusable test setup and teardown", "tags": ["testing", "pytest"]},
         "entry_b": {"content": "In pytest, fixtures handle test setup and cleanup in a reusable way", "tags": ["testing", "pytest"]},
         "expected_duplicate": True},
        {"id": "dedup-006",
         "entry_a": {"content": "JWT tokens should have short expiry times for security", "tags": ["security", "jwt"]},
         "entry_b": {"content": "Redis pub/sub enables real-time messaging between services", "tags": ["database", "redis"]},
         "expected_duplicate": False},
        {"id": "dedup-007",
         "entry_a": {"content": "TypeScript generics enable type-safe reusable components", "tags": ["javascript", "typescript"]},
         "entry_b": {"content": "Generics in TypeScript allow writing components that are both reusable and type-safe", "tags": ["javascript", "typescript"]},
         "expected_duplicate": True},
        {"id": "dedup-008",
         "entry_a": {"content": "Alembic migrations must be run in order for schema consistency", "tags": ["database", "migration"]},
         "entry_b": {"content": "XSS attacks inject malicious scripts into web pages", "tags": ["security", "xss"]},
         "expected_duplicate": False},
        {"id": "dedup-009",
         "entry_a": {"content": "asyncio.gather runs multiple coroutines concurrently", "tags": ["python", "asyncio"]},
         "entry_b": {"content": "Python asyncio.gather executes coroutines in parallel concurrently", "tags": ["python", "asyncio"]},
         "expected_duplicate": True},
        {"id": "dedup-010",
         "entry_a": {"content": "Terraform plan shows infrastructure changes before apply", "tags": ["devops", "terraform"]},
         "entry_b": {"content": "Next.js getServerSideProps runs on every request for SSR", "tags": ["javascript", "nextjs"]},
         "expected_duplicate": False},
        {"id": "dedup-011",
         "entry_a": {"content": "CORS preflight requests use the OPTIONS HTTP method", "tags": ["security", "cors"]},
         "entry_b": {"content": "The OPTIONS method is used for CORS preflight checks", "tags": ["security", "cors"]},
         "expected_duplicate": True},
        {"id": "dedup-012",
         "entry_a": {"content": "Code coverage tools measure which lines are executed during tests", "tags": ["testing", "coverage"]},
         "entry_b": {"content": "Embedding vectors must be normalized for cosine similarity", "tags": ["ai-ml", "embeddings"]},
         "expected_duplicate": False},
        {"id": "dedup-013",
         "entry_a": {"content": "Microservices communicate via API gateways or message queues", "tags": ["architecture", "microservices"]},
         "entry_b": {"content": "In microservice architectures, services talk through API gateways or message queues", "tags": ["architecture", "microservices"]},
         "expected_duplicate": True},
        {"id": "dedup-014",
         "entry_a": {"content": "LLM fine-tuning requires curated high-quality training data", "tags": ["ai-ml", "fine-tuning"]},
         "entry_b": {"content": "PostgreSQL partitioning improves query performance on large tables", "tags": ["database", "postgres"]},
         "expected_duplicate": False},
        {"id": "dedup-015",
         "entry_a": {"content": "mypy --strict catches missing type annotations in Python code", "tags": ["python", "mypy"]},
         "entry_b": {"content": "Running mypy in strict mode detects missing type annotations", "tags": ["python", "mypy"]},
         "expected_duplicate": True},
        {"id": "dedup-016",
         "entry_a": {"content": "Docker volumes persist data across container restarts", "tags": ["devops", "docker"]},
         "entry_b": {"content": "Integration tests verify component interactions end-to-end", "tags": ["testing", "integration"]},
         "expected_duplicate": False},
        {"id": "dedup-017",
         "entry_a": {"content": "RAG systems split documents into chunks for vector retrieval", "tags": ["ai-ml", "rag"]},
         "entry_b": {"content": "Retrieval-augmented generation chunks documents for vector-based search", "tags": ["ai-ml", "rag"]},
         "expected_duplicate": True},
        {"id": "dedup-018",
         "entry_a": {"content": "RBAC policies should follow least-privilege principle", "tags": ["security", "rbac"]},
         "entry_b": {"content": "Node.js process.env reads environment variables at runtime", "tags": ["javascript", "node"]},
         "expected_duplicate": False},
        {"id": "dedup-019",
         "entry_a": {"content": "Event-driven systems use message brokers for async communication", "tags": ["architecture", "event-driven"]},
         "entry_b": {"content": "Async communication in event-driven architectures relies on message brokers", "tags": ["architecture", "event-driven"]},
         "expected_duplicate": True},
        {"id": "dedup-020",
         "entry_a": {"content": "SQLite FTS5 provides full-text search with ranking functions", "tags": ["database", "sqlite"]},
         "entry_b": {"content": "Zod schemas validate runtime data in TypeScript applications", "tags": ["javascript", "zod"]},
         "expected_duplicate": False},
        {"id": "dedup-021",
         "entry_a": {"content": "Clean architecture puts business logic in the domain layer", "tags": ["architecture", "clean"]},
         "entry_b": {"content": "In clean architecture, the domain layer contains all business logic", "tags": ["architecture", "clean"]},
         "expected_duplicate": True},
        {"id": "dedup-022",
         "entry_a": {"content": "Kubernetes namespaces isolate resources within a cluster", "tags": ["devops", "kubernetes"]},
         "entry_b": {"content": "bcrypt is recommended for password hashing over SHA-256", "tags": ["security", "encryption"]},
         "expected_duplicate": False},
        {"id": "dedup-023",
         "entry_a": {"content": "TDD requires writing failing tests before implementation code", "tags": ["testing", "tdd"]},
         "entry_b": {"content": "Test-driven development means writing tests that fail before writing the code", "tags": ["testing", "tdd"]},
         "expected_duplicate": True},
        {"id": "dedup-024",
         "entry_a": {"content": "CI/CD pipelines should run linting before expensive test suites", "tags": ["devops", "ci-cd"]},
         "entry_b": {"content": "DDD aggregate roots enforce consistency boundaries", "tags": ["architecture", "ddd"]},
         "expected_duplicate": False},
        {"id": "dedup-025",
         "entry_a": {"content": "LLM prompting benefits from few-shot examples in the context", "tags": ["ai-ml", "prompting"]},
         "entry_b": {"content": "Including few-shot examples in LLM prompts improves output quality", "tags": ["ai-ml", "prompting"]},
         "expected_duplicate": True},
        {"id": "dedup-026",
         "entry_a": {"content": "Redis TTL keys enable automatic cache expiration", "tags": ["database", "redis"]},
         "entry_b": {"content": "React Server Components reduce client-side JavaScript bundle size", "tags": ["javascript", "react"]},
         "expected_duplicate": False},
        {"id": "dedup-027",
         "entry_a": {"content": "Vitest supports concurrent test execution for faster feedback", "tags": ["javascript", "vitest"]},
         "entry_b": {"content": "Concurrent test running in Vitest speeds up the test feedback loop", "tags": ["javascript", "vitest"]},
         "expected_duplicate": True},
        {"id": "dedup-028",
         "entry_a": {"content": "Monitoring dashboards need alerting thresholds for SLO violations", "tags": ["devops", "monitoring"]},
         "entry_b": {"content": "CQRS enables independent scaling of read and write workloads", "tags": ["architecture", "cqrs"]},
         "expected_duplicate": False},
        {"id": "dedup-029",
         "entry_a": {"content": "Agent tool-calling requires structured output parsing for reliability", "tags": ["ai-ml", "agents"]},
         "entry_b": {"content": "Reliable agent tool use depends on structured output parsing", "tags": ["ai-ml", "agents"]},
         "expected_duplicate": True},
        {"id": "dedup-030",
         "entry_a": {"content": "PostgreSQL indexes should be analyzed after bulk data loads", "tags": ["database", "postgres"]},
         "entry_b": {"content": "OAuth2 authorization code flow is preferred over implicit flow", "tags": ["security", "auth"]},
         "expected_duplicate": False},
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"pairs": pairs}, indent=2) + "\n",
        encoding="utf-8",
    )
