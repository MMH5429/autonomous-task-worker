# Autonomous Task Worker

A prototype AI worker that takes a task in natural language, works out what to
do, does it against real systems, and then **independently checks whether it
actually worked** before telling you it is done.

Built for the CentrAlign AI intern problem statement. The company it operates
inside is simulated; everything the agent does to that company is real — real
files on disk, real HTTP calls, real rows in SQLite. The scoping decision was
deliberate: the assignment says a narrow prototype that genuinely works beats a
broad one that mostly simulates, so nothing inside the agent is mocked.

```
python run.py "Find the latest invoice from Northwind Logistics, extract the
               amount and due date, enter it into our internal system, and
               tell me once it is done."
```

---

## The loop

```
Goal → Understand → Plan → Execute → Observe → Adapt → Verify → Complete
```

Each phase is a named function in `agent/loop.py`, each one writes to the trace,
and the control flow is plain Python — no agent framework, nothing hidden.

| Phase | What it does here |
|---|---|
| **Understand** | Restates the goal and records it. Success criteria are produced by the planning call, which needs the tool catalogue to know what success can even mean. |
| **Plan** | `planner.make_plan` asks the model for an ordered plan given the goal and the **tool registry**. There is no step list anywhere in the source. |
| **Execute** | The model picks the next tool call given the plan and everything observed so far. The loop binds arguments against the declared schema and dispatches. Writes pass an approval gate first. |
| **Observe** | Results are recorded into working memory. Missing extraction fields and successful writes get special handling, because both change what happens next. |
| **Adapt** | On a tool failure, `planner.revise_plan` produces a new plan from the failure and current state. Bounded to 3 replans so it cannot thrash. |
| **Verify** | Two independent legs (below). This is the part that decides the outcome. |
| **Complete** | Prints OUTCOME, what was done, and evidence read back from the system of record — not from the write response. |

Bounded by `MAX_STEPS = 25`, which exits with a clear "step budget exhausted".

---

## Verification — the part that matters

An agent that says "done" is worth nothing. Verification has two legs, and
**either one failing fails the run**, even if every individual step succeeded.

**Leg 1 — state.** Do not trust the write response. The ERP's `POST` reply is a
claim made by the same call that might have gone wrong. Verification re-fetches
each created entry with `GET /entries/{id}` and diffs it field by field against
what was sent.

**Leg 2 — claim.** An agent can also fail by simply *asserting* it did the work.
The final summary is audited against the trace of what actually executed. A
summary claiming an ERP entry was created, when no write tool ever succeeded,
fails the run.

Leg 2 is not decoration — it was added because a run actually did this. Against
a weak local model, the agent announced *"the accounts-payable entry has been
successfully created"*, having never called a write tool, and the run came out
**SUCCESS**. Leg 1 could not catch it: there was no write to re-read. That run
is why read-only tasks get a real check instead of a free pass.

---

## Quickstart

Requires Python 3.10+.

```bash
git clone https://github.com/MMH5429/autonomous-task-worker.git
cd autonomous-task-worker
pip install -r requirements.txt

cp .env.example .env        # then put your key in it (see below)
```

**Get a free LLM key** (no credit card): https://console.groq.com/keys
Put it in `.env` as `GROQ_API_KEY=gsk_...`.

**Terminal 1 — the simulated company:**
```bash
python mockco/erp.py --port 8099 --reset
```
`--reset` clears the ERP database first, so each demo starts from a clean
system. Leave it off to accumulate entries across runs.

**Terminal 2 — the agent:**
```bash
python run.py "Find the latest invoice from Northwind Logistics, extract the amount and due date, enter it into our internal system, and tell me once it is done." --today 2026-10-04
```

Add `--auto-approve` to skip the interactive approval prompt before writes.
`--today YYYY-MM-DD` fixes the agent's notion of today (the sample invoices are
dated 2026).

Tests: `python -m pytest -q`

### Running fully offline

No key, no internet:

```bash
ollama serve && ollama pull qwen2.5
# in .env:
LLM_PROVIDER=ollama
```

This works — it is how the loop was first validated — but small local models
plan poorly and will not reliably complete the tasks below. The provider is a
one-line change because of `agent/llm.py`, not a rewrite.

---

## Demo scenarios

Run each from a fresh `mockco` process. All five use the **same binary with no
code changes**.

**1 · Happy path — resolving ambiguity**
```bash
python run.py "Find the latest invoice from Northwind Logistics, extract the amount and due date, enter it into our internal system, and tell me once it is done." --auto-approve --today 2026-10-04
```
*Look for:* there are **two** Northwind invoices. "Latest" is never resolved for
the agent — it reads both and compares issue dates, picking `NWL-2026-0915`
(Sept) over `NWL-2026-0117` (Jan). Ends SUCCESS with the stored row as evidence.

**2 · Transient failure — retry and recover**
```bash
python mockco/erp.py --port 8099 --flaky     # terminal 1
```
*Look for:* the first write returns `500 upstream timeout` and persists nothing.
The ERP client retries with exponential backoff (0.5s, 1s, 2s) and succeeds. The
failure stays visible in the trace on purpose — it is not swallowed.

**3 · Missing data — ask, do not guess**
```bash
python run.py "Enter the Vertex Paper invoice into our system." --auto-approve --today 2026-10-04
```
*Look for:* `VPX-2026-0727.txt` has **no due date**. Extraction returns
`due_date: null` and flags it in `missing_fields`. The agent stops and asks a
specific question, then uses the answer. It could have inferred "Net 30" from
the other invoices — guessing a due date in an ERP is worse than asking.

**4 · Verification catches a silent corruption** ← *the important one*
```bash
python mockco/erp.py --port 8099 --corrupt-writes     # terminal 1
python run.py "Find the latest invoice from Northwind Logistics, extract the amount and due date, enter it into our internal system, and tell me once it is done." --auto-approve --today 2026-10-04
```
*Look for:* the ERP stores `amount` as `0.0` while returning the **correct**
amount in the POST response. Every step succeeds. The run still comes out
**FAILED**, with a field-level diff — because verification re-reads from the
system of record instead of trusting the write it just made.

**5 · Generalization — different tasks, zero code change**
```bash
python run.py "Which invoices are overdue as of 4 October 2026? Give me a summary." --today 2026-10-04
python run.py "Total up everything we owe Northwind Logistics across all their invoices." --today 2026-10-04
```
*Look for:* no new tools, no changes to `agent/`. Neither task writes anything,
so leg 1 has nothing to check — leg 2 (claim vs trace) is what verifies them.

---

## Architecture

```
                    run.py  (CLI)
                      │
                      ▼
          ┌───────────────────────────┐
          │      agent/loop.py        │   Understand → Plan → Execute →
          │   explicit control flow   │   Observe → Adapt → Verify → Complete
          └───────────────────────────┘
             │        │        │     │
   ┌─────────┘        │        │     └──────────────┐
   ▼                  ▼        ▼                    ▼
planner.py        registry.py  memory.py         verify.py
(model makes      (tools +     (facts, history,  ├─ leg 1: re-read ERP + diff
 and revises       schemas,     writes to be     └─ leg 2: audit summary
 the plan)         writes flag) verified)                 against the trace
   │                  │                                      │
   └──────┬───────────┴──────────────┬─────────────┬─────────┘
          ▼                          ▼             ▼
      agent/llm.py              agent/tools/    agent/trace.py
   (Groq | Ollama —            files · extract  (append-only JSONL
    the only provider-          erp · human      = the evidence)
    aware file)
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │  mockco/ (simulated)  │
                         │  FastAPI ERP + SQLite │
                         │  5 invoice documents  │
                         └───────────────────────┘
```

### Design decisions

**1 · No agent framework — rejected LangChain/LangGraph.**
The whole submission rests on being able to explain the retry, replan and
approval semantics line by line. Frameworks hide exactly the control flow I most
need to reason about, and a hidden retry is indistinguishable from a bug. The
loop is ~200 lines of plain Python.

**2 · Simulated company, real I/O — rejected browser automation.**
Driving a real website with Selenium or Playwright is the most likely thing to
break during a live walkthrough, and it would demonstrate scraping, not autonomy.
The assignment explicitly permits a simulated company application. So the
*company* is simulated and the agent's actions against it are not: real files,
real HTTP, real rows. The honest cost is in Known Limitations.

**3 · Verification re-reads from source — rejected trusting the write response.**
A write response is a claim by the component that may have failed. `--corrupt-writes`
exists to prove the difference is real and not theoretical.

**4 · Verification also audits the claim — rejected verifying state alone.**
State verification is blind to work that never happened. Added after a real
false-success run (see above). It is also the only thing that verifies read-only
tasks.

**5 · The plan is model-generated — rejected a hardcoded DAG.**
A state machine would run the invoice task perfectly and prove nothing. The test
of autonomy is whether a goal nobody anticipated still produces a sensible
sequence from the same binary. The planner gets the goal and the tool registry,
nothing else.

**6 · Approval gate enforced from the registry — rejected prompt-level rules.**
Tools declare `writes=True`; `loop._run_tool` checks that flag and gates on human
approval. A model cannot reason its way past a Python `if`. Asking permission is
the right default inside a company, and the prompt is not where safety belongs.

**7 · Model-agnostic LLM boundary — rejected SDK-coupled calls.**
Every provider detail lives in `agent/llm.py`. This started as a constraint —
no Anthropic key was available — and became a feature: this project ran on
Groq and on local Ollama with a one-line `.env` change, and no agent code knows
which.

**8 · JSONL trace as the evidence artifact — rejected plain logging.**
One record per phase with inputs, outputs, durations and errors. It is
machine-readable, which is what makes the replay harness in "what's next"
possible, and it is what the run hands back as evidence.

### Adding a new capability

To teach the agent a new company system — say Slack, or a CRM:

1. One new file in `agent/tools/` with `@register(...)` decorators.
2. Import it in `agent/loop.py`'s tool-import line.

Nothing in `loop.py`, `planner.py` or `verify.py` changes. The planner discovers
the tool through `registry.catalogue()`; the executor dispatches it through the
same path as everything else. Mark it `writes=True` and it automatically
inherits the approval gate.

---

## What is real vs simulated

| Component | Status | Notes |
|---|---|---|
| Agent loop, planning, replanning | **Real** | Model-generated plans, live tool calls |
| LLM reasoning | **Real** | Groq `openai/gpt-oss-120b`, or local Ollama |
| Field extraction | **Real** | A live LLM call per document, schema-validated in Python |
| Invoice documents | **Real files** | Plain `.txt` on disk, read through the filesystem |
| ERP writes and reads | **Real HTTP + real SQLite** | Rows persist in `mockco/erp.db`; inspect with `sqlite3` |
| Retry / backoff | **Real** | Genuine 500s from the server, genuine retries |
| Verification | **Real** | Independent `GET` re-read; real field diff |
| Approval gate | **Real** | Blocks on stdin unless `--auto-approve` |
| **The company itself** | **Simulated** | `mockco/` is a stand-in for a real ERP |
| **Fault injection** | **Simulated** | `--flaky` and `--corrupt-writes` are deliberate test hooks |
| Browser / desktop control | **Not built** | See limitations |

---

## Known limitations

1. **No browser or desktop control.** The agent operates files and an HTTP API.
   Real company work also happens in web apps and desktop software; that is the
   largest gap between this and the full problem statement.
2. **Memory is per-run.** `WorkingMemory` is discarded when the process exits.
   The agent re-learns the document store on every run and cannot build up
   knowledge of how this company operates.
3. **One company, one domain.** The tools are invoice/ERP shaped. Generalization
   is demonstrated *across tasks*, not across business domains.
4. **Steps are sequential.** Five invoices are extracted one at a time when they
   are plainly parallelisable. No task queue, no background execution.
5. **Extraction is single-pass with no confidence score.** The model reads each
   document once. There is no second opinion and no measure of how sure it is —
   only a hard distinction between "found" and "null".
6. **No authentication, authorization or multi-tenancy.** The ERP trusts every
   caller. Nothing here is safe to expose.
7. **The claim-check auditor is itself an LLM call.** If it fails, it fails open
   (state verification still stands) so the auditor cannot block a good run —
   but a determined wrong answer from the auditor is possible. It is checked
   first by a cheap structural rule that needs no model.
8. **Free-tier rate limits dominate wall-clock time.** At 8k tokens/minute, runs
   spend much of their duration waiting. The retry logic honours the server's
   hint, but the latency is real and visible.
9. **No evaluation harness.** Correctness is judged by reading traces, not by a
   scored regression suite over many runs.

---

## What I would build next

1. **Persistent cross-run company memory.** The single biggest gap. A store of
   durable facts — vendor → payment terms, which approver owns which cost
   centre, that Vertex Paper's terms are pending — written at the end of a run
   and loaded into the planner at the start. The Vertex Paper question should
   only ever be asked once.
2. **A replay harness that turns traces into regression tests.** The JSONL trace
   already records every input and output. Replaying recorded traces with the
   tool layer stubbed turns each past run into a deterministic test, so prompt
   and model changes can be evaluated without re-running live. This is what
   makes agent changes safe to ship.
3. **Real browser and desktop tools behind the same registry.** A Playwright
   tool registers exactly like `erp.py` does. The loop, planner and verifier do
   not change — which is the claim the current design makes and that this would
   actually test.
4. **A typed permission and policy layer.** Today `writes=True` is a single
   boolean. Real companies need value thresholds, per-role approvers, and an
   audit trail of who approved what — the gate is the right place, it just needs
   to be richer than a yes/no.
5. **Parallel step execution with a dependency graph.** The planner already
   produces ordered steps; making dependencies explicit would let independent
   branches run concurrently, which matters as soon as a task touches dozens of
   documents.

---

## Assumptions

- A simulated company application is acceptable in place of real third-party
  systems; the assignment states this explicitly.
- Invoices are plain text. Real ones are PDFs and scans; OCR is a separate
  problem from the agentic one being demonstrated.
- One operator at a terminal — no concurrent users, no queueing.
- The sample documents are dated 2026, so `--today` is passed in demos to make
  date-relative tasks ("overdue as of...") reproducible.
- Vendor names are taken verbatim from documents rather than normalised against
  a vendor master, so `NORTHWIND LOGISTICS PVT LTD` is stored as written.
- `amount` is the invoice total payable; tax is not decomposed.

---

## Models, APIs and components used

| Thing | What for |
|---|---|
| **Groq** — `openai/gpt-oss-120b` | Planning, tool selection, extraction, claim audit |
| **Ollama** — `qwen2.5`, `mistral` | Offline fallback provider |
| **FastAPI + Uvicorn** | The simulated ERP |
| **SQLite** (stdlib) | ERP persistence |
| **httpx** | HTTP client for both the LLM and the ERP |
| **pydantic** | Request validation in the ERP |
| **python-dotenv** | Config |
| **pytest** | Tests |

No agent framework, no orchestration library, no vector database.

---

## AI tool usage

Claude Code was used throughout, and the assignment states this is not a
disadvantage — so, precisely:

- **It generated** the bulk of the implementation: the FastAPI ERP, the tool
  modules, the trace and memory classes, the prompts, the tests, and this README.
- **I directed the architecture and the decisions**: the choice to simulate the
  company but keep all I/O real; the two-leg verification; enforcing approval
  from the registry rather than the prompt; the provider adapter; extraction
  returning null rather than guessing; and every trade-off recorded above.
- **Debugging was driven by running it.** Several things in this repo exist only
  because a real run failed and the trace showed why: the claim-check leg (an
  agent announced success having written nothing), extraction taking a filename
  instead of pasted text (context blowout and rate limits), document previews in
  `list_invoices` (the agent asked a human a question it could answer itself),
  argument binding against the schema (a model-invented kwarg crashed a step
  with `TypeError`), UTF-8 console output, and salvaging the provider's
  `tool_use_failed` responses.

The commit history shows this sequence honestly rather than as one clean drop.
