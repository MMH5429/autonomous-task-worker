# Demo runbook

Two terminals. Terminal 1 = the company (mockco). Terminal 2 = the agent.
Each agent run takes 60-130s (free-tier rate limits), so narrate while it runs.

## 1. Happy path — resolving ambiguity  (~70s)
T1:  python mockco/erp.py --port 8099 --reset
T2:  python run.py "Find the latest invoice from Northwind Logistics, extract the amount and due date, enter it into our internal system, and tell me once it is done." --auto-approve --today 2026-10-04
Say: "Two Northwind invoices exist. Nothing tells it which is latest - it reads
      both and compares dates. And this plan came from the model; there is no
      step list in my source."
Then: python -c "import sqlite3;print(list(sqlite3.connect('mockco/erp.db').execute('select * from entries')))"

## 2. Transient failure  (~65s)
T1:  Ctrl+C, then  python mockco/erp.py --port 8099 --flaky --reset
T2:  (same command as scenario 1)
Say: "First write 500s. Retry with backoff, succeeds. The failure stays in the
      trace on purpose - I want it visible, not swallowed."

## 3. Missing data - asks instead of guessing  (~60s)
T1:  Ctrl+C, then  python mockco/erp.py --port 8099 --reset
T2:  python run.py "Enter the Vertex Paper invoice into our system." --today 2026-10-04
     (answer the due-date prompt with: 2026-11-15, then approve the write with: y)
Say: "No due date in that document. It could have inferred Net 30 from the
      others. Guessing a due date in an ERP is worse than asking."

## 4. THE IMPORTANT ONE - verification catches a silent corruption  (~75s)
T1:  Ctrl+C, then  python mockco/erp.py --port 8099 --corrupt-writes --reset
T2:  (same command as scenario 1)
Say: "The ERP stores zero but returns the correct amount in the write response.
      Every step succeeded. The run still comes out FAILED, with the diff,
      because verification re-reads from the system of record instead of
      trusting the write it just made."

## 5. Generalization  (~30s, or just show scrollback)
T2:  python run.py "Which invoices are overdue as of 4 October 2026? Give me a summary." --today 2026-10-04
Say: "Same binary, different goal, no code change. git status shows nothing."

## Close (~20s)
"Biggest limitation: memory is per-run, so it re-learns the company every time.
 Next thing I would build is persistent company memory, and a replay harness
 that turns these JSONL traces into regression tests."
