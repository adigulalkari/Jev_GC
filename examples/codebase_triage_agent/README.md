# Codebase triage agent (Mistral)

A bug-triage agent that investigates real bug reports against **this
actual repository** -- jev-gc's own source, tests, and git history. No
mocked data anywhere: every tool call is a real subprocess or filesystem
read.

| Tool | What it actually does |
|---|---|
| `grep_codebase(pattern, path)` | Real `git grep` against tracked files |
| `read_file(path, start_line, end_line)` | Reads a real line range; raises `FileNotFoundError` on a bad path |
| `git_log(path, max_commits)` | Real `git log` for a file/dir |
| `run_pytest(test_path)` | Actually runs a test file with pytest and reports pass/fail |

Tuned with a tight `keep_last_n_turns=1` so, across a real multi-turn
investigation (real source/tests/history, not one-liners), more spans
land in the genuinely-ambiguous bucket jev-gc actually sends to Jev,
rather than everything getting resolved for free by the recency window
alone.

## Run it

```bash
pip install -e ".[examples]"
export JEV_API_KEY=...        # from the TypeSafe dashboard; omit to run against FakeJevClient
export MISTRAL_API_KEY=...    # required -- from console.mistral.ai
python examples/codebase_triage_agent/agent.py
```

**Mistral gotcha:** a freshly created API key can still return `429 Rate
limit exceeded` on every call until you explicitly activate a plan
(even the free "Experiment" tier) in the console's Billing section --
an unactivated workspace has an effective rate limit of zero, which is
easy to mistake for a transient rate limit or a hang.

It runs four investigations (a policy/test correctness question, a real
git-history lookup, an actual test run, and a deliberate bad file path
to produce a genuine error span), then prints the assembled
budget-fit context and `JevGC.stats()`.
