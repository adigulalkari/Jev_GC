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

Compare with [`examples/strands_dummy_agent/`](../strands_dummy_agent/):
that demo is 4 short turns and barely touches Jev (1 call out of 27
spans) because almost everything stays inside the recency window. This
example runs a longer, denser investigation (4 multi-tool-call turns
against real source/tests/history) with a tighter `keep_last_n_turns=1`,
so more spans land in the genuinely-ambiguous bucket jev-gc sends to Jev.

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
