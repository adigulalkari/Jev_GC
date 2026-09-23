"""Fail if the published page disagrees with the committed measurements.

`docs/index.html` prints numbers that `results/benchmark.json` produced. Those
are two files that have to be edited together and will not be, eventually: a
re-run updates the JSON, the tables keep the old figures, and the page quietly
starts claiming something no committed artifact supports. The page's whole
argument is that its numbers are reproducible, so drift there is not a
cosmetic bug -- it makes the page dishonest.

This checker regenerates the expected cell values from the JSON and compares
them against what the HTML actually renders. It also re-runs the worked
example twice, because the page claims that script is deterministic and a
claim printed on a page should be enforced rather than trusted.

Run it locally or in CI:

    python scripts/check_docs_consistency.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_JSON = REPO_ROOT / "results" / "benchmark.json"
DOCS_HTML = REPO_ROOT / "docs" / "index.html"
WORKED_EXAMPLE = REPO_ROOT / "scripts" / "worked_example.py"

#: Captions are the only stable handle on a table here -- the markup has no
#: ids, and matching on document order would break the moment a table moves.
OVERHEAD_CAPTION = "Median of 5 sweeps"
MEMORY_CAPTION = "Median of 5."
TOKENS_CAPTION = "Identical across all 5 repeats"


class TableCollector(HTMLParser):
    """Collects (caption, rows) for every table, where each row is its cell
    text in document order. Deliberately ignores attributes: a row gaining a
    highlight class is a presentation change, not a change to the numbers."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[tuple[str, list[list[str]]]] = []
        self._caption = ""
        self._rows: list[list[str]] = []
        self._row: list[str] = []
        self._cell: list[str] = []
        self._in_caption = False
        self._in_cell = False
        self._in_table = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table":
            self._in_table = True
            self._caption = ""
            self._rows = []
        elif tag == "caption":
            self._in_caption = True
        elif tag == "tr":
            self._row = []
        elif tag in ("td", "th"):
            self._in_cell = True
            self._cell = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "table":
            self.tables.append((self._caption.strip(), self._rows))
            self._in_table = False
        elif tag == "caption":
            self._in_caption = False
        elif tag == "tr":
            if self._row:
                self._rows.append(self._row)
        elif tag in ("td", "th"):
            self._in_cell = False
            self._row.append("".join(self._cell).strip())

    def handle_data(self, data: str) -> None:
        if self._in_caption:
            self._caption += data
        elif self._in_cell:
            self._cell.append(data)


def _table(collector: TableCollector, caption_prefix: str) -> list[list[str]]:
    for caption, rows in collector.tables:
        if caption.startswith(caption_prefix):
            # Drop the header row; only the data rows carry figures.
            return [row for row in rows if row and row[0].replace(",", "").isdigit()]
    raise SystemExit(f"FAIL: no table captioned {caption_prefix!r} in {DOCS_HTML}")


def expected_overhead(summary: list[dict[str, Any]]) -> list[list[str]]:
    rows: list[list[str]] = []
    for entry in summary:
        obs = entry["observe_per_span_us"]
        total = entry["observe_total_s"]
        rows.append(
            [
                f"{entry['spans']:,}",
                f"{obs['median']:.1f}",
                f"{obs['min']:.1f}–{obs['max']:.1f}",
                f"{total['median'] * 1000:.1f}",
                f"{_median(entry, 'build_context_ms'):.3f}",
                f"{_median(entry, 'search_cold_ms'):.3f}",
            ]
        )
    return rows


def expected_memory(summary: list[dict[str, Any]]) -> list[list[str]]:
    rows: list[list[str]] = []
    for entry in summary:
        entries = int(entry["archive_entry_count"])
        retained = int(entry["archive_retained_bytes"])
        rows.append(
            [
                f"{entry['spans']:,}",
                f"{_median(entry, 'peak_rss_mb'):.1f}",
                f"{_median(entry, 'peak_rss_delta_mb'):.2f}",
                str(entries),
                f"{retained / 1024:.1f}",
                f"{retained / entries:.0f}",
            ]
        )
    return rows


def expected_tokens(summary: list[dict[str, Any]]) -> list[list[str]]:
    rows: list[list[str]] = []
    for entry in summary:
        full = int(entry["tokens_full_content"])
        saved = int(entry["tokens_saved_estimate"])
        rows.append(
            [
                f"{entry['spans']:,}",
                str(entry["tier_hot"]),
                str(entry["tier_warm"]),
                str(entry["tier_cold"]),
                f"{int(entry['tokens_rendered']):,}",
                f"{full:,}",
                f"{saved:,}",
                f"{100 * saved / full:.1f}%",
            ]
        )
    return rows


def _median(entry: dict[str, Any], field: str) -> float:
    stat = entry[field]
    return float(stat["median"])


def _compare(label: str, expected: list[list[str]], actual: list[list[str]]) -> list[str]:
    if expected == actual:
        return []
    failures = [f"{label}: table does not match results/benchmark.json"]
    for index, exp in enumerate(expected):
        act = actual[index] if index < len(actual) else None
        if act != exp:
            failures.append(f"  row {index}: page has {act}, JSON gives {exp}")
    if len(actual) > len(expected):
        failures.append(f"  page has {len(actual) - len(expected)} extra row(s)")
    return failures


def check_tables() -> list[str]:
    data = json.loads(BENCHMARK_JSON.read_text())
    if "summary" not in data:
        return [
            "results/benchmark.json has no 'summary' key -- regenerate it with "
            "`--repeat 5`, since published figures should not come from one sweep"
        ]

    collector = TableCollector()
    collector.feed(DOCS_HTML.read_text())
    summary = data["summary"]

    failures: list[str] = []
    failures += _compare("overhead", expected_overhead(summary), _table(collector, OVERHEAD_CAPTION))
    failures += _compare("memory", expected_memory(summary), _table(collector, MEMORY_CAPTION))
    failures += _compare("tokens", expected_tokens(summary), _table(collector, TOKENS_CAPTION))
    return failures


def check_headline_figures() -> list[str]:
    """The four big numbers above the tables are not table cells, so they drift
    independently and need their own check.

    Scoped to the figure tiles rather than searched for across the whole page:
    several of these values also appear in the tables and the prose, so a
    document-wide substring test passes even when the tile itself is wrong --
    which makes the check worse than none at all.
    """
    data = json.loads(BENCHMARK_JSON.read_text())
    summary = data["summary"]
    first, last = summary[0], summary[-1]

    growth = _median(last, "observe_per_span_us") / _median(first, "observe_per_span_us")
    saved_pct = 100 * last["tokens_saved_estimate"] / last["tokens_full_content"]
    archive_share = 100 * (last["archive_retained_bytes"] / 1024 / 1024) / _median(
        last, "peak_rss_delta_mb"
    )

    expected = [
        f"{growth:.2f}×",
        f"{_median(last, 'observe_per_span_us'):.0f} µs",
        f"{saved_pct:.1f}%",
        f"{archive_share:.1f}%",
    ]

    block = re.search(r'<div class="figures">(.*?)</div>\s*</div>', DOCS_HTML.read_text(), re.S)
    if block is None:
        return ["could not find the <div class=\"figures\"> block in docs/index.html"]

    actual = re.findall(r'<span class="value">(.*?)</span>', block.group(1), re.S)
    actual = [value.replace("&micro;", "µ").strip() for value in actual]

    if actual != expected:
        return [
            "headline figures do not match results/benchmark.json",
            f"  page shows {actual}",
            f"  JSON gives {expected}",
        ]
    return []


def check_worked_example_is_deterministic() -> list[str]:
    """The page states this script reproduces byte for byte. Enforce it."""
    runs: list[str] = []
    for _ in range(2):
        result = subprocess.run(
            [sys.executable, str(WORKED_EXAMPLE)],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        if result.returncode != 0:
            return [f"worked_example.py exited {result.returncode}:\n{result.stderr[-2000:]}"]
        runs.append(result.stdout)

    if runs[0] != runs[1]:
        return ["worked_example.py is not deterministic: two runs produced different output"]
    return []


def main() -> int:
    failures: list[str] = []
    failures += check_tables()
    failures += check_headline_figures()
    failures += check_worked_example_is_deterministic()

    if failures:
        print("Documentation is out of step with the committed measurements:\n")
        for line in failures:
            print(f"  {line}")
        print(
            "\nRegenerate rather than hand-editing:\n"
            "  python scripts/benchmark_overhead.py --repeat 5 --json > results/benchmark.json\n"
            "then update docs/index.html and results/report.md to match."
        )
        return 1

    print("docs/index.html agrees with results/benchmark.json; worked example is deterministic.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
