#!/usr/bin/env python3
"""Turn results.csv into a Markdown table you can paste into the README."""
import csv, sys

path = sys.argv[1] if len(sys.argv) > 1 else "results.csv"
rows = list(csv.DictReader(open(path)))
cols = [("label", "Run"), ("concurrency", "Conc."), ("ttft_p50_s", "TTFT p50 (s)"),
        ("e2e_p50_s", "E2E p50 (s)"), ("e2e_p95_s", "E2E p95 (s)"),
        ("tpot_mean_ms", "TPOT (ms)"), ("output_tok_per_s", "Tokens/s")]
print("| " + " | ".join(h for _, h in cols) + " |")
print("|" + "---|" * len(cols))
for r in rows:
    cells = []
    for k, _ in cols:
        v = r[k]
        try:
            f = float(v)
            cells.append(f"{f:.0f}" if k in ("concurrency", "output_tok_per_s") else f"{f:.2f}")
        except ValueError:
            cells.append(v)
    print("| " + " | ".join(cells) + " |")
