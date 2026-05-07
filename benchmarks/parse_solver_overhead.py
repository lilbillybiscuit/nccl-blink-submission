#!/usr/bin/env python3
"""Aggregate `Blink timing:` lines from solver_overhead/*.log into a CSV."""
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "results" / "solver_overhead"
TIMING = re.compile(
    r"Blink timing: extract=([\d.]+)us MWU=([\d.]+)us refine=([\d.]+)us "
    r"chain=([\d.]+)us total=([\d.]+)us"
)
EDGES = re.compile(r"Blink: extracted graph with (\d+) GPUs, (\d+) directed edges")
MWU = re.compile(r"Blink MWU: (\d+) vertices, (\d+) edges, epsilon=([\d.]+), maxIter=(\d+)")
CANDS = re.compile(r"Blink MWU: found (\d+) candidate trees")
REF = re.compile(r"Blink refine: (\d+) unique trees from (\d+) candidates, target=(\d+)")
SEL = re.compile(r"Blink refine: selected (\d+) trees, total rate=([\d.]+) GB/s")
CHAN = re.compile(r"Blink: computed (\d+) tree channels, bwIntra=([\d.]+) GB/s")


def parse_one(path):
    text = path.read_text(errors="ignore")
    timings = TIMING.findall(text)
    if not timings:
        return None
    times = [tuple(float(x) for x in t) for t in timings]
    avg = lambda i: sum(t[i] for t in times) / len(times)
    edges = EDGES.search(text)
    mwu = MWU.search(text)
    cands = CANDS.search(text)
    ref = REF.search(text)
    sel = SEL.search(text)
    chan = CHAN.search(text)
    return {
        "topology": path.stem,
        "n_inits": len(times),
        "extract_us_mean": avg(0),
        "mwu_us_mean": avg(1),
        "refine_us_mean": avg(2),
        "chain_us_mean": avg(3),
        "total_us_mean": avg(4),
        "total_us_min": min(t[4] for t in times),
        "total_us_max": max(t[4] for t in times),
        "n_gpus": int(edges.group(1)) if edges else "",
        "n_edges": int(edges.group(2)) if edges else "",
        "epsilon": float(mwu.group(3)) if mwu else "",
        "max_iter": int(mwu.group(4)) if mwu else "",
        "candidate_trees": int(cands.group(1)) if cands else "",
        "unique_trees": int(ref.group(1)) if ref else "",
        "target_channels": int(ref.group(3)) if ref else "",
        "selected_trees": int(sel.group(1)) if sel else "",
        "total_rate_gbps": float(sel.group(2)) if sel else "",
        "channels": int(chan.group(1)) if chan else "",
        "bw_intra_gbps": float(chan.group(2)) if chan else "",
    }


def main():
    rows = []
    for log in sorted(ROOT.glob("*.log")):
        r = parse_one(log)
        if r:
            rows.append(r)
    if not rows:
        print("no data parsed")
        return
    out = ROOT.parent / "solver_overhead.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")
    # Quick summary
    totals = [r["total_us_mean"] for r in rows]
    mwus = [r["mwu_us_mean"] for r in rows]
    cands = [r["candidate_trees"] for r in rows if r["candidate_trees"] != ""]
    sel = [r["selected_trees"] for r in rows if r["selected_trees"] != ""]
    print(f"total_us mean over topologies: {sum(totals)/len(totals):.1f}us "
          f"(min={min(totals):.1f}, max={max(totals):.1f})")
    print(f"mwu_us mean over topologies: {sum(mwus)/len(mwus):.1f}us "
          f"(min={min(mwus):.1f}, max={max(mwus):.1f})")
    print(f"candidate_trees: mean={sum(cands)/len(cands):.1f}, max={max(cands)}")
    print(f"selected_trees: mean={sum(sel)/len(sel):.1f}, max={max(sel)}")


if __name__ == "__main__":
    main()
