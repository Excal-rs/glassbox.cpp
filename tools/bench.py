#!/usr/bin/env python3

# Drives the engine's --benchmark mode and makes sense of what it emits.
#
#   python3 tools/bench.py                      # sweep with repeats, aggregate
#   python3 tools/bench.py --repeats 7 --note "avx2 scores"
#   python3 tools/bench.py --no-run             # re-read the CSV, measure nothing
#   python3 tools/bench.py --self-test
#
# The engine times itself and writes one long-format row per (pass, layer,
# component). This script owns everything the engine cannot know about itself:
# which machine it ran on, which commit, how many repeats, and what the numbers
# mean once they are stacked up.

import argparse
import csv
import glob
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, asdict, field

# --------- Constants ---------

REPO   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BINARY = os.path.join(REPO, "build", "app", "glassbox_cli")
MODEL  = os.path.join(REPO, "model")
DETAIL = os.path.join(REPO, "benchmarks.csv")               # written by the engine
RUNS   = os.path.join(REPO, "tools", "bench_runs.csv")      # written here

PROMPT  = "The capital of France is"
SWEEP   = (16,)
REPEATS = 5

# One row per invocation. The engine's rows join to these on `tag`, which keeps
# a 40-character CPU name out of every one of the ~2000 rows it emits.
RUN_FIELDS = ("tag", "commit", "date", "prompt_chars", "n_tokens", "repeat",
              "cpu", "cores", "l1d", "l2", "l3", "governor", "note")

# Stages rather than work inside a block. The run-level ones happen once and
# survive the pass-0 filter; `pass` is per-pass and has to be dropped with the
# components it brackets, or the two sides describe different passes.
RUN_STAGES = ("load", "encode")
STAGES     = RUN_STAGES + ("decode", "pass")


# --------- Local Types ---------

@dataclass
class Machine:
    cpu:      str
    cores:    int
    l1d:      str
    l2:       str
    l3:       str
    governor: str


@dataclass
class Totals:
    ns:    int = 0
    flops: int = 0
    bytes: int = 0
    alloc: int = 0
    calls: int = 0
    layers: dict = field(default_factory=dict)   # layer -> ns, for spotting an outlier


# --------- Machine ---------

def probe_machine():
    cpu = ""
    for line in read_text("/proc/cpuinfo").splitlines():
        if line.startswith("model name"):
            cpu = line.split(":", 1)[1].strip()
            break

    caches = {}
    for d in sorted(glob.glob("/sys/devices/system/cpu/cpu0/cache/index*")):
        level = read_text(f"{d}/level")
        kind  = read_text(f"{d}/type")
        caches[f"l{level}{'d' if kind == 'Data' else ''}"] = read_text(f"{d}/size")

    return Machine(
        cpu      = cpu,
        cores    = os.cpu_count() or 0,
        l1d      = caches.get("l1d", ""),
        l2       = caches.get("l2", ""),
        l3       = caches.get("l3", ""),
        governor = read_text("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor", "unknown"),
    )


# The commit the binary was *run* from. The engine bakes one in at configure
# time, which goes stale the moment you rebuild without re-running cmake, so
# this is the one worth trusting -- mismatches are reported, not silently kept.
#
# The CSVs this script writes are excluded from the dirty check: appending
# measurements to them is not a change to the code that produced them, and once
# they are tracked every run would otherwise report the next one as dirty.
def probe_commit(written):
    commit = git("rev-parse", "--short", "HEAD")
    changed = [line[3:] for line in git("status", "--porcelain", "--untracked-files=no").splitlines()]
    dirty   = [path for path in changed if os.path.abspath(os.path.join(REPO, path)) not in written]
    return commit + ("-dirty" if dirty else "")


# --------- Measurement ---------

def run_engine(binary, model_dir, prompt, n_tokens, detail, tag):
    cmd = [binary, model_dir, "-p", prompt, "-n", str(n_tokens), "-o", os.devnull,
           "-b", "--bench-out", detail, "--bench-tag", tag]

    done = subprocess.run(cmd, capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} exited {done.returncode}: {done.stderr.strip()}")
    return done.stderr.strip()


# Runs the whole matrix, returning one metadata row per invocation.
def sweep(binary, model_dir, prompt, sweep_tokens, repeats, detail, commit, machine, note, verbose):
    stamp = time.strftime("%Y%m%dT%H%M%S")
    rows  = []

    for n_tokens in sweep_tokens:
        for repeat in range(repeats):
            tag = f"{commit}-{stamp}-n{n_tokens}-r{repeat}"
            started = time.perf_counter()
            run_engine(binary, model_dir, prompt, n_tokens, detail, tag)
            elapsed = time.perf_counter() - started

            rows.append({
                "tag":          tag,
                "commit":       commit,
                "date":         time.strftime("%Y-%m-%dT%H:%M:%S"),
                "prompt_chars": len(prompt),
                "n_tokens":     n_tokens,
                "repeat":       repeat,
                **asdict(machine),
                "note":         note,
            })

            if verbose:
                print(f"  -n {n_tokens:<3} repeat {repeat}  {elapsed:.3f}s wall")

    return rows


# --------- Aggregation ---------

def load_detail(path, tags):
    if not os.path.exists(path):
        die(f"no engine output at {path} -- run without --no-run first")

    with open(path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if not tags or r["tag"] in tags]

    if not rows:
        die(f"{path} holds no rows for this session")

    for r in rows:
        for key in ("pass_index", "seq_len", "layer", "calls", "ns", "flops", "bytes", "alloc"):
            r[key] = int(r[key])

    return rows


# Folds the engine's rows into one entry per component, per repeat.
# Pass 0 is dropped by default: it touches every weight page for the first time,
# so it is reliably slower than the passes after it and is not a repeat of them.
def fold(rows, drop_first):
    per_tag = {}

    for r in rows:
        if drop_first and r["component"] not in RUN_STAGES and r["pass_index"] == 0:
            continue

        totals = per_tag.setdefault(r["tag"], {}).setdefault(r["component"], Totals())
        totals.ns    += r["ns"]
        totals.flops += r["flops"]
        totals.bytes += r["bytes"]
        totals.alloc += r["alloc"]
        totals.calls += r["calls"]
        totals.layers[r["layer"]] = totals.layers.get(r["layer"], 0) + r["ns"]

    return per_tag


# Median across repeats for every component, with the shares that make the
# medians comparable. Median rather than mean: one thermally-stalled repeat
# should move the answer by nothing.
def summarise(per_tag):
    components = sorted({c for tags in per_tag.values() for c in tags})
    summary    = {}

    for component in components:
        seen = [t[component] for t in per_tag.values() if component in t]
        if not seen:
            continue

        summary[component] = {
            "ns":     statistics.median(s.ns for s in seen),
            "spread": (max(s.ns for s in seen) - min(s.ns for s in seen)) / max(1, statistics.median(s.ns for s in seen)),
            "flops":  statistics.median(s.flops for s in seen),
            "bytes":  statistics.median(s.bytes for s in seen),
            "alloc":  statistics.median(s.alloc for s in seen),
            "calls":  statistics.median(s.calls for s in seen),
            "layers": seen[0].layers,
            "runs":   len(seen),
        }

    return summary


def report(summary):
    inside = {c: s for c, s in summary.items() if c not in STAGES}
    total  = sum(s["ns"] for s in inside.values())

    print(f"{'component':<12}{'share':>8}{'ms':>10}{'GFLOP/s':>10}{'GB/s':>8}"
          f"{'fl/byte':>9}{'calls':>8}{'spread':>8}")

    for component, s in sorted(inside.items(), key=lambda kv: -kv[1]["ns"]):
        seconds   = s["ns"] / 1e9
        gflops    = s["flops"] / s["ns"] if s["ns"] else 0        # flops/ns == Gflop/s
        gbs       = s["bytes"] / s["ns"] if s["ns"] else 0
        intensity = s["flops"] / s["bytes"] if s["bytes"] else 0

        print(f"{component:<12}{s['ns'] / total * 100:>7.1f}%{seconds * 1e3:>10.1f}"
              f"{gflops:>10.2f}{gbs:>8.2f}{intensity:>9.2f}{int(s['calls']):>8}"
              f"{s['spread'] * 100:>7.1f}%")

    print(f"{'':<12}{'':>8}{'':>10}")
    for stage in STAGES:
        if stage in summary:
            print(f"{stage:<12}{'':>8}{summary[stage]['ns'] / 1e6:>10.1f} ms")

    return inside, total


# The engine's own totals have to agree with the parts it reports. A gap means
# work is happening between the probes, which is worth finding before any of
# these percentages are trusted.
def check_coverage(summary, inside, total):
    if "pass" not in summary:
        return ["no `pass` rows, so the component total cannot be checked against anything"]

    passes = summary["pass"]["ns"]
    gap    = (passes - total) / passes * 100 if passes else 0

    if abs(gap) > 5:
        return [f"components account for {100 - gap:.1f}% of the timed passes -- "
                f"{gap:.1f}% is happening between the probes"]

    print(f"\ncoverage: components sum to {total / passes * 100:.1f}% of the timed passes")
    return []


# Cross-checks the engine's FLOP counting against an independent model, so the
# two cannot quietly drift apart. They are derived separately on purpose.
def check_flops(summary, config, seq_lens, passes):
    counted = sum(s["flops"] for c, s in summary.items() if c not in STAGES)
    if not counted:
        return ["the engine reported zero FLOPs"]

    expected = sum(flops_pass(config, seq) for seq in seq_lens)
    off      = abs(counted - expected) / expected * 100 if expected else 0

    if off > 5:
        return [f"engine counted {counted / 1e9:.2f} GFLOP, this script's model says "
                f"{expected / 1e9:.2f} GFLOP ({off:.0f}% apart) -- one of them is wrong"]

    print(f"flop model: engine {counted / 1e9:.2f} GFLOP vs model {expected / 1e9:.2f} GFLOP, {off:.1f}% apart")
    return []


# Same arithmetic as the engine's cost helpers, derived from config alone.
def flops_pass(config, seq):
    n, layers = config["n_embd"], config["n_layer"]
    return layers * (24 * seq * n * n + 4 * seq * seq * n) + 2 * config["vocab_size"] * n


# --------- Self Test ---------

def self_test():
    cases = []

    rows = [
        {"tag": "a", "pass_index": 0, "seq_len": 5, "layer": 0, "component": "scores",
         "calls": 1, "ns": 900, "flops": 100, "bytes": 50, "alloc": 10},
        {"tag": "a", "pass_index": 1, "seq_len": 6, "layer": 0, "component": "scores",
         "calls": 1, "ns": 100, "flops": 100, "bytes": 50, "alloc": 10},
        {"tag": "a", "pass_index": 1, "seq_len": 6, "layer": 1, "component": "scores",
         "calls": 1, "ns": 200, "flops": 100, "bytes": 50, "alloc": 10},
        {"tag": "a", "pass_index": 1, "seq_len": 6, "layer": 0, "component": "pass",
         "calls": 1, "ns": 310, "flops": 0, "bytes": 0, "alloc": 0},
    ]

    folded = fold(rows, drop_first=True)
    cases.append(("pass 0 dropped, layers summed", folded["a"]["scores"].ns == 300))

    folded_all = fold(rows, drop_first=False)
    cases.append(("--keep-first keeps pass 0", folded_all["a"]["scores"].ns == 1200))

    stage_rows = [
        {"tag": "a", "pass_index": 0, "seq_len": 5, "layer": 12, "component": "pass",
         "calls": 1, "ns": 900, "flops": 0, "bytes": 0, "alloc": 0},
        {"tag": "a", "pass_index": 1, "seq_len": 6, "layer": 12, "component": "pass",
         "calls": 1, "ns": 300, "flops": 0, "bytes": 0, "alloc": 0},
        {"tag": "a", "pass_index": 0, "seq_len": 5, "layer": 12, "component": "load",
         "calls": 1, "ns": 500, "flops": 0, "bytes": 0, "alloc": 0},
    ]
    staged = fold(stage_rows, drop_first=True)
    cases.append(("pass 0 dropped from `pass` too, load kept",
                  staged["a"]["pass"].ns == 300 and staged["a"]["load"].ns == 500))

    summary = summarise({"x": {"scores": Totals(ns=100)}, "y": {"scores": Totals(ns=300)},
                         "z": {"scores": Totals(ns=200)}})
    cases.append(("median across repeats, not mean", summary["scores"]["ns"] == 200))

    inside = {"scores": {"ns": 300}}
    cases.append(("coverage gap is reported",
                  bool(check_coverage({"pass": {"ns": 1000}, **inside}, inside, 300))))

    config = {"n_embd": 768, "n_layer": 12, "vocab_size": 50257}
    cases.append(("flop model counts the quadratic term",
                  flops_pass(config, 100) > flops_pass(config, 50) * 2))

    for name, ok in cases:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")

    failed = sum(not ok for _, ok in cases)
    print(f"\n{len(cases) - failed}/{len(cases)} checks passed")
    return 1 if failed else 0


# --------- Main ---------

def main():
    ap = argparse.ArgumentParser(description="Drive and aggregate the engine's --benchmark mode.")
    ap.add_argument("--binary",  default=BINARY, metavar="PATH")
    ap.add_argument("--model",   default=MODEL,  metavar="DIR")
    ap.add_argument("--detail",  default=DETAIL, metavar="PATH", help="CSV the engine appends to")
    ap.add_argument("--runs",    default=RUNS,   metavar="PATH", help="CSV of per-invocation metadata")
    ap.add_argument("--prompt",  default=PROMPT)
    ap.add_argument("--sweep",   default=",".join(map(str, SWEEP)), help="token counts to measure")
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--note",    default="", help="free text stored with every run row")
    ap.add_argument("--no-run",     action="store_true", help="aggregate the existing CSV, measure nothing")
    ap.add_argument("--keep-first", action="store_true", help="keep pass 0, which touches cold weight pages")
    ap.add_argument("--self-test",  action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    machine = probe_machine()
    commit  = probe_commit({os.path.abspath(args.detail), os.path.abspath(args.runs)})

    warnings = []
    if machine.governor != "performance":
        warnings.append(f"governor is {machine.governor}, not performance")
    if commit.endswith("-dirty"):
        warnings.append("working tree is dirty, so this commit does not identify the code that ran")

    print(f"{commit}  {machine.cpu}  {machine.cores} cores  governor={machine.governor}")

    tags = set()
    if args.no_run:
        print("--no-run: aggregating what is already in the CSV\n")
    else:
        if not os.path.exists(args.binary):
            die(f"no engine at {args.binary} -- build it, or pass --binary")

        sweep_tokens = tuple(int(s) for s in args.sweep.split(","))
        print(f"sweep {sweep_tokens} x {args.repeats} repeats, prompt {args.prompt!r}\n")

        runs = sweep(args.binary, args.model, args.prompt, sweep_tokens, args.repeats,
                     args.detail, commit, machine, args.note, args.verbose)
        tags = {r["tag"] for r in runs}
        append_rows(args.runs, RUN_FIELDS, runs)

    rows    = load_detail(args.detail, tags)
    folded  = fold(rows, drop_first=not args.keep_first)
    summary = summarise(folded)

    builds = {r["build"] for r in rows}
    if builds - {"release"}:
        warnings.append(f"rows from a {'/'.join(sorted(builds))} build")

    baked = {r["commit"] for r in rows}
    if not args.no_run and baked != {commit.removesuffix("-dirty")}:
        warnings.append(f"engine reports commit {'/'.join(sorted(baked))}, git says {commit} "
                        "-- the baked-in commit is set at cmake time")

    print()
    inside, total = report(summary)

    warnings += check_coverage(summary, inside, total)

    config = load_json(f"{args.model}/config.json")
    if config:
        seq_lens = sorted({r["seq_len"] for r in rows if r["component"] == "pass"})
        warnings += check_flops(summary, config, seq_lens, len(seq_lens))

    if "pass" in summary and summary["pass"]["ns"]:
        passes = summary["pass"]["calls"]
        print(f"tok/s: {passes / (summary['pass']['ns'] / 1e9):.2f} over {int(passes)} passes, "
              f"load excluded")

    for warning in warnings:
        print(f"  warning: {warning}")

    if not args.no_run:
        print(f"\ndetail in {args.detail}, run metadata in {args.runs}")

    return 1 if any("wrong" in w or "between the probes" in w for w in warnings) else 0


# --------- Helper Function Definitions ---------

def die(msg):
    print(f"error: {msg}")
    raise SystemExit(1)


def read_text(path, default=""):
    try:
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return default


def load_json(path):
    import json
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except OSError:
        return None


def git(*args):
    try:
        done = subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True, check=True)
        return done.stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def append_rows(path, fields, rows):
    fresh = not os.path.exists(path) or os.path.getsize(path) == 0

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if fresh:
            writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    sys.exit(main())


# --------- What each side owns ---------
#
# The engine reports facts about itself: the commit baked in at cmake time,
# whether NDEBUG was defined, and for every (pass, layer, component) the calls,
# nanoseconds, flops, bytes moved and bytes allocated. It does one run and
# aggregates nothing -- passes are never averaged together, because with no KV
# cache each pass runs over a longer prefix and they are different measurements.
#
# This script owns everything the engine cannot know: which machine, which real
# commit (git, not the baked one), how many repeats, and what the numbers mean
# stacked up. Hardware lives in the runs CSV, joined on `tag`, rather than being
# repeated onto a couple of thousand detail rows.
#
# Three things are checked rather than assumed:
#
#   - components must sum to within 5% of the engine's own pass timing, or work
#     is happening between the probes and every share below is misleading;
#   - the engine's counted FLOPs must agree with this script's independent model
#     to within 5%, since a hand-written cost at a call site is easy to get wrong;
#   - the commit git reports must match the one the binary was built with, which
#     catches a rebuild that skipped cmake.
