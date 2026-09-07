#!/usr/bin/env python3

# Tier 2a validation of GBIC dumps. numpy only, no torch, no HuggingFace.
#
#   python3 tools/validate.py              # generate dumps and run the suite
#   python3 tools/validate.py dump.gbic    # check one existing dump
#   python3 tools/validate.py -v --keep D  # every check, dumps left in D

import argparse
import json
import os
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

# --------- Constants ---------

REPO   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BINARY = os.path.join(REPO, "build", "app", "glassbox_cli")
MODEL  = os.path.join(REPO, "model")

PROMPT = "The capital of France is"

HEADER = 32

# Mirrors AblationType in include/glassbox/interp.h, spelled the way the CLI spells it.
ABLATIONS = {
    0: "none",
    1: "zero-attn",
    2: "zero-mlp",
    3: "patch-attn",
    4: "patch-mlp",
}

# One block's buffers, in the order dump_cache() writes them.
FIELDS = ("attention_output", "stream_post_attention", "mlp_output", "stream_post_mlp")


# --------- Local Types ---------

@dataclass
class Cache:
    n_layer:    int
    n_embd:     int
    seq_len:    int
    n_prompt:   int
    ablation:   str
    abl_layer:  int | None
    ids:        np.ndarray
    embeddings: np.ndarray
    layers:     list[dict[str, np.ndarray]]


@dataclass
class Result:
    status:   str                                    # PASS, FAIL or SKIP
    message:  str
    problems: list[str] = field(default_factory=list)


@dataclass
class Case:
    group:  str
    name:   str
    build:  Callable[[str], str]   # given a directory, writes the dump and returns its path
    expect: dict[str, str]         # check -> required status; every other check must not FAIL


# --------- Reader ---------

# Parses a dump into (seq_len, n_embd) views over the file's own bytes.
# Nothing is repaired or guessed at: a size that misses by four bytes is a
# layout bug, and it is cheaper to catch here than as a delta ten checks later.
def read(path):
    with open(path, "rb") as f:
        raw = f.read()

    if raw[:4] != b"GBIC":
        raise ValueError(f"bad magic {raw[:4]!r}")

    version, n_layer, n_embd, seq_len, n_prompt, abl_type, abl_layer = struct.unpack("<7I", raw[4:HEADER])

    if version != 2:                die(f"unsupported version {version}, expected 2", ValueError)
    if abl_type not in ABLATIONS:   die(f"unknown ablation type {abl_type}", ValueError)
    if not (n_layer and n_embd and seq_len):
        die(f"degenerate header: n_layer={n_layer} n_embd={n_embd} seq_len={seq_len}", ValueError)

    want = HEADER + 4 * seq_len + 4 * (1 + 4 * n_layer) * seq_len * n_embd
    if len(raw) != want:
        die(f"size mismatch: {len(raw)} bytes on disk, {want} implied by the header", ValueError)

    ids  = np.frombuffer(raw, "<u4", count=seq_len, offset=HEADER)
    bufs = np.frombuffer(raw, "<f4", offset=HEADER + 4 * seq_len).reshape(-1, seq_len, n_embd)

    return Cache(
        n_layer    = n_layer,
        n_embd     = n_embd,
        seq_len    = seq_len,
        n_prompt   = n_prompt,
        ablation   = ABLATIONS[abl_type],
        abl_layer  = None if abl_type == 0 else abl_layer,
        ids        = ids,
        embeddings = bufs[0],
        layers     = [dict(zip(FIELDS, bufs[1 + 4 * i : 5 + 4 * i])) for i in range(n_layer)],
    )


# Reads named F32 tensors out of a safetensors file.
def load_tensors(model_dir, names):
    with open(f"{model_dir}/model.safetensors", "rb") as f:
        header = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
        base   = f.tell()

        out = {}
        for name in names:
            meta = header[name]
            if meta["dtype"] != "F32":
                die(f"{name} is {meta['dtype']}, expected F32", ValueError)

            start, end = meta["data_offsets"]
            f.seek(base + start)
            out[name] = np.frombuffer(f.read(end - start), "<f4").reshape(meta["shape"])

    return out


# --------- Checks ---------

# The stream entering block i: the embeddings at 0, the previous block's output after.
def stream_in(c, i):
    return c.embeddings if i == 0 else c.layers[i - 1]["stream_post_mlp"]


# Bitwise float32 equality, plus a description of the worst mismatch.
def exact(a, b):
    a = a.astype(np.float32, copy=False)
    if np.array_equal(a, b):
        return True, ""

    d = np.abs(a.astype(np.float64) - b.astype(np.float64))

    # A NaN difference is the worst kind, but argmax walks straight past it.
    t, ch = np.unravel_index(int(np.where(np.isnan(d), np.inf, d).argmax()), d.shape)
    return False, f"{int((a != b).sum())}/{b.size} elements differ, max |Δ| {d[t, ch]:.3e} at [t={t}, c={ch}]"


# Every buffer in the dump, named the way the report should name it.
def buffers(c):
    yield "init_embeddings", c.embeddings
    for i, layer in enumerate(c.layers):
        for name in FIELDS:
            yield f"L{i:02d}.{name}", layer[name]


# Both residual adds of every block, as (layer, sublayer, ok, why).
def identities(c):
    for i, layer in enumerate(c.layers):
        yield (i, "attn", *exact(stream_in(c, i) + layer["attention_output"], layer["stream_post_attention"]))
        yield (i, "mlp",  *exact(layer["stream_post_attention"] + layer["mlp_output"], layer["stream_post_mlp"]))


# The (layer, sublayer) whose identity the ablation is meant to break.
def target_of(c):
    if c.ablation == "none":
        return None
    return c.abl_layer, "attn" if c.ablation.endswith("attn") else "mlp"


# The dump's geometry is this model's geometry, and its ids are real tokens.
def check_header(c, model_dir):
    try:
        with open(f"{model_dir}/config.json", encoding="utf-8") as f:
            config = json.load(f)
    except OSError as exc:
        return Result("SKIP", f"no config.json ({exc.strerror}: {model_dir})")

    n_ctx    = config.get("n_ctx", config.get("n_positions"))
    n_vocab  = config["vocab_size"]
    problems = []

    if c.n_layer != config["n_layer"]:
        problems.append(f"n_layer {c.n_layer} != config {config['n_layer']}")
    if c.n_embd != config["n_embd"]:
        problems.append(f"n_embd {c.n_embd} != config {config['n_embd']}")
    if c.seq_len > n_ctx:
        problems.append(f"seq_len {c.seq_len} exceeds the model's context {n_ctx}")
    if not 1 <= c.n_prompt <= c.seq_len:
        problems.append(f"n_prompt {c.n_prompt} is not in 1..{c.seq_len}")

    bad = np.flatnonzero(c.ids >= n_vocab)
    if len(bad):
        problems.append(f"token ids out of [0, {n_vocab}) at positions {bad.tolist()[:8]}")

    return verdict(f"{c.n_layer}x{c.n_embd} matches config, ids within [0, {n_vocab})", problems)


# No NaN or inf anywhere, and no buffer the engine forgot to write.
def check_values(c):
    problems = []

    for name, buf in buffers(c):
        if not np.isfinite(buf).all():
            t, ch = np.unravel_index(int(np.argmin(np.isfinite(buf))), buf.shape)
            problems.append(f"{name}: {int(np.isnan(buf).sum())} NaN, {int(np.isinf(buf).sum())} inf, "
                            f"first at [t={t}, c={ch}]")
        elif not buf.any():
            problems.append(f"{name}: entirely zero -- the buffer was never written")

    return verdict(f"{4 * c.n_layer + 1} buffers finite and nonzero", problems)


# init_embeddings[t] == wte[ids[t]] + wpe[t], bitwise. Nothing arithmetic sits
# between embed() and the capture, so recomputing it also proves the header's
# ids are the ones these activations came from.
def check_embeddings(c, model_dir):
    try:
        w = load_tensors(model_dir, ["wte.weight", "wpe.weight"])
    except (OSError, KeyError, ValueError) as exc:
        return Result("SKIP", f"no usable model.safetensors ({exc})")

    wte, wpe = w["wte.weight"], w["wpe.weight"]

    if wte.shape[1] != c.n_embd or c.seq_len > wpe.shape[0] or int(c.ids.max()) >= wte.shape[0]:
        return Result("FAIL", "1 problem(s)", [f"dump {c.seq_len}x{c.n_embd}, ids up to {int(c.ids.max())}, "
                                               f"does not fit wte {wte.shape} / wpe {wpe.shape}"])

    ok, why = exact(wte[c.ids] + wpe[: c.seq_len], c.embeddings)
    return verdict(f"all {c.seq_len} rows are wte[id] + wpe[t]", [] if ok else [why])


# Every residual add reproduces, except the one an ablation is supposed to break.
def check_identities(c):
    target   = target_of(c)
    problems = []
    held     = 0

    for i, kind, ok, why in identities(c):
        if (i, kind) == target:
            if ok:
                problems.append(f"L{i} {kind}: the identity still holds at the ablated layer -- the cache "
                                "stored the ablated output rather than the clean one")
        elif ok:
            held += 1
        else:
            problems.append(f"L{i} {kind}: {why}")

    total   = 2 * c.n_layer - (1 if target else 0)
    message = f"{held}/{total} residual adds bitwise exact"
    if target:
        message += f" (L{target[0]} {target[1]} excluded, it is the ablated one)"

    return verdict(message, problems)


# The ablated sublayer was skipped, and the cache kept its clean output anyway.
def check_ablation(c):
    if c.ablation == "none":
        return Result("SKIP", "clean dump, nothing was ablated")
    if c.ablation.startswith("patch"):
        return Result("SKIP", "patch ablation, the patch values are not in the dump")

    i, kind = target_of(c)
    layer   = c.layers[i]

    if kind == "attn":
        checkpoint            = "stream_post_attention"
        before, after, clean  = stream_in(c, i), layer["stream_post_attention"], layer["attention_output"]
    else:
        checkpoint            = "stream_post_mlp"
        before, after, clean  = layer["stream_post_attention"], layer["stream_post_mlp"], layer["mlp_output"]

    problems = []

    ok, why = exact(before, after)
    if not ok:
        problems.append(f"{checkpoint}[{i}] is not the stream that entered the sublayer: {why}")
    if not clean.any():
        problems.append(f"the cached {kind} output at layer {i} is all zero, so it is not the clean value")

    return verdict(f"{checkpoint}[{i}] is bitwise the stream before it, and the clean {kind} output is still cached",
                   problems)


# Every check that can run on one file. A bad header ends the list there.
def evaluate(path, model_dir):
    try:
        c = read(path)
    except (OSError, ValueError, struct.error) as exc:
        return {"reader": Result("FAIL", "1 problem(s)", [str(exc)])}

    ablation = c.ablation if c.ablation == "none" else f"{c.ablation}@{c.abl_layer}"

    return {
        "reader":     Result("PASS", f"GBIC v2 {c.n_layer}x{c.n_embd} seq_len={c.seq_len} "
                                     f"({c.n_prompt} prompt) ablation={ablation}"),
        "header":     check_header(c, model_dir),
        "values":     check_values(c),
        "embeddings": check_embeddings(c, model_dir),
        "identities": check_identities(c),
        "ablation":   check_ablation(c),
    }


# --------- Suite ---------

# Runs the engine once, returning the dump it wrote.
def run_engine(binary, model_dir, prompt, n_tokens, dump, ablate=None):
    cmd = [binary, model_dir, "-p", prompt, "-n", str(n_tokens), "-i", dump]
    if ablate:
        cmd += ["-a", ablate]

    done = subprocess.run(cmd, capture_output=True, text=True)
    if done.returncode != 0:
        die(f"{' '.join(cmd)} exited {done.returncode}: {done.stderr.strip()}", RuntimeError)

    return dump


# Byte range of payload buffer `index`: 0 is init_embeddings, then 1 + 4*layer + field.
def buffer_span(raw, index):
    _, _, n_embd, seq_len, *_ = struct.unpack("<7I", raw[4:HEADER])

    stride = 4 * seq_len * n_embd
    start  = HEADER + 4 * seq_len + index * stride
    return slice(start, start + stride)


# Introduces exactly one defect, of the kind a real layout or hook bug produces.
def corrupt(raw, how):
    if how == "truncate": return raw[:-4]
    if how == "extend":   return raw + b"\0\0\0\0"

    out = bytearray(raw)

    if how == "nan":                                     # L3 mlp_output[0][0]
        at = buffer_span(raw, 1 + 4 * 3 + 2).start
        out[at : at + 4] = struct.pack("<f", float("nan"))
    elif how == "unwritten":                             # L4 attention_output left blank
        span = buffer_span(raw, 1 + 4 * 4 + 0)
        out[span] = bytes(span.stop - span.start)
    elif how == "mislabel":                              # a clean dump claiming zero-attn:5
        out[24:HEADER] = struct.pack("<2I", 1, 5)
    else:
        die(f"no such corruption: {how}", ValueError)

    return bytes(out)


# Twenty cases: five clean, five zero-attn, five zero-mlp, five corrupted.
def build_cases(binary, model_dir):
    cases = []

    def engine(name, prompt, n_tokens, ablate=None):
        return lambda d: run_engine(binary, model_dir, prompt, n_tokens, os.path.join(d, name), ablate)

    # The lengths vary because a single fixed prompt never reaches the geometry
    # the header check is there for: one token, a long prompt, and two dumps
    # that cover generated tokens as well, where n_prompt < seq_len.
    clean = (
        ("baseline",            PROMPT,  1),
        ("single-token prompt", "Hello", 1),
        ("long prompt",         "In a shocking finding, scientists discovered a herd of unicorns "
                                "living in a remote valley in the Andes mountains.", 1),
        ("8 tokens generated",  "Hello", 8),
        ("16 tokens generated", PROMPT,  16),
    )
    for i, (name, prompt, n_tokens) in enumerate(clean):
        cases.append(Case("clean", name, engine(f"clean-{i}.gbic", prompt, n_tokens), {"ablation": "SKIP"}))

    # Layer 0 earns its place: there stream_in is init_embeddings rather than a
    # previous block's output, which is a different path through forward().
    for kind, layers in (("zero-attn", (0, 3, 5, 8, 11)), ("zero-mlp", (0, 2, 6, 9, 11))):
        for layer in layers:
            spec = f"{kind}:{layer}"
            cases.append(Case(kind, f"layer {layer}",
                              engine(f"{kind}-{layer}.gbic", PROMPT, 1, spec),
                              {"ablation": "PASS", "identities": "PASS"}))

    # Dumps that must be rejected, each by the check named alongside it. This is
    # the group that keeps the other fifteen honest.
    def mutant(how):
        def build(d):
            base = run_engine(binary, model_dir, PROMPT, 1, os.path.join(d, f"base-{how}.gbic"))
            path = os.path.join(d, f"{how}.gbic")
            with open(base, "rb") as f, open(path, "wb") as g:
                g.write(corrupt(f.read(), how))
            return path
        return build

    broken = (
        ("truncate",  "four bytes short",                {"reader": "FAIL"}),
        ("extend",    "four trailing bytes",             {"reader": "FAIL"}),
        ("nan",       "NaN in L3 mlp_output",            {"values": "FAIL", "identities": "FAIL"}),
        ("unwritten", "L4 attention_output all zero",    {"values": "FAIL", "identities": "FAIL"}),
        ("mislabel",  "clean dump labelled zero-attn:5", {"identities": "FAIL", "ablation": "FAIL"}),
    )
    for how, name, expect in broken:
        cases.append(Case("corruption", name, mutant(how), expect))

    return cases


# A case passes when the named checks have the named statuses and nothing else failed.
def judge(results, expect):
    problems = []

    for name, want in expect.items():
        got = results.get(name)
        if got is None:
            problems.append(f"{name} did not run (expected {want})")
        elif got.status != want:
            problems.append(f"{name} is {got.status}, expected {want}")

    for name, result in results.items():
        if name not in expect and result.status == "FAIL":
            problems.append(f"{name} FAIL: {'; '.join(result.problems) or result.message}")

    return problems


def run_suite(binary, model_dir, keep, verbose):
    if not os.path.exists(binary):
        print(f"no engine at {binary} -- build it, or pass --binary")
        return 1

    cases = build_cases(binary, model_dir)
    print(f"{len(cases)} cases  binary={binary}  model={model_dir}\n")

    scratch   = tempfile.TemporaryDirectory(prefix="glassbox-validate-")
    directory = keep or scratch.name
    if keep:
        scratch.cleanup()
        os.makedirs(keep, exist_ok=True)

    failed = 0
    group  = None

    try:
        for case in cases:
            if case.group != group:
                group = case.group
                print(group)

            try:
                results  = evaluate(case.build(directory), model_dir)
                problems = judge(results, case.expect)
            except (RuntimeError, OSError) as exc:
                results, problems = {}, [str(exc)]

            wanted = ", ".join(f"{k} {v}" for k, v in case.expect.items())
            print(f"  {'FAIL' if problems else 'PASS'}  {case.name:<34} expects {wanted}")
            for problem in problems:
                print(f"        {problem}")
            if verbose and results:
                report(results, indent="        ")

            failed += bool(problems)
    finally:
        if not keep:
            scratch.cleanup()

    print(f"\n{len(cases) - failed}/{len(cases)} cases passed")
    if keep:
        print(f"dumps kept in {keep}")

    return 1 if failed else 0


def main():
    ap = argparse.ArgumentParser(description="Tier 2a validation of GBIC v2 interp dumps.")
    ap.add_argument("dump", nargs="?", help="check one existing dump instead of running the suite")
    ap.add_argument("--model",  default=MODEL,  metavar="DIR")
    ap.add_argument("--binary", default=BINARY, metavar="PATH")
    ap.add_argument("--keep",   metavar="DIR", help="write the suite's dumps here instead of a temp dir")
    ap.add_argument("-v", "--verbose", action="store_true", help="print every check of every case")
    args = ap.parse_args()

    if args.dump:
        results = evaluate(args.dump, args.model)
        print(args.dump)
        report(results)
        return 1 if any(r.status == "FAIL" for r in results.values()) else 0

    return run_suite(args.binary, args.model, args.keep, args.verbose)


# --------- Helper Function Definitions ---------

# Refuses to carry on, in the spirit of die() in src/utils.cpp.
def die(msg, kind=ValueError):
    raise kind(msg)


# A check's outcome: what it proved, or everything it found wrong.
def verdict(message, problems):
    if problems:
        return Result("FAIL", f"{len(problems)} problem(s)", problems)
    return Result("PASS", message)


# Prints one check per line, with any problems indented beneath it.
def report(results, indent="  "):
    for name, result in results.items():
        print(f"{indent}{name:<11}{result.status}  {result.message}")
        for problem in result.problems:
            print(f"{indent}    {problem}")


if __name__ == "__main__":
    sys.exit(main())


# --------- GBIC v2 layout (dump_cache in src/interp.cpp) ---------
#
#   offset  field
#   0       "GBIC"                          4 bytes
#   4       version, n_layer, n_embd,       7 x u32
#           seq_len, n_prompt_tokens,
#           ablation_type, ablation_layer
#   32      token_ids                       u32[seq_len]
#   ...     init_embeddings                 f32[seq_len][n_embd]
#   ...     per layer, in this order:       f32[seq_len][n_embd] each
#             attention_output, stream_post_attention, mlp_output, stream_post_mlp
#
#   size = 32 + 4*seq_len + 4*(1 + 4*n_layer)*seq_len*n_embd, exactly
#
# --------- Why the comparisons are bitwise ---------
#
# The identities recompute the same float32 adds the engine did, in the same
# order, so they must reproduce exactly. A tolerance here would pass a hook
# placed after the residual add instead of before it, which is the bug the
# check exists to find.
#
# An ablation is meant to break exactly one identity. The engine caches the
# sublayer's clean output and then ablates, so on a zero-attn:k dump the stream
# passes layer k untouched while the cache still holds what attention would
# have contributed -- and recomputing that add no longer reproduces the stream.
# Both halves have to hold: a hook placed after the ablation would cache zeros,
# the identity would still reproduce, and only the second half would catch it.
