# glassbox.cpp

**An interpretability-first GPT-2 inference engine in C++.**

Most inference engines treat the model as a blackbox, `glassbox.cpp` is designed to be open, it's a from-scratch inference engine (currently only being built for GPT-2) that where every activation can be named, accessed, and edited. The goal is a purpose-built tool, that allows researchers to study models like GPT-2, so we can understand them better through **Mechanistic Interpretability**.

> **Status: v0.1 — the engine works.**
> The full pipeline (BPE tokenizer → safetensors loader → 12-block forward
> pass → greedy decoding → detokenizer) is implemented and validated
> token-for-token against HuggingFace GPT-2 (see
> [`doc/.markdown/benchmarks.md`](doc/.markdown/benchmarks.md)). The
> interpretability toolkit is the next major phase.

---

## What it is

A single C++ library (`glassbox`) implementing the full GPT-2 pipeline, plus a
thin CLI driver (`glassbox_cli`). No deep-learning framework — the matmuls,
LayerNorm, attention, and softmax are written directly so that every step is
legible and hookable.

The reference model is GPT-2 small (124M): `vocab=50257`, `n_embd=768`,
`n_layer=12`, `n_head=12`, `n_ctx=1024`, weights loaded from a standard
`safetensors` checkpoint.

---

## Roadmap

### Done (v0.1)
- [x] Byte-level BPE tokenizer (encode + decode)
- [x] safetensors weight loader
- [x] Full forward pass: embedding, LayerNorm, attention, MLP, tied unembedding
- [x] Greedy generation loop + `glassbox_cli`
- [x] Token-for-token validation against HuggingFace GPT-2

### Engine improvements *(near term)*
- [ ] KV cache — per-token cost is currently O(seq²) from recomputing attention over the whole prefix
- [ ] Sampling strategies: temperature, top-k, top-p
- [ ] Multithreaded / SIMD matmuls (single-threaded scalar loops today; see [`benchmarks.md`](doc/.markdown/benchmarks.md))

### Interpretability toolkit *(next major phase)*
The reason the project exists. A `glassbox-cli` that loads GPT-2 and lets you
**inspect, ablate, modify, and patch** activations and weights by name — driven
entirely from the command line or an experiment file (see below), no code changes. Built on a single **named hook-point** mechanism wired into the forward pass, with a registry
so `list`/`inspect` can enumerate valid targets.

### Intervention language
A small declarative **command-file format** — the unit of reproducible,
shareable experiments. Each line is one intervention:

```gbx
# ablate.gbx
ablate  blocks.9.mlp.post   2073        # zero a neuron
scale   blocks.5.attn.head  7   0.0     # silence a head
patch   blocks.2.resid_post run_b       # paste activations from another run
```

Deliberately *not* a programming language (no loops/variables) — a flat,
declarative list that covers the standard mech-interp moves. A curated library of
**known circuits** (induction heads, IOI, …) from published research may ship
later as named presets — circuits are *discovered*, not read off the weights.

### CUDA support
The forward pass is CPU-first for clarity and correctness. A CUDA backend will
follow for throughput once the reference path is trusted — the named-tensor /
hook design is meant to survive the move to GPU so interventions keep working.

---

## Building

```bash
cmake -S . -B build        # defaults to a Release build
cmake --build build
```

Requires a C++17 compiler and CMake. The build defaults to `Release` — the
forward pass is ~20× slower unoptimized (pass `-DCMAKE_BUILD_TYPE=Debug` when
debugging).

## Getting the model files

GPT-2 weights are not included. Download these four files from the
[HuggingFace `gpt2` repo](https://huggingface.co/openai-community/gpt2/tree/main)
into a `model/` directory:

```
model/
├── config.json
├── vocab.json
├── merges.txt
└── model.safetensors
```

## Usage

```bash
./build/app/glassbox_cli <model_dir> "<prompt>" [n_tokens]
```

`n_tokens` defaults to 50. Decoding is greedy (deterministic), streamed
token-by-token; generation stops early on `<|endoftext|>` or at the 1024-token
context limit. Example:

```
$ ./build/app/glassbox_cli model "The cat sat on" 10
The cat sat on the floor, and the cat was still asleep.
```


---

## Project layout

```
glassbox.cpp/
├── include/glassbox/   # public headers
├── src/                # library implementation (the engine)
├── app/                # glassbox_cli driver
└── CMakeLists.txt      # library + thin driver
```

