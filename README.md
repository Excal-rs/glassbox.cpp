# glassbox.cpp

**An interpretability-first GPT-2 inference engine in C++.**

Most inference engines treat the model as a blackbox, `glassbox.cpp` is designed to be open, it's a from-scratch inference engine (currently only being built for GPT-2) that where every activation can be named, accessed, and edited. The goal is a purpose-built tool, that allows researchers to study models like GPT-2, so we can understand them better through **Mechanistic Interpretability**.

> **Status: early - In Progress** 
> The byte-level BPE tokenizer and the safetensors weight
> loader are in place; the transformer forward pass is being built now. The
> interpretability toolkit is designed but intentionally deferred until the
> engine is complete.

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

## Coming in future 

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
cmake -S . -B build
cmake --build build

./build/app/glassbox_cli
```

Requires a C++17 compiler and CMake. GPT-2 weights are not included — supply your
own `safetensors` checkpoint and GPT-2 vocab files.

Model files can be downloaded from huggingface. 


---

## Project layout

```
glassbox.cpp/
├── include/glassbox/   # public headers
├── src/                # library implementation (the engine)
├── app/                # glassbox_cli driver
└── CMakeLists.txt      # library + thin driver
```

