# glassbox.cpp

**An interpretability-first GPT-2 inference engine in C++.**

Most inference engines treat the model as a blackbox, `glassbox.cpp` is designed to be open, it's a from-scratch inference engine (currently only being built for GPT-2) that where every activation can be named, accessed, and edited. The goal is a purpose-built tool, that allows researchers to study models like GPT-2, so we can understand them better through **Mechanistic Interpretability**.

> **Status: v0.1, the engine runs and you can now look inside it.**
> The full pipeline (BPE tokenizer, safetensors loader, 12-block forward pass,
> greedy decoding, detokenizer) is implemented and validated token-for-token
> against HuggingFace GPT-2. On top of that, a run can export every activation
> it produced, and you can knock out one layer's attention or MLP and watch the
> output change.

## What it is

A single C++ library (`glassbox`) implementing the full GPT-2 pipeline, plus a
thin CLI driver (`glassbox_cli`). No deep-learning framework: the matmuls,
LayerNorm, attention, and softmax are written from scratch.

The reference model is GPT-2 small (124M): `vocab=50257`, `n_embd=768`,
`n_layer=12`, `n_head=12`, `n_ctx=1024`, weights loaded from a standard
`safetensors` checkpoint.

## Where the project is now

The engine is done and trusted. It produces the same tokens as HuggingFace
GPT-2 on the same prompt, and the per-sublayer activations agree to within
float32 rounding (worst relative difference around 6e-6, which is roughly what
summing 768 terms in a different order costs you).

The interpretability toolkit has its foundation in place. A forward pass can
record four buffers per block:

| buffer | what it holds |
|---|---|
| `attention_output` | the attention sublayer's contribution, before the residual add |
| `stream_post_attention` | the residual stream after that add |
| `mlp_output` | the MLP sublayer's contribution, before the residual add |
| `stream_post_mlp` | the residual stream after that add |

plus the embeddings entering the first block. That is 49 buffers for GPT-2
small, each `seq_len x n_embd` floats, written to one file you can load straight
into NumPy.

Intervention works at layer granularity: zero a block's attention or MLP output
and the residual stream carries on without it. The captured buffers always hold
the *clean* value, so a dump shows you both what the sublayer computed and what
the model actually did with it. Heads and individual neurons are the next step.

## Building

```bash
cmake -S . -B build        # defaults to a Release build
cmake --build build
```

Requires CMake and a C++20 compiler. PCRE2 and nlohmann_json are used for the
tokenizer and config parsing; if they are not installed, the build fetches and
builds them, so the first configure needs network access.

Release is the default because the forward pass is roughly 20x slower
unoptimized. Pass `-DCMAKE_BUILD_TYPE=Debug` when you need symbols.

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
./build/app/glassbox_cli <model_dir> [options]
```

| flag | meaning |
|---|---|
| `-p, --prompt <text>` | prompt, inline |
| `-f, --prompt-file <path>` | read the prompt from a file |
| `-o, --output <path>` | write the full response to a file |
| `-n, --n-tokens <count>` | tokens to generate (default 50) |
| `-i, --interp <path>` | dump the activation cache |
| `-a, --ablate <kind>:<n>` | `zero-attn:<n>` or `zero-mlp:<n>` |

With neither `-p` nor `-f`, the prompt is read from stdin. Decoding is greedy
and streamed token by token, stopping at `<|endoftext|>` or the 1024-token
context limit.

```
$ ./build/app/glassbox_cli model -p "The cat sat on" -n 10
The cat sat on the floor, and the cat was still asleep.

$ ./build/app/glassbox_cli model -p "The cat sat on" -n 10 -a zero-mlp:9
The cat sat on the ground, and the dog was still in the
```

### Activation dumps

`-i <path>` writes the run's last forward pass to a binary file: the prompt plus
every token generated before it. Since attention is causal and there is no KV
cache, that last pass recomputes every earlier one exactly, so a single file
covers the whole run.

The layout is little-endian with no padding. A 32-byte header (`GBIC`, version,
`n_layer`, `n_embd`, `seq_len`, `n_prompt_tokens`, ablation type, ablated layer)
is followed by `seq_len` token ids as `u32`, then the payload as float32: the
initial embeddings, then the four buffers above for each block in order. Every
buffer is `seq_len * n_embd` values, row-major, so element `[t][c]` sits at flat
index `t * n_embd + c`.

The ablation fields matter when reading a dump back. An ablated pass is
otherwise indistinguishable from a clean one, and its residual arithmetic is
*supposed* to stop adding up at the layer you targeted.

## Roadmap

### Done
- [x] Byte-level BPE tokenizer (encode + decode)
- [x] safetensors weight loader
- [x] Full forward pass: embedding, LayerNorm, attention, MLP, tied unembedding
- [x] Greedy generation loop + `glassbox_cli`
- [x] Token-for-token validation against HuggingFace GPT-2
- [x] Activation capture across every block, exported for NumPy
- [x] Layer-granularity ablation of attention and MLP sublayers

### Engine improvements
- [ ] KV cache, since per-token cost is currently O(seq²) from recomputing attention over the whole prefix
- [ ] Sampling strategies: temperature, top-k, top-p
- [ ] Multithreaded / SIMD matmuls (single-threaded scalar loops today)

### Finer-grained interventions
Layer granularity is the coarsest useful unit. Next is naming things inside a
block: individual heads (hooked before the output projection), individual
neurons, and patching activations in from a second run rather than only zeroing
them. Attention pattern capture, the post-softmax matrix behind the familiar
attention-map plots, belongs here too.

### Intervention language
A small declarative command-file format, the unit of reproducible, shareable
experiments. Each line is one intervention:

```gbx
# ablate.gbx
ablate  blocks.9.mlp.post   2073        # zero a neuron
scale   blocks.5.attn.head  7   0.0     # silence a head
patch   blocks.2.resid_post run_b       # paste activations from another run
```

Deliberately not a programming language (no loops or variables), just a flat
list covering the standard mech-interp moves. A curated library of known
circuits (induction heads, IOI, and so on) from published research may ship
later as named presets, since circuits are discovered rather than read off the
weights.

### CUDA support
The forward pass is CPU-first for clarity and correctness. A CUDA backend will
follow for throughput once the reference path is trusted. The named-tensor and
hook design is meant to survive the move to GPU so interventions keep working.

## Project layout

```
glassbox.cpp/
├── include/glassbox/   # public headers
├── src/                # library implementation (the engine)
├── app/                # glassbox_cli driver
└── CMakeLists.txt      # library + thin driver
```
