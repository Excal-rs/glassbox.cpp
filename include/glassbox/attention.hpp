#pragma once

// Include `glassbox` libraries
#include "glassbox/model.hpp"
#include "glassbox/interp.hpp"

struct Profile;   // glassbox/benchmarking.hpp
struct KV_cache;  // glassbox/kvcache.hpp

// --------- Public API ---------

// Runs one block's attention sublayer on x (shape {seq, n_embd})
Tensor attention(const Tensor& x, const LayerNorm& ln_1, const Attention& attn, const Config& config, const InterpContext& interpctx, KV_cache* kv, size_t layer_idx, Profile* profile = nullptr);
