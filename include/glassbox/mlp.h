#pragma once

#include "glassbox/model.h"
#include "glassbox/interp.h"

struct Profile;   // glassbox/bench.h

// --------- Public API ---------

// Runs one block's MLP sublayer on x (shape {seq, n_embd})
Tensor mlp(const Tensor& x, const LayerNorm& ln_2, const MLP& mlp, const Config& config, const InterpContext& interpctx, size_t layer_idx, Profile* profile = nullptr);
