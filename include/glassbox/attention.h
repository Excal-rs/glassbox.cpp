#pragma once

#include "glassbox/model.h"

// --------- Public API ---------

// Runs one block's attention sublayer on x (shape {seq, n_embd})
Tensor attention(const Tensor& x, const LayerNorm& ln_1, const Attention& attn, const Config& config);
