#pragma once

#include "glassbox/model.h"

// --------- Public API ---------

// Takes in a Tensor of token vectors (shape {seq, n_embd}) and a LayerNorm,
// Will return a new Tensor of the same shape with each row normalised independently
Tensor layernorm(const Tensor& x, const LayerNorm& ln, float eps);
