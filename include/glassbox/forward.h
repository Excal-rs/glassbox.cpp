#pragma once

#include <vector>
#include "glassbox/model.h"
#include "glassbox/interp.h"

struct Profile;   // glassbox/bench.h

// --------- Public API ---------

// Runs the full forward pass: embed ids, apply all blocks, then ln_f.
// Returns the final hidden states (shape {seq, n_embd})
Tensor forward(const std::vector<int>& ids, const Model& model, const InterpContext& interpctx = {}, Profile* profile = nullptr);

// Projects the last token's hidden state through the tied wte matrix.
// Returns one next-token score per vocab entry (size n_vocab)
std::vector<float> lm_logits(const Tensor& x, const Model& model, Profile* profile = nullptr);
