#include <vector>
#include "glassbox/embed.h"
#include "glassbox/layernorm.h"
#include "glassbox/attention.h"
#include "glassbox/mlp.h"
#include "glassbox/forward.h"

// --------- Public API ---------

// Contract is documented in forward.h.
Tensor forward(const std::vector<int>& ids, const Model& model)
{
    const Config&  config  = model.config;
    const Network& network = model.network;

    Tensor x { embed(ids, model) };

    // Both sublayers handle their own pre-norm and residual add
    for (const Block& block : network.h){
        x = attention(x, block.ln_1, block.attn, config);
        x = mlp(x, block.ln_2, block.mlp, config);
    }

    return layernorm(x, network.ln_f, config.ln_eps);
}

// logits[t] = dot(h, wte[t]) with h the last row of x — the tied wte matrix
// used in the output direction (the one transposed matmul in GPT-2). 
std::vector<float> lm_logits(const Tensor& x, const Model& model)
{
    const Tensor& wte     = model.network.wte;
    const size_t  last    = x.shape[0] - 1;
    const size_t  n_embd  = x.shape[1];
    const size_t  n_vocab = model.config.n_vocab;

    std::vector<float> logits(n_vocab);
    for (size_t t = 0; t < n_vocab; ++t){
        float sum = 0.0f;
        for (size_t j = 0; j < n_embd; ++j){
            sum += x(last, j) * wte(t, j);
        }
        logits[t] = sum;
    }

    return logits;
}
