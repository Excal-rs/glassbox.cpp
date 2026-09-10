#include <cmath>
#include "glassbox/layernorm.h"
#include "glassbox/utils.h"
#include "glassbox/mlp.h"
#include "glassbox/benchmarking.h"

// --------- Constants ---------
static constexpr float GELU_SCALE = 0.7978845608f; // This is what GPT-2 was trained with as the approximation, therefore using this instead of exact value


// --------- Static Forward Declarations ---------
static inline float tanh_approx(float u);

// --------- Public API ---------

// Runs one block's MLP sublayer on x (shape {seq, n_embd})
Tensor mlp(const Tensor& x, const LayerNorm& ln, const MLP& mlp, const Config& config, const InterpContext& interpctx, size_t layer_idx, Profile* profile)
{
    const size_t seq    = x.shape[0];
    const size_t n_embd = config.n_embd;

    Tensor xn;
    {
        ScopeTimer timer { profile, Component::LN, layer_idx, elementwise_cost(seq * n_embd, 5) };
        xn = layernorm(x, ln, config.ln_eps);
    }

    // Up Projection. The bias add shares the bucket with the matmul it follows.
    Tensor h;
    {
        ScopeTimer timer { profile, Component::MLP_FC, layer_idx, matmul_cost(seq, n_embd, 4 * n_embd) };
        h = matmul(xn, mlp.c_fc.w);
        for (size_t i = 0; i < seq; ++i){
            for (size_t j = 0; j < n_embd * 4; ++j) {
                h(i, j) += mlp.c_fc.b.data[j];
            }
        }
    }

    // GELU 
    {
        ScopeTimer timer { profile, Component::GELU, layer_idx, elementwise_cost(seq * n_embd * 4, 10) };
        for (float& v : h.data){
            v = tanh_approx(v);
        }
    }

    // Down projection
    Tensor out;
    {
        ScopeTimer timer { profile, Component::MLP_PROJ, layer_idx, matmul_cost(seq, 4 * n_embd, n_embd) };
        out = matmul(h, mlp.c_proj.w);

        // Pass 1: add the c_proj bias only 
        for (size_t i = 0; i < seq; ++i){
            for (size_t j = 0; j < n_embd; ++j) {
                out(i, j) += mlp.c_proj.b.data[j];
            }
        }
    }

    if (interpctx.cache) {
        interpctx.cache->layers[layer_idx].mlp_output = out.data;
    }
    ablate_mlp(out.data, interpctx.ablation, layer_idx);

    // Pass 2: add the residual
    {
        ScopeTimer timer { profile, Component::RESIDUAL, layer_idx, elementwise_cost(seq * n_embd, 1) };
        for (size_t i = 0; i < seq; ++i){
            for (size_t j = 0; j < n_embd; ++j) {
                out(i, j) += x(i, j);
            }
        }
    }

    return out;
}

// --------- Helper Function Definitions ---------

// GELU tanh approximation
static inline float tanh_approx(float u)
{
    return 0.5f * u * (1.0f + std::tanh(GELU_SCALE * (u + 0.044715f * u * u * u)));
}
