#include <cmath>
#include "glassbox/layernorm.h"
#include "glassbox/utils.h"
#include "glassbox/mlp.h"

// --------- Constants ---------
static constexpr float GELU_SCALE = 0.7978845608f; // This is what GPT-2 was trained with as the approximation, therefore using this instead of exact value


// --------- Static Forward Declarations ---------
static inline float tanh_approx(float u);

// --------- Public API ---------

// Runs one block's MLP sublayer on x (shape {seq, n_embd})
Tensor mlp(const Tensor& x, const LayerNorm& ln, const MLP& mlp, const Config& config)
{
    const size_t seq    = x.shape[0];
    const size_t n_embd = config.n_embd;

    Tensor xn { layernorm(x, ln, config.ln_eps) };
    
    // Up Projection
    Tensor h  { matmul(xn, mlp.c_fc.w) };
    for (size_t i = 0; i < seq; ++i){
        for (size_t j = 0; j < n_embd * 4; ++j) {
            h(i, j) += mlp.c_fc.b.data[j];
        }
    }

    // GELU 
    for (float& v : h.data){
        v = tanh_approx(v);
    }

    // DOwn projection
    Tensor out { matmul(h, mlp.c_proj.w) };
    for (size_t i = 0; i < seq; ++i){
        for (size_t j = 0; j < n_embd; ++j) {
            out(i, j) += mlp.c_proj.b.data[j] + x(i, j);
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
