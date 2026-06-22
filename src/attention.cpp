#include <array>
#include "glassbox/layernorm.h"
#include "glassbox/utils.h"
#include "glassbox/attention.h"

// --------- Static Forward Declarations ---------

static std::array<Tensor, 3> qkv_projection(const Tensor& xn, const Attention& attn);
static Tensor split_heads(const Tensor& m, size_t n_head);
static Tensor attention_head(const Tensor& Q_h, const Tensor& K_h, const Tensor& V_h, float scale);
static Tensor merge_heads(const Tensor& heads, size_t n_head);
static Tensor output_projection(const Tensor& concat, const Attention& attn);


// --------- Public API ---------

// Runs one block's attention sublayer on x (shape {seq, n_embd}).
// Does the following:
// - Applies LayerNorm 
// - QKV Projection
// - Split into Heads
// - Compute Scores, Mask, Softmax and Weighted Sum
// - Merge Heads
// - Output Projection
// - Add to input
Tensor attention(const Tensor& ids, const LayerNorm& ln_1, const Attention& attn, const Config& config)
{
    Tensor xn { layernorm(ids, ln_1, config.ln_eps) };

    return {};
}


// --------- Helper Function Definitions ---------

static std::array<Tensor, 3> qkv_projection(const Tensor& xn, const Attention& attn)
{
    return {};
}

static Tensor split_heads(const Tensor& m, size_t n_head)
{
    return {};
}

static Tensor attention_head(const Tensor& Q_h, const Tensor& K_h, const Tensor& V_h, float scale)
{
    return {};
}

static Tensor merge_heads(const Tensor& heads, size_t n_head)
{
    return {};
}

static Tensor output_projection(const Tensor& concat, const Attention& attn)
{
    return {};
}

