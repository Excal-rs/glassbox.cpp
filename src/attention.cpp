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

// Extract and Create the Query, Key and Value Matrices from the attention block
static std::array<Tensor, 3> qkv_projection(const Tensor& xn, const Attention& attn, const Config& config)
{
    const size_t seq      = xn.shape[0];
    const size_t n_embd   = config.n_embd;
    const size_t n_stride = n_embd * 3;

    // Perform (xn . w) + b
    Tensor n { matmul(xn, attn.c_attn.w) };
    for (size_t i = 0; i < n.data.size(); ++i){
        n.data[i] += attn.c_attn.b.data[i % n_stride];
    }

    // [Q, K, V]
    std::array<Tensor, 3> qkv {};
    for (Tensor& t : qkv){
        t.shape = {seq, n_embd};
        t.data  = std::vector<float>(seq * n_embd); 
    }

    // Extract data
    for (size_t t = 0; t < seq; ++t){
        for (size_t c = 0; c < n_embd; ++c){
            const size_t dst = t * n_embd + c;
            const size_t src = t * n_stride + c;
            qkv[0].data.at(dst) = n.data[src];
            qkv[1].data.at(dst) = n.data[src + n_embd];
            qkv[2].data.at(dst) = n.data[src + n_embd * 2];
        }
    }

    return qkv;
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

