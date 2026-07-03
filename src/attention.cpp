#include <array>
#include <cmath>
#include <limits>
#include "glassbox/layernorm.h"
#include "glassbox/utils.h"
#include "glassbox/attention.h"

// --------- Constants ---------

// Used to mask out future positions before softmax; exp(-inf) == 0.
static constexpr float NEG_INF = -std::numeric_limits<float>::infinity();

// --------- Static Forward Declarations ---------

static std::array<Tensor, 3> qkv_projection(const Tensor& xn, const Attention& attn, const Config& config);
static std::vector<Tensor> split_heads(const Tensor& m, const Config& config);
static Tensor attention_head(const Tensor& Q, const Tensor& K, const Tensor& V, float scale);
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
    Tensor xn                 { layernorm(ids, ln_1, config.ln_eps) };
    std::array<Tensor, 3> QKV { qkv_projection(xn, attn, config) };
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
    for (size_t t = 0; t < seq; ++t){
        for (size_t c = 0; c < n_stride; ++c){
            n(t, c) += attn.c_attn.b.data[c];
        }
    }

    // [Q, K, V]
    std::array<Tensor, 3> qkv {};
    for (Tensor& t : qkv){
        t.shape = {seq, n_embd};
        t.data  = std::vector<float>(seq * n_embd);
    }

    // Extract data: n's columns are laid out as [Q | K | V], each n_embd wide.
    for (size_t t = 0; t < seq; ++t){
        for (size_t c = 0; c < n_embd; ++c){
            qkv[0](t, c) = n(t, c);
            qkv[1](t, c) = n(t, c + n_embd);
            qkv[2](t, c) = n(t, c + n_embd * 2);
        }
    }

    return qkv;
}

static std::vector<Tensor> split_heads(const Tensor& m, const Config& config)
{
    auto seq      = m.shape[0];
    auto n_embd   = config.n_embd;
    auto n_head   = config.n_head;
    auto head_dim = n_embd / n_head;
    
    // Initiating Tensors
    std::vector<Tensor> heads(n_head);
    for (size_t i = 0; i < n_head; ++i){
        heads[i].data = std::vector<float>(seq * head_dim);
        heads[i].shape = {seq, head_dim};
    }

    // Assigning Tensor values
    for (size_t i = 0; i < n_head; ++i){
        for (size_t j = 0; j < seq; ++j){
            for (size_t k = 0; k < head_dim; ++k){
                heads[i](j, k) = m(j, i * head_dim + k);
            }
        }
    }

    return heads;
}

static Tensor attention_head(const Tensor& Q, const Tensor& K, const Tensor& V, float scale)
{
    auto seq = Q.shape[0];
    Tensor S = matmul(Q, transpose(K));

    for (size_t i = 0; i < seq; ++i){
        // Scale and mask future positions
        float row_max = NEG_INF;
        for (size_t j = 0; j < seq; ++j){
            const float s = (j > i) ? NEG_INF : S(i, j) * scale;
            S(i, j) = s;
            if (s > row_max) row_max = s;
        }

        // Softmax over the row
        float sum = 0.0f;
        for (size_t j = 0; j < seq; ++j){
            const float e = std::exp(S(i, j) - row_max);
            S(i, j) = e;
            sum += e;
        }
        for (size_t j = 0; j < seq; ++j){
            S(i, j) /= sum;
        }
    }

    return matmul(S, V);
}

static Tensor merge_heads(const Tensor& heads, size_t n_head)
{
    return {};
}

static Tensor output_projection(const Tensor& concat, const Attention& attn)
{
    return {};
}

