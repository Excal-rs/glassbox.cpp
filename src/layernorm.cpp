#include <cmath>
#include "glassbox/layernorm.h"

// --------- Static Forward Declarations
static inline void layernorm_vector(const float* in, float* out, size_t dim, const LayerNorm& ln, float eps);


// --------- Public API ---------

// Takes in a Tensor of token vectors (shape {seq, n_embd}) and a LayerNorm,
// Will return a new Tensor of the same shape with each row normalised independently
Tensor layernorm(const Tensor& x, const LayerNorm& ln, float eps){
    const size_t seq = x.shape[0];
    const size_t dim = x.shape[1];

    Tensor out {
        .data  = std::vector<float>(x.data.size()),
        .shape = x.shape
    };

    for (size_t r = 0; r < seq; ++r){
        const size_t base = r * dim;
        layernorm_vector(&x.data[base], &out.data[base], dim, ln, eps);
    }
    return out;
}


// --------- Helper Function Definitions ---------

// Normalises a single vector (one token's features) in place from in -> out.
// mean and variance are computed across this vector's `dim` features only,
// then each element is scaled and shifted by the learned weight/bias.
static inline void layernorm_vector(const float* in, float* out, size_t dim, const LayerNorm& ln, float eps)
{
    float sum    = 0;
    float sum_sq = 0;
    for (size_t i = 0; i < dim; ++i){
        sum    += in[i];
        sum_sq += in[i] * in[i];
    }
    float mean     =  sum / dim;
    float variance = (sum_sq / dim) - (mean * mean);

    for (size_t i = 0; i < dim; ++i){
        out[i] = (in[i] - mean) / std::sqrt(variance + eps)
                 * ln.weight.data[i]
                 + ln.bias.data[i];
    }
}

// --------- Math (per vector x of length n) ---------
//
//   mean      mu    = (1/n) * sum_i( x_i )
//   variance  var   = (1/n) * sum_i( x_i^2 ) - mu^2
//   normalise x_hat = (x_i - mu) / sqrt(var + eps)
//   output    y_i   = x_hat_i * weight_i + bias_i
//
// eps is added inside the sqrt for numerical stability.

