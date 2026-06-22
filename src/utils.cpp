#include "glassbox/utils.h"

// --------- Public API ---------

// Matrix multiply of two 2D tensors: A (shape {m, k}) by B (shape {k, n}).
// Returns a new Tensor of shape {m, n}
Tensor matmul(const Tensor& a, const Tensor& b)
{
    // Matrix Multiplication only works on certain shaped matrices
    if (a.shape[1] != b.shape[0]){
        abort();
    }

    const size_t m = a.shape[0];
    const size_t k = a.shape[1];
    const size_t n = b.shape[1];

    Tensor product {
        .data  = std::vector<float>(m * n),
        .shape = {m, n}
    };

    // i,k,j ordering: the inner loop walks a row of B and a row of out
    // contiguously, keeping memory access cache-friendly.
    for (size_t i = 0; i < m; ++i){
        for (size_t p = 0; p < k; ++p){
            const float a_ip = a.data[i * k + p];
            for (size_t j = 0; j < n; ++j){
                product.data[i * n + j] += a_ip * b.data[p * n + j];
            }
        }
    }
    
    return product;
}
