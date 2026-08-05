#include <cstdlib>
#include <iostream>
#include <string>
#include "glassbox/utils.h"

// --------- Public API ---------

// Prints "error: <msg>" to stderr and terminates the process with status 1.
void die(const std::string& msg)
{
    std::cerr << "error: " << msg << "\n";
    std::exit(1);
}

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

// Transpose of a 2D tensor: A (shape {m, n}) -> shape {n, m}, out[j][i] = A[i][j].
Tensor transpose(const Tensor& a)
{
    const size_t m = a.shape[0];
    const size_t n = a.shape[1];

    Tensor out {
        .data  = std::vector<float>(m * n),
        .shape = {n, m}
    };

    for (size_t i = 0; i < m; ++i){
        for (size_t j = 0; j < n; ++j){
            out(j, i) = a(i, j);
        }
    }

    return out;
}
