#pragma once
// Include `glassbox` libraries
#include "glassbox/model.h"

// Include stdlib
#include <cstdlib>
#include <string>


// --------- Public API ---------

// Prints "error: <msg>" to stderr and terminates the process with status 1.
void die(const std::string& msg);

// Matrix multiplication of 2D tensors: A {m, k} by B {k, n}. TA/TB may each be
// Tensor or TensorView, read through operator()
// return type stays Tensor regardless of the inputs.
// Returns a new Tensor A @ B {m, n}
template <typename TA, typename TB>
Tensor matmul(const TA& a, const TB& b)
{
    if (a.shape[1] != b.shape[0]){
        abort();
    }

    const size_t m { a.shape[0] };
    const size_t k { a.shape[1] };
    const size_t n { b.shape[1] };

    Tensor product {
        .data  = std::vector<float>(m * n),
        .shape = {m, n}
    };

    for (size_t i {0}; i < m; ++i){
        for (size_t p {0}; p < k; ++p){
            const float a_ip = a(i, p);
            for (size_t j {0}; j < n; ++j){
                product(i, j) += a_ip * b(p, j);
            }
        }
    }

    return product;
}

// Transpose of a 2D tensor. TA may be Tensor or TensorView.
template <typename TA>
Tensor transpose(const TA& a)
{
    const size_t m { a.shape[0] };
    const size_t n { a.shape[1] };

    Tensor out {
        .data  = std::vector<float>(m * n),
        .shape = {n, m}
    };

    for (size_t i {0}; i < m; ++i){
        for (size_t j {0}; j < n; ++j){
            out(j, i) = a(i, j);
        }
    }

    return out;
}
