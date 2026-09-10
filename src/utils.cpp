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

// Matrix multiplication of 2D tensors: A {m, k} by B {k, n}
// Returns a new Tensor A @ B {m, n}
Tensor matmul(const Tensor& a, const Tensor& b)
{
    // Matrix Multiplication only works on certain shaped matrices
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

    // Auto-vectorised for loop
    for (size_t i {0}; i < m; ++i){
        for (size_t p {0}; p < k; ++p){
            const float a_ip = a.data[i * k + p];
            for (size_t j {0}; j < n; ++j){
                product.data[i * n + j] += a_ip * b.data[p * n + j];
            }
        }
    }

    return product;
}

// Transpose of a 2D tensor
Tensor transpose(const Tensor& a)
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
