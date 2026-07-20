#pragma once

#include <string>
#include "glassbox/model.h"

// --------- Public API ---------

// Prints "error: <msg>" to stderr and terminates the process with status 1.
// The standard fatal-error exit for glassbox.
void die(const std::string& msg);

// Matrix multiply of two 2D tensors: A (shape {m, k}) by B (shape {k, n}).
// Returns a new Tensor of shape {m, n} where out[i][j] = sum_p A[i][p] * B[p][j].
Tensor matmul(const Tensor& a, const Tensor& b);

// Transpose of a 2D tensor: A (shape {m, n}) -> shape {n, m}, out[j][i] = A[i][j].
Tensor transpose(const Tensor& a);
