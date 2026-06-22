#pragma once

#include "glassbox/model.h"

// --------- Public API ---------

// Matrix multiply of two 2D tensors: A (shape {m, k}) by B (shape {k, n}).
// Returns a new Tensor of shape {m, n} where out[i][j] = sum_p A[i][p] * B[p][j].
Tensor matmul(const Tensor& a, const Tensor& b);
