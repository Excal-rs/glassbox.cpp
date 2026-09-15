#pragma once
// Include `glassbox` libraries
#include "glassbox/model.h"

// Include stdlib
#include <string>


// --------- Public API ---------

// Prints "error: <msg>" to stderr and terminates the process with status 1.
void die(const std::string& msg);

// Matrix multiplication (Tensors must be 2D)
// Returns a @ b
Tensor matmul(const Tensor& a, const Tensor& b);

// Transpose a 2D tensor
Tensor transpose(const Tensor& a);
