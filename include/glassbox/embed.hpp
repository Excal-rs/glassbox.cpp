#pragma once

// Include `glassbox` libraries
#include "glassbox/model.hpp"

// Include stdlib
#include <vector>

// --------- Public API ---------
Tensor embed(const std::vector<int>& tokens, const Model& model, size_t pos_offset = 0);
