#pragma once

// Include `glassbox` libraries
#include "glassbox/model.h"

// Include stdlib
#include <vector>

// --------- Public API ---------
Tensor embed(const std::vector<int>& tokens, const Model& model);
