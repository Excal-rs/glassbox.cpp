#pragma once

#include <vector>
#include "glassbox/model.h"

// --------- Public API ---------
Tensor embed(const std::vector<int>& tokens, const Model& model);

