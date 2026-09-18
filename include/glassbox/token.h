#pragma once

// Include `glassbox` libraries
#include "glassbox/model.h"  // Vocab, Merge

// Include stdlib
#include <string>
#include <vector>

inline constexpr size_t END_OF_TEXT = 50256; // Token indicating the end of a prompt

// --------- Public API ---------

// Encode UTF-8 text into GPT-2 BPE token ids.
std::vector<int> encode(const std::string& text, const Vocab& vocab, const Merge& merge);

// Decode GPT-2 BPE token ids back into UTF-8 text (inverse of encode).
std::string decode(const std::vector<int>& ids, const Vocab& vocab);
