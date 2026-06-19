#pragma once

#include <string>
#include <vector>
#include <unordered_map>

#include "glassbox/model.h"  // Vocab, Merge


// Encode UTF-8 text into GPT-2 BPE token ids.
std::vector<int> encode(const std::string& text, const Vocab& vocab,const Merge& merge);

