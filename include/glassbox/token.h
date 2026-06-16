#pragma once

#include <string>
#include <vector>
#include <unordered_map>

// --------- Types ---------
using Vocab = std::unordered_map<std::string, u_int16_t>;
using Merge = std::unordered_map<std::string, u_int16_t>;



// Encode UTF-8 text into GPT-2 BPE token ids.
std::vector<int> encode(const std::string& text, const Vocab& vocab,const Merge& merge);

// Loads the needed files
Vocab load_vocab(const std::string& path);
Merge load_merge(const std::string& path);

