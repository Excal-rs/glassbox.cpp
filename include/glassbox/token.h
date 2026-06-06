#pragma once

#include <string>
#include <vector>

// Encode UTF-8 text into GPT-2 BPE token ids.
//
// Must match the GPT-2 byte-level BPE exactly (same ids as tiktoken's "gpt2"
// encoding and the HuggingFace vocab.json + merges.txt pair in model/).
//
// No special/control tokens are added: encode("") returns an empty vector.
//
// How the vocab/merges get loaded is an implementation detail of token.cpp
// (e.g. lazy-load from model/ on first call, or via an env var / fixed path).
// The tests only depend on this signature.
std::vector<int> encode(const std::string& text);
