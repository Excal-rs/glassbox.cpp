#include "glassbox/token.h"

// TODO(you): implement GPT-2 byte-level BPE encoding here.
//
//   1. Load the vocab (model/vocab.json) and merge ranks (model/merges.txt).
//   2. Pre-tokenize with the GPT-2 regex pattern and apply the byte->unicode
//      mapping so every input byte maps to a printable code point.
//   3. Greedily apply the lowest-rank merge until no merge applies, then map
//      each resulting token string to its vocab id.
//
// Until this is implemented, encode() returns an empty vector, so the cases in
// tests/test_token.cpp fail with clear expected-vs-got diffs.

std::vector<int> encode(const std::string& text) {
    (void)text;
    return {};
}
