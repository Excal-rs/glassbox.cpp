#include <array>
#include <string>
#include <vector>
#include <unordered_map>
#include <utility>
#include <cassert>
#include <climits>
#include <fstream>
#include <iostream>
#include <limits>
#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>
#include "glassbox/token.h"


// --------- Constants ---------

// Byte values that bytes_to_unicode leaves unchanged (each maps to itself):
// visible ASCII (! to ~), then the two printable runs of Latin-1, skipping the
// C1 control block (127-160) and the soft hyphen (173).
static constexpr std::array<std::pair<int, int>, 3> printableRanges {{
    {'!', '~'},
    {0xa1, 0xac},
    {0xae, 0xff},
}};

static constexpr std::string_view GPT2_REGEX {R"('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)"};

// --------- Forward declarations ---------

static std::vector<std::string> chop(std::string input);
static std::array<std::string, 256> bytes_to_unicode();
static std::vector<std::string> merge_chunk(std::vector<std::string> chunk, const Merge &merge);
static bool isPrintable(int c);
static std::string mini_utf8(unsigned int code);

// --------- Public API ---------

// Contract is documented in token.h.
std::vector<int> encode(const std::string& text, const Vocab& vocab,const Merge& merge)
{
    auto mapping = bytes_to_unicode();
    std::vector<int> ids {};

    for (const std::string& chunk : chop(text)){
        std::vector<std::string> mapped {};
        for (unsigned char b : chunk){
            mapped.push_back(mapping[b]);
        }

        // Merge the chunk, then look up each final token's id.
        for (const std::string& token : merge_chunk(mapped, merge)){
            ids.push_back(vocab.at(token));
        }
    }

    return ids;
}



// Splits text into GPT-2's pre-tokenization chunks via the BPE regex.
// Takes the input string; returns the chunks in order, covering the whole
// string with no gaps or overlaps.
static std::vector<std::string> chop(std::string input)
{
    std::vector<std::string> chunks {};

    // Compile Regex Pattern
    int errcode;
    PCRE2_SIZE erroffset;
    pcre2_code *rcode {pcre2_compile(
        (PCRE2_SPTR) GPT2_REGEX.data(),
        GPT2_REGEX.size(),
        PCRE2_UTF | PCRE2_UCP,
        &errcode,
        &erroffset, NULL
    )};
    assert(rcode != NULL && "No error when compiling regex");

    // Allocate match data
    pcre2_match_data *match {pcre2_match_data_create_from_pattern(
        rcode, NULL
    )};

    // Perform Matches
    PCRE2_SIZE offset {0};
    while (offset < input.size())
    {
        pcre2_match(
            rcode,
            (PCRE2_SPTR) input.data(),
            input.size(),
            offset,
            0,
            match,
            NULL
        );

        // Get start and end positions for appending and offset update
        PCRE2_SIZE *overctor {pcre2_get_ovector_pointer(match)};
        chunks.push_back(input.substr(overctor[0], overctor[1] - overctor[0]));
        offset = overctor[1];
    }

    // Free heap memory
    pcre2_match_data_free(match);
    pcre2_code_free(rcode);


    return chunks;
}

// Builds GPT-2's byte→unicode table, mapping all 256 byte values to printable
// stand-in characters so the BPE chunk never hold raw control or whitespace bytes.
static std::array<std::string, 256> bytes_to_unicode() 
{
    std::array<std::string, 256> byteMapping{};
    int n = 0;
    for (int i = 0; i < 256; ++i) {
        if (isPrintable(i)) {
            byteMapping[i] = mini_utf8(i);
        } else {
            byteMapping[i] = mini_utf8(256 + n);  // not printable: next free code point above 255
            ++n;
        }
    }
    return byteMapping;
}

// True if byte value c is a printable character that maps to itself in
// bytes_to_unicode (visible ASCII, or the printable parts of Latin-1).
static bool isPrintable(int c) 
{
    for (auto [lo, hi] : printableRanges) {
        if (lo <= c && c <= hi)
            return true;
    }
    return false;
}

// Encodes a Unicode code point as UTF-8, but only across the 1- and 2-byte range
// (code points up to U+07FF), all that bytes_to_unicode ever needs, hence the
// assert. Swap in the full 4-byte encoder if we ever need to go higher.
static std::string mini_utf8(unsigned int code) 
{
    assert(code <= 0x7ff && "mini_utf8() only works for 1-2 byte ranges");
    std::string out;
    if (code <= 0x7f) {
        out.append(1, static_cast<char>(code));
    } else if (code <= 0x7ff) {
        out.append(1, static_cast<char>(0xc0 | ((code >> 6) & 0x1f)));
        out.append(1, static_cast<char>(0x80 | (code & 0x3f)));
    }
    return out;
}


// Follows the given merge rules and merges a chunk into its final tokens.
// The chunk starts as one mapped byte per element and combines as merges are
// applied; returns the merged token strings.
static std::vector<std::string> merge_chunk(std::vector<std::string> chunk, const Merge &merge)
{
    // Repeatedly merge the lowest-rank adjacent pair, until nothing is mergeable
    while (chunk.size() > 1) {
        std::pair<int, int> best = {-1, std::numeric_limits<int>::max()};   // {index, rank}

        for (int i = 0; i + 1 < chunk.size(); ++i) {
            auto it = merge.find(chunk[i] + " " + chunk[i + 1]);
            if (it != merge.end() && it->second < best.second) {
                best = {i, it->second};
            }
        }

        if (best.first == -1) break;

        int i = best.first;
        chunk[i] += chunk[i + 1];
        chunk.erase(chunk.begin() + i + 1);
    }

    return chunk;
}