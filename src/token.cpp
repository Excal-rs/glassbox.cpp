#include <array>
#include <string>
#include <vector>
#include <utility>
#include <cassert>
#include "glassbox/token.h"

// --------- Constants ---------

// Byte values that bytes_to_unicode leaves unchanged (each maps to itself):
// visible ASCII (! to ~), then the two printable runs of Latin-1, skipping the
// C1 control block (127–160) and the soft hyphen (173).
static constexpr std::array<std::pair<int, int>, 3> printableRanges {{
    {'!', '~'},
    {0xa1, 0xac},
    {0xae, 0xff},
}};

// --------- Forward declarations ---------

static bool isPrintable(int c);
static std::string mini_utf8(unsigned int code);

// --------- Public API ---------

// Contract is documented in token.h.
std::vector<int> encode(const std::string& text) {
    (void)text;
    return {};  // TODO: implement the encode pipeline
}

// --------- Helper functions ---------

// Builds GPT-2's byte→unicode table. Each of the 256 byte values is mapped to a
// printable stand-in character, so the BPE pieces never hold raw control or
// whitespace bytes (which the merge logic and the vocab files can't carry
// cleanly). Printable bytes map to themselves; the rest are bumped to code
// points from U+0100 upward. Returns each byte value → its stand-in, as a UTF-8 string.
std::array<std::string, 256> bytes_to_unicode() {
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
static bool isPrintable(int c) {
    for (auto [lo, hi] : printableRanges) {
        if (lo <= c && c <= hi)
            return true;
    }
    return false;
}

// Encodes a Unicode code point as UTF-8, but only across the 1- and 2-byte range
// (code points up to U+07FF) — all that bytes_to_unicode ever needs, hence the
// assert. Swap in the full 4-byte encoder if you ever need to go higher.
static std::string mini_utf8(unsigned int code) {
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