// Compares encode() against GPT-2 reference ids produced by tiktoken.
//
// The expected ids live in token_test_data.h, regenerated with:
//     python3 tools/gen_token_reference.py
//
// No test framework: prints one line per case and returns non-zero if any
// case mismatches, so it plugs straight into CTest / CI.

#include "glassbox/token.h"
#include "token_test_data.h"

#include <cstdio>
#include <string>
#include <vector>

namespace {

std::string format_ids(const std::vector<int>& ids) {
    std::string out = "[";
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i) out += ", ";
        out += std::to_string(ids[i]);
    }
    out += "]";
    return out;
}

// Escape non-printable bytes so the diff stays on one readable line.
std::string escape(const std::string& s) {
    std::string out;
    for (unsigned char c : s) {
        switch (c) {
            case '\n': out += "\\n"; break;
            case '\t': out += "\\t"; break;
            case '\r': out += "\\r"; break;
            default:
                if (c < 0x20 || c >= 0x7f) {
                    char buf[5];
                    std::snprintf(buf, sizeof(buf), "\\x%02x", c);
                    out += buf;
                } else {
                    out += static_cast<char>(c);
                }
        }
    }
    return out;
}

}  // namespace

int main() {
    const auto& cases = token_cases();
    int failures = 0;

    for (const auto& c : cases) {
        const std::vector<int> got = encode(c.text);
        if (got == c.expected) {
            std::printf("ok    %-16s (%zu tokens)\n", c.name, c.expected.size());
        } else {
            ++failures;
            std::printf("FAIL  %-16s\n", c.name);
            std::printf("        input    : \"%s\"\n", escape(c.text).c_str());
            std::printf("        expected : %s\n", format_ids(c.expected).c_str());
            std::printf("        got      : %s\n", format_ids(got).c_str());
        }
    }

    std::printf("\n%d/%zu cases failed\n", failures, cases.size());
    return failures == 0 ? 0 : 1;
}
