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
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

// Encodes a Unicode code point as UTF-8 (code points in the model files stay
// well within the 3-byte range), matching token.cpp's byte->unicode mapping.
std::string utf8_encode(unsigned int cp) {
    std::string out;
    if (cp <= 0x7f) {
        out += static_cast<char>(cp);
    } else if (cp <= 0x7ff) {
        out += static_cast<char>(0xc0 | (cp >> 6));
        out += static_cast<char>(0x80 | (cp & 0x3f));
    } else {
        out += static_cast<char>(0xe0 | (cp >> 12));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3f));
        out += static_cast<char>(0x80 | (cp & 0x3f));
    }
    return out;
}

// Reads a JSON string beginning at s[i] (the opening quote), unescaping \", \\,
// the usual short escapes, and \uXXXX. Advances i to just past the closing quote.
std::string parse_json_string(const std::string& s, size_t& i) {
    std::string out;
    ++i;  // skip opening quote
    while (i < s.size() && s[i] != '"') {
        if (s[i] != '\\') {
            out += s[i++];
            continue;
        }
        switch (s[++i]) {
            case '"':  out += '"';  break;
            case '\\': out += '\\'; break;
            case '/':  out += '/';  break;
            case 'n':  out += '\n'; break;
            case 't':  out += '\t'; break;
            case 'r':  out += '\r'; break;
            case 'b':  out += '\b'; break;
            case 'f':  out += '\f'; break;
            case 'u':
                out += utf8_encode(std::stoul(s.substr(i + 1, 4), nullptr, 16));
                i += 4;
                break;
        }
        ++i;
    }
    ++i;  // skip closing quote
    return out;
}

// Loads GPT-2's vocab.json into a token->id map.
Vocab load_vocab(const std::string& path) {
    std::ifstream file(path);
    std::stringstream buffer;
    buffer << file.rdbuf();
    const std::string s = buffer.str();

    Vocab vocab {};
    size_t i = 0;
    while (true) {
        while (i < s.size() && s[i] != '"') ++i;   // next key opens with a quote
        if (i >= s.size()) break;

        const std::string key = parse_json_string(s, i);
        while (i < s.size() && s[i] != ':') ++i;    // skip to ':'
        ++i;
        while (i < s.size() && s[i] == ' ') ++i;     // skip space before the value

        const size_t start = i;
        while (i < s.size() && (s[i] == '-' || (s[i] >= '0' && s[i] <= '9'))) ++i;
        vocab[key] = static_cast<u_int16_t>(std::stoi(s.substr(start, i - start)));
    }
    return vocab;
}

// Loads GPT-2's merges.txt into a "A B" -> rank map (rank = order in the file).
Merge load_merges(const std::string& path) {
    std::ifstream file(path);
    Merge merge {};
    std::string line;
    u_int16_t rank = 0;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;   // skip blank lines + #version
        merge[line] = rank++;
    }
    return merge;
}

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

    const Vocab vocab = load_vocab(std::string(MODEL_DIR) + "/vocab.json");
    const Merge merge = load_merges(std::string(MODEL_DIR) + "/merges.txt");

    for (const auto& c : cases) {
        const std::vector<int> got = encode(c.text, vocab, merge);
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
