#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>
#include "glassbox/model.h"
#include "glassbox/token.h"
#include "glassbox/forward.h"
#include "glassbox/utils.h"

// --------- Constants ---------

static constexpr size_t END_OF_TEXT = 50256;



// --------- CLI Options ---------
//
// parse_args() turns argv into this struct; nothing downstream reads argv.
// Adding a flag = add a field here + one branch in parse_args().
struct Options {
    std::string                model_dir;         // required positional
    std::optional<std::string> prompt;            // -p / --prompt        (inline)
    std::optional<std::string> prompt_file;       // -f / --prompt-file
    std::optional<std::string> output;            // -o / --output
    std::optional<std::string> interp_dump;       // -i / --interp <path> (presence = on)
    int                        n_tokens = 50;     // -n / --n-tokens
};

void print_usage(const char* prog) {
    std::cerr <<
        "usage: " << prog << " <model_dir> [options]\n"
        "  -p, --prompt <text>       prompt text, inline\n"
        "  -f, --prompt-file <path>  read the prompt from a file\n"
        "                            (neither -p nor -f: read the prompt from stdin)\n"
        "  -o, --output <path>       write the complete response to a file\n"
        "  -n, --n-tokens <count>    number of tokens to generate (default 50)\n"
        "  -i, --interp <dump file>  run one forward pass and dump the interp cache\n"
        "  -h, --help                show this help\n";
}

Options parse_args(int argc, char* argv[]) {
    Options opt;
    std::vector<std::string_view> positionals;

    for (int i = 1; i < argc; ++i) {
        std::string_view a = argv[i];

        // Fetch the value that follows a value-taking flag, with bounds checking.
        auto value = [&](std::string_view flag) -> std::string_view {
            if (i + 1 >= argc) die(std::string(flag) + " needs an argument");
            return argv[++i];
        };

        if      (a == "-p" || a == "--prompt")      opt.prompt      = std::string(value(a));
        else if (a == "-f" || a == "--prompt-file") opt.prompt_file = std::string(value(a));
        else if (a == "-o" || a == "--output")      opt.output      = std::string(value(a));
        else if (a == "-i" || a == "--interp")      opt.interp_dump = std::string(value(a));
        else if (a == "-n" || a == "--n-tokens")    opt.n_tokens    = std::stoi(std::string(value(a)));
        else if (a == "-h" || a == "--help")        { print_usage(argv[0]); std::exit(0); }
        else if (!a.empty() && a[0] == '-')         die("unknown flag: " + std::string(a));
        else                                        positionals.push_back(a);
    }

    if (positionals.empty()) { print_usage(argv[0]); die("model_dir is required"); }
    opt.model_dir = std::string(positionals[0]);

    if (opt.prompt && opt.prompt_file)
        die("use either --prompt or --prompt-file, not both");

    return opt;
}

// Resolve the prompt from the selected source: inline, file, or stdin (default).
std::string resolve_prompt(const Options& opt) {
    if (opt.prompt) return *opt.prompt;

    if (opt.prompt_file) {
        std::ifstream f(*opt.prompt_file);
        if (!f) die("cannot open prompt file: " + *opt.prompt_file);
        std::stringstream ss;
        ss << f.rdbuf();
        return ss.str();
    }

    std::cout << "prompt> " << std::flush;
    std::string line;
    std::getline(std::cin, line);
    return line;
}


// --------- Entry Point ---------

int main(int argc, char* argv[]) {
    const Options opt = parse_args(argc, argv);

    Model model = load_model(opt.model_dir);

    const std::string prompt = resolve_prompt(opt);
    std::vector<int>  ids    = encode(prompt, model.vocab, model.merge);
    if (ids.empty()) die("empty prompt");

    if (opt.interp_dump) {
        // TODO
        die("interp dump mode is not implemented yet (dump target: " + *opt.interp_dump + ")");
    }

    // Generation
    std::string response = prompt;
    std::cout << prompt << std::flush;

    for (int i = 0; i < opt.n_tokens && ids.size() < model.config.n_ctx; ++i) {
        Tensor x { forward(ids, model) };
        std::vector<float> logits { lm_logits(x, model) };

        // Greedy decoding, currently being used for testing, TODO: Add other modes and flags for this
        size_t best = 0;
        for (size_t t = 1; t < logits.size(); ++t) {
            if (logits[t] > logits[best]) best = t;
        }

        if (best == END_OF_TEXT) break;

        ids.push_back(static_cast<int>(best));
        const std::string piece = decode({static_cast<int>(best)}, model.vocab);
        response += piece;
        std::cout << piece << std::flush;
    }
    std::cout << "\n";

    if (opt.output) {
        std::ofstream out(*opt.output);
        if (!out) die("cannot open output file: " + *opt.output);
        out << response << "\n";
    }

    return 0;
}
