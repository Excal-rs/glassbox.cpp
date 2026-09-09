// CLI application headers
#include "input.h"

// Include `glassbox` Libraries
#include "glassbox/benchmarking.h"
#include "glassbox/utils.h"

// stdlib includes
#include <charconv>
#include <cstdlib>
#include <fstream>
#include <format>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>


// --------- Static Forward Declarations ---------

static int parse_number(std::string_view text, std::string_view flag);
static AblationConfig parse_ablation(std::string_view spec);


// --------- CLI Flag Table ---------

static constexpr size_t HELP_COLUMN {28};

struct Flag {
    std::string_view short_name, long_name, arg, help;
    void (*apply)(Options&, std::string_view);
};

static constexpr Flag FLAGS[] = {
    {"-p", "--prompt", "<text>", "prompt text, inline",
     [](Options& o, std::string_view v) { o.prompt = std::string(v); }},

    {"-f", "--prompt-file", "<path>",
     "read the prompt from a file\n"
     "(neither -p nor -f: read the prompt from stdin)",
     [](Options& o, std::string_view v) { o.prompt_file = std::string(v); }},

    {"-o", "--output", "<path>", "write the complete response to a file",
     [](Options& o, std::string_view v) { o.output = std::string(v); }},

    {"-n", "--n-tokens", "<count>", "number of tokens to generate (default 50)",
     [](Options& o, std::string_view v) { o.n_tokens = parse_number(v, "--n-tokens"); }},

    {"-i", "--interp", "<dump file>",
     "dump the interp cache - the run's last forward\n"
     "pass, so the prompt plus the tokens generated\n"
     "before it",
     [](Options& o, std::string_view v) { o.interp_dump_out = std::string(v); }},

    {"-a", "--ablate", "<kind>:<n>",
     "drop one layer's sublayer output, so the stream\n"
     "passes it by: zero-attn:<n> or zero-mlp:<n>",
     [](Options& o, std::string_view v) { o.ablation = parse_ablation(v); }},

    {"-b", "--benchmark", "", "time every component and append them to a CSV",
     [](Options& o, std::string_view) { o.benchmarking = true; }},

    {"", "--bench-out", "<path>", "where those rows go (default benchmarking.csv)",
     [](Options& o, std::string_view v) { o.benchmarking_out = std::string(v); }},

    {"", "--bench-tag", "<text>", "stored on every row, for a sweep to label runs by",
     [](Options& o, std::string_view v) { o.benchmarking_tag = std::string(v); }},

    {"-h", "--help", "", "show this help",
     [](Options&, std::string_view) { print_usage(std::cout); std::exit(0); }},
};


// --------- Public API ---------

void print_usage(std::ostream& os) {
    os << "usage: glassbox-cli <model_dir> [options]\n";

    for (const Flag& flag : FLAGS) {
        std::string lead { flag.short_name.empty()
            ? std::format("      {} {}", flag.long_name, flag.arg)
            : std::format("  {}, {} {}", flag.short_name, flag.long_name, flag.arg) };

        for (size_t pos {0}; pos < flag.help.size(); ) {
            const size_t nl { flag.help.find('\n', pos) };
            os << std::format("{:<{}}{}\n", lead, HELP_COLUMN, flag.help.substr(pos, nl - pos));

            lead.clear();
            pos = (nl == std::string_view::npos) ? flag.help.size() : nl + 1;
        }
    }
}

Options parse_arguments(int argc, char* argv[]) {
    Options options;
    std::vector<std::string_view> positionals;

    for (int i {1}; i < argc; ++i) {
        std::string_view arg { argv[i] };

        const Flag* match { nullptr };
        for (const Flag& flag : FLAGS) {
            if (arg == flag.long_name || (!flag.short_name.empty() && arg == flag.short_name)) {
                match = &flag;
                break;
            }
        }

        if (!match) {
            if (!arg.empty() && arg[0] == '-') {
                print_usage(std::cerr);
                die("unknown flag: " + std::string(arg));
            }
            positionals.push_back(arg);
            continue;
        }

        std::string_view value;
        if (!match->arg.empty()) {
            if (i + 1 >= argc) {
                print_usage(std::cerr);
                die(std::string(arg) + " needs an argument");
            }
            value = argv[++i];
        }

        match->apply(options, value);
    }

    if (positionals.empty()) {
        print_usage(std::cerr);
        die("model_dir is required");
    }
    options.model_dir = std::string(positionals[0]);

    if (options.prompt && options.prompt_file)
        die("use either --prompt or --prompt-file, not both");

    return options;
}


// Resolve the prompt from the selected source: inline, file, or stdin (default).
std::string resolve_prompt(const Options& opt) {
    if (opt.prompt) return *opt.prompt;

    if (opt.prompt_file) {
        std::ifstream file(*opt.prompt_file);
        if (!file) die("cannot open prompt file: " + *opt.prompt_file);

        std::stringstream buffer;
        buffer << file.rdbuf();
        return buffer.str();
    }

    std::cout << "prompt> " << std::flush;

    std::string line;
    std::getline(std::cin, line);
    return line;
}

// One-line note for a run whose output would otherwise be unexplained.
std::string describe_ablation(const AblationConfig& ablation) {
    const std::string layer { " at layer " + std::to_string(ablation.target_layer) };

    switch (ablation.type) {
        case AblationType::ZERO_ATTENTION:  return "zeroing attention output" + layer;
        case AblationType::ZERO_MLP:        return "zeroing MLP output" + layer;
        case AblationType::PATCH_ATTENTION: return "patching attention output" + layer;
        case AblationType::PATCH_MLP:       return "patching MLP output" + layer;
        case AblationType::NONE:            break;
    }

    return "none";
}


// --------- Helper Function Definitions ---------

// Used for parsing numbers for flags that take a number argument
static int parse_number(std::string_view text, std::string_view flag) {
    int value {};
    const auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), value);

    if (ec != std::errc{} || end != text.data() + text.size())
        die(std::string(flag) + " wants a number, got: " + std::string(text));

    return value;
}

// Turns interp commands (e.g. "zero-attn:5") into a config
static AblationConfig parse_ablation(std::string_view spec) {
    const size_t colon { spec.find(':') };
    if (colon == std::string_view::npos)
        die("--ablate wants <kind>:<layer>, e.g. zero-attn:5");

    const std::string_view kind  { spec.substr(0, colon) };
    const std::string_view layer { spec.substr(colon + 1) };

    AblationConfig ablation;
    if      (kind == "zero-attn") ablation.type = AblationType::ZERO_ATTENTION;
    else if (kind == "zero-mlp")  ablation.type = AblationType::ZERO_MLP;
    else                          die("--ablate kind must be zero-attn or zero-mlp, got: " + std::string(kind));

    const int layer_index { parse_number(layer, "--ablate layer") };
    if (layer_index < 0)
        die("--ablate layer must be non-negative, got: " + std::string(layer));

    ablation.target_layer = static_cast<size_t>(layer_index);
    return ablation;
}
