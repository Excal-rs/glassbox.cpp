#include <algorithm>
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
#include "glassbox/interp.h"
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
    AblationConfig             ablation;          // -a / --ablate        (default: NONE)
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
        "  -i, --interp <dump file>  dump the interp cache — the run's last forward\n"
        "                            pass, so the prompt plus the tokens generated\n"
        "                            before it\n"
        "  -a, --ablate <kind>:<n>   drop one layer's sublayer output, so the stream\n"
        "                            passes it by: zero-attn:<n> or zero-mlp:<n>\n"
        "  -h, --help                show this help\n";
}

// Turns "zero-attn:5" into a config. The patch variants have no spelling here:
// they need a values-file format, so v1 exposes them programmatically only.
AblationConfig parse_ablation(std::string_view spec) {
    const size_t colon = spec.find(':');
    if (colon == std::string_view::npos)
        die("--ablate wants <kind>:<layer>, e.g. zero-attn:5");

    const std::string_view kind  = spec.substr(0, colon);
    const std::string_view layer = spec.substr(colon + 1);

    AblationConfig ablation;
    if      (kind == "zero-attn") ablation.type = AblationType::ZERO_ATTENTION;
    else if (kind == "zero-mlp")  ablation.type = AblationType::ZERO_MLP;
    else                          die("--ablate kind must be zero-attn or zero-mlp, got: " + std::string(kind));

    // Checked here so a typo reports itself, rather than std::stoul throwing or
    // "-1" wrapping into a layer index no error message can explain.
    if (layer.empty() || layer.find_first_not_of("0123456789") != std::string_view::npos)
        die("--ablate layer must be a non-negative integer, got: " + std::string(layer));

    ablation.target_layer = std::stoul(std::string(layer));
    return ablation;
}

// One-line note for a run whose output would otherwise be unexplained.
std::string describe_ablation(const AblationConfig& ablation) {
    const std::string layer = " at layer " + std::to_string(ablation.target_layer);

    switch (ablation.type) {
        case AblationType::ZERO_ATTENTION:  return "zeroing attention output" + layer;
        case AblationType::ZERO_MLP:        return "zeroing MLP output" + layer;
        case AblationType::PATCH_ATTENTION: return "patching attention output" + layer;
        case AblationType::PATCH_MLP:       return "patching MLP output" + layer;
        case AblationType::NONE:            break;
    }

    return "none";
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
        else if (a == "-a" || a == "--ablate")      opt.ablation    = parse_ablation(value(a));
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
    const size_t n_prompt_tokens { ids.size() };

    // Interp init. Ablation is independent of capture — either, both, or neither.
    ModelCache       cache {};
    InterpContext    interpctx {};
    std::vector<int> cached_ids;

    interpctx.ablation = opt.ablation;
    if (opt.ablation.type != AblationType::NONE) {
        // forward() validates as well; doing it here too reports a bad flag
        // before any generated text has reached stdout.
        validate_ablation(opt.ablation, model.config, ids.size());
        std::cerr << "ablation: " << describe_ablation(opt.ablation) << "\n";
    }

    if (opt.interp_dump) {
        // Size for the longest pass this run can reach, so the per-pass hook copies
        // reuse the capacity instead of reallocating every token.
        const size_t max_seq_len { std::min(n_prompt_tokens + static_cast<size_t>(opt.n_tokens),
                                            model.config.n_ctx) };
        cache = init_cache(model.config, max_seq_len);
        interpctx.cache = &cache;
    }

    // Generation
    std::string response { prompt };
    std::cout << prompt << std::flush;

    for (int i = 0; i < opt.n_tokens && ids.size() < model.config.n_ctx; ++i) {
        // Each pass overwrites the cache, so what survives the loop is the last one.
        cached_ids = ids;

        Tensor x { forward(ids, model, interpctx) };
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

    if (opt.interp_dump) {
        if (cached_ids.empty())
            die("no forward pass ran, so there are no activations to dump");

        std::ofstream file(*opt.interp_dump, std::ios::binary);
        if (!file) die("cannot open dump file: " + *opt.interp_dump);

        dump_cache(file, cache, model.config, cached_ids, n_prompt_tokens, opt.ablation);

        // ofstream's destructor flushes but swallows any error, so close explicitly.
        file.close();
        if (!file) die("cannot finish writing dump file: " + *opt.interp_dump);

        std::cerr << "wrote " << *opt.interp_dump << " (" << cached_ids.size()
                  << " tokens, " << n_prompt_tokens << " from the prompt, "
                  << model.config.n_layer << " layers)\n";
    }

    if (opt.output) {
        std::ofstream out(*opt.output);
        if (!out) die("cannot open output file: " + *opt.output);
        out << response << "\n";
    }

    return 0;
}
