// CLI application headers
#include "input.h"

// Include `glassbox` libraries
#include "glassbox/model.h"
#include "glassbox/token.h"
#include "glassbox/forward.h"
#include "glassbox/interp.h"
#include "glassbox/bench.h"
#include "glassbox/utils.h"

// stdlib includes
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>




// --------- Entry Point ---------

int main(int argc, char* argv[]) {
    const Options opt = parse_arguments(argc, argv);

#ifndef NDEBUG
    if (opt.benchmarking)
        std::cerr << "warning: benchmarking a build without NDEBUG - the numbers will not mean much\n";
#endif

    const auto load_started = std::chrono::steady_clock::now();
    Model model = load_model(opt.model_dir);
    const size_t load_ns { elapsed_ns(load_started) };

    const std::string prompt = resolve_prompt(opt);

    const auto encode_started = std::chrono::steady_clock::now();
    std::vector<int> ids = encode(prompt, model.vocab, model.merge);
    const size_t encode_ns { elapsed_ns(encode_started) };

    if (ids.empty()) die("empty prompt");
    const size_t n_prompt_tokens { ids.size() };

    // Interp init. Ablation is independent of capture - either, both, or neither.
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

    // Benchmark init. The profile is owned here and handed to forward() as a
    // pointer, so a run without -b pays one null check per component.
    Profile  profile {};
    Profile* profilep { nullptr };
    if (opt.benchmarking) {
        profile  = init_profile(model.config);
        profilep = &profile;
        record_stage(profile, Component::LOAD, load_ns);
        record_stage(profile, Component::ENCODE, encode_ns);
    }

    if (opt.interp_dump_out) {
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

        profile.pass_index = static_cast<size_t>(i);
        profile.seq_len    = ids.size();

        const auto pass_started = std::chrono::steady_clock::now();
        Tensor x { forward(ids, model, interpctx, profilep) };
        std::vector<float> logits { lm_logits(x, model, profilep) };
        const size_t pass_ns { elapsed_ns(pass_started) };

        // Greedy decoding, currently being used for testing, TODO: Add other modes and flags for this
        size_t best = 0;
        for (size_t t = 1; t < logits.size(); ++t) {
            if (logits[t] > logits[best]) best = t;
        }

        if (best == END_OF_TEXT) break;

        ids.push_back(static_cast<int>(best));

        const auto decode_started = std::chrono::steady_clock::now();
        const std::string piece = decode({static_cast<int>(best)}, model.vocab);
        const size_t decode_ns { elapsed_ns(decode_started) };

        if (opt.benchmarking) {
            record_stage(profile, Component::DECODE, decode_ns);
            record_stage(profile, Component::PASS, pass_ns);
            finish_pass(profile, static_cast<size_t>(i), profile.seq_len);
        }

        response += piece;
        std::cout << piece << std::flush;
    }
    std::cout << "\n";

    if (opt.benchmarking) {
        write_profile(opt.benchmarking_out, profile, opt.benchmarking_tag);
        std::cerr << "wrote " << profile.rows.size() << " benchmark rows to " << opt.benchmarking_out << "\n";
    }

    if (opt.interp_dump_out) {
        if (cached_ids.empty())
            die("no forward pass ran, so there are no activations to dump");

        std::ofstream file(*opt.interp_dump_out, std::ios::binary);
        if (!file) die("cannot open dump file: " + *opt.interp_dump_out);

        dump_cache(file, cache, model.config, cached_ids, n_prompt_tokens, opt.ablation);

        // ofstream's destructor flushes but swallows any error, so close explicitly.
        file.close();
        if (!file) die("cannot finish writing dump file: " + *opt.interp_dump_out);

        std::cerr << "wrote " << *opt.interp_dump_out << " (" << cached_ids.size()
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
