// CLI application headers
#include "input.hpp"

// Include `glassbox` libraries
#include "glassbox/model.hpp"
#include "glassbox/token.hpp"
#include "glassbox/forward.hpp"
#include "glassbox/interp.hpp"
#include "glassbox/kvcache.hpp"
#include "glassbox/benchmarking.hpp"
#include "glassbox/utils.hpp"

// Include stdlib
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

int main(int argc, char* argv[]) {
    const Options opt { parse_arguments(argc, argv) };

#ifndef NDEBUG
    if (opt.benchmarking)
        std::cerr << "warning: benchmarking a build without NDEBUG - the numbers will not mean much\n";
#endif

    // Load (and time) model
    const auto load_started { std::chrono::steady_clock::now() };
    Model model { load_model(opt.model_dir) };
    const size_t load_ns { elapsed_ns(load_started) };

    const std::string prompt { resolve_prompt(opt) };

    // Encode prompt into tokens (and time)
    const auto encode_started { std::chrono::steady_clock::now() };
    std::vector<int> ids { encode(prompt, model.vocab, model.merge) };
    const size_t encode_ns { elapsed_ns(encode_started) };

    if (ids.empty()) die("empty prompt");
    const size_t n_prompt_tokens { ids.size() };

    // Interp init
    ModelCache       cache {};
    InterpContext    interpctx {};
    std::vector<int> cached_ids;

    interpctx.ablation = opt.ablation;
    if (opt.ablation.type != AblationType::NONE) {
        validate_ablation(opt.ablation, model.config, ids.size());
        std::cerr << "ablation: " << describe_ablation(opt.ablation) << "\n";
    }

    // Benchmark init
    Profile  profile {};
    Profile* profilep { nullptr };
    if (opt.benchmarking) {
        profile  = init_profile(model.config);
        profilep = &profile;
        record_stage(profile, Component::LOAD, load_ns);
        record_stage(profile, Component::ENCODE, encode_ns);
    }

    if (opt.interp_dump_out) {
        const size_t max_seq_len { std::min(n_prompt_tokens + static_cast<size_t>(opt.n_tokens),
                                            model.config.n_ctx) };
        cache = init_cache(model.config, max_seq_len);
        interpctx.cache = &cache;
    }

    std::optional<KV_cache> kv;
    if (opt.kv_cache) kv.emplace(model.config);
    KV_cache* kvp { kv ? &*kv : nullptr };

    // Generation
    std::cout << prompt << std::flush;

    for (int i {0}; i < opt.n_tokens && ids.size() < model.config.n_ctx; ++i) {
        // Each pass overwrites the cache, only final one written
        cached_ids = ids;

        profile.pass_index = static_cast<size_t>(i);
        profile.seq_len    = ids.size();

        std::cerr << "\rGenerating Token " << (i + 1) << "/" << opt.n_tokens << "..." << std::flush;

        const auto pass_started { std::chrono::steady_clock::now() };
        Tensor x { forward(ids, model, interpctx, kvp, profilep) };
        std::vector<float> logits { lm_logits(x, model, profilep) };
        const size_t pass_ns { elapsed_ns(pass_started) };

        // Greedy decoding, TODO: Add other modes and flags for this
        size_t best {0};
        for (size_t t {1}; t < logits.size(); ++t) {
            if (logits[t] > logits[best]) best = t;
        }

        if (best == END_OF_TEXT) break;

        ids.push_back(static_cast<int>(best));

        if (opt.benchmarking) {
            record_stage(profile, Component::PASS, pass_ns);
            finish_pass(profile, static_cast<size_t>(i), profile.seq_len);
        }
    }
    std::cerr << "\n";

    // Decode
    const auto decode_started { std::chrono::steady_clock::now() };
    const std::vector<int> generated_ids(ids.begin() + static_cast<std::ptrdiff_t>(n_prompt_tokens), ids.end());
    const std::string generated { decode(generated_ids, model.vocab) };
    const size_t decode_ns { elapsed_ns(decode_started) };

    if (opt.benchmarking) record_stage(profile, Component::DECODE, decode_ns);

    const std::string response { prompt + generated };
    std::cout << generated << "\n";

    if (opt.benchmarking) {
        write_profile(opt.benchmarking_out, profile, opt.benchmarking_tag);
        std::cerr << "wrote " << profile.rows.size() << " benchmark rows to " << opt.benchmarking_out << "\n";
    }

    if (opt.interp_dump_out) {
        if (cached_ids.empty()) die("no forward pass ran, so there are no activations to dump");

        std::ofstream file(*opt.interp_dump_out, std::ios::binary);
        if (!file) die("cannot open interp dump file: " + *opt.interp_dump_out);

        dump_cache(file, cache, model.config, cached_ids, n_prompt_tokens, opt.ablation);

        file.close();
        if (!file) die("cannot finish writing interp dump file: " + *opt.interp_dump_out);

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
