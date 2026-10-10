#pragma once

// Include `glassbox` libraries
#include "glassbox/interp.hpp"

// Include stdlib
#include <optional>
#include <ostream>
#include <string>

// --------- CLI Flags ---------
struct Options {
    std::string                 model_dir;

    std::optional<std::string>  prompt;
    std::optional<std::string>  prompt_file;
    std::optional<std::string>  output;
    int                         n_tokens {50};

    AblationConfig              ablation;
    std::optional<std::string>  interp_dump_out;

    bool                        kv_cache {true};

    bool                        benchmarking {false};
    std::string                 benchmarking_out {"benchmarking.csv"};
    std::string                 benchmarking_tag;
};

// --------- Public API ---------
void        print_usage(std::ostream& os);
Options     parse_arguments(int argc, char* argv[]);
std::string resolve_prompt(const Options& opt);
std::string describe_ablation(const AblationConfig& ablation);
