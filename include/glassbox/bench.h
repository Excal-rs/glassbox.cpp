#pragma once

#include <chrono>
#include <cstddef>
#include <string>
#include <vector>
#include "glassbox/model.h"

// --------- Component Types ---------

// Written into CSV rows as text, so append new ones rather than renumbering.
enum class Component {
    EMBED = 0, LN, QKV, SPLIT, TRANSPOSE, SCORES, SOFTMAX, AV, MERGE, ATTN_PROJ,
    RESIDUAL, MLP_FC, GELU, MLP_PROJ, LN_F, LOGITS, LOAD, ENCODE, DECODE, PASS,
    COUNT
};

inline constexpr size_t COMPONENT_COUNT = static_cast<size_t>(Component::COUNT);

// --------- Profile Types ---------

// What one call to a kernel costs, worked out from its shapes rather than
// counted as it runs - the shapes are known, and a counter in the inner loop
// would cost more than the loop it was counting.
struct Cost {
    size_t flops = 0;
    size_t bytes = 0;   // moved: weights read, activations read and written
    size_t alloc = 0;   // freshly allocated, since every kernel returns by value
};

struct Sample {
    size_t calls = 0;
    size_t ns    = 0;
    Cost   cost;
};

// One finished measurement, in the shape the CSV wants it.
struct Row {
    size_t    pass_index;
    size_t    seq_len;
    size_t    layer;        // n_layer for anything that is not inside a block
    Component component;
    Sample    sample;
};

struct Profile {
    size_t n_layer    = 0;
    size_t pass_index = 0;
    size_t seq_len    = 0;

    std::vector<Sample> slots;   // (n_layer + 1) * COMPONENT_COUNT, the pass in flight
    std::vector<Row>    rows;    // every pass that has finished
};

// Times its own scope and adds the result to (layer, component). A null profile
// costs one branch and nothing else, so call sites need no condition of their own.
struct ScopeTimer {
    ScopeTimer(Profile* profile, Component component, size_t layer, Cost cost);
    ~ScopeTimer();

    Profile*  profile;
    Component component;
    size_t    layer;
    Cost      cost;

    std::chrono::steady_clock::time_point start;
};

// --------- Public API ---------
Profile init_profile(const Config& config);
void    finish_pass(Profile& profile, size_t pass_index, size_t seq_len);
void    record_stage(Profile& profile, Component component, size_t ns);
void    write_profile(const std::string& path, const Profile& profile, const std::string& tag);

// Nanoseconds since `start`, for the stages main times by hand.
size_t elapsed_ns(std::chrono::steady_clock::time_point start);

// --------- Cost Helpers ---------
Cost matmul_cost(size_t m, size_t k, size_t n);
Cost copy_cost(size_t elements);
Cost elementwise_cost(size_t elements, size_t flops_each);
