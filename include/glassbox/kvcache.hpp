#pragma once

// Include `glassbox` libraries
#include "glassbox/model.hpp"

// Include stdlib
#include <cstddef>
#include <vector>

struct KV_layer {
    Tensor k; // k.shape = {n_ctx, n_embd}
    Tensor v; // v.shape = {n_ctx, n_embd}

    explicit KV_layer(const Config& config)
        : k { {config.n_ctx, config.n_embd} },
          v { {config.n_ctx, config.n_embd} } {}
};

struct KV_cache {
    std::vector<KV_layer> layers; // layers.size() == config.n_layer
    size_t                length;

    explicit KV_cache(const Config& config)
        : layers(config.n_layer, KV_layer { config }),
          length { 0 } {}
};
