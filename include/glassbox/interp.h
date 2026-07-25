#pragma once

#include <vector>
#include <cstddef>
#include <ostream>
#include "model.h"

// --------- Interp Cache Types ---------
struct LayerCache {
    std::vector<float> attention_output;
    std::vector<float> stream_post_attention;
    std::vector<float> mlp_output;
    std::vector<float> stream_post_mlp;
};

struct ModelCache {
    std::vector<float>      init_embeddings;
    std::vector<LayerCache> layers;
};

struct InterpContext {
    ModelCache* cache = nullptr;
};

// --------- Public API ---------
ModelCache init_cache(const Config& config, const size_t seq_len);
void       dump_cache(std::ostream& file, const ModelCache& cache, const Config& config, const size_t seq_len);
