#include "glassbox/interp.h"

// --------- Public API ---------

ModelCache init_cache(const Config& config, const size_t seq_len){
    const size_t n_layer     { config.n_layer };
    const size_t stream_size { config.n_embd * seq_len };

    ModelCache cache {
        .init_embeddings = std::vector<float>(stream_size),
        .layers = std::vector<LayerCache>(n_layer)
    };

    for (size_t i = 0; i < n_layer; ++i){
        cache.layers[i].attention_output      = std::vector<float>(stream_size);
        cache.layers[i].stream_post_attention = std::vector<float>(stream_size);
        cache.layers[i].mlp_output            = std::vector<float>(stream_size);
        cache.layers[i].stream_post_mlp       = std::vector<float>(stream_size);
    }
    
    return cache;
}