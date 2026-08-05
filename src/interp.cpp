#include <algorithm>
#include <cstdint>
#include <ostream>
#include <string>
#include "glassbox/interp.h"
#include "glassbox/utils.h"

// --------- Static Forward Declarations ---------

static void write_u32(std::ostream& file, const size_t value);
static void write_buffer(std::ostream& file, const std::vector<float>& buffer, const size_t stream_size);


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


void dump_cache(std::ostream& file, const ModelCache& cache, const Config& config, const std::vector<int>& ids, const size_t n_prompt_tokens, const AblationConfig& ablation)
{
    const size_t seq_len     { ids.size() };
    const size_t stream_size { config.n_embd * seq_len };

    if (seq_len == 0)                                die("dump_cache: no tokens to describe");
    if (n_prompt_tokens > seq_len)                   die("dump_cache: prompt is longer than the cached pass");
    if (cache.init_embeddings.size() != stream_size) die("dump_cache: init_embeddings is not sized for this pass");
    if (cache.layers.size() != config.n_layer)       die("dump_cache: layer count does not match config");

    // Header: 32 bytes, then one u32 per token. The ablation fields say how the
    // pass was perturbed — without them an ablated dump reads as a clean one.
    // patch_values are not recorded: whoever supplied them already has them.
    file.write("GBIC", 4);
    write_u32(file, 2);
    write_u32(file, config.n_layer);
    write_u32(file, config.n_embd);
    write_u32(file, seq_len);
    write_u32(file, n_prompt_tokens);
    write_u32(file, static_cast<size_t>(ablation.type));
    write_u32(file, ablation.target_layer);   // NO_LAYER narrows to 0xFFFFFFFF

    for (const int id : ids){
        if (id < 0) die("dump_cache: negative token id");
        write_u32(file, static_cast<size_t>(id));
    }

    // File Body
    write_buffer(file, cache.init_embeddings, stream_size);
    for (const LayerCache& layer : cache.layers){
        write_buffer(file, layer.attention_output,      stream_size);
        write_buffer(file, layer.stream_post_attention, stream_size);
        write_buffer(file, layer.mlp_output,            stream_size);
        write_buffer(file, layer.stream_post_mlp,       stream_size);
    }

    if (!file) die("dump_cache: write failed");
}


// The layer check also handles AblationType::NONE, whose NO_LAYER target
// matches no real index — so an interp-free run pays one integer compare.
void ablate_attention(std::vector<float>& sublayer_out, const AblationConfig& ablation, const size_t layer_idx)
{
    if (layer_idx != ablation.target_layer) return;

    if      (ablation.type == AblationType::ZERO_ATTENTION)  std::fill(sublayer_out.begin(), sublayer_out.end(), 0.0f);
    else if (ablation.type == AblationType::PATCH_ATTENTION) sublayer_out = ablation.patch_values;
}


void ablate_mlp(std::vector<float>& sublayer_out, const AblationConfig& ablation, const size_t layer_idx)
{
    if (layer_idx != ablation.target_layer) return;

    if      (ablation.type == AblationType::ZERO_MLP)  std::fill(sublayer_out.begin(), sublayer_out.end(), 0.0f);
    else if (ablation.type == AblationType::PATCH_MLP) sublayer_out = ablation.patch_values;
}


void validate_ablation(const AblationConfig& ablation, const Config& config, const size_t seq_len)
{
    if (ablation.type == AblationType::NONE) return;

    if (ablation.target_layer >= config.n_layer)
        die("ablation targets layer " + std::to_string(ablation.target_layer) +
            ", but the model has " + std::to_string(config.n_layer));

    const bool   patching    { ablation.type == AblationType::PATCH_ATTENTION ||
                               ablation.type == AblationType::PATCH_MLP };
    const size_t stream_size { config.n_embd * seq_len };

    if (patching && ablation.patch_values.size() != stream_size)
        die("patch_values holds " + std::to_string(ablation.patch_values.size()) +
            " floats, but this pass needs " + std::to_string(stream_size));
}


// --------- Helper Function Definitions ---------

// Writes one uint32_t value as 4 raw little-endian bytes
static void write_u32(std::ostream& file, const size_t value)
{
    const uint32_t narrowed = static_cast<uint32_t>(value);
    file.write(reinterpret_cast<const char*>(&narrowed), sizeof(narrowed));
}

// Writes one activation buffer as raw float32 bytes
static void write_buffer(std::ostream& file, const std::vector<float>& buffer, const size_t stream_size)
{
    if (buffer.size() != stream_size) die("dump_cache: buffer is not sized for this pass");

    file.write(reinterpret_cast<const char*>(buffer.data()),
               static_cast<std::streamsize>(buffer.size() * sizeof(float)));
}