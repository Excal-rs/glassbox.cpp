#pragma once

#include <vector>
#include <cstddef>
#include <limits>
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

// --------- Ablation Types ---------

// Unsigned so it never trips -Wsign-compare against a layer index.
inline constexpr size_t NO_LAYER = std::numeric_limits<size_t>::max();

// These values are written into GBIC dumps, so they are part of the file
// format: append new ones, never renumber the existing ones.
enum class AblationType {
    NONE            = 0,
    ZERO_ATTENTION  = 1,
    ZERO_MLP        = 2,
    PATCH_ATTENTION = 3,
    PATCH_MLP       = 4,
};

struct AblationConfig {
    AblationType       type         = AblationType::NONE;
    size_t             target_layer = NO_LAYER;

    // PATCH_* only: replaces the sublayer output wholesale, so these are
    // post-bias values. Sized for one pass (seq_len * n_embd) — seq_len grows
    // with every generated token, so a patch outlives only a single forward().
    std::vector<float> patch_values;
};

struct InterpContext {
    ModelCache*    cache = nullptr;
    AblationConfig ablation;
};

// --------- Public API ---------
ModelCache init_cache(const Config& config, const size_t seq_len);
void       dump_cache(std::ostream& file, const ModelCache& cache, const Config& config, const std::vector<int>& ids, const size_t n_prompt_tokens, const AblationConfig& ablation);

// Replace a sublayer's pre-residual output when the config targets it; a no-op
// otherwise. Call after the cache hook, so the cache keeps the clean value.
void ablate_attention(std::vector<float>& sublayer_out, const AblationConfig& ablation, const size_t layer_idx);
void ablate_mlp(std::vector<float>& sublayer_out, const AblationConfig& ablation, const size_t layer_idx);

// Rejects an ablation this pass cannot honour. Dies rather than silently doing nothing.
void validate_ablation(const AblationConfig& ablation, const Config& config, const size_t seq_len);
