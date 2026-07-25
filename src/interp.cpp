#include <cstdint>
#include <ostream>
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


void dump_cache(std::ostream& file, const ModelCache& cache, const Config& config, const size_t seq_len)
{
    const size_t stream_size { config.n_embd * seq_len };

    if (cache.init_embeddings.size() != stream_size) die("dump_cache: init_embeddings is not sized for this pass");
    if (cache.layers.size() != config.n_layer)       die("dump_cache: layer count does not match config");

    // Header: 20 bytes.
    file.write("GBIC", 4);              
    write_u32(file, 1);
    write_u32(file, config.n_layer);
    write_u32(file, config.n_embd);
    write_u32(file, seq_len);

    // File Body
    write_buffer(file, cache.init_embeddings, stream_size);
    for (const LayerCache& layer : cache.layers){
        write_buffer(file, layer.attention_output,      stream_size);
        write_buffer(file, layer.stream_post_attention, stream_size);
        write_buffer(file, layer.mlp_output,            stream_size);
        write_buffer(file, layer.stream_post_mlp,       stream_size);
    }

    // ostream failure is sticky, so one check here catches a fault anywhere above.
    if (!file) die("dump_cache: write failed");
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