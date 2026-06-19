#pragma once

#include <vector>
#include <unordered_map>
#include <string>
#include <cstdint>
#include <cstddef>

// All comments regarding shape are specifc to GPT2
// --------- Types ---------
using Vocab  = std::unordered_map<std::string, u_int16_t>;
using Merge  = std::unordered_map<std::string, u_int16_t>;


// --------- Model Network Types ---------
struct Tensor {
    std::vector<float>  data;
    std::vector<size_t> shape;
};

struct Linear {
    Tensor w; // w.shape = {in, out}
    Tensor b; // b.shape = {out}
};

struct LayerNorm {
    Tensor weight;
    Tensor bias;

    // weight.shape = {768}
    // bias.shape   = {768}
};

struct Attention {
    Linear c_attn;
    // attn.b shape = {2304}
    // attn.w shape = {768, 2304}

    Linear c_proj;
    // proj.b shape = {768}
    // proj.w shape = {768, 768}
};

struct MLP {
    Linear c_fc;
    // fc.b shape = {3072}
    // fc.w shape = {768, 3072}

    Linear c_proj;
    // proj.b shape = {768}
    // proj.w shape = {3072, 768}
};

struct Block {
    LayerNorm ln_1, ln_2;
    Attention attn;
    MLP       mlp;
    // h.size() == n_layer == 12
};

struct Network {
    Tensor wpe;
    // Contains the positional embedding vectors
    // Shape = {1024, 768}

    Tensor wte;
    // Contains the token-specific embedding vectors
    // Shape  = {50257, 768}

    LayerNorm ln_f;
    std::vector<Block> h;
};


// --------- Other Model Types ---------

// GPT-2 parameters, read from config.json.
struct Config {
    size_t n_vocab;   // config.json: vocab_size
    size_t n_ctx;     // config.json: n_ctx / n_positions
    size_t n_embd;    // config.json: n_embd
    size_t n_head;    // config.json: n_head
    size_t n_layer;   // config.json: n_layer
    float  ln_eps;    // config.json: layer_norm_epsilon
};


struct Model {
    Network network;
    Config  config;
    Vocab   vocab;
    Merge   merge;
};


// --------- Public API ---------
Model load_model(const std::string& model_dir);

