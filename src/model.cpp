#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "glassbox/model.h"

using json = nlohmann::json;

// --------- Local Types ---------
struct SafeTensors {
    json          header;
    std::ifstream file;
    size_t        data_start;
};

// --------- Static Forward Declarations ---------
static Vocab   load_vocab   (const std::string& path);
static Merge   load_merge   (const std::string& path);
static Config  load_config  (const std::string& path);
static Network load_network (const std::string& path, const Config& config);

static SafeTensors open_safetensors (const std::string& path);
static Tensor      load_tensor      (SafeTensors& st, const std::string& name);

// --------- Function Definitions ---------

// Takes a path to a model directory and loads the model files into a `Model` struct
Model load_model(const std::string& model_dir){
    Config config = load_config(model_dir + "/config.json");

    Model model;
    model.config  = config;
    model.vocab   = load_vocab  (model_dir + "/vocab.json");
    model.merge   = load_merge  (model_dir + "/merges.txt");
    model.network = load_network(model_dir + "/model.safetensors", config);
    return model;
}

// Helper function for `load_model`, loads the vocab files.
static Vocab load_vocab(const std::string& path){
    std::ifstream vocabf(path);
    if (!vocabf){
        std::cerr << "Error loading vocab file!";
        exit(1);
    }

    json j;
    vocabf >> j;

    Vocab vocab{};
    for (auto& [token, id] : j.items()) {
        vocab[token] = id.get<uint16_t>();
    }

    return vocab;
}

// Helper function for `load_model`, loads the merge files.
static Merge load_merge(const std::string& path){
    std::ifstream mergesf(path);
    if (!mergesf){
        std::cerr << "Error opening merges file!";
        exit(1);
    }

    Merge merge{};
    std::string line;

    std::getline(mergesf, line);

    uint16_t rank = 0;
    while (std::getline(mergesf, line)){
        if (line.empty()) continue;
        merge[line] = rank++;
    }

    return merge;
}

// Helper function for `load_model`, loads the config files.
static Config load_config(const std::string& path){
    std::ifstream configf(path);
    if (!configf) {
        std::cerr << "Error loading Model config file!";
        exit(1);
    }

    json j;
    configf >> j;

    Config config {
        .n_vocab = j.at("vocab_size"),
        .n_ctx   = j.at("n_ctx"),
        .n_embd  = j.at("n_embd"),
        .n_head  = j.at("n_head"),
        .n_layer = j.at("n_layer"),
        .ln_eps  = j.at("layer_norm_epsilon"),
    };

    return config;
}

// Helper function for `load_model`, loads safetensor file into `SafeTensors` struct
static SafeTensors open_safetensors(const std::string& path){
    std::ifstream safetensorsf(path, std::ios::binary);
    if (!safetensorsf){
        std::cerr << "Error opening weights file!";
        exit(1);
    }

    uint64_t header_len;
    safetensorsf.read(reinterpret_cast<char*>(&header_len), 8);

    std::string hdr(header_len, '\0');
    safetensorsf.read(hdr.data(), header_len);

    SafeTensors st;
    st.header     = json::parse(hdr);
    st.data_start = 8 + header_len;
    st.file       = std::move(safetensorsf);
    return st;
}

// Extracts a named tensor from a given safetensor file
static Tensor load_tensor(SafeTensors& st, const std::string& name){
    auto& meta = st.header.at(name);

    Tensor t;
    t.shape = meta.at("shape").get<std::vector<size_t>>();

    auto   offsets = meta.at("data_offsets");
    size_t begin   = offsets.at(0);
    size_t end     = offsets.at(1);

    t.data.resize((end - begin) / sizeof(float));
    st.file.seekg(st.data_start + begin);
    st.file.read(reinterpret_cast<char*>(t.data.data()), end - begin);
    return t;
}

// Loads all tensors into the Network Struct
static Network load_network(const std::string& path, const Config& config){
    SafeTensors st = open_safetensors(path);

    Network net;
    net.wte         = load_tensor(st, "wte.weight");
    net.wpe         = load_tensor(st, "wpe.weight");
    net.ln_f.weight = load_tensor(st, "ln_f.weight");
    net.ln_f.bias   = load_tensor(st, "ln_f.bias");

    net.h.resize(config.n_layer);
    for (size_t i = 0; i < config.n_layer; ++i){
        std::string p = "h." + std::to_string(i) + ".";
        Block& b = net.h[i];

        b.ln_1.weight = load_tensor(st, p + "ln_1.weight");
        b.ln_1.bias   = load_tensor(st, p + "ln_1.bias");
        b.ln_2.weight = load_tensor(st, p + "ln_2.weight");
        b.ln_2.bias   = load_tensor(st, p + "ln_2.bias");

        b.attn.c_attn.w = load_tensor(st, p + "attn.c_attn.weight");
        b.attn.c_attn.b = load_tensor(st, p + "attn.c_attn.bias");
        b.attn.c_proj.w = load_tensor(st, p + "attn.c_proj.weight");
        b.attn.c_proj.b = load_tensor(st, p + "attn.c_proj.bias");

        b.mlp.c_fc.w   = load_tensor(st, p + "mlp.c_fc.weight");
        b.mlp.c_fc.b   = load_tensor(st, p + "mlp.c_fc.bias");
        b.mlp.c_proj.w = load_tensor(st, p + "mlp.c_proj.weight");
        b.mlp.c_proj.b = load_tensor(st, p + "mlp.c_proj.bias");
    }

    return net;
}
