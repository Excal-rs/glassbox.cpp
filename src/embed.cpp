#include "glassbox/embed.h"


// --------- Public API ---------

// Takes in a vector of tokens then will add their positional and token vectors together, 
// Will return a Tensor contianing the embedded vectors
Tensor embed(const std::vector<int>& tokens, const Model& model){
    const size_t vctr_dim = model.config.n_embd;
    Tensor out {
        .data  = std::vector<float>(tokens.size() * vctr_dim),
        .shape = {tokens.size(), vctr_dim}
    };

    const auto& wte = model.network.wte.data;   
    const auto& wpe = model.network.wpe.data;

    for (size_t i = 0; i < tokens.size(); ++i){
        const size_t tok_base = tokens[i] * vctr_dim;
        const size_t pos_base = i * vctr_dim; 
        const size_t dst_base = i * vctr_dim;

        for (size_t j = 0; j < vctr_dim; ++j){
            out.data[dst_base + j] = wte[tok_base + j] + wpe[pos_base + j];
        }
    }
    return out;
}
