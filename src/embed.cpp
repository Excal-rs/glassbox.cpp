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

    const Tensor& wte = model.network.wte;
    const Tensor& wpe = model.network.wpe;

    for (size_t i = 0; i < tokens.size(); ++i){
        for (size_t j = 0; j < vctr_dim; ++j){
            out(i, j) = wte(tokens[i], j) + wpe(i, j);
        }
    }
    return out;
}
