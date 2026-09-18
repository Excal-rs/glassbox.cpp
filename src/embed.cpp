// Include `glassbox` libraries
#include "glassbox/embed.h"

// --------- Public API ---------

// Takes in a vector of tokens and adds their positional and token vectors together.
// Returns a Tensor containing the embedded vectors.
Tensor embed(const std::vector<int>& tokens, const Model& model){
    const size_t n_embd { model.config.n_embd };
    Tensor out {
        .data  = std::vector<float>(tokens.size() * n_embd),
        .shape = {tokens.size(), n_embd}
    };

    const Tensor& wte = model.network.wte;
    const Tensor& wpe = model.network.wpe;

    for (size_t i {0}; i < tokens.size(); ++i){
        for (size_t j {0}; j < n_embd; ++j){
            out(i, j) = wte(tokens[i], j) + wpe(i, j);
        }
    }
    return out;
}
