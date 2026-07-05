#include <iostream>
#include <string>
#include <vector>
#include "glassbox/model.h"
#include "glassbox/token.h"
#include "glassbox/forward.h"

// --------- Constants ---------

static constexpr size_t END_OF_TEXT = 50256;

// --------- Entry Point ---------

int main(int argc, char* argv[])
{
    if (argc < 3){
        std::cerr << "usage: " << argv[0] << " <model_dir> \"<prompt>\" [n_tokens]\n";
        return 1;
    }

    const std::string model_dir {argv[1]};
    const std::string prompt    {argv[2]};
    const int n_tokens = (argc > 3) ? std::stoi(argv[3]) : 50;

    Model model = load_model(model_dir);
    std::vector<int> ids = encode(prompt, model.vocab, model.merge);

    std::cout << prompt << std::flush;

    for (int i = 0; i < n_tokens && ids.size() < model.config.n_ctx; ++i){
        Tensor x { forward(ids, model) };
        std::vector<float> logits { lm_logits(x, model) };

        // Greedy decoding: take the highest-scoring token id
        size_t best = 0;
        for (size_t t = 1; t < logits.size(); ++t){
            if (logits[t] > logits[best]) best = t;
        }

        if (best == END_OF_TEXT) break;

        ids.push_back(static_cast<int>(best));
        std::cout << decode({static_cast<int>(best)}, model.vocab) << std::flush;
    }

    std::cout << "\n";
    return 0;
}
