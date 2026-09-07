#include <chrono>
#include <ctime>
#include <fstream>
#include <string>
#include "glassbox/bench.h"
#include "glassbox/utils.h"

// Set by CMake from `git rev-parse --short HEAD` at configure time, so it goes
// stale if you rebuild without re-running cmake. tools/bench.py reports git's
// answer alongside it and warns when the two disagree.
#ifndef GLASSBOX_COMMIT
#define GLASSBOX_COMMIT "unknown"
#endif

// --------- Constants ---------

#ifdef NDEBUG
static constexpr const char* BUILD = "release";
#else
static constexpr const char* BUILD = "debug";
#endif

static constexpr const char* CSV_HEADER =
    "commit,build,date,tag,pass_index,seq_len,layer,component,calls,ns,flops,bytes,alloc\n";

// --------- Static Forward Declarations ---------

static const char* component_name(Component component);
static std::string timestamp();


// --------- Public API ---------

ScopeTimer::ScopeTimer(Profile* profile, Component component, size_t layer, Cost cost)
    : profile(profile), component(component), layer(layer), cost(cost)
{
    if (profile) start = std::chrono::steady_clock::now();
}


ScopeTimer::~ScopeTimer()
{
    if (!profile) return;

    Sample& slot = profile->slots[layer * COMPONENT_COUNT + static_cast<size_t>(component)];

    slot.calls      += 1;
    slot.ns         += elapsed_ns(start);
    slot.cost.flops += cost.flops;
    slot.cost.bytes += cost.bytes;
    slot.cost.alloc += cost.alloc;
}


// One slot per (layer, component), plus a final row of slots for the stages
// that sit outside any block: the embedding, ln_f, the logits, the run itself.
Profile init_profile(const Config& config)
{
    Profile profile {};
    profile.n_layer = config.n_layer;
    profile.slots   = std::vector<Sample>((config.n_layer + 1) * COMPONENT_COUNT);
    return profile;
}


// Moves the pass in flight into rows and clears the slots for the next one.
// Passes are never folded together here: without a KV cache each one runs over
// a longer prefix, so they are different measurements rather than repeats.
void finish_pass(Profile& profile, size_t pass_index, size_t seq_len)
{
    for (size_t layer = 0; layer <= profile.n_layer; ++layer){
        for (size_t c = 0; c < COMPONENT_COUNT; ++c){
            const Sample& slot = profile.slots[layer * COMPONENT_COUNT + c];
            if (slot.calls == 0) continue;

            profile.rows.push_back(Row{
                .pass_index = pass_index,
                .seq_len    = seq_len,
                .layer      = layer,
                .component  = static_cast<Component>(c),
                .sample     = slot,
            });
        }
    }

    profile.slots.assign(profile.slots.size(), Sample{});
}


// For work outside a forward pass: loading the model, encoding, decoding.
void record_stage(Profile& profile, Component component, size_t ns)
{
    profile.rows.push_back(Row{
        .pass_index = profile.pass_index,
        .seq_len    = profile.seq_len,
        .layer      = profile.n_layer,
        .component  = component,
        .sample     = Sample{ .calls = 1, .ns = ns, .cost = {} },
    });
}


// Appends every row, writing the header only into an empty file, so a sweep can
// point run after run at the same path.
void write_profile(const std::string& path, const Profile& profile, const std::string& tag)
{
    const bool fresh { !std::ifstream(path).good() };

    std::ofstream file(path, std::ios::app);
    if (!file) die("cannot open benchmark file: " + path);
    if (fresh) file << CSV_HEADER;

    const std::string date = timestamp();

    for (const Row& row : profile.rows){
        file << GLASSBOX_COMMIT           << ',' << BUILD             << ','
             << date                      << ',' << tag               << ','
             << row.pass_index            << ',' << row.seq_len       << ','
             << row.layer                 << ','
             << component_name(row.component)                         << ','
             << row.sample.calls          << ',' << row.sample.ns     << ','
             << row.sample.cost.flops     << ',' << row.sample.cost.bytes << ','
             << row.sample.cost.alloc     << '\n';
    }

    file.close();
    if (!file) die("benchmark write failed: " + path);
}


size_t elapsed_ns(std::chrono::steady_clock::time_point start)
{
    const auto elapsed = std::chrono::steady_clock::now() - start;
    return static_cast<size_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count());
}


// --------- Cost Helpers ---------

// A (shape {m, k}) by B (shape {k, n}): two flops per multiply-add, and every
// element of both inputs and of the output crosses the bus once.
Cost matmul_cost(size_t m, size_t k, size_t n)
{
    return Cost{
        .flops = 2 * m * k * n,
        .bytes = 4 * (m * k + k * n + m * n),
        .alloc = 4 * m * n,
    };
}


// A reshape or transpose: no arithmetic, one read and one write per element.
Cost copy_cost(size_t elements)
{
    return Cost{ .flops = 0, .bytes = 8 * elements, .alloc = 4 * elements };
}


// Work done in place over `elements`, at a nominal cost per element. The flop
// count is a fiction for anything containing exp() - which is why softmax and
// GELU get their own buckets, to be read on time and bytes rather than rate.
Cost elementwise_cost(size_t elements, size_t flops_each)
{
    return Cost{ .flops = elements * flops_each, .bytes = 8 * elements, .alloc = 0 };
}


// --------- Helper Function Definitions ---------

static const char* component_name(Component component)
{
    switch (component) {
        case Component::EMBED:     return "embed";
        case Component::LN:        return "ln";
        case Component::QKV:       return "qkv";
        case Component::SPLIT:     return "split";
        case Component::TRANSPOSE: return "transpose";
        case Component::SCORES:    return "scores";
        case Component::SOFTMAX:   return "softmax";
        case Component::AV:        return "av";
        case Component::MERGE:     return "merge";
        case Component::ATTN_PROJ: return "attn_proj";
        case Component::RESIDUAL:  return "residual";
        case Component::MLP_FC:    return "mlp_fc";
        case Component::GELU:      return "gelu";
        case Component::MLP_PROJ:  return "mlp_proj";
        case Component::LN_F:      return "ln_f";
        case Component::LOGITS:    return "logits";
        case Component::LOAD:      return "load";
        case Component::ENCODE:    return "encode";
        case Component::DECODE:    return "decode";
        case Component::PASS:      return "pass";
        case Component::COUNT:     break;
    }
    return "unknown";
}


static std::string timestamp()
{
    const std::time_t now = std::time(nullptr);
    char buffer[32];
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%S", std::localtime(&now));
    return buffer;
}
