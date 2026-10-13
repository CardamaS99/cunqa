#pragma once

#include <array>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "sim/noise/channels.hpp"
#include "sim/noise/device_noise.hpp"
#include "sim/timing/random_delay.hpp"
#include "utils/json.hpp"

namespace cunqa {
namespace sim {

// How the random delays of a shot (EPR generation times, classical jitter) are treated.
enum class WaitStrategy {
    // Every shot samples its own delays, so each shot is one realization of the timeline.
    // Sampled delays can be binned (time_bin) to emulate grouping shots by realization.
    EXACT,
    // Delays are not sampled: the timeline uses their mean and every random wait is turned
    // into the averaged relaxation channel E[R_T], built from the Laplace transform of the
    // waiting time. Exact for one qubit, it misses the correlations of waits shared by
    // several qubits.
    AVERAGED
};

enum class NoiseMode {
    AUTO,   // EXACT when the simulator supports Kraus channels, PAULI otherwise
    EXACT,
    PAULI
};

// One pair delivered by an entanglement generator: its generation time and the Bell
// coefficients {Phi+, Psi+, Psi-, Phi-} of the state at heralding.
struct EprDraw {
    double delay;
    noise::BellCoefficients bell;
};

// Source of entangled pairs of a link. The parametric source models a heralded generator
// (N ~ Geom(p) attempts of duration attempt_time, fixed Bell-diagonal state); the tabulated
// one replays (delay, Bell coefficients) samples precomputed offline, e.g. with a network
// simulator such as Q2NS or NetSquid.
class EprSource {
public:
    static std::shared_ptr<const EprSource> from_json(const JSON& epr);

    EprDraw sample(timing::Rng& rng) const;

    const timing::RandomDelayPtr& delay() const { return delay_; }
    const noise::BellCoefficients& mean_bell() const { return mean_bell_; }

private:
    timing::RandomDelayPtr delay_;
    noise::BellCoefficients mean_bell_{1.0, 0.0, 0.0, 0.0};

    // Tabulated sources keep their samples, (delay, weight) and Bell coefficients, in the
    // given order, so that a draw returns the delay and the state of the same sample.
    std::shared_ptr<const timing::DiscreteDelay> table_;
    std::vector<std::pair<double, double>> rows_;
    std::vector<double> cumulative_;
    std::vector<noise::BellCoefficients> table_bell_;
};

struct ClassicalLink {
    double latency = 0.0;
    timing::RandomDelayPtr jitter; // nullptr when there is no jitter

    static ClassicalLink from_json(const JSON& classical_link);
};

struct LinkModel {
    std::shared_ptr<const EprSource> epr;
    ClassicalLink classical;
};

// Default durations of the operations whose duration is not given by the noise properties.
struct Durations {
    double gate_1q = 0.0;
    double gate_2q = 0.0;
    double gate_multi = 0.0;
    double measure = 0.0;
    double reset = 0.0;
};

// Timing and noise of one vQPU of the quantum-communications scheme.
struct VQPUModel {
    std::string id;
    std::size_t first_qubit = 0;
    std::size_t num_data_qubits = 0;
    std::size_t num_comm_qubits = 0;

    std::optional<noise::Coherence> data_qubits;
    std::optional<noise::Coherence> comm_qubits;
    Durations durations;
    double feed_forward_latency = 0.0;
    std::shared_ptr<const noise::DeviceNoiseModel> device;

    bool is_comm_qubit(std::size_t local_qubit) const { return local_qubit >= num_data_qubits; }

    // Coherence of a local qubit: the communication-qubit times when given, then the noise
    // properties, then the default data-qubit times.
    const noise::Coherence* coherence(std::size_t local_qubit) const;
};

// Simulated-time and noise model of the vQPUs that share the quantum-communications
// executor. It is assembled from the "comm_model" (and "noise_model") sections of the
// backend of every vQPU, optionally overridden for a single run by a "comm_model" run
// parameter.
struct CommModel {
    bool enabled = false;
    WaitStrategy strategy = WaitStrategy::EXACT;
    double time_bin = 0.0;
    NoiseMode noise_mode = NoiseMode::AUTO;

    std::vector<VQPUModel> vqpus;

    // vQPU-level parameters taken from the backend of every vQPU. The backend JSON is the
    // one each vQPU publishes in qpus.json, so the noise properties are read from disk here.
    static CommModel from_backends(const std::vector<std::pair<std::string, JSON>>& backends);

    // Applies the "comm_model" run parameter (a null JSON keeps the deployment model).
    CommModel with_run_overrides(const JSON& run_comm_model) const;

    const LinkModel& link(const std::string& qpu_a, const std::string& qpu_b) const;

    std::size_t vqpu_of_qubit(std::size_t global_qubit) const;

private:
    // Network-level parameters as given in the backends, to be resolved with the run ones.
    JSON network_json_ = JSON::object();
    std::map<std::string, LinkModel> links_;
    LinkModel default_link_;

    void build_network_(const JSON& network);
};

std::string link_key(const std::string& qpu_a, const std::string& qpu_b);

} // End of sim namespace
} // End of cunqa namespace
