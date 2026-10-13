#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "qc_engine.hpp"
#include "dynamic_circuit/instruction_type.hpp"
#include "sim/noise/channel_applier.hpp"
#include "sim/noise/channels.hpp"
#include "sim/timing/random_delay.hpp"

#include "comm_managers/qc_manager.hpp"
#include "comm_managers/cc_manager.hpp"

#include "logger.hpp"

namespace {

using namespace cunqa;
using namespace cunqa::sim;

using MeasCounter = std::unordered_map<std::string, std::unordered_map<std::string, std::size_t>>;

enum class BlockType {
    NO_BLOCK,
    BY_ENT,
    BY_CC
};

struct InstructionStream
{
    const std::vector<QuantumTask>& quantum_tasks;
    std::size_t num_quantum_tasks;
    std::vector<bool> finished;
    std::vector<BlockType> blocked;
    std::size_t pointed_quantum_task;
    std::vector<std::size_t> pointed_instruction;
    std::size_t steps_since_progress{0};

    InstructionStream(
        const std::vector<QuantumTask>& quantum_tasks_,
        const std::vector<std::size_t>& zero_clbits_,
        const std::vector<std::size_t>& zero_qubits_
    )
        : quantum_tasks{quantum_tasks_}
        , num_quantum_tasks{quantum_tasks_.size()}
        , finished(quantum_tasks_.size(), false)
        , blocked(quantum_tasks_.size(), BlockType::NO_BLOCK)
        , pointed_quantum_task{0}
        , pointed_instruction(quantum_tasks_.size(), 0)
    {
        for (std::size_t t = 0; t < num_quantum_tasks; t++) {
            for (auto& instr : quantum_tasks[t].circuit->instructions) {
                std::visit([&](auto& payload) {
                    using T = std::decay_t<decltype(payload)>;

                    if constexpr (!std::is_same_v<T, std::monostate>) {
                        if constexpr (requires { payload.qubits; }) {
                            for (auto& q : payload.qubits) {
                                q += zero_qubits_[t];
                            }
                        } else if constexpr (requires { payload.qubit; }) {
                            payload.qubit += zero_qubits_[t];
                        }

                        if constexpr (requires { payload.clbits; }) {
                            for (auto& c : payload.clbits)
                                c += zero_clbits_[t];
                        } else if constexpr (requires { payload.clbit; }) {
                            payload.clbit += zero_clbits_[t];
                        } else if constexpr (std::is_same_v<T, cunqa::Copy>) {
                            for (std::size_t i = 0; i < payload.l_clbits.size(); i++) {
                                payload.l_clbits[i] += zero_clbits_[t];
                                payload.r_clbits[i] += zero_clbits_[t];
                            }
                        }
                    }
                    if constexpr (std::is_same_v<T, cunqa::ClassicalComm>) {
                        if (instr.type == cunqa::InstructionType::SEND)
                            payload.qpus.insert(payload.qpus.begin(), quantum_tasks[t].config.qpu_id);
                        else
                            payload.qpus.push_back(quantum_tasks[t].config.qpu_id);
                        payload.qpus = std::vector<std::string>{std::accumulate(
                                payload.qpus.begin(),
                                payload.qpus.end(),
                                std::string{},
                                [](std::string acc, const std::string& s) {
                                    return acc.empty() ? s : acc + "|" + s;
                                }
                            )};
                    }
                    if constexpr (std::is_same_v<T, cunqa::GenEnt>) {
                        if (payload.tag == "NO_TAG") {
                            std::sort(payload.qpus.begin(), payload.qpus.end());
                            payload.tag = std::accumulate(
                                payload.qpus.begin(),
                                payload.qpus.end(),
                                std::string{},
                                [](std::string acc, const std::string& s) {
                                    return acc.empty() ? s : acc + "|" + s;
                                }
                            );
                        }
                    }
                }, instr.payload);
            }
        }
    }

    inline void reset()
    {
        std::fill(finished.begin(), finished.end(), false);
        std::fill(blocked.begin(), blocked.end(), BlockType::NO_BLOCK);
        pointed_quantum_task = 0;
        steps_since_progress = 0;
        std::fill(pointed_instruction.begin(), pointed_instruction.end(), 0);
    }

    inline void complete_and_advance()
    {
        const auto total = quantum_tasks[pointed_quantum_task].circuit->instructions.size();

        if (blocked[pointed_quantum_task] == BlockType::NO_BLOCK) {
            pointed_instruction[pointed_quantum_task]++;
            steps_since_progress = 0; // made progress
        } else {
            if (++steps_since_progress >= static_cast<std::size_t>(std::count(finished.begin(), finished.end(), false)))
                throw std::runtime_error(
                    "Deadlock: every pending quantum task is blocked "
                    "(circular SEND/RECV or unsatisfiable entanglement).");
        }

        if (pointed_instruction[pointed_quantum_task] >= total)
            finished[pointed_quantum_task] = true;

        if (!all_finished()) {
            do {
                pointed_quantum_task = (pointed_quantum_task + 1) % num_quantum_tasks;
            } while (finished[pointed_quantum_task]);
        }
    }

    inline const cunqa::Instruction& get_current() noexcept
    {
        return quantum_tasks[pointed_quantum_task]
                    .circuit->instructions[pointed_instruction[pointed_quantum_task]];
    }

    inline cunqa::InstructionType skip_next()
    {
        pointed_instruction[pointed_quantum_task]++;
        if (pointed_instruction[pointed_quantum_task] == quantum_tasks[pointed_quantum_task].circuit->instructions.size())
            throw std::runtime_error("You reached the end of the instruction stream skipping.");

        return quantum_tasks[pointed_quantum_task]
                    .circuit->instructions[pointed_instruction[pointed_quantum_task]].type;
    }

    inline void block_current(BlockType block_type) noexcept
    {
        blocked[pointed_quantum_task] = block_type;
    }

    inline void unblock_current() noexcept
    {
        blocked[pointed_quantum_task] = BlockType::NO_BLOCK;
    }

    inline bool is_current_blocked() const noexcept
    {
        return (blocked[pointed_quantum_task] != BlockType::NO_BLOCK);
    }

    inline bool all_finished() const
    {
        return std::all_of(finished.begin(), finished.end(), [](bool b) { return b; });
    }
};

// Shot with ideal, instantaneous entanglement: the behaviour without a communication model.
void execute_shot_(
    std::shared_ptr<cunqa::sim::Simulator> simulator,
    InstructionStream& stream,
    const std::vector<std::vector<std::size_t>> communication_qubits_
)
{
    cunqa::QuantumCommManager qc_manager{communication_qubits_};
    cunqa::ClassicalCommManager cc_manager;
    while(!stream.all_finished()) {
        const auto& instr = stream.get_current();

        std::visit([&](const auto& payload) {
            using T = std::decay_t<decltype(payload)>;
            const auto& type = instr.type;

            if constexpr (std::is_same_v<T, cunqa::ClassicalIf>) {
                if (type == cunqa::InstructionType::CIF){
                    // Operates on the values provided, with the specified operation.
                    //If condition is 0, the operation negates the second operand (check cif_ops)
                    bool result = std::accumulate(
                        payload.clbits.begin() + 1, payload.clbits.end(),
                        simulator->creg[payload.clbits[0]], // Starting value
                        [&](bool acc, std::size_t clbit) {
                            return cunqa::cif_ops[payload.operation](acc, simulator->creg[clbit]);
                        });

                    // If the condition is not met, skip every gate until the
                    // matching ENDCIF arrives.
                    // NOTE: nesting of CIF blocks is NOT supported. This stops at
                    // the first ENDCIF, so a nested block would be mis-handled.
                    // Circuits must not contain nested CIF blocks.
                    if (result != payload.condition){
                        cunqa::InstructionType type;
                        do {
                            type = stream.skip_next();
                        } while (type != cunqa::InstructionType::ENDCIF);
                    }
                }
            } else if constexpr (std::is_same_v<T, cunqa::ClassicalComm>) {
                if (type == cunqa::InstructionType::SEND) {
                    for (std::size_t i=0; i<payload.clbits.size(); i++)
                        cc_manager.send(simulator->creg[payload.clbits[i]], payload.qpus[0]);
                } else {
                    bool value_i;
                    for (std::size_t i=0; i<payload.clbits.size(); i++) {
                        if (cc_manager.recv(payload.qpus[0], value_i)) {
                            simulator->creg[payload.clbits[i]] = value_i;
                            simulator->save_clbit[payload.clbits[i]] = false;
                            if (stream.is_current_blocked())
                                stream.unblock_current();
                        } else
                            stream.block_current(BlockType::BY_CC);
                    }
                }
            } else if constexpr (std::is_same_v<T, cunqa::GenEnt>) {
                qc_manager.get_ghz(simulator, payload);
            } else if constexpr (std::is_same_v<T, std::monostate>) {
                throw std::runtime_error("Empty circuit received.");
            }
            else {
                if constexpr (requires { payload.qubits; }) {
                    for (auto& q : payload.qubits) {
                        if (qc_manager.is_pending.contains(q) && qc_manager.is_pending.at(q)) {
                            stream.block_current(BlockType::BY_ENT);
                            return;
                        }
                    }
                } else if constexpr (requires { payload.qubit; }) {
                    if (qc_manager.is_pending.contains(payload.qubit) && qc_manager.is_pending.at(payload.qubit)) {
                        stream.block_current(BlockType::BY_ENT);
                        return;
                    }
                }
                if (stream.is_current_blocked())
                    stream.unblock_current();
                simulator->apply_gate(type, payload);
            }

        }, instr.payload);
        stream.complete_and_advance();
    }
}

void update_measures_(
    std::shared_ptr<cunqa::sim::Simulator> simulator,
    const std::vector<QuantumTask>& quantum_tasks,
    const std::vector<std::size_t>& zero_clbits,
    MeasCounter& meas_counter
)
{
    for (std::size_t t = 0; t < quantum_tasks.size(); ++t) {
        std::string result;

        const auto begin = zero_clbits[t];
        const auto end = begin + quantum_tasks[t].config.num_clbits;

        for (std::size_t c = begin; c < end; ++c) {
            auto it = simulator->creg.find(c);
            if (it != simulator->creg.end()) {
                if (simulator->save_clbit.at(c))
                    result = (it->second ? '1' : '0') + result;
            }
        }
        if (result != "")
            meas_counter[quantum_tasks[t].config.qpu_id][result]++;
    }
}


// ---------------------------------------------------------------------------------------
// Simulated time
// ---------------------------------------------------------------------------------------

// Time at which a qubit or a classical bit becomes available. With the averaged strategy
// the timeline runs on the mean of every random delay, and `delays` lists the random delays
// (ids of TimedShot::delays_) whose value this time depends on.
struct STime {
    double t = 0.0;
    std::vector<std::uint32_t> delays;
};

inline const STime& later(const STime& a, const STime& b)
{
    return (b.t > a.t) ? b : a;
}

inline STime after(STime time, double duration)
{
    time.t += duration;
    return time;
}

struct TaskInfo {
    std::size_t vqpu;
    std::string qpu_id;
    std::size_t first_qubit;
    std::size_t num_data_qubits;
    std::size_t num_qubits;
    std::size_t first_clbit;
    std::size_t num_clbits;
};

struct RunningStats {
    std::size_t count = 0;
    double mean = 0.0;
    double m2 = 0.0;
    double min = std::numeric_limits<double>::infinity();
    double max = -std::numeric_limits<double>::infinity();

    void add(double x)
    {
        ++count;
        const double delta = x - mean;
        mean += delta / count;
        m2 += delta * (x - mean);
        min = std::min(min, x);
        max = std::max(max, x);
    }

    JSON to_json() const
    {
        return {
            {"mean", mean},
            {"std", count > 1 ? std::sqrt(m2 / (count - 1)) : 0.0},
            {"min", count ? min : 0.0},
            {"max", count ? max : 0.0}
        };
    }
};

struct Message {
    bool value;
    STime arrival;
};

struct EntanglementRequest {
    std::size_t task;
    std::size_t qubit;
    STime time;
};

template <typename Payload>
std::vector<std::size_t> qubits_of(const Payload& payload)
{
    if constexpr (requires { payload.qubits; })
        return std::vector<std::size_t>(payload.qubits.begin(), payload.qubits.end());
    else if constexpr (requires { payload.qubit; })
        return {payload.qubit};
    else
        return {};
}

std::pair<std::string, std::string> channel_endpoints(const std::string& channel)
{
    const auto separator = channel.find('|');
    if (separator == std::string::npos)
        throw std::runtime_error("Malformed classical channel " + channel + ".");
    return {channel.substr(0, separator), channel.substr(separator + 1)};
}

// One shot of the quantum-communications scheme with simulated time.
class TimedShot {
public:
    TimedShot(
        Simulator& simulator,
        const CommModel& model,
        const std::vector<TaskInfo>& tasks,
        const std::vector<const noise::Coherence*>& coherence,
        noise::ChannelApplier& applier,
        timing::Rng& rng
    )
        : sim_{simulator}
        , model_{model}
        , tasks_{tasks}
        , coherence_{coherence}
        , applier_{applier}
        , rng_{rng}
        , qubit_free_(coherence.size())
        , qubit_noise_(coherence.size())
        , clbit_time_(std::accumulate(tasks.begin(), tasks.end(), std::size_t{0},
              [](std::size_t acc, const TaskInfo& task) { return acc + task.num_clbits; }))
        , condition_(tasks.size())
        , requests_(tasks.size(), 0)
        , generation_time_(tasks.size(), 0.0)
    { }

    void run(InstructionStream& stream)
    {
        while (!stream.all_finished()) {
            const auto& instr = stream.get_current();
            const auto task = stream.pointed_quantum_task;

            std::visit([&](const auto& payload) {
                using T = std::decay_t<decltype(payload)>;
                const auto& type = instr.type;

                if constexpr (std::is_same_v<T, ClassicalIf>) {
                    if (type == InstructionType::CIF)
                        conditional_(stream, task, payload);
                    else
                        condition_[task].reset();
                } else if constexpr (std::is_same_v<T, ClassicalComm>) {
                    if (type == InstructionType::SEND)
                        send_(task, payload);
                    else
                        recv_(stream, task, payload);
                } else if constexpr (std::is_same_v<T, GenEnt>) {
                    request_entanglement_(task, payload);
                } else if constexpr (std::is_same_v<T, std::monostate>) {
                    throw std::runtime_error("Empty circuit received.");
                } else {
                    const auto qubits = qubits_of(payload);
                    for (const auto q : qubits) {
                        if (pending_.contains(q)) {
                            stream.block_current(BlockType::BY_ENT);
                            return;
                        }
                    }
                    if (stream.is_current_blocked())
                        stream.unblock_current();

                    if constexpr (std::is_same_v<T, Measure>)
                        measure_(task, payload);
                    else if constexpr (std::is_same_v<T, Reset>)
                        reset_(task, type, payload, qubits);
                    else if constexpr (std::is_same_v<T, Copy>)
                        copy_(task, type, payload);
                    else
                        gate_(task, type, payload, qubits);
                }
            }, instr.payload);
            stream.complete_and_advance();
        }
    }

    // Simulated time at which the task finishes: its last qubit or classical bit.
    double completion_time(std::size_t task) const
    {
        const auto& info = tasks_[task];
        double end = 0.0;
        for (std::size_t q = info.first_qubit; q < info.first_qubit + info.num_qubits; ++q)
            end = std::max(end, qubit_free_[q].t);
        for (std::size_t c = info.first_clbit; c < info.first_clbit + info.num_clbits; ++c)
            end = std::max(end, clbit_time_[c].t);
        return end;
    }

    std::size_t requests(std::size_t task) const { return requests_[task]; }
    double generation_time(std::size_t task) const { return generation_time_[task]; }

private:
    Simulator& sim_;
    const CommModel& model_;
    const std::vector<TaskInfo>& tasks_;
    const std::vector<const noise::Coherence*>& coherence_;
    noise::ChannelApplier& applier_;
    timing::Rng& rng_;

    std::vector<STime> qubit_free_;   // when the qubit can start its next operation
    std::vector<STime> qubit_noise_;  // up to when its noise has been applied
    std::vector<STime> clbit_time_;   // when the value of the bit is known
    std::vector<std::optional<STime>> condition_; // condition of the open cif block, per task

    std::unordered_map<std::string, std::deque<Message>> messages_;
    std::unordered_map<std::string, std::vector<EntanglementRequest>> waiting_;
    std::unordered_set<std::size_t> pending_;

    std::vector<timing::RandomDelayPtr> delays_; // random delays of the shot (averaged strategy)

    std::vector<std::size_t> requests_;
    std::vector<double> generation_time_;

    const VQPUModel& vqpu_(std::size_t task) const { return model_.vqpus[tasks_[task].vqpu]; }

    STime start_time_(std::size_t task) const
    {
        return condition_[task] ? *condition_[task] : STime{};
    }

    // Adds a random delay to a time: sampled with the exact strategy, as its mean (and
    // remembered) with the averaged one.
    STime delayed_(STime time, const timing::RandomDelayPtr& delay)
    {
        if (!delay)
            return time;
        if (model_.strategy == WaitStrategy::EXACT)
            return after(std::move(time), timing::bin_delay(delay->sample(rng_), model_.time_bin));
        if (delay->is_deterministic())
            return after(std::move(time), delay->mean());

        const auto id = static_cast<std::uint32_t>(delays_.size());
        delays_.push_back(delay);
        time.t += delay->mean();
        time.delays.insert(std::upper_bound(time.delays.begin(), time.delays.end(), id), id);
        return time;
    }

    // Relaxation of a qubit idling from `from` to `to`.
    noise::Relaxation idle_relaxation_(const noise::Coherence& coherence, const STime& from, const STime& to) const
    {
        const double length = std::max(0.0, to.t - from.t);
        if (from.delays.empty() && to.delays.empty())
            return noise::relaxation(coherence, length);

        // Random delays that stretch the interval (only its end depends on them) and that
        // shrink it (only its start does). Delays shared by both ends cancel out.
        std::vector<std::uint32_t> stretch, shrink;
        std::set_difference(to.delays.begin(), to.delays.end(), from.delays.begin(), from.delays.end(),
                            std::back_inserter(stretch));
        std::set_difference(from.delays.begin(), from.delays.end(), to.delays.begin(), to.delays.end(),
                            std::back_inserter(shrink));

        // Deterministic part of the interval, with every random delay removed.
        double deterministic = length;
        for (const auto id : stretch)
            deterministic -= delays_[id]->mean();
        for (const auto id : shrink)
            deterministic += delays_[id]->mean();

        auto by_mean = [this](std::uint32_t a, std::uint32_t b) { return delays_[a]->mean() < delays_[b]->mean(); };

        auto decay = [&](double rate) {
            if (rate == 0.0)
                return 1.0;
            if (stretch.empty() && shrink.empty())
                return std::exp(-rate * std::max(0.0, deterministic));

            if (stretch.empty()) {
                // Interval max(0, d - sum Y): the largest delay is kept random, the rest at
                // their mean.
                const auto main = *std::max_element(shrink.begin(), shrink.end(), by_mean);
                double offset = deterministic;
                for (const auto id : shrink)
                    if (id != main)
                        offset -= delays_[id]->mean();
                return delays_[main]->deficit_laplace(rate, offset);
            }

            // Interval max(0, d + sum X - sum Y), with the shrinking delays at their mean.
            double offset = deterministic;
            for (const auto id : shrink)
                offset -= delays_[id]->mean();
            if (offset >= 0.0) {
                double value = std::exp(-rate * offset);
                for (const auto id : stretch)
                    value *= delays_[id]->laplace(rate);
                return value;
            }
            const auto main = *std::max_element(stretch.begin(), stretch.end(), by_mean);
            for (const auto id : stretch)
                if (id != main)
                    offset += delays_[id]->mean();
            return delays_[main]->excess_laplace(rate, -offset);
        };

        return {decay(coherence.rate1()), decay(coherence.rate2())};
    }

    void idle_(std::size_t qubit, const STime& until)
    {
        const auto* coherence = coherence_[qubit];
        if (coherence && coherence->decoheres())
            applier_.relaxation(qubit, idle_relaxation_(*coherence, qubit_noise_[qubit], until));
        qubit_noise_[qubit] = until;
    }

    double default_duration_(const VQPUModel& vqpu, std::size_t num_qubits) const
    {
        if (num_qubits <= 1)
            return vqpu.durations.gate_1q;
        return (num_qubits == 2) ? vqpu.durations.gate_2q : vqpu.durations.gate_multi;
    }

    template <typename Payload>
    void gate_(std::size_t task, InstructionType type, const Payload& payload, const std::vector<std::size_t>& qubits)
    {
        STime start = start_time_(task);
        for (const auto q : qubits)
            start = later(start, qubit_free_[q]);
        for (const auto q : qubits)
            idle_(q, start);

        sim_.apply_gate(type, payload);

        const auto& vqpu = vqpu_(task);
        const noise::GateNoise* gate_noise = nullptr;
        if (vqpu.device) {
            std::vector<std::size_t> local(qubits);
            for (auto& q : local)
                q -= vqpu.first_qubit;
            gate_noise = vqpu.device->gate(std::string(instruction_type_name(type)), local);
        }

        const double duration = gate_noise ? gate_noise->duration : default_duration_(vqpu, qubits.size());
        if (gate_noise)
            applier_.depolarizing(qubits, gate_noise->depolarizing);

        const STime end = after(start, duration);
        for (const auto q : qubits) {
            qubit_free_[q] = end;
            if (gate_noise && gate_noise->relaxation) {
                // The gate error already includes the relaxation during the gate.
                if (const auto* coherence = coherence_[q])
                    applier_.relaxation(q, noise::relaxation(*coherence, duration));
                qubit_noise_[q] = end;
            }
            // Otherwise the qubit relaxes during the gate as if idle, when next used.
        }
    }

    void measure_(std::size_t task, const Measure& payload)
    {
        const auto q = payload.qubit;
        const STime start = later(start_time_(task), qubit_free_[q]);
        idle_(q, start);

        sim_.apply_gate(InstructionType::MEASURE, payload);

        const auto& vqpu = vqpu_(task);
        const auto local = q - vqpu.first_qubit;
        double duration = vqpu.durations.measure;
        if (vqpu.device) {
            if (applier_.flip(vqpu.device->readout_error(local)))
                sim_.creg[payload.clbit] = !sim_.creg[payload.clbit];
            duration = vqpu.device->readout_duration(local).value_or(duration);
        }

        qubit_free_[q] = after(start, duration);
        clbit_time_[payload.clbit] = qubit_free_[q];
    }

    void reset_(std::size_t task, InstructionType type, const Reset& payload, const std::vector<std::size_t>& qubits)
    {
        STime start = start_time_(task);
        for (const auto q : qubits)
            start = later(start, qubit_free_[q]);
        for (const auto q : qubits)
            idle_(q, start);

        sim_.apply_gate(type, payload);

        const STime end = after(start, vqpu_(task).durations.reset);
        for (const auto q : qubits)
            qubit_free_[q] = qubit_noise_[q] = end;
    }

    void copy_(std::size_t task, InstructionType type, const Copy& payload)
    {
        sim_.apply_gate(type, payload);
        STime time = start_time_(task);
        for (const auto c : payload.r_clbits)
            time = later(time, clbit_time_[c]);
        for (const auto c : payload.l_clbits)
            clbit_time_[c] = time;
    }

    void conditional_(InstructionStream& stream, std::size_t task, const ClassicalIf& payload)
    {
        const bool result = std::accumulate(
            payload.clbits.begin() + 1, payload.clbits.end(),
            sim_.creg[payload.clbits[0]],
            [&](bool acc, std::size_t clbit) {
                return cif_ops[payload.operation](acc, sim_.creg[clbit]);
            });

        // The controller decides once the bits are known and processed.
        STime decided = start_time_(task);
        for (const auto c : payload.clbits)
            decided = later(decided, clbit_time_[c]);
        decided = after(decided, vqpu_(task).feed_forward_latency);

        if (result != payload.condition) {
            // Skipped operations do not touch their qubits, but those cannot be used before
            // the controller knows that they are skipped. Nested blocks are not supported.
            InstructionType skipped;
            do {
                skipped = stream.skip_next();
                if (skipped != InstructionType::ENDCIF) {
                    std::visit([&](const auto& skipped_payload) {
                        for (const auto q : qubits_of(skipped_payload))
                            qubit_free_[q] = later(qubit_free_[q], decided);
                    }, stream.get_current().payload);
                }
            } while (skipped != InstructionType::ENDCIF);
        } else {
            condition_[task] = decided;
        }
    }

    void send_(std::size_t task, const ClassicalComm& payload)
    {
        STime departure = start_time_(task);
        for (const auto c : payload.clbits)
            departure = later(departure, clbit_time_[c]);

        const auto& channel = payload.qpus[0];
        const auto [sender, receiver] = channel_endpoints(channel);
        const auto& link = model_.link(sender, receiver).classical;
        const STime arrival = delayed_(after(departure, link.latency), link.jitter);

        auto& queue = messages_[channel];
        for (const auto c : payload.clbits)
            queue.push_back({sim_.creg[c], arrival});
    }

    void recv_(InstructionStream& stream, std::size_t task, const ClassicalComm& payload)
    {
        (void) task;
        auto& queue = messages_[payload.qpus[0]];
        if (queue.size() < payload.clbits.size()) {
            stream.block_current(BlockType::BY_CC);
            return;
        }
        for (const auto c : payload.clbits) {
            sim_.creg[c] = queue.front().value;
            sim_.save_clbit[c] = false;
            clbit_time_[c] = queue.front().arrival;
            queue.pop_front();
        }
        if (stream.is_current_blocked())
            stream.unblock_current();
    }

    void request_entanglement_(std::size_t task, const GenEnt& payload)
    {
        // The pair is requested on demand: once the communication qubit is free and the
        // vQPU has finished the work issued before the request (its data qubits), so the
        // consumer waits for the whole generation. Requesting earlier in the circuit
        // prefetches the pair, which then waits in memory instead.
        const auto& info = tasks_[task];
        STime requested = later(start_time_(task), qubit_free_[payload.qubit]);
        for (std::size_t q = info.first_qubit; q < info.first_qubit + info.num_data_qubits; ++q)
            requested = later(requested, qubit_free_[q]);

        auto& waiting = waiting_[payload.tag];
        waiting.push_back({task, payload.qubit, requested});
        if (waiting.size() == payload.qpus.size()) {
            deliver_(waiting);
            waiting.clear();
        } else {
            pending_.insert(payload.qubit);
        }
    }

    // Generates the entangled state requested by every participant: a Bell pair for two of
    // them, a GHZ state assembled from a star of links rooted at the first one otherwise.
    void deliver_(std::vector<EntanglementRequest> participants)
    {
        std::sort(participants.begin(), participants.end(),
            [this](const EntanglementRequest& a, const EntanglementRequest& b) {
                return tasks_[a.task].qpu_id < tasks_[b.task].qpu_id;
            });

        const auto& root = participants.front();
        const auto& root_id = tasks_[root.task].qpu_id;
        STime requested = root.time;
        for (const auto& participant : participants)
            requested = later(requested, participant.time);

        const std::size_t n = participants.size();
        std::vector<noise::BellCoefficients> bell(n, {1.0, 0.0, 0.0, 0.0});
        std::vector<double> link_ready(n, requested.t);
        STime ready;

        if (model_.strategy == WaitStrategy::EXACT) {
            double latest = requested.t;
            for (std::size_t k = 1; k < n; ++k) {
                const auto& link = model_.link(root_id, tasks_[participants[k].task].qpu_id);
                const auto draw = link.epr->sample(rng_);
                link_ready[k] = requested.t + timing::bin_delay(draw.delay, model_.time_bin);
                bell[k] = draw.bell;
                latest = std::max(latest, link_ready[k]);
            }
            ready = requested;
            ready.t = latest;
        } else {
            std::vector<timing::RandomDelayPtr> generation;
            for (std::size_t k = 1; k < n; ++k) {
                const auto& link = model_.link(root_id, tasks_[participants[k].task].qpu_id);
                generation.push_back(link.epr->delay());
                bell[k] = link.epr->mean_bell();
            }
            if (generation.empty())
                ready = requested;
            else
                ready = delayed_(requested, (generation.size() == 1)
                    ? generation.front()
                    : timing::DiscreteDelay::max_of(generation));
        }

        // Heralded pair: the memories are re-initialized and loaded with the state.
        if (sim_.supports_reset()) {
            for (const auto& participant : participants)
                applier_.reset(participant.qubit);
        }
        sim_.apply_gate(InstructionType::H, OneQubitNoParam{root.qubit});
        for (std::size_t k = 1; k < n; ++k)
            sim_.apply_gate(InstructionType::CX, TwoQubitNoParam{{root.qubit, participants[k].qubit}});
        for (std::size_t k = 1; k < n; ++k)
            applier_.pauli(participants[k].qubit, noise::bell_to_pauli(bell[k]));

        // A GHZ state is complete when its slowest link heralds; the pairs that arrived
        // earlier have been decohering meanwhile.
        if (n > 2 && model_.strategy == WaitStrategy::EXACT) {
            const double first = *std::min_element(link_ready.begin() + 1, link_ready.end());
            for (std::size_t k = 0; k < n; ++k) {
                const auto* coherence = coherence_[participants[k].qubit];
                const double waited = ready.t - (k == 0 ? first : link_ready[k]);
                if (coherence && waited > 0.0)
                    applier_.relaxation(participants[k].qubit, noise::relaxation(*coherence, waited));
            }
        }

        for (const auto& participant : participants) {
            qubit_free_[participant.qubit] = ready;
            qubit_noise_[participant.qubit] = ready;
            pending_.erase(participant.qubit);
            requests_[participant.task]++;
            generation_time_[participant.task] += ready.t - requested.t;
        }
    }
};

noise::ChannelMode resolve_channel_mode(const CommModel& model, const Simulator& simulator)
{
    switch (model.noise_mode) {
        case NoiseMode::EXACT:
            return noise::ChannelMode::EXACT;
        case NoiseMode::PAULI:
            return noise::ChannelMode::PAULI;
        default:
            return simulator.supports_kraus() ? noise::ChannelMode::EXACT : noise::ChannelMode::PAULI;
    }
}

JSON run_comm_model(const std::vector<QuantumTask>& quantum_tasks)
{
    JSON run_model;
    for (const auto& task : quantum_tasks) {
        const auto& specifics = task.config.simulator_specifics;
        const JSON model = specifics.contains("comm_model") ? specifics.at("comm_model") : JSON();
        if (&task == &quantum_tasks.front())
            run_model = model;
        else if (model != run_model)
            throw std::runtime_error("Every quantum task with quantum communications has to "
                                     "be run with the same comm_model.");
    }
    return run_model;
}

} // End of anonymous namespace


namespace cunqa {
namespace sim {

QCEngine::QCEngine(std::shared_ptr<Simulator> simulator, CommModel model)
    : simulator_{std::move(simulator)}
    , model_{std::move(model)}
{ }

std::vector<JSON> QCEngine::run(std::vector<QuantumTask>& quantum_tasks)
{
    if (quantum_tasks.empty())
        return {};

    const auto model = model_.with_run_overrides(run_comm_model(quantum_tasks));

    // Layout of every task in the shared simulator.
    std::vector<TaskInfo> tasks;
    std::vector<std::size_t> zero_qubits, zero_clbits;
    std::size_t accumulated_clbits = 0;
    for (const auto& quantum_task : quantum_tasks) {
        const auto it = std::find_if(model.vqpus.begin(), model.vqpus.end(),
            [&](const VQPUModel& vqpu) { return vqpu.id == quantum_task.config.qpu_id; });
        if (it == model.vqpus.end())
            throw std::runtime_error("Quantum task for unknown vQPU " + quantum_task.config.qpu_id + ".");

        tasks.push_back({
            static_cast<std::size_t>(it - model.vqpus.begin()),
            it->id,
            it->first_qubit,
            it->num_data_qubits,
            it->num_data_qubits + it->num_comm_qubits,
            accumulated_clbits,
            static_cast<std::size_t>(quantum_task.config.num_clbits)
        });
        zero_qubits.push_back(it->first_qubit);
        zero_clbits.push_back(accumulated_clbits);
        accumulated_clbits += quantum_task.config.num_clbits;
    }

    std::vector<std::vector<std::size_t>> communication_qubits;
    for (const auto& vqpu : model.vqpus) {
        std::vector<std::size_t> comm_qubits(vqpu.num_comm_qubits);
        std::iota(comm_qubits.begin(), comm_qubits.end(), vqpu.first_qubit + vqpu.num_data_qubits);
        communication_qubits.push_back(std::move(comm_qubits));
    }

    InstructionStream stream(quantum_tasks, zero_clbits, zero_qubits);
    MeasCounter meas_counter;

    // Seeded runs give every shot its own seed, drawn from the run seed: re-seeding the
    // simulator with the same value would repeat the same outcomes in every shot.
    const int run_seed = simulator_->config.seed;
    timing::Rng rng(run_seed != NO_SEED ? static_cast<std::uint64_t>(run_seed) : std::random_device{}());
    auto next_shot_seed = [&]() {
        if (run_seed != NO_SEED)
            simulator_->config.seed = static_cast<int>(rng() % 2147483647ULL);
    };

    std::vector<RunningStats> completion(tasks.size());
    std::vector<std::size_t> requests(tasks.size(), 0);
    std::vector<double> generation_time(tasks.size(), 0.0);

    const auto shots = static_cast<std::size_t>(simulator_->config.shots);
    auto start = std::chrono::high_resolution_clock::now();
    if (!model.enabled) {
        for (std::size_t i = 0; i < shots; i++) {
            next_shot_seed();
            simulator_->initialize();

            execute_shot_(simulator_, stream, communication_qubits);
            stream.reset();

            update_measures_(simulator_, quantum_tasks, zero_clbits, meas_counter);
            simulator_->clear();
        }
    } else {
        // Coherence of every qubit of the shared simulator.
        const auto& last = model.vqpus.back();
        std::vector<const noise::Coherence*> coherence(
            last.first_qubit + last.num_data_qubits + last.num_comm_qubits, nullptr);
        double shortest_t2 = noise::NO_DECOHERENCE;
        for (const auto& vqpu : model.vqpus) {
            for (std::size_t q = 0; q < vqpu.num_data_qubits + vqpu.num_comm_qubits; ++q) {
                coherence[vqpu.first_qubit + q] = vqpu.coherence(q);
                if (coherence[vqpu.first_qubit + q])
                    shortest_t2 = std::min(shortest_t2, coherence[vqpu.first_qubit + q]->T2);
            }
        }
        if (model.time_bin > 0.01 * shortest_t2)
            LOGGER_WARN("comm_model.time_bin ({} s) is not much smaller than the shortest T2 ({} s): "
                        "binning the waiting times will bias the idle noise.", model.time_bin, shortest_t2);

        noise::ChannelApplier applier(*simulator_, rng, resolve_channel_mode(model, *simulator_));
        if (!simulator_->supports_reset())
            LOGGER_WARN("{} cannot reset qubits: communication qubits are assumed to be in |0> "
                        "when an entangled pair is delivered.", simulator_->get_name());

        for (std::size_t i = 0; i < shots; i++) {
            next_shot_seed();
            simulator_->initialize();

            TimedShot shot(*simulator_, model, tasks, coherence, applier, rng);
            shot.run(stream);
            stream.reset();

            update_measures_(simulator_, quantum_tasks, zero_clbits, meas_counter);
            for (std::size_t t = 0; t < tasks.size(); ++t) {
                completion[t].add(shot.completion_time(t));
                requests[t] += shot.requests(t);
                generation_time[t] += shot.generation_time(t);
            }
            simulator_->clear();
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;
    simulator_->config.seed = run_seed;

    std::vector<JSON> results;
    for (std::size_t t = 0; t < quantum_tasks.size(); ++t) {
        JSON result = {
            {"counts", meas_counter[quantum_tasks[t].config.qpu_id]},
            {"time_taken", duration.count()}
        };
        if (model.enabled) {
            result["simulated_time"] = completion[t].to_json();
            result["comm_stats"] = {
                {"entanglement_requests_per_shot", shots ? static_cast<double>(requests[t]) / shots : 0.0},
                {"mean_generation_time", requests[t] ? generation_time[t] / requests[t] : 0.0}
            };
        }
        results.push_back(std::move(result));
    }
    return results;
}

} // End of sim namespace
} // End of cunqa namespace
