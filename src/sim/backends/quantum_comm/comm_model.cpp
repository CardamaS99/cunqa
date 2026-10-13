#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>
#include <string>

#include "comm_model.hpp"

namespace cunqa {
namespace sim {

namespace {

const std::set<std::string> VQPU_KEYS{
    "data_qubits", "comm_qubits", "durations", "feed_forward_latency"
};
const std::set<std::string> NETWORK_KEYS{
    "strategy", "time_bin", "noise_mode", "epr", "classical_link", "links"
};

void check_keys(const JSON& json, const std::set<std::string>& allowed, const std::string& where)
{
    if (!json.is_object())
        throw std::invalid_argument(where + " must be a JSON object.");
    for (const auto& [key, value] : json.items()) {
        if (!allowed.contains(key)) {
            std::string valid;
            for (const auto& name : allowed)
                valid += (valid.empty() ? "" : ", ") + name;
            throw std::invalid_argument("Unknown key \"" + key + "\" in " + where +
                                        ". Valid keys: " + valid + ".");
        }
    }
}

double non_negative(const JSON& json, const std::string& key, double default_value, const std::string& where)
{
    if (!json.contains(key) || json.at(key).is_null())
        return default_value;
    const double value = json.at(key).get<double>();
    if (!(value >= 0.0) || !std::isfinite(value))
        throw std::invalid_argument(where + "." + key + " must be a finite non-negative number.");
    return value;
}

noise::Coherence coherence_from_json(const JSON& json, const std::string& where)
{
    check_keys(json, {"T1", "T2"}, where);
    const double t1 = json.contains("T1") ? json.at("T1").get<double>() : noise::NO_DECOHERENCE;
    const double t2 = json.contains("T2") ? json.at("T2").get<double>() : noise::NO_DECOHERENCE;
    return noise::Coherence(t1, t2);
}

void apply_vqpu_fields(VQPUModel& vqpu, const JSON& comm_model)
{
    if (comm_model.contains("data_qubits"))
        vqpu.data_qubits = coherence_from_json(comm_model.at("data_qubits"), "comm_model.data_qubits");
    if (comm_model.contains("comm_qubits"))
        vqpu.comm_qubits = coherence_from_json(comm_model.at("comm_qubits"), "comm_model.comm_qubits");
    if (comm_model.contains("durations")) {
        const auto& durations = comm_model.at("durations");
        check_keys(durations, {"gate_1q", "gate_2q", "gate_multi", "measure", "reset"}, "comm_model.durations");
        auto& d = vqpu.durations;
        d.gate_1q = non_negative(durations, "gate_1q", d.gate_1q, "comm_model.durations");
        d.gate_2q = non_negative(durations, "gate_2q", d.gate_2q, "comm_model.durations");
        d.gate_multi = non_negative(durations, "gate_multi", d.gate_2q, "comm_model.durations");
        d.measure = non_negative(durations, "measure", d.measure, "comm_model.durations");
        d.reset = non_negative(durations, "reset", d.reset, "comm_model.durations");
    }
    vqpu.feed_forward_latency = non_negative(
        comm_model, "feed_forward_latency", vqpu.feed_forward_latency, "comm_model");
}

JSON network_part(const JSON& comm_model)
{
    JSON network = JSON::object();
    for (const auto& key : NETWORK_KEYS)
        if (comm_model.contains(key))
            network[key] = comm_model.at(key);
    return network;
}

JSON merged(JSON base, const JSON& patch)
{
    if (patch.is_object())
        base.merge_patch(patch);
    return base;
}

} // End of anonymous namespace


std::string link_key(const std::string& qpu_a, const std::string& qpu_b)
{
    return (qpu_a < qpu_b) ? qpu_a + "|" + qpu_b : qpu_b + "|" + qpu_a;
}


std::shared_ptr<const EprSource> EprSource::from_json(const JSON& epr)
{
    auto source = std::make_shared<EprSource>();
    const std::string kind = epr.value("source", std::string{"parametric"});

    if (kind == "parametric") {
        check_keys(epr, {"source", "fidelity", "bell_coefficients", "success_probability", "attempt_time"},
                   "comm_model.epr");
        if (epr.contains("fidelity") && epr.contains("bell_coefficients"))
            throw std::invalid_argument("comm_model.epr takes either \"fidelity\" or \"bell_coefficients\", not both.");

        source->mean_bell_ = epr.contains("bell_coefficients")
            ? noise::normalized_bell(epr.at("bell_coefficients").get<noise::BellCoefficients>())
            : noise::werner(epr.value("fidelity", 1.0));
        source->delay_ = std::make_shared<timing::GeometricAttemptsDelay>(
            epr.value("success_probability", 1.0),
            non_negative(epr, "attempt_time", 0.0, "comm_model.epr")
        );
    } else if (kind == "tabulated") {
        check_keys(epr, {"source", "samples", "path"}, "comm_model.epr");
        JSON samples;
        if (epr.contains("path")) {
            const auto table = read_file(epr.at("path").get<std::string>());
            samples = table.is_array() ? table : table.at("samples");
        } else {
            samples = epr.at("samples");
        }
        if (!samples.is_array() || samples.empty())
            throw std::invalid_argument("A tabulated EPR source needs a non-empty list of samples.");

        std::vector<std::pair<double, double>> points;
        double total_weight = 0.0;
        noise::BellCoefficients mean_bell{0.0, 0.0, 0.0, 0.0};
        for (const auto& sample : samples) {
            check_keys(sample, {"delay", "fidelity", "bell_coefficients", "weight"}, "comm_model.epr.samples");
            const double delay = non_negative(sample, "delay", 0.0, "comm_model.epr.samples");
            const double weight = sample.value("weight", 1.0);
            const auto bell = sample.contains("bell_coefficients")
                ? noise::normalized_bell(sample.at("bell_coefficients").get<noise::BellCoefficients>())
                : noise::werner(sample.value("fidelity", 1.0));
            points.emplace_back(delay, weight);
            source->table_bell_.push_back(bell);
            total_weight += weight;
            for (std::size_t k = 0; k < 4; ++k)
                mean_bell[k] += weight * bell[k];
        }
        for (auto& coefficient : mean_bell)
            coefficient /= total_weight;

        // The table keeps one entry per sample, in the given order, so that a draw reads
        // the delay and the Bell coefficients of the same sample.
        source->table_ = std::make_shared<const timing::DiscreteDelay>(points);
        source->delay_ = source->table_;
        source->mean_bell_ = mean_bell;
        source->rows_ = std::move(points);
        source->cumulative_.reserve(source->rows_.size());
        double acc = 0.0;
        for (const auto& row : source->rows_) {
            acc += row.second / total_weight;
            source->cumulative_.push_back(acc);
        }
        source->cumulative_.back() = 1.0;
    } else {
        throw std::invalid_argument("Unknown EPR source \"" + kind + "\": use \"parametric\" or \"tabulated\".");
    }
    return source;
}

EprDraw EprSource::sample(timing::Rng& rng) const
{
    if (!table_)
        return {delay_->sample(rng), mean_bell_};

    const double u = std::uniform_real_distribution<double>(0.0, 1.0)(rng);
    const auto it = std::upper_bound(cumulative_.begin(), cumulative_.end(), u);
    const auto index = std::min<std::size_t>(it - cumulative_.begin(), rows_.size() - 1);
    return {rows_[index].first, table_bell_[index]};
}


ClassicalLink ClassicalLink::from_json(const JSON& classical_link)
{
    check_keys(classical_link, {"latency", "jitter"}, "comm_model.classical_link");
    ClassicalLink link;
    link.latency = non_negative(classical_link, "latency", 0.0, "comm_model.classical_link");

    if (classical_link.contains("jitter") && !classical_link.at("jitter").is_null()) {
        const auto& jitter = classical_link.at("jitter");
        const std::string distribution = jitter.value("distribution", std::string{"none"});
        if (distribution == "uniform") {
            check_keys(jitter, {"distribution", "width"}, "comm_model.classical_link.jitter");
            const double width = non_negative(jitter, "width", 0.0, "comm_model.classical_link.jitter");
            if (width > 0.0)
                link.jitter = std::make_shared<timing::UniformDelay>(0.0, width);
        } else if (distribution == "exponential") {
            check_keys(jitter, {"distribution", "mean"}, "comm_model.classical_link.jitter");
            const double mean = non_negative(jitter, "mean", 0.0, "comm_model.classical_link.jitter");
            if (mean > 0.0)
                link.jitter = std::make_shared<timing::ExponentialDelay>(0.0, mean);
        } else if (distribution != "none") {
            throw std::invalid_argument("Unknown jitter distribution \"" + distribution +
                                        "\": use \"none\", \"uniform\" or \"exponential\".");
        }
    }
    return link;
}


const noise::Coherence* VQPUModel::coherence(std::size_t local_qubit) const
{
    if (is_comm_qubit(local_qubit) && comm_qubits)
        return &*comm_qubits;
    if (device) {
        if (const auto* from_properties = device->coherence(local_qubit))
            return from_properties;
    }
    return data_qubits ? &*data_qubits : nullptr;
}


CommModel CommModel::from_backends(const std::vector<std::pair<std::string, JSON>>& backends)
{
    CommModel model;
    std::size_t first_qubit = 0;

    for (const auto& [qpu_id, backend] : backends) {
        VQPUModel vqpu;
        vqpu.id = qpu_id;
        vqpu.first_qubit = first_qubit;
        vqpu.num_data_qubits = backend.at("num_qubits")[0].get<std::size_t>();
        vqpu.num_comm_qubits = backend.at("num_qubits")[1].get<std::size_t>();
        first_qubit += vqpu.num_data_qubits + vqpu.num_comm_qubits;

        if (backend.contains("noise_model") && !backend.at("noise_model").is_null()
                && !backend.at("noise_model").empty()) {
            const auto& noise_model = backend.at("noise_model");
            const auto properties = read_file(noise_model.at("noise_properties_path").get<std::string>());
            vqpu.device = std::make_shared<const noise::DeviceNoiseModel>(
                properties,
                noise_model.value("thermal_relaxation", true),
                noise_model.value("readout_error", true),
                noise_model.value("gate_error", true)
            );
            model.enabled = true;
        }

        if (backend.contains("comm_model") && !backend.at("comm_model").is_null()) {
            const auto& comm_model = backend.at("comm_model");
            std::set<std::string> allowed = VQPU_KEYS;
            allowed.insert(NETWORK_KEYS.begin(), NETWORK_KEYS.end());
            check_keys(comm_model, allowed, "comm_model of vQPU " + qpu_id);
            apply_vqpu_fields(vqpu, comm_model);

            // The network is shared by every vQPU, so their backends must agree on it.
            const JSON network = network_part(comm_model);
            for (const auto& [key, value] : network.items()) {
                if (model.network_json_.contains(key) && model.network_json_.at(key) != value)
                    throw std::invalid_argument(
                        "The vQPUs define different values for comm_model." + key +
                        "; give it as a run parameter to set it for the whole network.");
                model.network_json_[key] = value;
            }
            model.enabled = true;
        }

        model.vqpus.push_back(std::move(vqpu));
    }

    model.build_network_(model.network_json_);
    return model;
}

CommModel CommModel::with_run_overrides(const JSON& run_comm_model) const
{
    if (run_comm_model.is_null())
        return *this;

    std::set<std::string> allowed = VQPU_KEYS;
    allowed.insert(NETWORK_KEYS.begin(), NETWORK_KEYS.end());
    check_keys(run_comm_model, allowed, "the comm_model run parameter");

    CommModel model = *this;
    model.enabled = true;
    for (auto& vqpu : model.vqpus)
        apply_vqpu_fields(vqpu, run_comm_model);

    JSON network = network_json_;
    const JSON run_network = network_part(run_comm_model);
    for (const auto& [key, value] : run_network.items()) {
        if (value.is_object() && network.contains(key) && network.at(key).is_object())
            network[key] = merged(network.at(key), value);
        else
            network[key] = value;
    }
    model.network_json_ = network;
    model.build_network_(network);
    return model;
}

void CommModel::build_network_(const JSON& network)
{
    const std::string strategy_name = network.value("strategy", std::string{"exact"});
    if (strategy_name == "exact")
        strategy = WaitStrategy::EXACT;
    else if (strategy_name == "averaged")
        strategy = WaitStrategy::AVERAGED;
    else
        throw std::invalid_argument("Unknown comm_model.strategy \"" + strategy_name +
                                    "\": use \"exact\" or \"averaged\".");

    time_bin = non_negative(network, "time_bin", 0.0, "comm_model");

    const std::string mode_name = network.value("noise_mode", std::string{"auto"});
    if (mode_name == "auto")
        noise_mode = NoiseMode::AUTO;
    else if (mode_name == "exact")
        noise_mode = NoiseMode::EXACT;
    else if (mode_name == "pauli")
        noise_mode = NoiseMode::PAULI;
    else
        throw std::invalid_argument("Unknown comm_model.noise_mode \"" + mode_name +
                                    "\": use \"auto\", \"exact\" or \"pauli\".");

    const JSON epr = network.value("epr", JSON::object());
    const JSON classical = network.value("classical_link", JSON::object());
    default_link_ = {EprSource::from_json(epr), ClassicalLink::from_json(classical)};

    links_.clear();
    for (const auto& link : network.value("links", JSON::array())) {
        check_keys(link, {"qpus", "circuits", "epr", "classical_link"}, "comm_model.links");
        if (!link.contains("qpus"))
            throw std::invalid_argument(
                "Every entry of comm_model.links needs the \"qpus\" it connects; the circuit ids "
                "given in \"circuits\" are translated by cunqa.qpu.run.");
        const auto qpus = link.at("qpus").get<std::vector<std::string>>();
        if (qpus.size() != 2)
            throw std::invalid_argument("Every entry of comm_model.links connects exactly two vQPUs.");

        links_[link_key(qpus[0], qpus[1])] = {
            EprSource::from_json(merged(epr, link.value("epr", JSON::object()))),
            ClassicalLink::from_json(merged(classical, link.value("classical_link", JSON::object())))
        };
    }
}

const LinkModel& CommModel::link(const std::string& qpu_a, const std::string& qpu_b) const
{
    const auto it = links_.find(link_key(qpu_a, qpu_b));
    return it != links_.end() ? it->second : default_link_;
}

std::size_t CommModel::vqpu_of_qubit(std::size_t global_qubit) const
{
    for (std::size_t i = 0; i < vqpus.size(); ++i) {
        const auto& vqpu = vqpus[i];
        if (global_qubit < vqpu.first_qubit + vqpu.num_data_qubits + vqpu.num_comm_qubits)
            return i;
    }
    throw std::out_of_range("Qubit " + std::to_string(global_qubit) + " does not belong to any vQPU.");
}

} // End of sim namespace
} // End of cunqa namespace
