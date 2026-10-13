#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>

#include <gtest/gtest.h>

#include "comm_model.hpp"

using namespace cunqa;
using namespace cunqa::sim;

namespace {

JSON backend(int data, int comm, JSON comm_model = nullptr)
{
    JSON b = {{"name", "QCBackend"}, {"num_qubits", {data, comm}}};
    if (!comm_model.is_null())
        b["comm_model"] = comm_model;
    return b;
}

} // End of anonymous namespace

TEST(CommModel, DisabledWithoutModelSections)
{
    const auto model = CommModel::from_backends({{"A", backend(2, 1)}, {"B", backend(3, 2)}});
    EXPECT_FALSE(model.enabled);
    ASSERT_EQ(model.vqpus.size(), 2u);
    EXPECT_EQ(model.vqpus[0].first_qubit, 0u);
    EXPECT_EQ(model.vqpus[1].first_qubit, 3u);
    EXPECT_EQ(model.vqpus[1].num_comm_qubits, 2u);
    EXPECT_EQ(model.vqpu_of_qubit(4), 1u);

    // Ideal links by default: certain, instantaneous and perfect pairs.
    const auto& link = model.link("A", "B");
    EXPECT_DOUBLE_EQ(link.epr->delay()->mean(), 0.0);
    EXPECT_DOUBLE_EQ(link.epr->mean_bell()[0], 1.0);
    EXPECT_DOUBLE_EQ(link.classical.latency, 0.0);
    EXPECT_EQ(link.classical.jitter, nullptr);
}

TEST(CommModel, BackendSectionsEnableTheModel)
{
    const JSON comm = {
        {"comm_qubits", {{"T1", 1e-3}, {"T2", 5e-4}}},
        {"durations", {{"gate_2q", 3e-7}, {"measure", 1e-6}}},
        {"feed_forward_latency", 2e-7},
        {"epr", {{"fidelity", 0.9}, {"success_probability", 0.1}, {"attempt_time", 1e-6}}},
        {"classical_link", {{"latency", 1e-6}, {"jitter", {{"distribution", "uniform"}, {"width", 2e-7}}}}}
    };
    const auto model = CommModel::from_backends({{"A", backend(1, 1, comm)}, {"B", backend(1, 1, comm)}});
    EXPECT_TRUE(model.enabled);
    EXPECT_EQ(model.strategy, WaitStrategy::EXACT);
    EXPECT_EQ(model.noise_mode, NoiseMode::AUTO);

    const auto& vqpu = model.vqpus[0];
    EXPECT_DOUBLE_EQ(vqpu.durations.gate_2q, 3e-7);
    EXPECT_DOUBLE_EQ(vqpu.durations.measure, 1e-6);
    EXPECT_DOUBLE_EQ(vqpu.durations.gate_1q, 0.0);
    EXPECT_DOUBLE_EQ(vqpu.feed_forward_latency, 2e-7);

    const auto& link = model.link("B", "A");
    EXPECT_NEAR(link.epr->delay()->mean(), 1e-5, 1e-18);
    EXPECT_NEAR(link.epr->mean_bell()[0], 0.9, 1e-12);
    EXPECT_DOUBLE_EQ(link.classical.latency, 1e-6);
    ASSERT_NE(link.classical.jitter, nullptr);
    EXPECT_NEAR(link.classical.jitter->mean(), 1e-7, 1e-18);
}

TEST(CommModel, CoherencePrecedence)
{
    const JSON comm = {
        {"data_qubits", {{"T1", 1e-4}, {"T2", 1e-4}}},
        {"comm_qubits", {{"T1", 1e-3}, {"T2", 2e-3}}}
    };
    const auto model = CommModel::from_backends({{"A", backend(2, 1, comm)}});
    const auto& vqpu = model.vqpus[0];
    ASSERT_NE(vqpu.coherence(0), nullptr);
    EXPECT_DOUBLE_EQ(vqpu.coherence(0)->T1, 1e-4);
    ASSERT_NE(vqpu.coherence(2), nullptr);
    EXPECT_DOUBLE_EQ(vqpu.coherence(2)->T2, 2e-3);

    // Communication qubits fall back to the data-qubit times.
    const auto data_only = CommModel::from_backends({{"A", backend(2, 1, {{"data_qubits", {{"T2", 3e-5}}}})}});
    ASSERT_NE(data_only.vqpus[0].coherence(2), nullptr);
    EXPECT_DOUBLE_EQ(data_only.vqpus[0].coherence(2)->T2, 3e-5);
    EXPECT_TRUE(std::isinf(data_only.vqpus[0].coherence(2)->T1));
}

TEST(CommModel, RunOverridesMergeIntoTheDeployment)
{
    const JSON comm = {{"epr", {{"fidelity", 0.9}, {"success_probability", 0.5}, {"attempt_time", 1e-6}}}};
    const auto deployed = CommModel::from_backends({{"A", backend(1, 1, comm)}, {"B", backend(1, 1, comm)}});

    const JSON run = {
        {"strategy", "averaged"},
        {"noise_mode", "pauli"},
        {"epr", {{"success_probability", 0.1}}},
        {"durations", {{"gate_1q", 5e-8}}}
    };
    const auto model = deployed.with_run_overrides(run);
    EXPECT_EQ(model.strategy, WaitStrategy::AVERAGED);
    EXPECT_EQ(model.noise_mode, NoiseMode::PAULI);
    // Only the success probability changes: fidelity and attempt time are kept.
    EXPECT_NEAR(model.link("A", "B").epr->delay()->mean(), 1e-5, 1e-18);
    EXPECT_NEAR(model.link("A", "B").epr->mean_bell()[0], 0.9, 1e-12);
    for (const auto& vqpu : model.vqpus)
        EXPECT_DOUBLE_EQ(vqpu.durations.gate_1q, 5e-8);

    // The deployment model is left untouched.
    EXPECT_EQ(deployed.strategy, WaitStrategy::EXACT);
    EXPECT_NEAR(deployed.link("A", "B").epr->delay()->mean(), 2e-6, 1e-18);

    // An empty run model enables the timeline with the deployment parameters.
    EXPECT_TRUE(CommModel::from_backends({{"A", backend(1, 1)}}).with_run_overrides(JSON::object()).enabled);
    EXPECT_FALSE(CommModel::from_backends({{"A", backend(1, 1)}}).with_run_overrides(JSON()).enabled);
}

TEST(CommModel, PerLinkParameters)
{
    const auto deployed = CommModel::from_backends({{"A", backend(1, 1)}, {"B", backend(1, 1)}, {"C", backend(1, 1)}});
    const JSON run = {
        {"epr", {{"fidelity", 0.95}, {"attempt_time", 1e-6}}},
        {"links", {
            {{"qpus", {"C", "A"}}, {"epr", {{"success_probability", 0.25}}},
             {"classical_link", {{"latency", 3e-6}}}}
        }}
    };
    const auto model = deployed.with_run_overrides(run);
    EXPECT_NEAR(model.link("A", "C").epr->delay()->mean(), 4e-6, 1e-18);
    EXPECT_NEAR(model.link("A", "C").epr->mean_bell()[0], 0.95, 1e-12);
    EXPECT_DOUBLE_EQ(model.link("A", "C").classical.latency, 3e-6);
    EXPECT_NEAR(model.link("A", "B").epr->delay()->mean(), 1e-6, 1e-18);
    EXPECT_DOUBLE_EQ(model.link("A", "B").classical.latency, 0.0);
}

TEST(CommModel, TabulatedSource)
{
    const JSON run = {{"epr", {
        {"source", "tabulated"},
        {"samples", {
            {{"delay", 1e-6}, {"fidelity", 1.0}, {"weight", 3.0}},
            {{"delay", 5e-6}, {"bell_coefficients", {0.5, 0.5, 0.0, 0.0}}, {"weight", 1.0}}
        }}
    }}};
    const auto model = CommModel::from_backends({{"A", backend(1, 1)}, {"B", backend(1, 1)}}).with_run_overrides(run);
    const auto& epr = *model.link("A", "B").epr;
    EXPECT_NEAR(epr.delay()->mean(), 2e-6, 1e-18);
    EXPECT_NEAR(epr.mean_bell()[0], 0.875, 1e-12);
    EXPECT_NEAR(epr.mean_bell()[1], 0.125, 1e-12);

    // Each draw keeps the delay and the state of the same sample.
    cunqa::timing::Rng rng(3);
    int slow = 0;
    for (int i = 0; i < 4000; ++i) {
        const auto draw = epr.sample(rng);
        if (draw.delay > 2e-6) {
            ++slow;
            EXPECT_DOUBLE_EQ(draw.bell[0], 0.5);
        } else {
            EXPECT_DOUBLE_EQ(draw.bell[0], 1.0);
        }
    }
    EXPECT_NEAR(slow / 4000.0, 0.25, 0.03);
}

TEST(CommModel, TabulatedSourceFromFile)
{
    const std::string path = ::testing::TempDir() + "epr_table.json";
    {
        std::ofstream file(path);
        file << R"json({"samples": [{"delay": 2e-6, "fidelity": 0.8}]})json";
    }
    const JSON run = {{"epr", {{"source", "tabulated"}, {"path", path}}}};
    const auto model = CommModel::from_backends({{"A", backend(1, 1)}, {"B", backend(1, 1)}}).with_run_overrides(run);
    EXPECT_NEAR(model.link("A", "B").epr->delay()->mean(), 2e-6, 1e-18);
    EXPECT_NEAR(model.link("A", "B").epr->mean_bell()[0], 0.8, 1e-12);
    std::remove(path.c_str());
}

TEST(CommModel, InvalidModelsAreRejected)
{
    auto build = [](const JSON& run) {
        return CommModel::from_backends({{"A", backend(1, 1)}, {"B", backend(1, 1)}}).with_run_overrides(run);
    };
    EXPECT_THROW(build({{"epr", {{"fidelty", 0.9}}}}), std::invalid_argument);
    EXPECT_THROW(build({{"unknown", 1}}), std::invalid_argument);
    EXPECT_THROW(build({{"strategy", "fast"}}), std::invalid_argument);
    EXPECT_THROW(build({{"noise_mode", "kraus"}}), std::invalid_argument);
    EXPECT_THROW(build({{"epr", {{"success_probability", 0.0}}}}), std::invalid_argument);
    EXPECT_THROW(build({{"epr", {{"fidelity", 0.9}, {"bell_coefficients", {1, 0, 0, 0}}}}}), std::invalid_argument);
    EXPECT_THROW(build({{"classical_link", {{"latency", -1.0}}}}), std::invalid_argument);
    EXPECT_THROW(build({{"classical_link", {{"jitter", {{"distribution", "gaussian"}}}}}}), std::invalid_argument);
    EXPECT_THROW(build({{"links", {{{"circuits", {"c1", "c2"}}}}}}), std::invalid_argument);
    EXPECT_THROW(build({{"time_bin", -1e-9}}), std::invalid_argument);
}

TEST(CommModel, BackendsMustAgreeOnTheNetwork)
{
    const JSON a = {{"epr", {{"fidelity", 0.9}}}};
    const JSON b = {{"epr", {{"fidelity", 0.8}}}};
    EXPECT_THROW(CommModel::from_backends({{"A", backend(1, 1, a)}, {"B", backend(1, 1, b)}}), std::invalid_argument);
    // A vQPU that does not set the network does not disagree.
    EXPECT_NO_THROW(CommModel::from_backends({{"A", backend(1, 1, a)}, {"B", backend(1, 1)}}));
}
