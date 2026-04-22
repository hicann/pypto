/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_osp_algorithms.cpp
 * \brief Unit test for OSP Algorithm files.
 */

#include <algorithm>
#include <iterator>
#include <limits>
#include <numeric>
#include <vector>
#include <string>
#include "gtest/gtest.h"

#define MODULE_NAME "OspTests"

#include "passes/algorithms/osp/auxiliary/datastructures/union_find_universe.h"
#include "passes/algorithms/osp/auxiliary/balanced_coin_flips.h"
#include "passes/algorithms/osp/bsp/model/bsp_architecture.h"
#include "passes/algorithms/osp/bsp/model/bsp_instance.h"
#include "passes/algorithms/osp/bsp/model/bsp_schedule.h"
#include "passes/algorithms/osp/bsp/model/util/set_schedule.h"
#include "passes/algorithms/osp/bsp/scheduler/greedy_schedulers/greedy_children.h"
#include "passes/algorithms/osp/bsp/scheduler/greedy_schedulers/greedy_meta_scheduler.h"
#include "passes/algorithms/osp/bsp/scheduler/greedy_schedulers/grow_local_auto_cores.h"
#include "passes/algorithms/osp/bsp/scheduler/improvement_scheduler.h"
#include "passes/algorithms/osp/bsp/scheduler/local_search/kernighan_lin/kl_improver.h"
#include "passes/algorithms/osp/bsp/scheduler/local_search/kernighan_lin/comm_cost_modules/kl_hyper_total_comm_cost.h"
#include "passes/algorithms/osp/dag_divider/isomorphism_divider/isomorphic_subgraph_scheduler.h"
#include "passes/algorithms/osp/dag_divider/isomorphism_divider/precomputed_hash_computer.h"
#include "passes/algorithms/osp/coarser/coarser_util.h"
#include "passes/algorithms/osp/coarser/sarkar/sarkar.h"
#include "passes/algorithms/osp/coarser/sarkar/sarkar_mul.h"
#include "passes/algorithms/osp/concepts/directed_graph_edge_desc_concept.h"
#include "passes/algorithms/osp/graph_algorithms/directed_graph_path_util.h"
#include "passes/algorithms/osp/graph_implementations/adj_list_impl/compact_sparse_graph.h"
#include "passes/algorithms/osp/graph_implementations/adj_list_impl/dag_vector_adapter.h"
#include "passes/algorithms/osp/graph_implementations/adj_list_impl/cdag_vertex_impl.h"
#include "passes/algorithms/osp/graph_implementations/adj_list_impl/computational_dag_vector_impl.h"
#include "passes/algorithms/osp/graph_implementations/integral_range.h"

namespace npu::tile_fwk {
namespace osp {

using VertType = int32_t;
using WorkType = int32_t;
using VTypeType = unsigned;

using GraphType = CompactSparseGraph<VertType, VertType, WorkType, WorkType, WorkType, VTypeType>;

using VertexImpl = osp::CDagVertexImpl<VertType, WorkType, WorkType, WorkType, VTypeType>;
using GraphAdapterType = osp::DagVectorAdapter<VertexImpl>;
using ConstrGraphType = osp::ComputationalDagVectorImpl<VertexImpl>;

class TestOspAlgorithms : public testing::Test {
public:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TestOspAlgorithms, UnionFind1)
{
    std::vector<std::string> names({"a", "b", "c", "d", "e", "f"});
    UnionFindUniverse<std::string, unsigned, int> testUniverse;
    for (const auto &name : names) {
        testUniverse.AddObject(name);
    }

    for (auto &name : names) {
        EXPECT_EQ(testUniverse.FindOriginByName(name), name);
    }

    EXPECT_EQ(testUniverse.GetNumberOfConnectedComponents(), 6);

    EXPECT_EQ(testUniverse.GetNumberOfConnectedComponents(), 6);

    testUniverse.JoinByName("a", "b");
    EXPECT_EQ(testUniverse.FindOriginByName("a"), testUniverse.FindOriginByName("b"));
    EXPECT_EQ(testUniverse.GetNumberOfConnectedComponents(), 5);

    testUniverse.JoinByName("b", "c");
    EXPECT_EQ(testUniverse.GetNumberOfConnectedComponents(), 4);
    EXPECT_EQ(testUniverse.FindOriginByName("a"), testUniverse.FindOriginByName("b"));
    EXPECT_EQ(testUniverse.FindOriginByName("a"), testUniverse.FindOriginByName("c"));
    EXPECT_EQ(testUniverse.FindOriginByName("b"), testUniverse.FindOriginByName("c"));

    testUniverse.JoinByName("d", "b");
    EXPECT_EQ(testUniverse.GetNumberOfConnectedComponents(), 3);
    EXPECT_EQ(testUniverse.FindOriginByName("a"), testUniverse.FindOriginByName("d"));
    EXPECT_EQ(testUniverse.FindOriginByName("d"), testUniverse.FindOriginByName("b"));

    testUniverse.JoinByName("a", "c");
    EXPECT_EQ(testUniverse.GetNumberOfConnectedComponents(), 3);
    EXPECT_EQ(testUniverse.FindOriginByName("a"), testUniverse.FindOriginByName("c"));

    testUniverse.JoinByName("a", "d");
    EXPECT_EQ(testUniverse.GetNumberOfConnectedComponents(), 3);
    EXPECT_EQ(testUniverse.FindOriginByName("a"), testUniverse.FindOriginByName("d"));

    testUniverse.JoinByName("e", "f");
    EXPECT_EQ(testUniverse.GetNumberOfConnectedComponents(), 2);
    EXPECT_EQ(testUniverse.FindOriginByName("e"), testUniverse.FindOriginByName("f"));
    EXPECT_NE(testUniverse.FindOriginByName("a"), testUniverse.FindOriginByName("f"));

    EXPECT_EQ(testUniverse.FindOriginByName("a"), testUniverse.FindOriginByName("b"));
    EXPECT_EQ(testUniverse.FindOriginByName("b"), testUniverse.FindOriginByName("c"));
    EXPECT_EQ(testUniverse.FindOriginByName("c"), testUniverse.FindOriginByName("d"));
    EXPECT_EQ(testUniverse.FindOriginByName("a"), testUniverse.FindOriginByName("d"));
    EXPECT_EQ(testUniverse.FindOriginByName("c"), testUniverse.FindOriginByName("b"));

    EXPECT_EQ(testUniverse.FindOriginByName("e"), testUniverse.FindOriginByName("f"));

    EXPECT_NE(testUniverse.FindOriginByName("a"), testUniverse.FindOriginByName("f"));
}

TEST_F(TestOspAlgorithms, UnionFind2)
{
    std::vector<std::string> names({"a", "b", "c", "d", "e", "f", "g", "h", "i"});
    UnionFindUniverse<std::string, unsigned, int> testUniverse;

    for (auto &name : names) {
        testUniverse.AddObject(name);
    }

    for (auto &name : names) {
        EXPECT_EQ(testUniverse.FindOriginByName(name), name);
    }

    for (auto &name : names) {
        EXPECT_EQ(testUniverse.FindOriginByName(name), name);
    }

    EXPECT_EQ(testUniverse.GetNumberOfConnectedComponents(), 9);

    testUniverse.JoinByName("a", "b");
    EXPECT_EQ(testUniverse.GetNumberOfConnectedComponents(), 8);
    testUniverse.JoinByName("b", "c");
    EXPECT_EQ(testUniverse.GetNumberOfConnectedComponents(), 7);
    testUniverse.JoinByName("c", "d");
    EXPECT_EQ(testUniverse.GetNumberOfConnectedComponents(), 6);
    testUniverse.JoinByName("d", "e");
    EXPECT_EQ(testUniverse.GetNumberOfConnectedComponents(), 5);
    testUniverse.JoinByName("e", "f");
    EXPECT_EQ(testUniverse.GetNumberOfConnectedComponents(), 4);

    testUniverse.JoinByName("c", "f");
    EXPECT_EQ(testUniverse.GetNumberOfConnectedComponents(), 4);

    testUniverse.JoinByName("g", "h");
    EXPECT_EQ(testUniverse.GetNumberOfConnectedComponents(), 3);
    testUniverse.JoinByName("h", "i");
    EXPECT_EQ(testUniverse.GetNumberOfConnectedComponents(), 2);

    testUniverse.JoinByName("b", "h");
    EXPECT_EQ(testUniverse.GetNumberOfConnectedComponents(), 1);

    EXPECT_EQ(testUniverse.FindOriginByName("a"), testUniverse.FindOriginByName("b"));
    EXPECT_EQ(testUniverse.FindOriginByName("b"), testUniverse.FindOriginByName("c"));
    EXPECT_EQ(testUniverse.FindOriginByName("c"), testUniverse.FindOriginByName("d"));
    EXPECT_EQ(testUniverse.FindOriginByName("a"), testUniverse.FindOriginByName("h"));
    EXPECT_EQ(testUniverse.FindOriginByName("c"), testUniverse.FindOriginByName("i"));
    EXPECT_EQ(testUniverse.FindOriginByName("f"), testUniverse.FindOriginByName("g"));

    testUniverse.Reset();
    EXPECT_EQ(testUniverse.GetNumberOfConnectedComponents(), 0);
}

TEST_F(TestOspAlgorithms, UnionFind3)
{
    std::vector<std::string> names({"a", "b", "c", "d", "e", "f"});
    std::vector<unsigned> weights({1, 2, 1, 3, 1, 1});

    UnionFindUniverse<std::string, unsigned, unsigned> testUniverse;
    for (std::size_t i = 0; i < names.size(); ++i) {
        testUniverse.AddObject(names[i], weights[i]);
    }

    for (size_t i = 0; i < names.size(); i++) {
        EXPECT_EQ(testUniverse.FindOriginByName(names[i]), names[i]);
        EXPECT_EQ(testUniverse.GetWeightOfComponentByName(names[i]), weights[i]);
    }

    EXPECT_EQ(testUniverse.GetNumberOfConnectedComponents(), 6);

    testUniverse.JoinByName("a", "b");
    testUniverse.JoinByName("b", "c");
    testUniverse.JoinByName("d", "b");
    testUniverse.JoinByName("a", "c");
    testUniverse.JoinByName("a", "d");

    testUniverse.JoinByName("e", "f");

    EXPECT_EQ(testUniverse.FindOriginByName("a"), testUniverse.FindOriginByName("b"));
    EXPECT_EQ(testUniverse.FindOriginByName("b"), testUniverse.FindOriginByName("c"));
    EXPECT_EQ(testUniverse.FindOriginByName("c"), testUniverse.FindOriginByName("d"));
    EXPECT_EQ(testUniverse.FindOriginByName("a"), testUniverse.FindOriginByName("d"));

    EXPECT_EQ(testUniverse.FindOriginByName("e"), testUniverse.FindOriginByName("f"));

    EXPECT_NE(testUniverse.FindOriginByName("a"), testUniverse.FindOriginByName("f"));

    EXPECT_EQ(testUniverse.GetWeightOfComponentByName("a"), 7);
    EXPECT_EQ(testUniverse.GetWeightOfComponentByName("b"), 7);

    std::vector<std::vector<std::string>> components = testUniverse.GetConnectedComponents();
    unsigned totalCompWeights = 0;
    unsigned totalElements = 0;
    for (auto &comp : components) {
        totalCompWeights += testUniverse.GetWeightOfComponentByName(comp.at(0));
        totalElements += static_cast<unsigned>(comp.size());
        for (auto &name : comp) {
            EXPECT_TRUE(std::any_of(
                names.cbegin(), names.cend(), [name](std::string other_name) { return name == other_name; }));
        }
    }

    unsigned totalWeight = 0;
    for (auto &wt : weights) {
        totalWeight += wt;
    }

    EXPECT_EQ(totalElements, names.size());
    EXPECT_EQ(totalWeight, totalCompWeights);

    for (auto &name : names) {
        EXPECT_TRUE(std::any_of(components.cbegin(), components.cend(), [name](std::vector<std::string> comp) {
            return std::any_of(
                comp.cbegin(), comp.cend(), [name](std::string other_name) { return name == other_name; });
        }));
    }
}

TEST_F(TestOspAlgorithms, IntSqrt)
{
    for (std::size_t root = 1U; root < 200U; ++root) {
        for (std::size_t num = root * root; num < (root + 1U) * (root + 1U); ++num) {
            EXPECT_EQ(IntSqrtFloor(num), root);
        }
    }

    for (int root = 1; root < 300; ++root) {
        for (int num = root * root; num < (root + 1) * (root + 1); ++num) {
            EXPECT_EQ(IntSqrtFloor(num), root);
        }
    }
}

TEST_F(TestOspAlgorithms, Divisors)
{
    for (std::size_t num = 1U; num < 1000U; ++num) {
        const std::vector<std::size_t> divs = DivisorsList(num);
        for (const std::size_t &div : divs) {
            EXPECT_EQ(num % div, 0U);
        }

        auto it = divs.begin();
        for (std::size_t i = 1U; i <= num; ++i) {
            if (num % i == 0) {
                EXPECT_TRUE(it != divs.end());
                EXPECT_EQ(i, *it);
                ++it;
            }
        }
        EXPECT_TRUE(it == divs.end());
    }
}

bool thueMorseGen(long unsigned int n)
{
    unsigned long int bin_sum = 0;
    while (n != 0) {
        bin_sum += n % 2;
        n /= 2;
    }
    return bool(bin_sum % 2);
}

TEST_F(TestOspAlgorithms, RandomBiasedCoin)
{
    BiasedRandom Coin;
    bool valAnd = true;
    bool valOr = false;
    for (int i = 0; i < 1000; i++) {
        const bool flip = Coin.GetFlip();
        valAnd &= flip;
        valOr |= flip;
    }

    // Can both technically fail but insanely unlikely
    EXPECT_FALSE(valAnd);
    EXPECT_TRUE(valOr);
}

TEST_F(TestOspAlgorithms, ThueMorse)
{
    ThueMorseSequence coin(0);

    std::vector<bool> beginning(
        {0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1});
    std::vector<bool> generated;
    for (long unsigned i = 0; i < beginning.size(); i++) {
        const bool next = coin.GetFlip();
        generated.emplace_back(next);
    }

    EXPECT_TRUE(beginning == generated);

    ThueMorseSequence testCoinInSeq(0);
    for (unsigned i = 0; i < 200u; i++) {
        EXPECT_EQ(testCoinInSeq.GetFlip(), thueMorseGen(i));
    }
}

TEST_F(TestOspAlgorithms, InPlaceInversePermutationRandom)
{
    std::vector<unsigned> vec(20);
    std::iota(vec.begin(), vec.end(), 0);
    std::vector<unsigned> sol(vec);

    std::random_device rd;
    std::mt19937 gen(rd());

    for (unsigned i = 0; i < 5U; ++i) {
        std::shuffle(vec.begin(), vec.end(), gen);

        std::vector<unsigned> invPerm(vec.size());
        for (unsigned j = 0; j < vec.size(); ++j) {
            invPerm[vec[j]] = j;
        }

        InversePermuteInplace(vec, invPerm);
        for (std::size_t j = 0; j < sol.size(); ++j) {
            EXPECT_EQ(vec[j], sol[j]);
        }
    }
}

TEST_F(TestOspAlgorithms, InPlaceInversePermutationChar)
{
    std::vector<char> vec({'a', 'b', 'c', 'd', 'e', 'f', 'g'});
    std::vector<std::size_t> perm({4, 0, 1, 2, 3, 6, 5});
    std::vector<char> sol({'e', 'a', 'b', 'c', 'd', 'g', 'f'});

    InversePermuteInplace(vec, perm);
    for (std::size_t j = 0; j < sol.size(); ++j) {
        EXPECT_EQ(vec[j], sol[j]);
    }
}

TEST_F(TestOspAlgorithms, Architecture)
{
    std::vector<std::vector<WorkType>> uniformSentCosts = {
        {0, 1, 1, 1},
        {1, 0, 1, 1},
        {1, 1, 0, 1},
        {1, 1, 1, 0}
    };

    BspArchitecture<GraphType> architecture;
    architecture.SetNumberOfProcessors(4);
    architecture.SetCommunicationCosts(2);
    architecture.SetSynchronisationCosts(3);

    EXPECT_EQ(architecture.NumberOfProcessors(), 4);
    EXPECT_EQ(architecture.CommunicationCosts(), 2);
    EXPECT_EQ(architecture.SynchronisationCosts(), 3);
    EXPECT_EQ(architecture.GetMemoryConstraintType(), MemoryConstraintType::NONE);
    EXPECT_EQ(architecture.GetNumberOfProcessorTypes(), 1);

    EXPECT_EQ(architecture.MemoryBound(0), 100);
    EXPECT_EQ(architecture.MemoryBound(1), 100);
    EXPECT_EQ(architecture.MemoryBound(2), 100);
    EXPECT_EQ(architecture.MemoryBound(3), 100);
    architecture.SetMemoryBound(200, 3);
    EXPECT_EQ(architecture.MemoryBound(2), 100);
    EXPECT_EQ(architecture.MemoryBound(3), 200);

    EXPECT_EQ(architecture.ProcessorTypes()[0], 0);
    EXPECT_EQ(architecture.ProcessorTypes()[1], 0);
    EXPECT_EQ(architecture.ProcessorTypes()[2], 0);
    EXPECT_EQ(architecture.ProcessorTypes()[3], 0);

    EXPECT_EQ(architecture.ProcessorType(0), 0);
    EXPECT_EQ(architecture.ProcessorType(1), 0);
    EXPECT_EQ(architecture.ProcessorType(2), 0);
    EXPECT_EQ(architecture.ProcessorType(3), 0);
    architecture.SetProcessorsWithTypes({0, 0, 0, 1});
    EXPECT_EQ(architecture.ProcessorType(2), 0);
    EXPECT_EQ(architecture.ProcessorType(3), 1);

    EXPECT_EQ(architecture.CommunicationCosts(0, 1), 2);
    EXPECT_EQ(architecture.CommunicationCosts(0, 0), 0);

    EXPECT_EQ(architecture.GetNumberOfProcessorTypes(), 2);

    EXPECT_TRUE(architecture.SendCost() == uniformSentCosts);
    EXPECT_EQ(architecture.CommunicationCosts(0, 1), 2);
    EXPECT_EQ(architecture.CommunicationCosts(0, 0), 0);
}

TEST_F(TestOspAlgorithms, EmptyGraph)
{
    GraphType graph;

    EXPECT_EQ(graph.NumVertices(), 0);
    EXPECT_EQ(graph.NumEdges(), 0);
}

TEST_F(TestOspAlgorithms, NoEdgesGraph)
{
    const std::vector<std::pair<VertType, VertType>> edges({});

    GraphType graph(10, edges);

    EXPECT_EQ(graph.NumVertices(), 10);
    EXPECT_EQ(graph.NumEdges(), 0);
}

GraphType LineGraph()
{
    const std::set<std::pair<VertType, VertType>> edges({
        {0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}, {6, 7}
    });

    return GraphType(8, edges);
}

TEST_F(TestOspAlgorithms, TestLineGraph)
{
    const GraphType graph = LineGraph();

    EXPECT_EQ(graph.NumVertices(), 8);
    EXPECT_EQ(graph.NumEdges(), 7);

    std::size_t cntr = 0;
    for (const auto &vert : graph.Vertices()) {
        EXPECT_EQ(vert, cntr);
        ++cntr;
    }
    EXPECT_EQ(graph.NumVertices(), cntr);

    for (const auto &vert : graph.Vertices()) {
        VertType notLast = static_cast<VertType>((vert != 7));

        EXPECT_EQ(graph.OutDegree(vert), notLast);
        for (const auto &chld : graph.Children(vert)) {
            EXPECT_EQ(chld, vert + notLast);
        }
        auto chldren = graph.Children(vert);
        EXPECT_EQ(chldren.crend() - chldren.crbegin(), graph.OutDegree(vert));
        for (auto it = chldren.crbegin(); it != chldren.crend(); ++it) {
            EXPECT_EQ(*it, vert + notLast);
        }
    }
    for (const auto &vert : graph.Vertices()) {
        VertType notFirst = static_cast<VertType>((vert != 0));

        EXPECT_EQ(graph.InDegree(vert), notFirst);
        for (const auto &par : graph.Parents(vert)) {
            EXPECT_EQ(par, vert - notFirst);
        }
        auto prnts = graph.Parents(vert);
        EXPECT_EQ(prnts.crend() - prnts.crbegin(), graph.InDegree(vert));
        for (auto it = prnts.crbegin(); it != prnts.crend(); ++it) {
            EXPECT_EQ(*it, vert - notFirst);
        }
    }

    for (const auto &vert : graph.Vertices()) {
        EXPECT_EQ(graph.VertexType(vert), 0);
    }
}

GraphType SimpleGraph()
{
    const std::vector<std::pair<VertType, VertType>> edges({
        {0,  1}, {2,  3}, {6, 10}, {7,  9}, {0,  2}, {4,  6}, {1,  6}, {6,  7}, {5,  6}, {3,  7}, {1,  2}
    });

    return GraphType(11, edges);
}

TEST_F(TestOspAlgorithms, Graph1)
{
    const GraphType graph = SimpleGraph();

    EXPECT_EQ(graph.NumVertices(), 11);
    EXPECT_EQ(graph.NumEdges(), 11);

    std::size_t cntr0{};
    std::size_t cntrChldEdges{};
    std::size_t cntrParEdges{};
    for (const auto &vert : graph.Vertices()) {
        EXPECT_EQ(vert, cntr0++);
        cntrChldEdges += graph.OutDegree(vert);
        cntrParEdges += graph.InDegree(vert);
    }
    EXPECT_EQ(graph.NumVertices(), cntr0);
    EXPECT_EQ(graph.NumEdges(), cntrChldEdges);
    EXPECT_EQ(graph.NumEdges(), cntrParEdges);

    const std::vector<std::vector<std::size_t>> outEdges({
        {1, 2}, {2, 6}, {3}, {7}, {6}, {6}, {7, 10}, {9}, {}, {}, {}
    });

    for (const auto &vert : graph.Vertices()) {
        EXPECT_EQ(graph.OutDegree(vert), outEdges[vert].size());
        std::size_t cntr = 0;
        for (const auto &chld : graph.Children(vert)) {
            EXPECT_EQ(chld, outEdges[vert][cntr++]);
        }
        auto chldrn = graph.Children(vert);
        EXPECT_EQ(chldrn.crend() - chldrn.crbegin(), graph.OutDegree(vert));
        for (auto it = chldrn.crbegin(); it != chldrn.crend(); ++it) {
            EXPECT_EQ(*it, outEdges[vert][--cntr]);
        }
    }

    const std::vector<std::vector<std::size_t>> inEdges({
        {}, {0}, {0, 1}, {2}, {}, {}, {1, 4, 5}, {3, 6}, {}, {7}, {6}
    });

    for (const auto &vert : graph.Vertices()) {
        EXPECT_EQ(graph.InDegree(vert), inEdges[vert].size());
        std::size_t cntr = 0;
        for (const auto &par : graph.Parents(vert)) {
            EXPECT_EQ(par, inEdges[vert][cntr++]);
        }
        auto prnts = graph.Parents(vert);
        EXPECT_EQ(prnts.crend() - prnts.crbegin(), graph.InDegree(vert));
        for (auto it = prnts.crbegin(); it != prnts.crend(); ++it) {
            EXPECT_EQ(*it, inEdges[vert][--cntr]);
        }
    }

    for (const auto &vert : graph.Vertices()) {
        EXPECT_EQ(graph.VertexType(vert), 0);
    }
}

TEST_F(TestOspAlgorithms, GraphWorkWeights)
{
    std::vector<unsigned> ww(11);
    std::iota(ww.begin(), ww.end(), 0);

    GraphType graph = SimpleGraph();
    for (auto vert : graph.Vertices()) {
        graph.SetVertexWorkWeight(vert, ww[vert]);
    }

    for (auto vert : graph.Vertices()) {
        EXPECT_EQ(graph.VertexWorkWeight(vert), ww[vert]);

        const unsigned wt = static_cast<unsigned>(rand());
        graph.SetVertexWorkWeight(vert, wt);
        EXPECT_EQ(graph.VertexWorkWeight(vert), wt);
    }
}

TEST_F(TestOspAlgorithms, GraphCommWeights)
{
    std::vector<unsigned> cw(11);
    std::iota(cw.begin(), cw.end(), 11);

    GraphType graph = SimpleGraph();
    for (auto vert : graph.Vertices()) {
        graph.SetVertexCommWeight(vert, cw[vert]);
    }

    for (auto vert : graph.Vertices()) {
        EXPECT_EQ(graph.VertexCommWeight(vert), cw[vert]);

        const unsigned wt = static_cast<unsigned>(rand());
        graph.SetVertexCommWeight(vert, wt);
        EXPECT_EQ(graph.VertexCommWeight(vert), wt);
    }
}

TEST_F(TestOspAlgorithms, GraphMemWeights)
{
    std::vector<unsigned> mw(11);
    std::iota(mw.begin(), mw.end(), 22);

    GraphType graph = SimpleGraph();

    for (auto vert : graph.Vertices()) {
        graph.SetVertexMemWeight(vert, mw[vert]);
    }

    for (auto vert : graph.Vertices()) {
        EXPECT_EQ(graph.VertexMemWeight(vert), mw[vert]);

        const unsigned wt = static_cast<unsigned>(rand());
        graph.SetVertexMemWeight(vert, wt);
        EXPECT_EQ(graph.VertexMemWeight(vert), wt);
    }
}

TEST_F(TestOspAlgorithms, GraphVtype)
{
    std::vector<unsigned> vt(11);
    std::iota(vt.begin(), vt.end(), 33);

    GraphType graph = SimpleGraph();

    for (auto vert : graph.Vertices()) {
        graph.SetVertexType(vert, vt[vert]);
    }

    for (auto vert : graph.Vertices()) {
        EXPECT_EQ(graph.VertexType(vert), vt[vert]);

        const unsigned wt = static_cast<unsigned>(rand());
        graph.SetVertexType(vert, wt);
        EXPECT_EQ(graph.VertexType(vert), wt);
    }
}

TEST_F(TestOspAlgorithms, ExpansionMapValidity)
{
    const std::vector<std::vector<VertexIdxT<GraphType>>> expansionmap1 = {{0}, {1}, {2}, {3}};
    EXPECT_TRUE(coarser_util::CheckValidExpansionMap<GraphType>(expansionmap1));

    const std::vector<std::vector<VertexIdxT<GraphType>>> expansionmap2 = {{0}, {2}, {3}};
    EXPECT_FALSE(coarser_util::CheckValidExpansionMap<GraphType>(expansionmap2));

    const std::vector<std::vector<VertexIdxT<GraphType>>> expansionmap3 = {{0, 3}};
    EXPECT_FALSE(coarser_util::CheckValidExpansionMap<GraphType>(expansionmap3));

    const std::vector<std::vector<VertexIdxT<GraphType>>> expansionmap4 = {{0, 3}, {2, 1, 4}, {5}};
    EXPECT_TRUE(coarser_util::CheckValidExpansionMap<GraphType>(expansionmap4));

    const std::vector<std::vector<VertexIdxT<GraphType>>> expansionmap5 = {{0}, {}, {2}, {3}, {1}};
    EXPECT_FALSE(coarser_util::CheckValidExpansionMap<GraphType>(expansionmap5));
}

TEST_F(TestOspAlgorithms, ContractionMapValidity)
{
    const std::vector<VertexIdxT<GraphType>> contractionMap1 = {0, 1, 2, 3};
    EXPECT_TRUE(coarser_util::CheckValidContractionMap<GraphType>(contractionMap1));

    const std::vector<VertexIdxT<GraphType>> contractionMap2 = {0, 1, 1, 1};
    EXPECT_TRUE(coarser_util::CheckValidContractionMap<GraphType>(contractionMap2));

    const std::vector<VertexIdxT<GraphType>> contractionMap3 = {0, 1, 1, 3};
    EXPECT_FALSE(coarser_util::CheckValidContractionMap<GraphType>(contractionMap3));

    const std::vector<VertexIdxT<GraphType>> contractionMap4 = {2, 1, 1, 3};
    EXPECT_FALSE(coarser_util::CheckValidContractionMap<GraphType>(contractionMap4));
}

TEST_F(TestOspAlgorithms, ContractionMapCoarsening)
{
    std::set<std::pair<VertexIdxT<GraphType>, VertexIdxT<GraphType>>> edges({
        {0, 1},
        {1, 2}
    });
    GraphType graph(6, edges);

    GraphType coarseGraph1;

    std::vector<VertexIdxT<GraphType>> contractionMap({0, 0, 1, 1, 2, 3});
    EXPECT_TRUE(coarser_util::CheckValidContractionMap<GraphType>(contractionMap));
    EXPECT_TRUE(coarser_util::ConstructCoarseDag(graph, coarseGraph1, contractionMap));
    EXPECT_TRUE(contractionMap == std::vector<VertexIdxT<GraphType>>({0, 0, 1, 1, 2, 3}));

    EXPECT_EQ(coarseGraph1.NumVertices(), 4);
    EXPECT_EQ(coarseGraph1.NumEdges(), 1);

    EXPECT_EQ(coarseGraph1.OutDegree(0), 1);
    EXPECT_EQ(coarseGraph1.OutDegree(1), 0);
    EXPECT_EQ(coarseGraph1.OutDegree(2), 0);

    EXPECT_EQ(coarseGraph1.InDegree(0), 0);
    EXPECT_EQ(coarseGraph1.InDegree(1), 1);
    EXPECT_EQ(coarseGraph1.InDegree(2), 0);

    for (const auto &vert : coarseGraph1.Children(0)) {
        EXPECT_EQ(vert, 1);
    }

    for (const auto &vert : coarseGraph1.Parents(1)) {
        EXPECT_EQ(vert, 0);
    }
}

TEST_F(TestOspAlgorithms, TestTopSort)
{
    const GraphType graph = SimpleGraph();

    std::vector<VertexIdxT<GraphType>> verts(graph.NumVertices());
    std::iota(verts.begin(), verts.end(), 0);
    const auto topOrderVec = GetTopOrder<GraphType>(graph);
    EXPECT_TRUE(std::is_permutation(topOrderVec.cbegin(), topOrderVec.cend(), verts.cbegin(), verts.cend()));
    for (const auto vert : graph.Vertices()) {
        for (const auto chld : graph.Children(vert)) {
            EXPECT_GT(std::distance(std::find(topOrderVec.cbegin(), topOrderVec.cend(), vert),
                                    std::find(topOrderVec.cbegin(), topOrderVec.cend(), chld)),
                      0);
        }
    }
}

TEST_F(TestOspAlgorithms, NodeDistances)
{
    const std::vector<unsigned> botDistAns = {5, 4, 3, 2, 3, 3, 2, 1, 0, 0, 0};
    const std::vector<unsigned> topDistAns = {0, 1, 2, 3, 0, 0, 2, 4, 0, 5, 3};

    const GraphType graph = SimpleGraph();
    const std::vector<unsigned> botDist = GetBottomNodeDistance(graph);
    const std::vector<unsigned> topDist = GetTopNodeDistance(graph);

    EXPECT_EQ(botDist.size(), graph.NumVertices());
    EXPECT_EQ(topDist.size(), graph.NumVertices());

    for (const auto vert : graph.Vertices()) {
        EXPECT_EQ(botDist[vert], botDistAns[vert]);
        EXPECT_EQ(topDist[vert], topDistAns[vert]);
    }
}

TEST_F(TestOspAlgorithms, TestIntegralRange)
{
    const std::size_t length0 = 0U;
    for (const auto val : IntegralRange(length0)) {
        EXPECT_TRUE(false);
        EXPECT_EQ(val, 100U);
    }

    const std::size_t length147 = 147U;
    std::size_t cntr147 = 0U;
    for (const auto val : IntegralRange(length147)) {
        EXPECT_EQ(val, cntr147++);
    }
    EXPECT_EQ(cntr147, length147);

    const std::size_t start67 = 67U;
    const std::size_t end67 = 67U;
    for (const auto val : IntegralRange(start67, end67)) {
        EXPECT_TRUE(false);
        EXPECT_EQ(val, 100U);
    }

    std::size_t start134 = 67U;
    const std::size_t end257 = 67U;
    for (const auto val : IntegralRange(start134, end257)) {
        EXPECT_EQ(val, start134++);
    }
    EXPECT_EQ(start134, end257);
}

template<typename OtherGraphType>
void testValidContractionMap(const GraphType &graph,
                             const OtherGraphType &coarseGraph,
                             const std::vector<VertexIdxT<GraphType>> &contractionMap)
{
    EXPECT_EQ(contractionMap.size(), graph.NumVertices());
    EXPECT_TRUE(coarser_util::CheckValidContractionMap<GraphType>(contractionMap));

    for (auto vert : graph.Vertices()) {
        EXPECT_EQ(graph.VertexType(vert), coarseGraph.VertexType(contractionMap[vert]));
    }

    // Acyclic check
    std::vector<VertexIdxT<GraphType>> coarseVerts(coarseGraph.NumVertices());
    std::iota(coarseVerts.begin(), coarseVerts.end(), 0);
    const auto coarseTopOrder = GetTopOrder<OtherGraphType>(coarseGraph);
    EXPECT_TRUE(std::is_permutation(
        coarseTopOrder.cbegin(), coarseTopOrder.cend(),
        coarseVerts.cbegin(), coarseVerts.cend()));
    for (const auto vert : coarseGraph.Vertices()) {
        for (const auto chld : coarseGraph.Children(vert)) {
            EXPECT_GT(std::distance(std::find(coarseTopOrder.cbegin(), coarseTopOrder.cend(), vert),
                                    std::find(coarseTopOrder.cbegin(), coarseTopOrder.cend(), chld)),
                      0);
        }
    }
    for (const auto vert : graph.Vertices()) {
        for (const auto chld : graph.Children(vert)) {
            EXPECT_GE(std::distance(std::find(coarseTopOrder.cbegin(), coarseTopOrder.cend(), contractionMap[vert]),
                                    std::find(coarseTopOrder.cbegin(), coarseTopOrder.cend(), contractionMap[chld])),
                      0);
        }
    }

    // Grouping of Vertex Types
    std::vector<unsigned> coarseTypes(coarseGraph.NumVertices(), std::numeric_limits<unsigned>::max());
    for (const auto vert : graph.Vertices()) {
        const auto coarseVert = contractionMap[vert];
        if (coarseTypes[coarseVert] != std::numeric_limits<unsigned>::max()) {
            EXPECT_EQ(coarseTypes[coarseVert], graph.VertexType(vert));
        }
        coarseTypes[coarseVert] = graph.VertexType(vert);
    }
}

GraphType SimpleGraphWithVertexTypes()
{
    std::vector<unsigned> vt(11, 0);
    vt[0] = 1U;
    vt[1] = 1U;
    vt[2] = 1U;

    GraphType graph = SimpleGraph();
    for (auto vert : graph.Vertices()) {
        graph.SetVertexType(vert, vt[vert]);
    }

    return graph;
}

void testCoarseningAlgorithm(Coarser<GraphType, GraphType> &coarser)
{
    GraphType graph = SimpleGraphWithVertexTypes();

    GraphType coarseGraph;
    std::vector<VertexIdxT<GraphType>> contractionMap;

    EXPECT_TRUE(coarser.CoarsenDag(graph, coarseGraph, contractionMap));
    testValidContractionMap(graph, coarseGraph, contractionMap);
}

TEST_F(TestOspAlgorithms, CoarsenSarkar)
{
    sarkar_params::Parameters<VWorkwT<GraphType>> params;
    params.mode_ = sarkar_params::Mode::LINES;
    params.commCost_ = 100;
    params.useTopPoset_ = true;

    Sarkar<GraphType, GraphType> coarser(params);

    testCoarseningAlgorithm(coarser);

    params.useTopPoset_ = false;
    coarser.SetParameters(params);
    testCoarseningAlgorithm(coarser);

    params.mode_ = sarkar_params::Mode::FAN_IN_FULL;
    coarser.SetParameters(params);
    testCoarseningAlgorithm(coarser);

    params.mode_ = sarkar_params::Mode::FAN_IN_PARTIAL;
    coarser.SetParameters(params);
    testCoarseningAlgorithm(coarser);

    params.mode_ = sarkar_params::Mode::FAN_OUT_FULL;
    coarser.SetParameters(params);
    testCoarseningAlgorithm(coarser);

    params.mode_ = sarkar_params::Mode::FAN_OUT_PARTIAL;
    coarser.SetParameters(params);
    testCoarseningAlgorithm(coarser);

    params.mode_ = sarkar_params::Mode::LEVEL_EVEN;
    coarser.SetParameters(params);
    testCoarseningAlgorithm(coarser);

    params.mode_ = sarkar_params::Mode::LEVEL_ODD;
    coarser.SetParameters(params);
    testCoarseningAlgorithm(coarser);

    params.mode_ = sarkar_params::Mode::FAN_IN_BUFFER;
    coarser.SetParameters(params);
    testCoarseningAlgorithm(coarser);

    params.mode_ = sarkar_params::Mode::FAN_OUT_BUFFER;
    coarser.SetParameters(params);
    testCoarseningAlgorithm(coarser);

    params.mode_ = sarkar_params::Mode::HOMOGENEOUS_BUFFER;
    coarser.SetParameters(params);
    testCoarseningAlgorithm(coarser);
}

TEST_F(TestOspAlgorithms, CoarsenSarkarML)
{
    sarkar_params::MulParameters<VWorkwT<GraphType>> params;
    params.commCostVec_ = {100};

    SarkarMul<GraphType, GraphType> coarser;
    coarser.SetParameters(params);
    testCoarseningAlgorithm(coarser);

    params.commCostVec_ = {1, 2, 10, 50, 100};
    params.bufferMergeMode_ = sarkar_params::BufferMergeMode::FULL;
    coarser.SetParameters(params);

    testCoarseningAlgorithm(coarser);
}

TEST_F(TestOspAlgorithms, DagAdaptorSimpleGraph)
{
    const std::vector<std::vector<VertType>> outEdges({
        {1, 2}, {2, 6}, {3}, {7}, {6}, {6}, {7, 10}, {9}, {}, {}, {}
    });

    const std::vector<std::vector<VertType>> inEdges({
        {}, {0}, {0, 1}, {2}, {}, {}, {1, 4, 5}, {3, 6}, {}, {7}, {6}
    });

    GraphAdapterType graph(outEdges, inEdges);

    std::size_t cntr0{};
    std::size_t cntrChldEdges{};
    std::size_t cntrParEdges{};
    for (const auto &vert : graph.Vertices()) {
        EXPECT_EQ(vert, cntr0++);
        cntrChldEdges += graph.OutDegree(vert);
        cntrParEdges += graph.InDegree(vert);
    }
    EXPECT_EQ(graph.NumEdges(), cntrChldEdges);
    EXPECT_EQ(graph.NumEdges(), cntrParEdges);

    for (const auto &vert : graph.Vertices()) {
        EXPECT_EQ(graph.InDegree(vert), inEdges[vert].size());
        std::size_t cntr = 0;
        for (const auto &par : graph.Parents(vert)) {
            EXPECT_EQ(par, inEdges[vert][cntr++]);
        }
        EXPECT_EQ(cntr, graph.InDegree(vert));
    }

    for (const auto &vert : graph.Vertices()) {
        EXPECT_EQ(graph.OutDegree(vert), outEdges[vert].size());
        std::size_t cntr = 0;
        for (const auto &chld : graph.Children(vert)) {
            EXPECT_EQ(chld, outEdges[vert][cntr++]);
        }
        EXPECT_EQ(cntr, graph.OutDegree(vert));
    }

    for (const auto &vert : graph.Vertices()) {
        EXPECT_EQ(graph.VertexType(vert), 0);
    }

    for (const auto vert : graph.Vertices()) {
        graph.SetVertexWorkWeight(vert, 4*vert + 0);
        graph.SetVertexCommWeight(vert, 4*vert + 1);
        graph.SetVertexMemWeight(vert, 4*vert + 2);
        graph.SetVertexType(vert, static_cast<VTypeType>(4*vert + 3));
    }
    for (const auto vert : graph.Vertices()) {
        EXPECT_EQ(graph.VertexWorkWeight(vert), 4*vert + 0);
        EXPECT_EQ(graph.VertexCommWeight(vert), 4*vert + 1);
        EXPECT_EQ(graph.VertexMemWeight(vert), 4*vert + 2);
        EXPECT_EQ(graph.VertexType(vert), static_cast<VTypeType>(4*vert + 3));
    }
}

TEST_F(TestOspAlgorithms, BspSchedulers)
{
    BspInstance<GraphType> bspInst;
    bspInst.GetArchitecture() = BspArchitecture<GraphType>(3U);
    bspInst.GetComputationalDag() = SimpleGraph();

    BspSchedule<GraphType> schedule(bspInst);

    GrowLocalAutoCores<GraphType> growlocal;
    growlocal.ComputeSchedule(schedule);
    EXPECT_TRUE(schedule.IsValid());

    GreedyChildren<GraphType> children;
    children.ComputeSchedule(schedule);
    EXPECT_TRUE(schedule.IsValid());

    KlImprover<GraphType, KlHyperTotalCommCostFunction<GraphType, double, 1>, 1, double> kl;
    kl.SetSuperstepRemoveStrengthParameter(1.0);
    kl.SetTimeQualityParameter(1.0);

    ComboScheduler<GraphType> growlocalKl(growlocal, kl);
    growlocalKl.ComputeSchedule(schedule);
    EXPECT_TRUE(schedule.IsValid());

    ComboScheduler<GraphType> childrenKl(children, kl);
    childrenKl.ComputeSchedule(schedule);
    EXPECT_TRUE(schedule.IsValid());

    GreedyMetaScheduler<GraphType> greedymeta;
    greedymeta.AddScheduler(growlocalKl);
    greedymeta.AddScheduler(childrenKl);
    greedymeta.AddSerialScheduler();

    greedymeta.ComputeSchedule(schedule);
    EXPECT_TRUE(schedule.IsValid());
}

TEST_F(TestOspAlgorithms, CoarsenMerkleBsp)
{
    BspInstance<GraphType> bspInst;
    bspInst.GetArchitecture() = BspArchitecture<GraphType>(3U);
    bspInst.GetArchitecture().SetProcessorsWithTypes(std::vector<unsigned>({0U, 0U, 1U}));
    bspInst.GetComputationalDag() = SimpleGraphWithVertexTypes();
    bspInst.SetDiagonalCompatibilityMatrix(2);

    GrowLocalAutoCores<ConstrGraphType> growlocal;
    GreedyChildren<ConstrGraphType> children;

    KlImprover<ConstrGraphType, KlHyperTotalCommCostFunction<ConstrGraphType, double, 1>, 1, double> kl;
    kl.SetSuperstepRemoveStrengthParameter(1.0);
    kl.SetTimeQualityParameter(1.0);

    ComboScheduler<ConstrGraphType> growlocalKl(growlocal, kl);
    ComboScheduler<ConstrGraphType> childrenKl(children, kl);

    GreedyMetaScheduler<ConstrGraphType> scheduler;
    scheduler.AddScheduler(growlocalKl);
    scheduler.AddScheduler(childrenKl);
    scheduler.AddSerialScheduler();

    std::vector<uint64_t> nodeHashList(bspInst.GetComputationalDag().NumVertices(), 7U);
    MerkleHashComputer<GraphType, PrecomBwdMerkleNodeHashFunc<GraphType>> hashComputer(
        bspInst.GetComputationalDag(), bspInst.GetComputationalDag(), nodeHashList);
    IsomorphicSubgraphScheduler<GraphType, ConstrGraphType> isoScheduler(scheduler, hashComputer);
    isoScheduler.SetWorkThreshold(200);
    isoScheduler.SetCriticalPathThreshold(500);

    const auto vertexContractionMap = isoScheduler.ComputePartition(bspInst);
    EXPECT_TRUE(coarser_util::CheckValidContractionMap<ConstrGraphType>(vertexContractionMap));

    ConstrGraphType coarseGraph;
    coarser_util::ConstructCoarseDag(bspInst.GetComputationalDag(), coarseGraph, vertexContractionMap);

    testValidContractionMap(bspInst.GetComputationalDag(), coarseGraph, vertexContractionMap);
}

} // namespace osp
} // namespace npu::tile_fwk