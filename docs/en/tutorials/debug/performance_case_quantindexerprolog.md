# QuantIndexerProlog Operator Performance Optimization Case

<!-- md-trans-meta sourceCommit=ef159c53aaffca68734dbf8caedc52ff7e051796 translatedAt=2026-08-11T09:34:19.413Z pushedAt=2026-09-04T09:41:49.303Z -->

## Task and Objective

The DeepSeekV3.2-Exp network introduces the Indexer module. The first half of the computation in this module is referred to as IndexerProlog computation, and the computation flow of its quantized version is as follows:

![](../figures/docs_models_deepseek-v3-2-exp_figures_IndexerPrologQuant.png)

This operator exhibits the following characteristics:

- The computation consists of three parts: Indexer Q, Indexer Cache, and Indexer Weight. These three parts are independent of each other, and each part is executed serially.
- Among the three independent computation flows, Indexer Q has the longest execution duration, which can mask Indexer Cache and Indexer Weight.
- In a typical scenario (Batch = 4, MTP1, KV Cache length 64k), the computation workload is relatively small and does not fully occupy all cores for computation. The performance bottleneck lies in memory-bound operations.

## Analyzing Key Bottlenecks

After the precision was verified, the initial performance was obtained, also referred to as the out-of-the-box performance, which is as follows:

![](../figures/pre_optimization_state.png)

From the out-of-the-box performance swimlane diagram, the following performance optimization points can be observed:

- Vector tasks are both numerous and sparse, with a large number of scheduling bubbles in the execution subgraph. The issue manifests as the dequantization and RoPE computations failing to be merged into the same subgraph, which increases the number of tasks, adds to the scheduling burden, and creates scheduling gaps. The root cause is an unreasonable TileShape configuration.
- Cube computation is time-consuming. Adjust TileShape to improve Cube performance.
- L1 Reuse is not yet enabled, and some subgraphs that can be merged remain unmerged, which increases the number of tasks and causes a large amount of redundant data movement for the right matrix. In typical scenarios, the operator is usually memory-bound; reducing the amount of data movement can improve performance.

## Main Tuning Process

- Tile block adjustment: After observation, in the dequantization and RoPE computation segments, there are many tasks that are not merged into a single isomorphic subgraph. If the expected Vector computations can be concentrated into one isomorphic subgraph, redundant data movement can be avoided. By adjusting the Tile blocks in the related computations to keep them consistent, the pass will place the related computations into the same isomorphic subgraph. After this optimization, the performance was optimized to 76 us, as shown in the following swimlane diagram.

    ![](../figures/vec_optimization_swimlane.png)

- Cube Tile block adjustment: The initial Tile blocks are \([128, 128], [128, 128], [128, 128]\), which typically yield decent performance. However, since m = 8 in this operator's implementation, more suitable Tile blocks can be set to achieve better performance. For example, the m axis can be cut to 16, the k axis to 512/1024, and the n axis to 64/32. After this optimization, the performance reached 56 us, as shown in the following swimlane diagram:

    ![](../figures/cube_optimization.png)

- The excessive number of Cube tasks for Q computation led to repeated loading of the right matrix. Therefore, L1Reuse was enabled to merge tasks and reduce redundant data movement. After optimization, the performance reached 49 us. The swimlane diagram is as follows:

    ![](../figures/optimized_swimlane.png)

## Obtaining the Complete Sample

The example code is located at [lightning_indexer_prolog_quant.py](../../../../models/deepseek_v32_exp/deepseekv32_lightning_indexer_prolog_quant.py). This file primarily demonstrates the specific implementation of QuantIndexerProlog.

In a typical scenario (Batch=4, MTP1, Kv Cache length 64k), the QuantIndexerProlog operator can be executed by running the following example script:

```bash
python3 models/deepseek_v32_exp/testdsv32_lightning_indexer_prolog_quant.py
```

This script provides a rich set of test cases. For different scenarios, you can modify the script to run different test cases as needed.
