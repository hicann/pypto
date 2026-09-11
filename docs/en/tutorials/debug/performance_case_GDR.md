# GatedDeltaRule Operator Performance Optimization Case

<!-- md-trans-meta sourceCommit=ef159c53aaffca68734dbf8caedc52ff7e051796 translatedAt=2026-08-11T09:28:19.690Z pushedAt=2026-09-04T09:40:15.183Z -->

## Task and Objective

With its powerful attention mechanism, the Transformer delivers exceptional sequence modeling capability and has achieved breakthroughs in the field of large language models. However, the computational complexity of the self-attention module grows quadratically with sequence length, posing significant computational challenges during both training and inference. To mitigate this issue, researchers have proposed linear Transformer alternatives that replace the conventional softmax attention with kernelized dot-product attention, reformulating it as a linear RNN with matrix states and substantially reducing the computational requirements in both training and inference phases.
The paper *Gated Delta Networks: Improving Mamba2 with Delta Rule* introduces the Gated DeltaNet architecture, which combines a gating mechanism with the Delta update rule to enhance the performance of linear Transformers in long-sequence modeling and information retrieval tasks.

Figure 1 Gated DeltaNet model architecture
![Figure description](../figures/gdr_1.1.png)

## Operator Framework

```py
@pypto.frontend.jit
def chunk_gated_delta_rule():
    # Loop b: parallel
    for b_idx in pypto.loop(B):
        # Loop n: parallel
        for nv_idx in pypto.loop(Nv):
            # Loop s: serial
            for s_idx in pypto.loop(0, S, L):
                # view
                ...

                # compute
                # Computation logic before recurrent_loop can be executed in parallel.
                ...
                # recurrent_loop must be serial because its output state is updated chunk by chunk.
                ...

                # assemble
                ...
```

## Tuning Process

### PyPTO Config Tuning

PyPTO provides various Config options, allowing users to perform in-depth custom optimization for their specific fused operator implementations, such as pass graph stitching and splitting strategy adjustment and runtime task scheduling strategy. Among these, the configuration most relevant to GDN operator performance tuning is stitch_function_max_num in runtime_options. Dynamic stitch is a key technology of the PyPTO MPMD architecture, where the AI CPU dynamically determines the control flow to be executed based on runtime inputs and dynamically computes dependencies, thereby combining and dispatching tasks via stitch. The stitch_function_max_num configuration represents the number of computation tasks for the first device task submitted to the schedule AI CPU for processing within the machine runtime control flow AI CPU. This value is used to control the startup overhead of the device machine, allowing the control flow AI CPU and schedule AI CPU computations to overlap as early as possible.
The following swimlane diagrams are all based on the data scenario of B=2, T=8192, H=4, D=128. As shown in Figure 3-1 and Figure 3-2, they are the swimlane diagram with stitch_function_max_num set to 32 and the swimlane diagram with stitch_function_max_num set to 128, respectively. A larger value allows tasks to be more fully parallelized, resulting in better performance; however, workspace usage also increases, requiring a trade-off.

Figure 2 Swimlane diagram with stitch_function_max_num set to 32
![Figure description](../figures/gdr_3.1.png)

Figure 3 Swimlane diagram with stitch_function_max_num set to 128
![Figure description](../figures/gdr_3.2.png)

### Ascend-Friendly Tile Shape Size Tuning

In PyPTO, all computations are performed based on tiles (hardware-aware data blocks), fully leveraging the parallel computing capability and memory hierarchy of the hardware. Tiles can be stored in AICore private caches (such as UB and L1), significantly improving data access efficiency. For each OP operation, the tile_shape size can be flexibly set to optimize compute load balancing, memory bandwidth utilization, and to maximize hardware resource usage efficiency, so as to fully utilize the UB capacity and align with Ascend architecture characteristics. Triton operators developed for GPUs typically set the chunk size to 64, resulting in a Tile shape of [64, 128]. When the dtype is FP32, the data volume is 32 KB. For the Ascend NPU with a UB of 192 KB, the appropriate Tile size ranges from 16 KB to 64 KB. After adjusting the chunk_size to 128, the data volume increases to 64 KB, which better aligns with the Ascend operator development strategy of transferring larger data blocks at a time, greatly reducing the performance overhead caused by data transfer and fully unleashing the parallel computing capability of the hardware. In addition, due to the chunk loop update algorithm characteristic of the GDN operator, processing a larger chunk of data at a time reduces the number of serial loop updates, alleviating the performance issue where serial computation logic is difficult to fully parallelize and saturate the cores under the existing MPMD scheduling strategy of PyPTO.
As shown in Figure 3-3, compared with Figure 3-2, this is the swimlane diagram where chunk_size is optimized from 64 to 128 and TileShape is optimized from [64, 128] to [128, 128]. The Tile block data size is more aligned with the Ascend NPU hardware storage size, which reduces the number of loop iterations, the number of root functions, and the number of tasks. With the number of tasks delivered by stitch binding unchanged (stitch_function_max_num=128), the number of stitches is reduced, which essentially reduces movement overhead and improves computation efficiency.

Figure 4 Swimlane diagram with chunk_size set to 128 and TileShape set to [128, 128]
![Figure description](../figures/gdr_3.3.png)

### PyPTO Loop Mode and Graph Stitching and Unrolling Optimization

#### Adjusting Dynamic Loop to Static Loop

PyPTO adopts the PTO (Parallel Tensor/Tile Operation) programming paradigm, with a tile-based programming model as its core design philosophy. Through multi-level compute graph representation, it compiles the AI model built by users via APIs from high-level tensor compute graphs down to hardware instructions, ultimately generating code that can be efficiently executed on the target platform. The device side then automatically schedules execution in MPMD (Multiple Program Multiple Data) mode. Understanding PyPTO loops and the graph stitching and splitting logic is a further step toward deeper comprehension of its frontend representation. First, the loops in PyPTO are primarily designed to handle dynamic shapes. In a full-network scenario, parameters such as batch_size and seq_length are often dynamic, while the tile size is typically fixed. The PyPTO framework translates pre-marked dynamic axes into CCE code using expressions composed of SymbolicScalar during compilation, and then resolves the specific sizes of these expressions at the Machine layer. This enables dynamic handling of on-device execution states for different shapes, significantly reducing the repetitive overhead of compilation graph construction and allowing different dynamic shape inputs to share the same frontend representation and graph IR. In the GDN operator, the inverse module performs row-by-row computation and update on tile blocks. When tail blocks are not considered (they can also be padded), the number of row-by-row loop iterations is fixed. However, the actual number of PyPTO loop iterations occupies the root function, which is the number of tasks that can be bound and dispatched in a single stitch. Therefore, in terms of programming paradigm, the inverse module is better suited to using Python's static loop approach. While this sacrifices some compilation performance, it greatly reduces the runtime overhead incurred by the inverse module—which requires extensive repetitive computation—when dynamic loops are used.

```py
# row_num is a static value.
# Original dynamic loop approach.
for i in pypto.loop(2, row_num, 1):
    # Row-by-row inverse update.

# Change to static loop mode.
for i in range(2, row_num, 1):
    # Update the inverse row by row.
```

#### Graph Stitching and Splitting Optimization

The compute graph of PyPTO consists of tensor data nodes and operation nodes. Through layer-by-layer pass optimizations, the Tensor Graph defined by the user frontend is eventually converted into an Execution Graph, which is then translated into CCE-executable code. The Execution Graph integrates computation subgraph information, including dependency relationships and scheduling information, and defines the specific operation combinations of Tiles on AIC/AIV. Different graph stitching and splitting strategies have a significant impact on performance. Typically, pass provides generalized optimization strategies to help users achieve good overall performance. However, if finer control over the specific computation flow of Tile blocks is required, users need to leverage the DFX visualization capability of the compute graph to verify whether the execution matches expectations. As shown in Figure 3-4, which presents the initial out-of-the-box swimlane diagram for B=1, T=128, H=1, numerous extremely short fragments and abnormally long execution graphs are observed. After inspecting and comparing the computation flow, it was found that the PyPTO framework itself had certain issues with its graph stitching and splitting functionality, causing modules such as the inverse module to be incorrectly stitched and split according to the frontend representation. After subsequent discussion with the framework developers, the issue was resolved. As shown in Figure 3-5, which presents the swimlane diagram after manual tuning of graph stitching and splitting for B=1, T=128, H=1, the computation logic graph is now highly consistent with the frontend representation. For example, the inverse module is an isomorphic subgraph consisting of eight 16×16 Tile blocks being updated row by row in parallel. PyPTO provides users with visualization DFX capabilities, enabling them to understand the runtime computation flow more simply and intuitively, thereby identifying and resolving issues more efficiently.

Figure 5 Initial out-of-the-box swimlane diagram for B=1, T=128, H=1
![Figure description](../figures/gdr_3.4.png)

Figure 6 Swimlane diagram after manual tuning of graph stitching and splitting for B=1, T=128, H=1
![Figure description](../figures/gdr_3.5.png)

#### unroll_list Optimization

After understanding runtime concepts such as root functions and stitch, we can observe from the swimlane diagram that the GDN operator consists of both parallel and serial computation logic. Due to the current stitching and scheduling strategies of the PyPTO framework, when S is large, the chunk-by-chunk state update struggles to fully occupy all cores. PyPTO provides the unroll_list capability, which fuses the innermost loop graphs. When the unroll count is n, the loop step becomes step x n, and each iteration executes the loop body n times. This effectively improves the parallel computation capability of the outer BN loop. In essence, it reduces the number of loop iterations, the number of root functions, and the number of tasks. Given that the number of tasks dispatched per stitch remains unchanged (stitch_function_max_num=128), the total number of stitches is reduced. As shown in Figure 3-6, compared with Figure 3-3, this is the swimlane diagram when unroll_list is set to [16] in Loop S.

```py
# Original
for s_idx in pypto.loop(0, s, l, name="LOOP_S_TND", idx_name="s_idx"):

# Set unroll_list=[16]
for s_idx in pypto.loop(0, s, l, name="LOOP_S_TND", idx_name="s_idx", unroll_list=[16]):
```

Figure 7 Swimlane diagram with unroll_list=[16]
![Figure description](../figures/gdr_3.6.png)

### DFX-based In-Depth Performance Tuning

As shown in Figure 3-8, in the initial out-of-the-box implementation of GDN, the inverse computation for a [128, 128] tile block was performed by first applying the inverse algorithm to eight [16, 16] blocks. This approach offers strong parallel computation capability in small-shape scenarios. However, as shown in Figure 3-9, when the data volume is large (with AIV Cores already fully utilized), the computation efficiency is not high. Based on the DFX capabilities provided by the compute graph and swimlane diagram, we proposed a tail-axis concatenation optimization scheme: instead of computing eight [16, 16] blocks separately, we concatenate them along the last axis into two [16, 64] blocks for computation, thereby reducing data movement overhead and improving computation efficiency. However, the performance gain did not meet expectations. As shown in Figure 3-7, the improvement was only from 8 × 22 us to 2 × 62 us, which fell short of the anticipated gain. Upon inspecting the computation flow in the compute graph, we found significant copy-in and copy-out overhead, as indicated by the green and red nodes in Figure 3-8.

Figure 8 Swimlane diagram of the initial tail-axis concatenation optimization for the inverse module
![Figure description](../figures/gdr_3.7.png)

Figure 9 Compute graph of the initial tail-axis concatenation optimization for the inverse module (green nodes represent copy-in, red nodes represent copy-out)
![Figure description](../figures/gdr_3.8.png)

By comparing the computation flow in the compute graph with the frontend code, we found that the issue was caused by the row-by-row updated matrix being frequently moved in and out of UB. However, it was expected that this matrix could reside permanently in memory, which would enable deeper optimization. As shown in Figure 3-9, in the optimized compute graph, data is copied in only at the far left of the computation flow and copied out at the far right, which aligns with expectations.

Figure 10 Compute graph of the tail-axis concatenation optimization for the inverse module
![Figure description](../figures/gdr_3.9.png)

Finally, the tail-axis concatenation scheme is optimized to concatenate them into a single [16, 128] block for row-by-row inversion. The resulting swimlane diagram of the final puncture scheme for inverse tail-axis concatenation optimization is shown in Figure 3-10. The inverse module time was reduced from 8 x 28 us to 49 us.

Figure 10 Swimlane diagram of the final tail-axis concatenation optimization for the inverse module
![Figure description](../figures/gdr_3.10.png)
