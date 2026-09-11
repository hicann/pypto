# Overview

<!-- md-trans-meta sourceCommit=d92d51ac286ac6802735f52dae2fade039bbc9b3 translatedAt=2026-08-11T09:54:27.077Z pushedAt=2026-09-04T09:54:32.342Z -->

PyPTO (pronounced: pie P-T-O) is an efficient programming framework launched by CANN for AI accelerators, designed to simplify the operator development process while maintaining high-performance computing capabilities. The framework adopts the innovative PTO (Parallel Tensor/Tile Operation) programming paradigm, with the Tile-based programming model as its core design philosophy. Through multi-level compute graph representation, it gradually compiles AI models built by users through APIs from high-level tensor compute graphs into hardware instructions, ultimately generating code that can be efficiently executed on the target platform, which is then executed on the device side using the MPMD (Multiple Program Multiple Data) scheduling method.

## Core Architecture

The PyPTO framework adopts a layered architecture design, spanning from the user API to the underlying hardware execution, and is divided into the following layers:

![](../figures/pypto_architecture.png)

- User Interface Layer: Serves as the interface layer between the PyPTO framework and developers. It provides a Python-friendly programming interface, enabling developers to express computation logic in an intuitive manner without needing in-depth knowledge of the underlying hardware implementation details.
- Compute Graph Compile Layer: PyPTO adopts a multi-level compute graph representation, supporting the optimization and transformation of compute graphs across multiple abstraction levels, from high to low.

    - Tensor Graph: High-level tensor operations, close to the mathematical expressions of algorithm designers.
    - Tile Graph: Hardware-aware tile operations that fully leverage hardware parallelism and memory hierarchy.
    - Block Graph: Subgraph partitioning that supports parallel execution and resource management.
    - Execute Graph: An execution graph that contains dependency relationships and scheduling information.

    The compilation process is implemented through modular passes. Each stage consists of multiple passes, each responsible for optimization or transformation tasks at a specific stage.

    - Tensor Graph phase: Implements hardware-independent graph optimization, including redundant operation elimination, type conversion, and memory conflict inference.
    - Tile Graph phase: Performs tile expansion based on TileShape and implements tile-level optimization, including memory type allocation, move operation generation, and subgraph partitioning.
    - Block Graph phase: Partitions and generates computation subgraphs, and performs Block-level optimization, including out-of-order scheduling, memory reuse, and synchronization point insertion.
    - Execute Graph phase: Integrates computation subgraph information and orchestrates the generation of the final Execute Graph.

- Code Generation Layer: Converts the optimized compute graph into executable code for the target platform.
    - Virtual instruction generation: Generates PTO virtual instructions from the Execute Graph.
    - Target platform compilation: Compiles virtual instructions into target platform code.

- Scheduling & Execution Layer: Responsible for scheduling and executing executable code on the device.
    - MPMD scheduling: Schedules executable code to device processor cores through MPMD on the device.
    - Control flow execution: Manages task dependencies and executes control flow logic.

## Core Features

- Technical innovations:
    - Tile-based programming model: Computation is performed based on tiles (hardware-aware data blocks), fully leveraging the parallel computing capabilities and memory hierarchy of the hardware.
    - Multi-level compute graph representation and optimization: The Compute Graph Compile Layer transforms the Tensor Graph into a Tile Graph, Block Graph, and Execute Graph, with each step involving a series of pass optimization processes.
    - Automated code generation: The compilation result is converted into PTO virtual instructions through the code generation layer, and then compiled into executable code for the target platform by the compiler.
    - MPMD execution scheduling: Executable code is loaded to the device side and scheduled to device processor cores through the MPMD scheduling method, achieving efficient parallel execution.
    - Complete toolchain support: Intermediate compilation artifacts and runtime performance data across the entire process can be visualized through the IDE-integrated toolchain to identify performance bottlenecks. Developers can also control compilation and scheduling behaviors through the toolchain.
    - Python-friendly API: Provides intuitive Tensor-level abstraction that is algorithm-friendly to developers' thinking patterns, supporting dynamic shapes and symbolic programming.
    - Layered abstraction design: Exposes different abstraction levels to different developers. Algorithm developers use the tensor level, performance experts use the tile level, and system developers use the block level.

## Scenarios

PyPTO is applicable to the following scenarios:

- Deep learning operator development: Quickly implements various neural network operators.
- Foundation model development: Supports foundation model components such as Attention, MoE, and FFN.
- Dynamic shape processing: Supports dynamic shape scenarios such as dynamic batch size.

## Design Philosophy

Traditional model development typically involves algorithm developers and operator developers. This division of labor stems from the complexity of high-performance operator development: operator developers must not only understand the mathematical computation attributes of operators, but also consider how to transform them into hardware-friendly execution approaches. This is similar to the early CPU era, when out-of-order execution and compiler technologies were not yet mature, and programmers had to manually arrange pipeline instructions.

To reduce this complexity, PyPTO proposes a new programming framework design philosophy that aims to simplify the operator development process while preserving the potential for high-performance computing.

- Computation Layer Design

    The design philosophy of the computation layer is to stay as close as possible to the mathematical expressions of algorithm designers, using tensors rather than individual elements to describe computation processes. The AI models built by users through APIs are expressed as a Tensor Graph, a design that preserves maximum optimization potential, including:

    - Memory layout optimization: Automatically optimizes the arrangement of data in memory.
    - Data transfer optimization: Minimizes data movement between different memory hierarchies.
    - Multi-operator joint optimization: Identifies and fuses optimizable operator combinations.

    By using tensor as the basic data unit, the computation layer can express complex mathematical operations more naturally while providing rich information for subsequent compilation optimization.

- Compilation Layer Design

    The compilation layer is the critical link between the computation layer and the execution layer, responsible for converting Tensor Graph into a hardware-friendly execution form. The compilation process is implemented through a multi-stage Lowering Pipeline:

    - Tensor Graph to Tile Graph: Through compilation passes, tensor operations are converted into tile operations, tiling strategies are selected, and layout transformation, tile fusion, and tile reordering are performed.
    - Tile Graph to Block Graph: The tile graph is partitioned into subgraphs, isomorphic subgraphs are detected, the Block Graph is normalized, and dependencies are tracked.
    - Block Graph to Execute Graph: The execution graph is built, with dependencies between Block Graphs analyzed, global resources planned, and scheduling hints generated.

    Each stage includes multiple optimization passes. Through a modular graph transformation and optimization pipeline, the optimization space preserved by the computation layer is converted into actual performance gains.

    The compilation layer provides the following core capabilities:

    - Fast availability: Ensures that runnable results are generated at the earliest opportunity, meeting the needs of rapid development.
    - Flexible tuning: Supports performance-sensitive configuration adjustments, allowing developers to optimize based on actual requirements.
    - Deep optimization: Allows advanced users to deeply customize the compilation process for ultimate performance.

- Execution Layer Design

    The execution layer is responsible for converting compiled code into hardware-friendly instructions and executing them. The execution process includes:

    - Code generation: The compilation result is used by CodeGen to generate low-level PTO virtual instructions.
    - Target platform compilation: The virtual instructions are compiled by the compiler into executable code for the target NPU platform.
    - MPMD scheduling: The executable code is loaded to the device side and scheduled to the processor cores on the device through MPMD scheduling.

    Through automated code generation technology, the execution layer can automatically generate optimal execution instructions based on hardware characteristics, fully unleashing hardware computing power. This design avoids the complexity of manually tuning hardware instructions in traditional operator development while ensuring high-performance computing.

- Toolchain Design

    PyPTO provides comprehensive toolchain support, including:

    - Compilation intermediate artifact visualization: Supports saving intermediate artifacts (compute graphs) at different compilation stages (such as Tensor Graph, Tile Graph, Block Graph, and Execute Graph) for debugging and analysis.
    - Runtime performance analysis: Collects and visualizes runtime performance data (swimlane diagram) to help identify performance bottlenecks.
    - Compilation and scheduling control: Developers can control the execution of compilation passes and scheduling behavior through the toolchain, enabling deep customization.

    Through the preceding design philosophy, PyPTO enables efficient collaboration between algorithm development and operator development, significantly reducing the complexity of operator development while retaining high-performance computing capabilities.

## Supported Product Models

PyPTO is supported on the following product models:

- Ascend 950PR/Ascend 950DT
- Atlas A3 training products/Atlas A3 inference products
- Atlas A2 training products/Atlas A2 inference products
