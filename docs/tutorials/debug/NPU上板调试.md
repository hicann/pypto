# NPU上板调试

如果图编译或图执行流程中出现错误或结果不符合预期，可以启用调试模式，生成不同阶段的计算图文件。计算图描述PyPTO程序计算流程的结构，由多个计算节点和数据节点组成。它通过有向无环图（DAG）的形式表示数据流动和计算逻辑，表征了PyPTO程序从抽象计算描述到硬件执行的完整编译流程。本节将介绍如何采集并查看计算图，并展示图中的关键信息。

## 开启调试模式

1.  开启图编译阶段调试模式开关。

    ```python
    @pypto.jit(
        debug_options={"compile_debug_mode": 1}
    )
    ```

2.  执行用例

    ```bash
    python3 examples/02_intermediate/operators/softmax/softmax.py
    ```

3.  执行成功，在$\{work\_path\}/output/output\_\*/目录（\*代表时间戳）下生成不同阶段的计算图文件（.json 格式）。

    ```txt
    ├── Pass_xx_xx
    │   ├── After_004_ExpandFunction_TENSOR_s0_Unroll1_PATH0_4.json # pass优化后的计算图文件
    │   ├── After_004_ExpandFunction_TENSOR_s0_Unroll1_PATH0_4.tifwkgr # 用户暂不需要关注
    │   ├── Before_004_ExpandFunction_TENSOR_s0_Unroll1_PATH0_4.json # pass优化前的计算图文件
    │   ├── Before_004_ExpandFunction_TENSOR_s0_Unroll1_PATH0_4.tifwkgr # 用户暂不需要关注
    │   └── ExpandFunctionTENSOR_s0_Unroll1_PATH0_4.log
    ├── program.json # 记录function name, semantic label等静态信息
    ├── ...
    ```

## 查看计算图

下面将选取计算图各编译阶段的最后一张计算图，并使用PyPTO Toolkit可视化工具，帮助用户了解各类计算图上的关键信息，帮助开发者进行问题定位。

-   `Tensor Graph`：Before\_004\_ExpandFunction\_TENSOR\_loop\_0\_Unroll1\_PATH0\_hiddenfunc0\_8.json
-   `Tile Graph`：Before\_026\_SubgraphToFunction\_TENSOR\_loop\_0\_Unroll1\_PATH0\_hiddenfunc0\_8.json
-   `Block Graph`：After\_036\_CodegenPreproc\_TENSOR\_loop\_0\_Unroll1\_PATH0\_hiddenfunc0\_8\_LEAF\_program\_id\_00\_15536366383870408930.json
-   `Execute Graph`：After\_036\_CodegenPreproc\_TENSOR\_loop\_0\_Unroll1\_PATH0\_hiddenfunc0\_8\_ROOT.json

1.  通过PyPTO Toolkit查看Tensor Graph。

    右键单击Before\_004\_ExpandFunction\_TENSOR\_loop\_0\_Unroll1\_PATH0\_hiddenfunc0\_8.json文件，在弹出的菜单中选择“使用PyPTO Toolkit打开”。

    ![](../figures/zh-cn_image_0000002499728650.png)

    右上角可以看到计算图类型为Tensor Graph，Tensor Graph由Tensor和Operation组成，图中的Tensor Shape和代码定义一致，且没有经过Tile展开。

2.  通过PyPTO Toolkit查看Tile Graph。

    右键单击Before\_026\_SubgraphToFunction\_TENSOR\_loop\_0\_Unroll1\_PATH0\_hiddenfunc0\_8.json文件，在弹出的菜单中选择“使用PyPTO Toolkit打开”。

    ![](../figures/zh-cn_image_0000002499888764.png)

    右上角可以看到计算图类型为Tile Graph，相比于Tile展开前，Tile Graph中增加了很多节点，这是因为原Shape为\(-1, 32, 1, 256\)的Tensor经过Tile展开，切分成Shape为\(1, 4 ,1, 64\)的Tile。同时，为Tile分配内存层级（对应图中asis-原地址，tobe-目的地址），并自动插入内存搬运节点（对应图中TILE\_COPY\_IN和TILE\_COPY\_OUT）。

3.  通过PyPTO Toolkit查看Block Graph。

    右键单击After\_036\_CodegenPreproc\_TENSOR\_loop\_0\_Unroll1\_PATH0\_hiddenfunc0\_8\_LEAF\_program\_id\_00\_15536366383870408930.json文件，在弹出的菜单中选择“使用PyPTO Toolkit打开”。

    ![](../figures/zh-cn_image_0000002531608703.png)

    右上角可以看到计算图类型为Block Graph，在Block Graph阶段，Tile Graph被切成若干子图，每一个子图对应一个Block Graph，因此相比Tile Graph，Block Graph的规模大量减少。

    当前sample被切分成多个结构相同的子图（简称同构子图），因此Pass\_36\_CodegenPreproc目录下仅有一个文件名称包含After\_036\_CodegenPreproc\_\*\_**LEAF**\_\*关键字的json文件。

4.  通过PyPTO Toolkit查看Execute Graph。

    右键单击After\_036\_CodegenPreproc\_TENSOR\_loop\_0\_Unroll1\_PATH0\_hiddenfunc0\_8\_ROOT.json文件，在弹出的菜单中选择“使用PyPTO Toolkit打开”。

    ![](../figures/zh-cn_image_0000002499728842.png)

    右上角可以看到计算图类型为Execute Graph，Execute Graph中包含Tensor节点和调用节点（带有fx标识，表示对Block Graph进行一次调用），双击调用节点，可以查看对应的Block Graph子图信息，了解具体的执行过程。

