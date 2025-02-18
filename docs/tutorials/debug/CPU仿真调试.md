# CPU仿真调试

在不具备昇腾设备时，也支持在CPU仿真环境中进行测试体验，并支持用户查看算子的核内流水数据。

若仅需要运行仿真，而且当前环境没有昇腾设备，请勿安装torch\_npu，否则可能运行失败。

## 运行模式选择逻辑

-   手动指定仿真模式：

    在算子代码中显式调用`@pypto.jit(runtime_options={"run_mode": 1})`，强制启用CPU仿真模式执行算子程序。

-   自动识别模式：
    -   未检测到CANN软件包：自动启用仿真模式（无需显式配置）。
    -   检测到CANN软件包：优先使用真实硬件执行，仿真模式不生效。

## 操作步骤

1.  指定运行参数run\_mode。

    ```python
    @pypto.jit(runtime_options={"run_mode": 1})
    ```

2.  执行算子，自动触发仿真运行

    ```bash
    python examples/hello_world/hello_world.py --run_mode=sim
    ```

3.  执行成功，在output目录下，生成以下文件信息。

    ![](../figures/zh-cn_image_0000002527468273.png)

4.  右键单击merged\_swimlane.json文件，在弹出的菜单中选择“使用PyPTO Toolkit打开”。

    ![](../figures/zh-cn_image_0000002495188648.png)

    泳道图展示每个核内任务调试情况，包含执行耗时、空闲间隔等，可根据具体情况对算子进行调优，如调整张量的分块形状。

