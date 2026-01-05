# 自定义节点颜色及节点信息

## 功能说明

支持自定义计算图节点的颜色及默认节点信息。

## 操作步骤

1.  在Visual Studio Code工作区，右键单击PyPTO Toolkit插件名称，选择“设置”，进入配置面板界面。
2.  选择“扩展 \> PyPTO Toolkit \> Computational Graph”自定义配置方案。

    支持通过注入JS函数脚本来实现。

    -   节点属性配置函数的输入参数为节点的json内容详情，返回一个key:value格式的字符串数组，节点将根据返回结果进行展示。
    -   节点颜色配置函数的输入参数为节点的json内容详情，返回一个HEX格式的颜色字符串，节点将根据返回结果展示颜色。

    以下是通过自定义配置调整颜色和展示内容的示例：

    ![](../figures/zh-cn_image_0000002521728497.png)

    分别表示：

    -   配置Operation节点展示opcode和magic
    -   配置Operation节点颜色为\#8d6f64
    -   配置Tensor节点展示magic和shape
    -   配置Tensor节点颜色为\#cc9595

3.  清理缓存使得配置生效。

    ![](../figures/zh-cn_image_0000002534151235.png)

4.  页面展示效果。

    ![](../figures/zh-cn_image_0000002489608896.png)

