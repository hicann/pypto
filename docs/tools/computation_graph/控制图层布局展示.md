# 控制图层布局展示

## 功能说明

计算图支持图层控制能力，可以通过图层控制计算图布局方向，设置节点显示类型。

## 前提条件

执行PyPTO程序，生成计算图文件。

## 操作步骤

1.  打开计算图文件，下面提供两种方式，用户任选其一即可：
    -   在Visual Studio Code侧边栏，单击![](../figures/zh-cn_image_0000002491082472.png)图标，打开PyPTO Toolkit，在运行结果界面打开计算图文件\(此种方式仅支持打开文件命名以After开头的图\)。
    -   在Visual Studio Code工作区，右键单击计算图文件，在弹出的菜单中选择“PyPTO Toolkit：打开文件”。

2.  单击计算图界面的![](../figures/zh-cn_image_0000002512322027.png)按钮。

    ![](../figures/zh-cn_image_0000002521609447.png)

3.  设置相应的图层选项。

    ![](../figures/zh-cn_image_0000002489609666.png)

    **表 1**  参数说明

    |参数|说明|
    |--|--|
    |图布局|左右：按照左右结构展示计算图</br>上下：按照上下结构展示计算图|
    |节点显示模式|Tensor+OP：计算图上显示Tensor节点和Operation节点。</br>OP：计算图上仅显示Operation节点。|
    |节点控制|指定Tensor节点和Operation节点上显示的参数信息。例如配置为shape，则仅展示shape信息。|


