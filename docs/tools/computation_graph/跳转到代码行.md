# 跳转到代码行

## 功能说明

在计算图中，可以跳转到与Operation节点相关的代码行，但并非所有Operation节点都支持此功能。仅在PyPTO程序中显式调用的operation，如matmul、add等，支持跳转。而那些在Pass编译过程中自动插入的Operation则无法实现跳转。

## 操作说明

1.  打开计算图文件，下面提供两种方式，用户任选其一即可：
    -   在Visual Studio Code侧边栏，单击![](../figures/zh-cn_image_0000002491082472.png)图标，打开PyPTO Toolkit，在运行结果界面打开计算图文件\(此种方式仅支持打开文件命名以After开头的图\)。
    -   在Visual Studio Code工作区，右键单击计算图文件，在弹出的菜单中选择“PyPTO Toolkit：打开文件”。

2.  右键单击Operation节点，选择“跳转到代码行”。

    ![](../figures/zh-cn_image_0000002502264604.png)

    显示以下界面：

    ![](../figures/zh-cn_image_0000002534144793.png)

