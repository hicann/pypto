# 贡献文档

欢迎参与PyPTO文档贡献，符合要求的文档将会在[PyPTO文档中心]( https://pypto.gitcode.com)中呈现。

本项目支持Markdown格式的内容贡献，对应地可创建`.md`为后缀的文档或修改已存在的文档。


## 更新/新增文档

相关文档写作要求请参考[文档写作要求](https://gitcode.com/cann/community/blob/master/contributor/docs/document_writing_specs.md)。

### 更新文档

如果您需要更新现有文档，可点击页面右上角的编辑按钮，跳转至源文件。修改该文件，并提交PR即可参与贡献。


### 新增文档

如果您需要新增文档，请在docs合适的目录新建Markdown，目录结构说明可参考[目录说明](./README.md#目录说明)。

1. 新建文件。

2. 编辑目录索引文件，将新建文件添加到网页。

    以API文档为例，新增接口时，先在`docs/api`目录下找到[index.md](./api/index.md)文件，该文件即对应API文档的组织结构。

    在对应的分类中添加新建的文件，也可新建分类后再添加。例如：

    ```{toctree}
    :maxdepth: 2

    tensor/index
    element/index
    operation/index
    datatype/index
    controlflow/index
    config/index
    symbolic/index
    newcategory/index 
    ```

    在newcategory/index.md中增加以下内容：

    ```{toctree}
    :maxdepth: 2

    pypto-newapi1
    pypto-newapi1
    ```

完成上述操作后，并提交PR即可参与贡献。

## 本地构建

用户提交PR后，可以在本地使用Sphinx工具自动生成文档，详细请参考[文档构建](./README.md#文档构建)。

## 远端构建

PR合入后，将自动触发网页内容更新，最终您将在[PyPTO文档中心]( https://pypto.gitcode.com)中查看到新增内容。