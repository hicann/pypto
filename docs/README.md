# 项目文档

## 简介

此目录提供PyPTO文档源文件信息，包括环境部署、编程指南、API参考等。

## 目录说明

关键目录结构如下：

```txt
├── install                    # 环境部署
├── invocation                 # 样例运行
├── tutorials                  # PyPTO 编程指南
├── api                        # PyPTO API参考
├── tools                      # PyPTO Toolkit工具用户指南
└── README
```

## 文档构建

 PyPTO的教程和API文档均可由Sphinx工具生成。构建文档前需要安装必要模块，以下是具体步骤。

1. 下载PyPTO仓代码。

   ```bash
   git clone https://gitcode.com/cann/pypto.git
   ```

2. 进入docs目录并安装该目录下`requirements.txt`所需依赖。

   ```bash
   cd docs
   pip install -r requirements.txt
   ```

3. 在docs目录下执行如下命令进行文档构建。

   ```bash
   make html
   ```

4. 构建完成后会新建_build/html目录，执行如下命令启动HTTP服务器以提供文档服务。

   ```bash
   cd _build/html
   python3 -m http.server 8000
   ```

   默认端口8000，也可自行指定端口。

5. 在浏览器中访问`http://localhost:8000`查看文档。