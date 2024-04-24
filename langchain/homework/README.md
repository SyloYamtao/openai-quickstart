## 搭建运行环境

本项目使用 Python v3.10 开发，完整 Python 依赖软件包见[requirements.txt](requirements.txt)。

**以下是详细的安装指导（以 Ubuntu 操作系统为例）**：

### 安装 Miniconda

```shell
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
```

安装完成后，建议新建一个 Python 虚拟环境，命名为 `homework`。

```shell
conda create -n homework python=3.10

# 激活环境
conda activate homework 
```

之后每次使用需要激活此环境。

### 安装 Python 依赖软件包

#### 在homework路径下运行一下命令

```shell
pip install -r requirements.txt
```

### 配置 OpenAI API Key

根据你使用的命令行工具，在 `~/.bashrc` 或 `~/.zshrc` 中配置 `OPENAI_API_KEY` 环境变量：

```shell
export OPENAI_API_KEY="xxxx"
```

### 运行项目

```shell
python multiple_types_sales_chatbot.py --enable_chat=False
```

| 参数          | 描述                                                                     | 
|-------------|------------------------------------------------------------------------|
| enable_chat | 是否启动chatgpt话术,如果遇到不确定的问题,答案是(True)使用llm回答,还是(False)预先设置好的话术回答,默认为False |

### 启动成功
```shell
/opt/miniconda3/envs/langchain/lib/python3.10/site-packages/langchain/vectorstores/__init__.py:35: LangChainDeprecationWarning: Importing vector stores from langchain is deprecated. Importing from langchain will no longer be supported as of langchain==0.2.0. Please import from langchain-community instead:

`from langchain_community.vectorstores import FAISS`.

To install langchain-community run `pip install -U langchain-community`.
  warnings.warn(
Running on local URL:  http://0.0.0.0:7860
Running on public URL: https://b711648c551d7b5aaa.gradio.live

This share link expires in 72 hours. For free permanent hosting and GPU upgrades, run `gradio deploy` from Terminal to deploy to Spaces (https://huggingface.co/spaces)

```
#### 如果出现`Running on local URL:  http://0.0.0.0:7860`和`Running on public URL: https://b711648c551d7b5aaa.gradio.live`表示启动成功,请在浏览器中输入任意一个访问路径即可访问

### 页面
![img.png](image/img.png)

## 许可证
该项目根据Apache-2.0许可证的条款进行许可。详情请参见[LICENSE](LICENSE)文件。


