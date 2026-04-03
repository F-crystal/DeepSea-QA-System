# DeepSea QA System

一个基于大语言模型的深海领域问答系统，支持多模型、多检索策略和消融实验评估。

## 项目目录结构

```
.
├── analysis_tools/          # 分析工具和可视化
├── config/                  # 配置文件
├── deepsea_qa/             # 核心代码库
├── prepare_data/           # 数据准备和预处理
├── qa_dataset_out/         # QA 数据集输出
├── qa_post/               # 数据后处理
├── server/                # 服务器代码
├── ui/                    # 用户界面
├── README.md              # 项目说明文档
└── requirements.txt       # 项目依赖文件
```

## 项目简介

本项目是一个端到端的深海领域问答系统，集成了查询分类、查询改写、多模态检索、答案生成和验证等功能。系统支持多种大语言模型，并提供了完整的评估框架和消融实验支持。

### 核心功能

- **智能查询理解**：支持查询分类和查询改写扩展
- **多模态检索**：结合 BM25 稀疏检索和 FAISS 稠密检索
- **答案生成与验证**：基于检索证据生成答案，并支持反向验证
- **多模型支持**：支持多种大语言模型
- **评估框架**：支持 THELMA 指标、分类指标、BLEU、ROUGE、BERTScore 等多种评估指标
- **消融实验**：支持对系统各模块进行消融实验，验证各组件的有效性

## 系统架构

```
query → query_bundle → retrieval → generation → verification → retry
  ↓          ↓              ↓           ↓            ↓
分类      改写扩展        多模态检索    答案生成      反向验证
```

## 安装与配置

### 环境要求

- Python 3.8+
- CUDA 12.8（如需使用 GPU 加速）

### 依赖说明

- **基础科学计算**：numpy、pandas、openpyxl
- **向量检索**：faiss-cpu（推荐）或 faiss-gpu
- **Embedding 模型**：sentence-transformers、transformers、accelerate（使用 BGE-M3 模型）
- **分词**：jieba
- **接口**：fastapi、uvicorn、pydantic
- **评估**：bert-score（使用 bert-chinese-uncased 模型）
- **其他**：tqdm、scikit-learn、sacrebleu、rouge

### 安装依赖

```bash
# 安装完整项目依赖
pip install -r requirements.txt
```

### 配置 API Key

创建 `config/api_keys.py` 文件，配置所需的 API Key 和服务地址：

```python
# -*- coding: utf-8 -*-
"""
api_keys.py

用途说明：
存储所有API key和隐私信息，避免在代码中硬编码
"""

import os

# API Key 配置
API_KEYS = {
    # OpenBayes API Key
    "OPENBAYES_API_KEY": os.getenv("OPENBAYES_API_KEY", "your_openbayes_key"),
    
    # OpenBayes 服务地址
    "OPENBAYES_BASE_URL": os.getenv("OPENBAYES_BASE_URL", "your_openbayes_base_url"),
    
    # 智谱AI API Key
    "ZHIPU_API_KEY": os.getenv("ZHIPU_API_KEY", "your_zhipu_key"),
    
    # DeepSeek API Key
    "DEEPSEEK_API_KEY": os.getenv("DEEPSEEK_API_KEY", "your_deepseek_key"),
    
    # DashScope API Key
    "DASHSCOPE_API_KEY": os.getenv("DASHSCOPE_API_KEY", "your_dashscope_key"),
    
    # OpenAI API Key
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", "your_openai_key"),
}


def get_api_key(key_name: str) -> str:
    """获取API key
    
    Args:
        key_name: API key名称
        
    Returns:
        API key值
    """
    return API_KEYS.get(key_name, "")


def set_api_key_env():
    """设置API key到环境变量"""
    for key, value in API_KEYS.items():
        if value:
            os.environ[key] = value
```

**注意**：建议使用环境变量方式配置 API Key，避免在代码中硬编码。将实际的 API Key 设置为环境变量，代码中保留默认值即可。

## 快速开始

### 1. 构建知识库索引

```bash
# 构建 BM25 索引
python -m deepsea_qa.run.run_build_bm25

# 构建 FAISS 索引
python -m deepsea_qa.run.run_build_faiss

# 构建完整知识库
python -m deepsea_qa.run.run_build_kb
```

### 2. 运行端到端问答

```bash
# 非流式问答
python -m deepsea_qa.run.run_qa_pipeline \
  --query "深海AUV声学通信怎么做抗多径？" \
  --llm_provider zhipu --llm_model glm-4-plus

# 流式问答
python -m deepsea_qa.run.run_qa_pipeline \
  --query "深海AUV声学通信怎么做抗多径？" \
  --stream \
  --llm_provider zhipu --llm_model glm-4-plus
```

### 3. 运行评估

```bash
# 运行完整评估
python -m deepsea_qa.run.run_eval_pipeline \
  --llm_provider zhipu \
  --llm_model glm-4-plus
```

## 项目结构

```
.
├── analysis_tools/          # 分析工具和可视化
├── config/                  # 配置文件
├── deepsea_qa/             # 核心代码库
├── prepare_data/           # 数据准备和预处理
├── qa_dataset_out/         # QA 数据集输出
├── qa_post/               # 数据后处理
├── server/                # 服务器代码
└── ui/                    # 用户界面
```

## 详细说明

### analysis\_tools/

数据分析工具和可视化脚本，用于分析评估结果。

#### 文件说明

- **analyze\_correlation.py**：分析评估指标之间的相关性
- **analyze\_domain\_confusion.py**：分析领域分类混淆矩阵
- **analyze\_domain\_metrics.py**：分析各领域的评估指标
- **analyze\_domain\_thelma.py**：分析各领域的 THELMA 指标
- **analyze\_response\_time.py**：分析响应时间统计
- **plot\_metrics.py**：绘制评估指标图表
- **process\_eval\_results.py**：处理评估结果数据

#### 子目录

- **data/**：存储分析数据
- **pics/**：存储可视化图表

### config/

项目配置文件，包含 API Key 和路径配置。

#### 文件说明

- **api\_keys.py**：API Key 配置文件

### deepsea\_qa/

核心代码库，包含问答系统的所有模块。

#### 目录结构

```
deepsea_qa/
├── artifacts/              # 生成的模型和索引文件
├── configs/               # 配置类定义
├── data/                  # 数据处理工具
├── eval/                  # 评估模块
├── generation/            # 答案生成模块
├── index/                 # 索引构建模块
├── llm/                   # LLM 接口封装
├── qa/                    # 端到端问答流程
├── query/                 # 查询处理模块
├── retrieval/             # 检索模块
└── run/                   # 运行脚本
```

#### configs/

配置类定义文件。

- **eval\_config.py**：评估配置
- **generation\_config.py**：生成配置
- **index\_config.py**：索引配置
- **paths.py**：路径配置
- **query\_config.py**：查询配置
- **retrieval\_config.py**：检索配置

#### data/

数据处理工具。

- **chunking.py**：文本分块工具

#### eval/

评估模块，包含多种评估指标。

- **api.py**：评估 API 接口
- **classification\_utils.py**：分类评估工具
- **evaluator.py**：端到端评估器
- **thelma/evaluator.py**：THELMA 指标评估器
- **thelma/modules/**：THELMA 评估模块
  - **aggregate.py**：聚合模块
  - **decompose.py**：分解模块
  - **match.py**：匹配模块

#### generation/

答案生成模块。

- **answerer.py**：答案生成器
- **api.py**：生成 API 接口
- **gbt7714.py**：GB/T 7714 参考文献格式化
- **meta\_resolver.py**：元数据解析器
- **pipeline.py**：生成流程
- **types.py**：生成相关数据类型
- **verifier.py**：答案验证器

#### index/

索引构建模块。

- **bm25\_index.py**：BM25 索引构建
- **build\_kb.py**：知识库构建工具
- **faiss\_index.py**：FAISS 索引构建

#### llm/

LLM 接口封装，支持多种大语言模型。

- **base.py**：LLM 基类
- **dashscope.py**：阿里云 DashScope 接口
- **deepseek.py**：DeepSeek 接口
- **openai.py**：OpenAI 接口
- **registry.py**：LLM 注册表
- **zhipu.py**：智谱 AI 接口

#### qa/

端到端问答流程。

- **pipeline.py**：问答流程

#### query/

查询处理模块。

- **api.py**：查询 API 接口
- **classifier.py**：查询分类器
- **label\_cards.py**：标签卡片存储
- **pipeline.py**：查询处理流程
- **rewrite\_expand.py**：查询改写扩展
- **types.py**：查询相关数据类型

#### retrieval/

检索模块，支持多模态检索。

- **api.py**：检索 API 接口
- **boost.py**：检索增强
- **dense.py**：稠密检索
- **fusion.py**：检索结果融合
- **pipeline.py**：检索流程
- **rerank.py**：重排序
- **sparse.py**：稀疏检索
- **types.py**：检索相关数据类型
- **utils.py**：检索工具函数

#### run/

运行脚本，提供各种功能的命令行接口。

- **run\_ablation\_pipeline.py**：运行消融实验
- **run\_build\_bm25.py**：构建 BM25 索引
- **run\_build\_faiss.py**：构建 FAISS 索引
- **run\_build\_kb.py**：构建完整知识库
- **run\_eval\_pipeline.py**：运行评估流程
- **run\_generation\_pipeline.py**：运行生成流程
- **run\_qa\_pipeline.py**：运行端到端问答
- **run\_query\_pipeline.py**：运行查询处理
- **run\_retrieval\_pipeline.py**：运行检索流程

### prepare\_data/

数据准备和预处理工具。

#### 文件说明

- **llm\_labeling.py**：使用 LLM 进行数据标注
- **prepare\_wos\_corpus.py**：准备语料库
- **token\_stats\_chunks.py**：统计分块 token 数量

#### 子目录

- **stopwords/**：停用词表
- **token\_stats/**：token 统计结果
- **分类标签/**：各领域的分类标签
- **题录信息/**：原始题录信息
- **题录信息\_中间结果/**：数据处理中间结果

### qa\_dataset\_out/

QA 数据集输出目录。

#### 文件说明

- **gen\_qa\_dataset.py**：生成 QA 数据集

#### 子目录

- **logs/**：日志文件

### qa\_post/

数据后处理和采样。

#### 文件说明

- **add\_llm\_labels.py**：添加 LLM 标签
- **analyze\_distribution.py**：分析数据分布
- **post\_filter\_qa\_dataset.py**：后处理过滤 QA 数据集
- **sample\_dataset.py**：采样数据集

#### 子目录

- **distribution/**：分布分析结果

### server/

服务器代码，提供 API 接口。

#### 文件说明

- **app.py**：应用主文件
- **requirements.txt**：服务器依赖
- **start.sh**：启动脚本

### ui/

用户界面，提供 Web UI。

#### 文件说明

- **app.py**：应用主文件
- **启动界面.txt**：启动说明

#### 子目录

- **static/**：静态资源
  - **css/style.css**：样式表
  - **js/script.js**：JavaScript 脚本
- **templates/**：HTML 模板
  - **index.html**：主页面

## 消融实验

系统支持多种消融实验类型，用于验证各模块的有效性：

- **no\_cls**：禁用查询分类
- **no\_rewrite**：禁用查询改写
- **no\_cls\_rewrite**：同时禁用查询分类和改写
- **no\_hybrid**：禁用混合检索
- **no\_rerank**：禁用检索重排序
- **no\_reverse\_verification**：禁用反向验证

### 运行消融实验

```bash
# 禁用查询分类
python -m deepsea_qa.run.run_ablation_pipeline \
  --ablation_type no_cls \
  --llm_provider zhipu \
  --llm_model glm-4-plus

# 禁用查询改写
python -m deepsea_qa.run.run_ablation_pipeline \
  --ablation_type no_rewrite \
  --llm_provider zhipu \
  --llm_model glm-4-plus

# 同时禁用分类和改写
python -m deepsea_qa.run.run_ablation_pipeline \
  --ablation_type no_cls_rewrite \
  --llm_provider zhipu \
  --llm_model glm-4-plus

# 禁用混合检索
python -m deepsea_qa.run.run_ablation_pipeline \
  --ablation_type no_hybrid \
  --llm_provider zhipu \
  --llm_model glm-4-plus

# 禁用重排序
python -m deepsea_qa.run.run_ablation_pipeline \
  --ablation_type no_rerank \
  --llm_provider zhipu \
  --llm_model glm-4-plus

# 禁用反向验证
python -m deepsea_qa.run.run_ablation_pipeline \
  --ablation_type no_reverse_verification \
  --llm_provider zhipu \
  --llm_model glm-4-plus
```

## 评估指标

系统支持多种评估指标：

- **THELMA 指标**：综合评估答案质量
- **分类指标**：评估查询分类准确性
- **BLEU**：评估答案与标准答案的相似度
- **ROUGE**：评估答案召回率
- **BERTScore**：基于 BERT 的语义相似度

## 支持的大语言模型

- **智谱 AI**：glm-4-plus、glm-4、glm-3-turbo
- **DeepSeek**：deepseek-chat
- **OpenAI**：gpt-4、gpt-3.5-turbo
- **阿里云**：qwen-turbo、qwen-plus

## 断点续传

系统支持断点续传功能，当评估过程中断时，可以自动从上次中断的地方继续：

```bash
# 自动续传
python -m deepsea_qa.run.run_ablation_pipeline \
  --ablation_type no_cls \
  --llm_provider zhipu \
  --llm_model glm-4-plus

# 指定续传
python -m deepsea_qa.run.run_ablation_pipeline \
  --ablation_type no_cls \
  --llm_provider zhipu \
  --llm_model glm-4-plus \
  --resume
```

## 常见问题

### 1. API Key 配置问题

确保在 `config/api_keys.py` 中正确配置了所有需要的 API Key，或使用环境变量方式配置。

### 2. 依赖安装问题

如果遇到 FAISS 安装问题，建议先使用 `faiss-cpu`，确保系统稳定运行后再考虑 `faiss-gpu`。

### 3. 检索服务问题

如果遇到 HTTP 502 错误，可能是检索服务暂时不可用，请稍后重试。

### 4. 内存不足

如果遇到内存不足问题，可以：

- 减少 `max_sparse_queries` 参数
- 减少 `final_top_n` 参数
- 使用更小的模型

## 贡献指南

欢迎提交 Issue 和 Pull Request 来改进本项目。

## 许可证

本项目仅用于学术研究和教育目的。

## 联系方式

如有问题，请通过 Issue 联系。

## 致谢

感谢所有为本项目做出贡献的开发者和研究人员。
