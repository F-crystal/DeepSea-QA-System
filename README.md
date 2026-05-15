# DeepSea QA System

基于 RAG 技术的深海科技领域知识问答系统，集成了查询理解、混合检索、结构化生成与反向验证的完整流水线。

## 系统架构

```
用户提问
  │
  ▼
┌─────────────────────────────────────────────────────┐
│  查询理解层                                           │
│  ┌───────────┐   ┌──────────────────┐                │
│  │ 领域分类   │──▶│ 查询改写与扩展    │                │
│  └───────────┘   └──────────────────┘                │
└──────────────────────┬──────────────────────────────┘
                       │ QueryBundle（多查询变体）
                       ▼
┌─────────────────────────────────────────────────────┐
│  混合检索层                                           │
│  ┌────────┐   ┌────────┐   ┌──────┐   ┌──────────┐  │
│  │ BM25   │   │ FAISS  │──▶│ RRF  │──▶│ 领域增强  │  │
│  │ 稀疏检索│   │ 稠密检索│   │ 融合  │   │  Boost   │  │
│  └────────┘   └────────┘   └──────┘   └────┬─────┘  │
└─────────────────────────────────────────────┼───────┘
                                              ▼
                                       ┌──────────┐
                                       │ Cross-   │
                                       │ Encoder  │
                                       │ 重排序    │
                                       └────┬─────┘
                                            ▼
┌─────────────────────────────────────────────────────┐
│  生成与验证层                                         │
│  ┌──────────┐   ┌──────────┐   ┌──────────────────┐  │
│  │ 结构化   │──▶│ 反向验证  │──▶│ 失败则重新检索    │  │
│  │ 答案生成  │   │ 引用核查  │   │ 与生成（闭环）    │  │
│  └──────────┘   └──────────┘   └──────────────────┘  │
└─────────────────────────────────────────────────────┘
```

## 核心特性

### 查询理解
- **领域分类**：将用户问题归类到深海感知与通信、深海可再生能源、深海矿产、深水油气四个领域，支持 LLM + 规则混合策略，输出 Top-K 候选而非单一标签
- **查询改写与扩展**：生成标准化问句、关键词短语、布尔表达式、英文术语等多路变体，仅用于稀疏检索，避免污染稠密向量空间

### 混合检索
- **BM25 稀疏检索**：中英文混合分词，支持多查询 RRF 融合
- **FAISS 稠密检索**：基于 BGE-M3 向量模型
- **RRF 融合**：秩倒数融合稀疏与稠密结果
- **领域增强 Boost**：基于分类置信度的软加权，不过滤以保留召回率
- **Cross-Encoder 重排序**：BGE-Reranker-v2-M3 精排，动态截断低置信度结果

### 生成与验证
- **结构化输出**：答案以 JSON 格式输出，每条引用附带 `chunk_id` 和原文 `quote`
- **GB/T 7714 引用格式**：参考文献按中国国家标准格式化
- **两层反向验证**：基础层校验引用是否存在及原文匹配；进阶层通过 LLM 将答案分解为原子声明，逐一检查来源支持度、覆盖度和相关性
- **闭环自纠正**：验证失败时自动触发重新检索与生成，最多重试 N 次

### 流式交互
- Web UI 支持 SSE 流式输出，先展示证据卡片，再逐步渲染答案

## 评估体系

| 指标类别 | 指标 | 说明 |
|---------|------|------|
| THELMA | Source Precision (SP1/SP2)、Groundedness、Coverage、Response Precision、Self-Distinctness | 基于声明分解与匹配的综合评估 |
| 文本相似度 | BLEU、ROUGE-1/2/L、BERTScore | 中文分词感知，支持中文答案质量评估 |
| 分类准确率 | domain\_correct、label\_correct、rank\_score | 带位置加权的分类评估 |
| 效率 | 响应时间统计 | 单问题端到端耗时 |

## 消融实验

系统支持 6 种消融配置，用于验证各模块的独立贡献：

| 编号 | 消融类型 | 移除模块 |
|------|---------|---------|
| A1 | `no_cls` | 领域分类 |
| A2 | `no_rewrite` | 查询改写与扩展 |
| A3 | `no_cls_rewrite` | 分类 + 改写 |
| A4 | `no_sparse` | BM25 稀疏检索 |
| A5 | `no_rerank` | Cross-Encoder 重排序 |
| A6 | `no_reverse_verification` | 反向验证与自纠正 |

支持断点续传，实验中断后可自动从上次进度继续。

## 支持的 LLM

| 提供商 | 默认模型 | 用途 |
|--------|---------|------|
| 智谱 AI | GLM-4-Plus | 主生成模型 |
| DeepSeek | deepseek-chat | 评估裁判模型 |
| 阿里云 DashScope | qwen-plus | 备选生成模型 |
| OpenAI | gpt-4o | 备选生成模型 |

## 快速开始

### 环境要求

- Python 3.8+
- CUDA（可选，用于 GPU 加速检索与重排序）

### 安装

```bash
pip install -r requirements.txt
```

### 配置 API Key

在 `config/api_keys.py` 中配置所需的 API Key，或通过环境变量设置：

```bash
export ZHIPU_API_KEY="your_key"
export DEEPSEEK_API_KEY="your_key"
# ... 其他 API Key
```

### 构建索引

```bash
python -m deepsea_qa.run.run_build_bm25     # BM25 稀疏索引
python -m deepsea_qa.run.run_build_faiss     # FAISS 稠密索引
python -m deepsea_qa.run.run_build_kb        # 完整知识库
```

### 运行问答

```bash
# 非流式
python -m deepsea_qa.run.run_qa_pipeline \
  --query "深海AUV声学通信怎么做抗多径？" \
  --llm_provider zhipu --llm_model glm-4-plus

# 流式
python -m deepsea_qa.run.run_qa_pipeline \
  --query "深海AUV声学通信怎么做抗多径？" \
  --stream \
  --llm_provider zhipu --llm_model glm-4-plus
```

### 运行评估

```bash
python -m deepsea_qa.run.run_eval_pipeline \
  --dataset qa_post/qa_sampled_dataset.jsonl \
  --llm_provider zhipu --llm_model glm-4-plus
```

### 消融实验

```bash
python -m deepsea_qa.run.run_ablation_pipeline \
  --ablation_type no_cls \
  --llm_provider zhipu --llm_model glm-4-plus

# 断点续传
python -m deepsea_qa.run.run_ablation_pipeline \
  --ablation_type no_cls --resume
```

### 启动服务

```bash
# 后端 API
cd server && uvicorn app:app

# Web UI
cd ui && python app.py
```

## 项目结构

```
.
├── deepsea_qa/                # 核心代码库
│   ├── configs/               # 配置类定义
│   ├── llm/                   # LLM 接口（智谱/DeepSeek/DashScope/OpenAI）
│   ├── query/                 # 查询理解（分类 + 改写扩展）
│   ├── retrieval/             # 检索（BM25 + FAISS + 融合 + Boost + 重排序）
│   ├── generation/            # 生成（结构化答案 + 引用格式化 + 反向验证）
│   ├── eval/                  # 评估（THELMA + 文本相似度 + 分类指标）
│   ├── qa/                    # 端到端流水线编排
│   ├── index/                 # 索引构建
│   ├── data/                  # 数据处理工具
│   └── run/                   # CLI 入口脚本
├── server/                    # FastAPI 后端（检索 + 嵌入 + BERTScore 服务）
├── ui/                        # Flask Web UI（单条/流式/批量问答）
├── analysis_tools/            # 分析与可视化脚本
├── prepare_data/              # 数据准备工具
├── qa_post/                   # 数据后处理
└── config/                    # API Key 配置
```

## 分析工具

```bash
python analysis_tools/ablation_analysis.py           # 消融实验综合分析
python analysis_tools/process_eval_results.py         # 评估结果处理
python analysis_tools/analyze_domain_metrics.py       # 分领域指标分析
python analysis_tools/analyze_domain_confusion.py     # 分类混淆矩阵
python analysis_tools/analyze_correlation.py          # 指标相关性分析
python analysis_tools/analyze_response_time.py        # 响应时间统计
python analysis_tools/plot_ablation_domain_thelma.py  # 消融领域 THELMA 可视化
```

## API 接口

### 后端服务 (`server/app.py`)

| 端点 | 方法 | 说明 |
|------|------|------|
| `/search` | POST | 稠密检索（简化接口） |
| `/retrieve` | POST | 完整混合检索（支持分类 + 多查询变体） |
| `/compute_embedding` | POST | 批量嵌入计算 |
| `/compute_similarity` | POST | 文本相似度计算 |
| `/compute_bertscore` | POST | BERTScore 计算 |
| `/health` | GET | 服务健康检查 |

### Web UI (`ui/app.py`)

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 问答界面 |
| `/api/qa` | POST | 单条问答 |
| `/api/qa/stream` | POST | 流式问答（SSE） |
| `/api/qa/batch` | POST | 批量问答（上传 txt 文件） |

## 许可证

本项目采用 MIT 许可证。
