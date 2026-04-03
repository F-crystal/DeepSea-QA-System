#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_qa_dataset.py

作者: 冯冉
创建时间: 2026-02-03
用途说明: 基于各方向题录语料（corpus_{domain}_clean.xlsx），调用 LLM（GLM / DeepSeek）自动构建候选问答数据集：
    1) 逐文件读取 corpus_{domain}_clean.xlsx（每个文件对应一个 domain）
    2) 仅使用 text 字段作为知识来源（text = title + [SEP] + abstract）
    3) 将 text 按段落优先切片（每片不超过 CHUNK_SIZE 字符），默认只取前 MAX_SLICES_PER_PAPER 片（建议=1）
    4) 对每个切片生成 K 组 QA（question/answer/type/evidence）
    5) 输出 JSONL（instruction/output/meta），并保留 meta.source_slice 供后控使用
    6) 数量控制：每篇上限、每域上限、全局上限；支持断点续跑（跳过已生成 paper_id）；抽样控制

补充说明:
- GLM 使用智谱官方 zai SDK：from zai import ZhipuAiClient
- DeepSeek 使用 OpenAI-compatible HTTP 接口（requests）
- 本脚本只产出候选，严格过滤见 post_filter_qa_dataset.py
"""

# 忽略warning
import warnings
warnings.filterwarnings('ignore')

import os
import re
import json
import time
import glob
import hashlib
from typing import List, Dict, Any, Optional

import pandas as pd
import requests


# ==========================================================
# TODO ：API 配置 & 预算/生成配置
# ==========================================================
# ---- 输入 ----
INPUT_DIR = 'prepare_data/题录信息_中间结果'
INPUT_GLOB = os.path.join(INPUT_DIR, "corpus_*_clean.xlsx")
# ---- 输出 ----
OUT_DIR = "." # 当前文件夹
os.makedirs(OUT_DIR, exist_ok=True)
# ---- 日志 ----
LOG_DIR = os.path.join(OUT_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# provider: "deepseek"
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "deepseek").strip().lower()

# ---- DeepSeek (HTTP) ----
# 从配置文件加载API key
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config.api_keys import set_api_key_env
set_api_key_env()

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "") or os.environ.get("LLM_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat") or os.environ.get("LLM_MODEL", "deepseek-chat")
DEEPSEEK_TEMPERATURE = float(os.environ.get("DEEPSEEK_TEMPERATURE", "0.3"))
HTTP_TIMEOUT = 120

# ---- 生成节奏 ----
SLEEP_S = 0.25

# ---- 数量控制 ----
CHUNK_SIZE = 1024
MAX_SLICES_PER_PAPER = 1
QA_PER_CHUNK = 4
MAX_QA_PER_PAPER = 6

GLOBAL_MAX_QA = 10000
DOMAIN_MAX_QA = 2500

RESUME_SKIP_EXISTING = True
DEDUP_WITHIN_DOMAIN = True

# 每个领域最多参与生成的文献数（抽样上限）
DOMAIN_MAX_PAPERS = 600       

# 随机抽样种子
DOMAIN_SAMPLE_SEED = 42

# ==========================================================
# 工具函数
# ==========================================================
def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

def norm_space(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()

def chunk_text_by_paragraph(text: str, chunk_size: int = 1024) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    if not paras:
        return []

    chunks, buf = [], ""
    for p in paras:
        if len(buf) + len(p) + 1 <= chunk_size:
            buf = (buf + "\n" + p).strip()
        else:
            if buf:
                chunks.append(buf)
            if len(p) > chunk_size:
                for i in range(0, len(p), chunk_size):
                    chunks.append(p[i:i+chunk_size])
                buf = ""
            else:
                buf = p
    if buf:
        chunks.append(buf)
    return chunks

def safe_json_loads(s: str) -> Optional[Any]:
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\[[\s\S]*\]", s)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
        return None

def load_done_paper_ids(out_jsonl: str) -> set:
    done = set()
    if not os.path.exists(out_jsonl):
        return done
    with open(out_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                pid = (obj.get("meta", {}) or {}).get("paper_id", "")
                if pid:
                    done.add(str(pid))
            except Exception:
                continue
    return done

def count_lines(fp: str) -> int:
    if not os.path.exists(fp):
        return 0
    n = 0
    with open(fp, "r", encoding="utf-8") as f:
        for _ in f:
            n += 1
    return n

def infer_domain_from_filename(xlsx_path: str) -> str:
    base = os.path.basename(xlsx_path)
    m = re.match(r"corpus_(.+?)_clean\.xlsx$", base)
    return m.group(1) if m else "unknown"

def sample_new_papers_for_domain(
    df: pd.DataFrame,
    paper_id_col: str,
    done_paper_ids: set,
    max_papers: int,
    seed: int = 42
) -> pd.DataFrame:
    """
    论文友好抽样 + 断点续跑（核心）：
    - 按 paper_id 去重得到 paper 粒度列表
    - 统计已生成过 QA 的 paper（done_paper_ids）
    - 目标：每个领域最多 max_papers 篇参与生成
    - 返回：本次“需要新增生成”的 paper 子集（不包含已生成过的 paper）
    """
    if paper_id_col not in df.columns:
        raise KeyError(f"缺少列 {paper_id_col}，请检查 corpus 文件列名")

    # 规范化 paper_id，避免空格/None 导致断点失效
    df = df.copy()
    df[paper_id_col] = df[paper_id_col].apply(norm_space)
    df = df[df[paper_id_col] != ""]  # 去掉空 paper_id

    # paper 粒度去重（每个 paper 只保留一行用于生成）
    df_unique = df.drop_duplicates(subset=[paper_id_col], keep="first")

    all_paper_ids = df_unique[paper_id_col].tolist()
    done_in_this_domain = set(pid for pid in all_paper_ids if pid in done_paper_ids)

    already_done_n = len(done_in_this_domain)
    remaining_quota = max(0, max_papers - already_done_n)

    # 如果已完成 paper 已达到上限，直接返回空（跳过该领域）
    if remaining_quota <= 0:
        return df_unique.iloc[0:0]

    # 只从未生成过的 paper 里抽样补齐
    df_remaining = df_unique[~df_unique[paper_id_col].isin(done_in_this_domain)]

    if len(df_remaining) <= remaining_quota:
        df_sampled = df_remaining
    else:
        df_sampled = df_remaining.sample(n=remaining_quota, random_state=seed)

    return df_sampled

def get_domain_logger(domain: str):
    """
    获取某一领域的追加式 logger（每个 domain 一个文件）
    """
    import logging

    log_path = os.path.join(LOG_DIR, f"domain_{domain}.log")

    logger = logging.getLogger(f"domain_logger_{domain}")
    logger.setLevel(logging.INFO)

    # 防止重复添加 handler（多次调用时）
    if not logger.handlers:
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fmt = logging.Formatter(
            "[%(asctime)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger

# ==========================================================
# LLM 客户端：GLM(zai) / DeepSeek(http)
# ==========================================================
class LLMClient:
    def __init__(self, provider: str):
        self.provider = provider

        if provider == "deepseek":
            if not DEEPSEEK_API_KEY:
                raise RuntimeError("未设置 DEEPSEEK_API_KEY（或 LLM_API_KEY）")
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def chat(self, system: str, user: str) -> str:
        if self.provider == "glm":
            return self._chat_glm(system, user)
        return self._chat_deepseek(system, user)

    def _chat_deepseek(self, system: str, user: str) -> str:
        url = f"{DEEPSEEK_BASE_URL}/v1/chat/completions"
        headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": DEEPSEEK_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": DEEPSEEK_TEMPERATURE,
        }
        r = requests.post(url, headers=headers, json=payload, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


# ==========================================================
# Prompt 模版
# ==========================================================
def build_gen_prompt(domain: str, qa_per_chunk: int, text_slice: str) -> Dict[str, str]:
    system = f"""
    你是一名{domain}领域的专业知识问答数据集构建专家，负责为中文知识问答系统生成高质量问答对。

    你的任务不是概括或复述某一篇具体论文，而是将文献片段中的信息，转化为**真实用户在知识问答系统中可能提出的领域通用问题**。
    """

    user = f"""
    请基于【给定文献片段】，生成 {qa_per_chunk} 组**中文**问答对，用于构建中文领域知识问答数据集。

    ========================
    【总约束（必须遵守）】
    ========================
    1）所有问题必须使用**中文**表述。
    2）问题中**严禁出现**以下指代方式：
    “本文”“本研究”“该研究”“这篇文章”“作者”等。
    3）问题应面向**领域知识本身**，模拟用户在不了解具体文献的前提下进行检索或提问。
    4）答案必须**完全基于给定文献片段**，不得引入片段之外的事实、背景知识或推断。

    ========================
    【问题类型（type）与句式模板绑定规则】
    ========================

    你生成的每一条问答，必须明确标注 type，
    且 **问题表述必须严格符合该 type 对应的句式风格**。

    --------------------------------
    type = "method"（方法 / 技术路线）
    --------------------------------
    问题句式必须采用以下模式之一（或等价改写）：
    - 在【研究对象 / 场景】中，通常采用哪些方法或技术来解决【具体问题】？
    - 针对【特定问题或需求】，现有研究主要采用了哪些技术手段？
    - 【研究对象】相关研究通常是通过怎样的技术路径来实现【研究目标】的？

    --------------------------------
    type = "finding"（结论 / 研究发现）
    --------------------------------
    问题句式必须采用以下模式之一（或等价改写）：
    - 现有研究表明，【研究对象 / 技术】在【特定条件或场景】下表现出哪些主要特征或结果？
    - 【某种方法或技术】在实验研究或工程应用中取得了哪些主要结果？
    - 针对【研究对象】，相关研究目前形成了哪些主要认识？

    --------------------------------
    type = "definition"（概念 / 定义）
    --------------------------------
    问题句式必须采用以下模式之一（或等价改写）：
    - 在【研究领域】中，什么是【专业术语或技术名称】？
    - 【某项技术或方法】主要指的是什么，其核心作用或含义是什么？

    --------------------------------
    type = "application"（应用场景）
    --------------------------------
    问题句式必须采用以下模式之一（或等价改写）：
    - 【某项技术或方法】通常应用于哪些场景或工程问题？
    - 在【特定环境或条件】下，【某项技术】主要用于解决哪些问题？

    --------------------------------
    type = "limitation"（局限性 / 挑战）
    --------------------------------
    问题句式必须采用以下模式之一（或等价改写）：
    - 【某项技术或方法】在实际应用过程中存在哪些局限性或挑战？
    - 在【特定条件】下，【研究对象或方法】面临哪些主要问题？

    --------------------------------
    type = "comparison"（对比，可选）
    --------------------------------
    仅在文献片段明确涉及对比时使用：
    - 与【方法 A】相比，【方法 B】在【性能 / 应用 / 特点】方面有哪些差异？
    - 不同【技术或方法】在【应用场景】中的效果有何不同？

    ========================
    【数量与覆盖要求】
    ========================
    - 你生成的问答对**必须严格满足以下数量与顺序要求**：

    - 第 1 条：type = "method"
    - 第 2 条：type = "finding"

    如果需要生成超过 2 条（如 qa_per_chunk > 2），则：
    - 第 3 条及之后 **不得再使用 type = "method" 或 type = "finding"**
    - 只能从以下类型中选择：
    - definition
    - application
    - limitation
    - comparison（仅当文献片段明确涉及对比时）

    ========================
    【输出格式要求】
    ========================
    请输出一个 JSON 数组（必须可被 json.loads 解析），每个元素包含：
    - question: 符合上述约束的中文问题
    - answer: 中文答案（建议 2–6 句，必要时可条列）
    - type: 问题类型（method / finding / definition / application / limitation / comparison）
    - evidence: 从文献片段中摘取的 1–2 句原文，用于支持答案

    ========================
    【给定文献片段】
    ========================
    {text_slice}
    """

    return {"system": system, "user": user}



# ==========================================================
# 单域文件生成：按域预算 + 断点续跑
# ==========================================================
def generate_for_domain_file(xlsx_path: str, llm: LLMClient, global_written: int) -> int:
    df = pd.read_excel(xlsx_path)

    domain = infer_domain_from_filename(xlsx_path)
    base = os.path.basename(xlsx_path)
    out_jsonl = os.path.join(OUT_DIR, f"qa_{domain}_{LLM_PROVIDER}.jsonl")

    done_paper_ids = load_done_paper_ids(out_jsonl) if RESUME_SKIP_EXISTING else set()
    domain_written_existing = count_lines(out_jsonl)
    domain_written = domain_written_existing  # 包含历史输出

    logger = get_domain_logger(domain)
    # ---- 开始记录 ---- 
    logger.info("RUN_START")
    logger.info(f"domain={domain}")
    logger.info(f"input_file={base}")
    logger.info(f"provider={LLM_PROVIDER}")
    logger.info(f"DOMAIN_MAX_PAPERS={DOMAIN_MAX_PAPERS}")
    logger.info(f"DOMAIN_MAX_QA={DOMAIN_MAX_QA}")
    logger.info(f"GLOBAL_MAX_QA={GLOBAL_MAX_QA}")
    logger.info(f"existing_papers={len(done_paper_ids)}")
    logger.info(f"existing_qas={domain_written_existing}")

    # =========================
    # 论文友好抽样 + 断点续跑：只生成“新增 paper”的缺口部分
    # =========================
    # 先按 paper_id 抽样（并排除已生成过的 paper）
    df_sampled = sample_new_papers_for_domain(
        df=df,
        paper_id_col="paper_id",
        done_paper_ids=done_paper_ids,
        max_papers=DOMAIN_MAX_PAPERS,
        seed=DOMAIN_SAMPLE_SEED
    )

    total_papers = df["paper_id"].apply(norm_space).replace("", pd.NA).dropna().nunique()

    # -- 日志记录 -- 
    logger.info("SAMPLING")
    logger.info(f"total_papers={total_papers}")
    logger.info(f"already_done_papers={len(done_paper_ids)}")
    logger.info(f"target_papers={DOMAIN_MAX_PAPERS}")
    logger.info(f"new_papers_to_generate={len(df_sampled)}")
    logger.info(f"sample_seed={DOMAIN_SAMPLE_SEED}")

    # -- 控制台输出 -- 
    print(f"[INFO] domain={domain} file={base}")
    print(f"[INFO] out={out_jsonl}")
    print(f"[INFO] total_papers={total_papers}, already_done_papers={len(done_paper_ids)}")
    print(f"[INFO] DOMAIN_MAX_PAPERS={DOMAIN_MAX_PAPERS}, new_papers_to_generate={len(df_sampled)}")
    print(f"[INFO] existing_lines={domain_written_existing}")
    print(f"[INFO] budgets: DOMAIN_MAX_QA={DOMAIN_MAX_QA}, GLOBAL_MAX_QA={GLOBAL_MAX_QA}")

    # 如果该领域已达到“paper 抽样上限”，直接跳过
    if df_sampled.empty:
        print(f"[SKIP] domain={domain} 已达到抽样上限（{DOMAIN_MAX_PAPERS}篇），无需继续生成")
        return global_written

    seen = set()
    mode = "a" if os.path.exists(out_jsonl) else "w"

    with open(out_jsonl, mode, encoding="utf-8") as fw:
        for idx, row in df_sampled.iterrows():
            if domain_written >= DOMAIN_MAX_QA or global_written >= GLOBAL_MAX_QA:
                break

            paper_id = norm_space(row.get("paper_id", "")) or f"{domain}__row{idx}"

            # 双保险：即使抽样没过滤干净，这里也会跳过
            if RESUME_SKIP_EXISTING and paper_id in done_paper_ids:
                continue

            text = norm_space(row.get("text", ""))
            if not text:
                continue

            slices = chunk_text_by_paragraph(text, CHUNK_SIZE)[:MAX_SLICES_PER_PAPER]

            per_paper_items = []
            for slice_idx, sl in enumerate(slices):
                if domain_written >= DOMAIN_MAX_QA or global_written >= GLOBAL_MAX_QA:
                    break
                if len(per_paper_items) >= MAX_QA_PER_PAPER:
                    break

                prompt = build_gen_prompt(domain, QA_PER_CHUNK, sl)

                try:
                    raw = llm.chat(prompt["system"], prompt["user"])
                except Exception as e:
                    print(f"[WARN] LLM call failed: domain={domain} paper={paper_id} err={e}")
                    time.sleep(max(SLEEP_S, 0.5))
                    continue

                items = safe_json_loads(raw)
                if not isinstance(items, list):
                    time.sleep(SLEEP_S)
                    continue

                for it in items:
                    if len(per_paper_items) >= MAX_QA_PER_PAPER:
                        break
                    if domain_written >= DOMAIN_MAX_QA or global_written >= GLOBAL_MAX_QA:
                        break
                    if not isinstance(it, dict):
                        continue

                    q = norm_space(it.get("question", ""))
                    a = norm_space(it.get("answer", ""))
                    t = norm_space(it.get("type", ""))
                    e = norm_space(it.get("evidence", ""))

                    if len(q) < 8 or len(a) < 30:
                        continue

                    if DEDUP_WITHIN_DOMAIN:
                        key = _sha1(q + "||" + a)
                        if key in seen:
                            continue
                        seen.add(key)

                    sample = {
                        "instruction": q,
                        "input": "",
                        "output": a,
                        "meta": {
                            "domain": domain,
                            "paper_id": paper_id,
                            "slice_id": f"{paper_id}__{slice_idx}",
                            "qa_type": t,
                            "evidence": e,
                            "source_slice": sl,
                            "xlsx": base,
                        }
                    }
                    per_paper_items.append(sample)

                time.sleep(SLEEP_S)

            for s in per_paper_items:
                if domain_written >= DOMAIN_MAX_QA or global_written >= GLOBAL_MAX_QA:
                    break
                fw.write(json.dumps(s, ensure_ascii=False) + "\n")
                domain_written += 1
                global_written += 1

            done_paper_ids.add(paper_id)

    new_qas_generated = domain_written - domain_written_existing
    new_papers_generated = len(df_sampled)

    logger.info("RUN_END")
    logger.info(f"domain={domain}")
    logger.info(f"new_papers_generated={new_papers_generated}")
    logger.info(f"new_qas_generated={new_qas_generated}")
    logger.info(f"domain_total_qas={domain_written}")
    logger.info(f"global_total_qas={global_written}")

    print(f"[OK] domain={domain} total_domain_lines={domain_written} global_written={global_written}")
    return global_written

# ==========================================================
# Main：跨域全局预算控制
# ==========================================================
def main():
    llm = LLMClient(LLM_PROVIDER)

    files = sorted(glob.glob(INPUT_GLOB))
    if not files:
        raise RuntimeError(f"未找到输入文件：{INPUT_GLOB}")

    global_written = 0
    for fp in files:
        if global_written >= GLOBAL_MAX_QA:
            break
        global_written = generate_for_domain_file(fp, llm, global_written)

    print(f"[DONE] global_written={global_written} (budget={GLOBAL_MAX_QA})")


if __name__ == "__main__":
    main()
