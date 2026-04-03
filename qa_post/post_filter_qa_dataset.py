#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
post_filter_qa_dataset.py

作者: 冯冉
创建时间: 2026-02-03
目标: 对 LLM 生成的候选 QA 数据进行“后控过滤”，按三步筛除低质量/低可追溯/低多样性样本：
    Step 1) 计算问题与答案的语义相似度（判断语义关联）
    Step 2) 检测答案与对应原文片段（meta.source_slice）的文本重合率（确保内容来自真实文献）
    Step 3) 利用 Self-BLEU 与重复率阈值过滤冗余表达，提高数据集多样性

输入:
- qa_dataset_out/*.jsonl（gen_qa_dataset.py 输出）
输出:
- qa_post/qa_filtered.jsonl
- qa_post/qa_rejected.jsonl（附 _reject_reason）
- qa_post/qa_scores.csv（每条评分明细）
"""

# 忽略warning
import warnings
warnings.filterwarnings('ignore')

import os
import re
import json
import glob
import random
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd

# ==========================================================
# TODO : I/O 路径与处理规模控制
# ==========================================================

# 输入：所有领域的候选 QA（jsonl）
IN_GLOB = "qa_dataset_out/*.jsonl"

# 输出目录（包含过滤后的 jsonl 与 csv）
OUT_DIR = "." # 当前文件夹
os.makedirs(OUT_DIR, exist_ok=True)

# --------- 处理规模控制（防止一次性跑爆算力）---------
# 全局最多处理多少条候选
GLOBAL_MAX_IN = 10000

# 每个领域（domain）最多处理多少条
DOMAIN_MAX_IN = 2500

# ==========================================================
# TODO : 各步骤阈值配置
# ==========================================================

# ---------- Step1：Q-A 语义一致性 ----------
TH_SIM_QA = 0.45

# ---------- Step2：Answer vs Source（可追溯性）----------
# 跨语言语义对齐阈值（中文答案 vs 英文证据句）
TH_AE_SIM = 0.55

# 词项重合率阈值（用于包含大量英文术语的答案）
TH_COVERAGE = 0.30
TH_JACCARD = 0.12

# “硬证据”覆盖率（数字 / 缩写 / 拉丁术语）
# 该指标主要用于审计解释，不作为硬性过滤条件
TH_HARD_COV = 0.10

# ---------- Step3：多样性 ----------
TH_SELF_BLEU = 0.65
TH_REP2 = 0.25

DO_SELF_BLEU = True
SELF_BLEU_REF_SAMPLE = 30

SEED = 42
random.seed(SEED)

# ==========================================================
# 文本处理工具函数
# ==========================================================

# 英文 token（用于 overlap / BLEU 等）
_word_pat = re.compile(r"[A-Za-z0-9]+(?:[-_/][A-Za-z0-9]+)*")

# 英文句子切分（用于 Step2 的 sentence-level 对齐）
_sent_split = re.compile(r"(?<=[\.\?\!])\s+")

# “硬证据”模式：缩写、数字、模型名、拉丁术语等
_hard_pat = re.compile(
    r"[A-Z]{2,}|[0-9]+(?:\.[0-9]+)?|[A-Za-z]+(?:[-_/][A-Za-z0-9]+)*"
)

def tokenize_en(s: str) -> List[str]:
    """英文 token 化（忽略大小写）"""
    return _word_pat.findall((s or "").lower())

def split_en_sentences(s: str) -> List[str]:
    """
    将英文 source_slice 切分为句子。
    注意：保留长度 >= 20 的句子，避免噪声短句。
    """
    s = (s or "").replace("[SEP]", ". ").strip()
    if not s:
        return []
    return [x.strip() for x in _sent_split.split(s) if len(x.strip()) >= 20]

def hard_tokens(s: str) -> List[str]:
    """
    抽取“硬证据 token”：
    - 数字（如 3.6-MW）
    - 缩写（FOWT, LNG）
    - 拉丁术语 / 模型名（TetraSpar, SIMA）
    """
    return [t.lower() for t in _hard_pat.findall(s or "")]

def bigrams(tokens: List[str]):
    return list(zip(tokens, tokens[1:]))

def jaccard(a: List[str], b: List[str]) -> float:
    """Jaccard 相似度：衡量词项集合重合程度"""
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

def coverage(a: List[str], b: List[str]) -> float:
    """覆盖率：答案 token 中有多少比例能在证据中找到"""
    if not a:
        return 0.0
    sb = set(b)
    return sum(1 for t in a if t in sb) / len(a)

def rep2_rate(tokens: List[str]) -> float:
    """
    二元重复率（rep-2）：
    - 衡量文本中 bigram 的重复程度
    - 值越大，表达越单一
    """
    bg = bigrams(tokens)
    if len(bg) <= 1:
        return 0.0
    return 1.0 - len(set(bg)) / len(bg)

# ==========================================================
# 语义相似度模块（统一使用跨语言模型）
# ==========================================================
class SemanticSim:
    """
    使用 bge-m3：
    - 支持中文 Q/A
    - 支持中文-英文跨语言对齐
    - 用于 Step1 与 Step2
    """
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("BAAI/bge-m3")

    def pair_sim(self, xs: List[str], ys: List[str]) -> List[float]:
        """
        批量计算成对文本的余弦相似度
        """
        x = self.model.encode(xs, normalize_embeddings=True, show_progress_bar=False)
        y = self.model.encode(ys, normalize_embeddings=True, show_progress_bar=False)
        return [float(np.dot(x[i], y[i])) for i in range(len(xs))]

# ==========================================================
# Step2：Answer vs Source 的句子级最大相似度
# ==========================================================
def ae_max_sim(
    sim: SemanticSim,
    answers: List[str],
    slices: List[str]
) -> Tuple[List[float], List[str]]:
    """
    对每条样本：
    - 将英文 source_slice 切成句子
    - 计算 answer 与每个句子的相似度
    - 取最大值作为 sim_ae_max
    - 同时返回最相似的证据句（用于审计）
    """
    sims, best = [], []
    for a, sl in zip(answers, slices):
        sents = split_en_sentences(sl)
        if not sents:
            sims.append(0.0)
            best.append("")
            continue

        aa = [a] * len(sents)
        ss = sim.pair_sim(aa, sents)
        k = int(np.argmax(ss))
        sims.append(float(ss[k]))
        best.append(sents[k])
    return sims, best

# ==========================================================
# Step3：Self-BLEU（多样性控制）
# ==========================================================
def self_bleu(texts: List[str], ref_sample: int = 30) -> List[float]:
    """
    对同一 domain 内的答案计算 Self-BLEU：
    - 每条作为 hypothesis
    - 随机采样若干其它答案作为 references
    """
    from sacrebleu.metrics import BLEU
    bleu = BLEU(effective_order=True)

    toks = [tokenize_en(t) for t in texts]
    n = len(toks)
    scores = []

    for i in range(n):
        idxs = [j for j in range(n) if j != i]
        if not idxs:
            scores.append(0.0)
            continue

        refs = random.sample(idxs, min(ref_sample, len(idxs)))
        hyp = " ".join(toks[i])
        ref = [" ".join(toks[j]) for j in refs]
        scores.append(bleu.sentence_score(hyp, ref).score / 100.0)

    return scores

# ==========================================================
# I/O 工具
# ==========================================================
def load_jsonl(fp):
    with open(fp, "r", encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]

def dump_jsonl(fp, items):
    with open(fp, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

# ==========================================================
# 评分结构（用于导出 CSV）
# ==========================================================
@dataclass
class ScoreRow:
    domain: str
    paper_id: str
    slice_id: str
    sim_qa: float
    sim_ae_max: float
    hard_cov: float
    jaccard: float
    coverage: float
    rep2: float
    self_bleu: float
    verdict: str
    reason: str
    best_evidence_sent: str

# ==========================================================
# 主流程
# ==========================================================
def main():
    # --------- 读取全部候选 ---------
    items = []
    for fp in glob.glob(IN_GLOB):
        items.extend(load_jsonl(fp))

    # 仅保留含有 source_slice 的样本（否则无法审计）
    usable = [it for it in items if (it.get("meta", {}) or {}).get("source_slice")]
    print(f"[INFO] usable={len(usable)}")

    # --------- 预算截断（按 domain + 全局）---------
    dom_cnt, trimmed = {}, []
    for it in usable:
        d = it["meta"].get("domain", "unknown")
        dom_cnt.setdefault(d, 0)
        if dom_cnt[d] >= DOMAIN_MAX_IN:
            continue
        trimmed.append(it)
        dom_cnt[d] += 1
        if len(trimmed) >= GLOBAL_MAX_IN:
            break
    usable = trimmed

    # --------- Step1 & Step2 准备 ---------
    sim = SemanticSim()

    Q = [it["instruction"] for it in usable]
    A = [it["output"] for it in usable]
    S = [it["meta"]["source_slice"] for it in usable]

    # Step1：Q-A
    sim_qa = sim.pair_sim(Q, A)

    # Step2：A-Source（句子级最大）
    sim_ae, best_sent = ae_max_sim(sim, A, S)

    # --------- 初筛（Step1 + Step2 + rep2）---------
    prekeep = [False] * len(usable)
    jac_l, cov_l, rep2_l, hard_l = [], [], [], []

    for i, it in enumerate(usable):
        a = it["output"]
        sl = it["meta"]["source_slice"]

        a_tok = tokenize_en(a)
        sl_tok = tokenize_en(sl)

        jac = jaccard(a_tok, sl_tok)
        cov = coverage(a_tok, sl_tok)
        rep2 = rep2_rate(a_tok)

        # 硬证据覆盖率（用于解释）
        a_hard = hard_tokens(a)
        sl_hard = set(hard_tokens(sl))
        hard_cov = sum(1 for t in a_hard if t in sl_hard) / max(1, len(a_hard)) if a_hard else 0.0

        jac_l.append(jac)
        cov_l.append(cov)
        rep2_l.append(rep2)
        hard_l.append(hard_cov)

        # Step2 通过条件：
        # 跨语言语义对齐通过或传统 overlap 指标通过
        pass_step2 = (sim_ae[i] >= TH_AE_SIM) or (
            (cov >= TH_COVERAGE) and (jac >= TH_JACCARD)
        )

        if sim_qa[i] >= TH_SIM_QA and pass_step2 and rep2 <= TH_REP2:
            prekeep[i] = True

    # --------- Step3：Self-BLEU（仅对初筛通过者）---------
    self_bleu_l = [0.0] * len(usable)
    if DO_SELF_BLEU:
        idxs = [i for i, ok in enumerate(prekeep) if ok]
        for d in set(usable[i]["meta"]["domain"] for i in idxs):
            didx = [i for i in idxs if usable[i]["meta"]["domain"] == d]
            scores = self_bleu([usable[i]["output"] for i in didx], SELF_BLEU_REF_SAMPLE)
            for k, i in enumerate(didx):
                self_bleu_l[i] = scores[k]

    # --------- 最终判定 + 输出 ---------
    kept, rej, rows = [], [], []
    for i, it in enumerate(usable):
        reasons = []

        if sim_qa[i] < TH_SIM_QA:
            reasons.append("low_sim_qa")

        if not (
            (sim_ae[i] >= TH_AE_SIM) or
            ((cov_l[i] >= TH_COVERAGE) and (jac_l[i] >= TH_JACCARD))
        ):
            reasons.append("low_evidence_alignment")

        if rep2_l[i] > TH_REP2:
            reasons.append("high_rep2")

        if prekeep[i] and DO_SELF_BLEU and self_bleu_l[i] > TH_SELF_BLEU:
            reasons.append("high_self_bleu")

        verdict = "keep" if not reasons else "reject"
        if verdict == "keep":
            kept.append(it)
        else:
            it["_reject_reason"] = ";".join(reasons)
            rej.append(it)

        rows.append(ScoreRow(
            domain=it["meta"]["domain"],
            paper_id=it["meta"]["paper_id"],
            slice_id=it["meta"]["slice_id"],
            sim_qa=sim_qa[i],
            sim_ae_max=sim_ae[i],
            hard_cov=hard_l[i],
            jaccard=jac_l[i],
            coverage=cov_l[i],
            rep2=rep2_l[i],
            self_bleu=self_bleu_l[i],
            verdict=verdict,
            reason=";".join(reasons),
            best_evidence_sent=best_sent[i]
        ))

    dump_jsonl(os.path.join(OUT_DIR, "qa_filtered.jsonl"), kept)
    dump_jsonl(os.path.join(OUT_DIR, "qa_rejected.jsonl"), rej)

    pd.DataFrame([asdict(r) for r in rows]).to_csv(
        os.path.join(OUT_DIR, "qa_scores.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    print(f"[OK] keep={len(kept)} reject={len(rej)}")

if __name__ == "__main__":
    main()
