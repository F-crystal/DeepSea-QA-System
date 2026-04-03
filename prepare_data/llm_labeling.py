#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
llm_labeling.py

作者: 冯冉
创建时间: 2026-02-09

用途说明:
  端到端：读取四方向 WoS 清洗结果（题录信息_中间结果/corpus_{domain}_clean.xlsx），
  自动检测分类标签 JSON（分类标签/{domain}.json），用 DeepSeek LLM 进行多标签分类，
  支持并发、断点续跑、缓存结果，并导出 Excel 结果用于知识库/知识问答。

关键设计（减少 *_OTHER 的策略）:
  1) 关键词预路由：利用 label card 的 positive_keywords 对文本做轻量匹配打分，
     先挑选 TopK 候选标签（排除 *_OTHER），让 LLM 在小集合中决策，降低“兜底”概率。
  2) 两阶段判别：若第一轮仍输出 *_OTHER，则第二轮强制在候选集合内选择（除非确无证据）。
  3) 只有在确无信息或两轮失败时，才兜底到 *_OTHER。

依赖:
  pip install pandas openpyxl openai

运行示例:
  export DEEPSEEK_API_KEY="你的key"
  python llm_labeling.py --workers 6

仅跑深水油气:
  python llm_labeling.py --domains 深水油气 --workers 6

参数说明:
  --in_dir       题录信息_中间结果
  --labels_dir   分类标签
  --out_dir      题录信息_中间结果
  --cache_dir    题录信息_中间结果/llm_cache
  --workers      并发线程数
  --sleep        每条请求后 sleep 秒（防限流，建议 0.1~0.3）
  --limit        每个方向仅处理前 N 条（调试用）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

try:
    from openai import OpenAI  # DeepSeek OpenAI-compat
except Exception:
    OpenAI = None

# 忽略warning
import warnings
warnings.filterwarnings('ignore')

# =========================
# 0) 参数配置区
# =========================
# 从配置文件加载API key
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config.api_keys import set_api_key_env
set_api_key_env()

# 可处理的方向列表（ALL 模式下自动遍历）
DOMAINS_ALL = ["深水油气", "深海矿产", "深海可再生能源", "深海感知与通信装备"]

# 默认路径（可用命令行覆盖）
IN_DIR_DEFAULT = Path("题录信息_中间结果")
LABELS_DIR_DEFAULT = Path("分类标签")
OUT_DIR_DEFAULT = Path("题录信息_中间结果")
CACHE_DIR_DEFAULT = OUT_DIR_DEFAULT / "llm_cache"

# DeepSeek OpenAI-compat 默认模型与 base_url
MODEL_DEFAULT = "deepseek-chat"
BASE_URL_DEFAULT = "https://api.deepseek.com"

# 注意：最稳版本不在代码中硬编码/覆盖 API Key；仅从环境变量或 --api_key 读取

# 输出策略：1 个主标签 + 最多 N 个次标签
MAX_SECONDARY_DEFAULT = 2

# 稳定性：限流/网络抖动重试相关
SLEEP_PER_CALL_DEFAULT = 0.15
MAX_RETRIES_DEFAULT = 3

# 减少 *_OTHER 的关键参数
CAND_TOPK_DEFAULT = 5            # 关键词预路由 TopK 候选标签数
CAND_MIN_SCORE_DEFAULT = 1       # 候选最小命中阈值（小于该值则不提供候选）
TEXT_TRUNC_CHARS_DEFAULT = 4500  # 输入截断长度上限，避免超长导致失败

# 进度汇报频率
REPORT_EVERY_DEFAULT = 50        # 每处理 N 条新任务输出一次进度日志


# =========================
# 1) 工具函数
# =========================
def sha1(s: str) -> str:
    """计算字符串 SHA-1，用于缓存键/文本指纹。"""
    return hashlib.sha1((s or "").encode("utf-8", errors="ignore")).hexdigest()


def strip_code_fence(s: str) -> str:
    """
    去除模型可能包裹的 Markdown code fence（```json ... ```），便于 JSON 解析。
    """
    s = (s or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_+-]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


def safe_json_loads(s: str) -> Optional[Dict[str, Any]]:
    """
    安全 JSON 解析：解析失败返回 None。
    """
    try:
        return json.loads(strip_code_fence(s))
    except Exception:
        return None


def truncate_text(text: str, max_chars: int) -> str:
    """
    文本截断：避免 LLM 输入过长导致超时或被拒。
    """
    text = str(text or "")
    return text if len(text) <= max_chars else (text[:max_chars] + " …")


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    """
    追加写入 JSONL：一行一个 JSON 对象。
    说明：并发模式下必须由主线程调用，避免多线程同时写同一文件造成行交错。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_jsonl_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    """
    读取 jsonl 缓存：每行一个 record，返回 {cache_key: record}
    如果缓存中存在重复 cache_key，以最后一次出现为准（便于覆盖修复）。
    """
    cache: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return cache

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                k = obj.get("cache_key")
                if k:
                    cache[k] = obj
            except Exception:
                # 遇到损坏行或非 JSON 行时跳过，保证整体可用
                continue

    return cache


# =========================
# 2) 读取 Label Cards（封闭接口）——通用版：自动识别 *_OTHER
# =========================
class LabelPack:
    """
    标签包：封装 label cards 的关键信息（允许集合、id->中文名映射、兜底类识别、关键词表）。
    """

    def __init__(self, domain: str, version: str, labels: List[Dict[str, Any]]):
        self.domain = domain
        self.version = version
        self.labels = labels

        # 允许的 label_id 列表（用于校验与约束输出）
        self.allowed_ids = [x["label_id"] for x in labels if x.get("label_id")]
        self.id2name = {x["label_id"]: x.get("label_name_zh", "") for x in labels if x.get("label_id")}

        # 自动识别兜底类：支持任意 *_OTHER、OTHER
        self.other_ids = [
            lid for lid in self.allowed_ids
            if (lid.endswith("_OTHER") or lid == "OTHER" or lid.endswith("OTHER"))
        ]
        self.has_other = len(self.other_ids) > 0
        self.fallback_id = self.other_ids[0] if self.has_other else self.allowed_ids[0]

        # 关键词表：用于预路由（英文优先，兼容中文）
        self.pos_keywords: Dict[str, List[str]] = {}
        for x in labels:
            lid = x.get("label_id")
            if not lid:
                continue

            kws_zh = [str(kw).strip().lower() for kw in (x.get("positive_keywords") or []) if str(kw).strip()]
            kws_en = [str(kw).strip().lower() for kw in (x.get("positive_keywords_en") or []) if str(kw).strip()]
            self.pos_keywords[lid] = list(dict.fromkeys(kws_en + kws_zh))


def load_label_pack(labels_fp: Path) -> LabelPack:
    """
    从标签 JSON 文件构造 LabelPack。
    """
    obj = json.loads(labels_fp.read_text(encoding="utf-8"))
    labels = obj.get("labels", [])
    if not labels:
        raise ValueError(f"标签json缺少 labels 或为空：{labels_fp}")

    version = obj.get("schema", {}).get("version", "unknown")
    domain = obj.get("meta", {}).get("domain", labels_fp.stem)
    return LabelPack(domain=domain, version=version, labels=labels)


def build_label_cards_text(pack: LabelPack) -> str:
    """
    压缩版 Label Cards（减少 token）：
    - 保留 domain/version/允许集合/兜底标签声明
    - 每个 label 保留：id/name/scope + 部分规则与关键词（截断避免过长）
    """
    lines = [
        f"【领域】{pack.domain}",
        f"【标签版本】{pack.version}",
        "【允许的label_id集合】" + ", ".join(pack.allowed_ids),
        f"【兜底标签】{pack.fallback_id}（仅在信息不足/无法匹配任何标签时使用）",
        "【标签定义卡】"
    ]

    for lc in pack.labels:
        lines.append(f"- label_id: {lc.get('label_id')}")
        lines.append(f"  name: {lc.get('label_name_zh')}")
        lines.append(f"  scope: {lc.get('scope_description')}")

        inc = lc.get("inclusion_rules") or []
        exc = lc.get("exclusion_rules") or []
        pos = lc.get("positive_keywords") or []
        neg = lc.get("negative_keywords") or []
        pri = lc.get("priority_rules") or []

        if inc:
            lines.append("  include_rules: " + "；".join([str(x) for x in inc[:5]]))
        if exc:
            lines.append("  exclude_rules: " + "；".join([str(x) for x in exc[:5]]))
        if pos:
            lines.append("  positive_keywords: " + "，".join([str(x) for x in pos[:15]]))
        if neg:
            lines.append("  negative_keywords: " + "，".join([str(x) for x in neg[:12]]))
        if pri:
            lines.append("  priority_rules: " + "；".join([str(x) for x in pri[:6]]))

    return "\n".join(lines)


# =========================
# 3) 关键词预路由（减少 *_OTHER）
# =========================
def keyword_route_candidates(
    pack: LabelPack,
    text: str,
    topk: int,
    min_score: int
) -> List[str]:
    """
    使用 positive_keywords 对 text 做轻量命中计数，选 TopK 候选标签。

    规则：
      - 排除兜底类（pack.other_ids）不参与候选
      - 若最高得分低于 min_score，则返回所有可能相关的标签，而不是空
      - 改进匹配逻辑：短词（<=3）用词边界匹配，长词允许子串匹配
      - 优先考虑得分高的标签，但确保至少返回一些候选
    """
    t = (text or "").lower()
    scores: List[Tuple[str, int]] = []

    for lid in pack.allowed_ids:
        if lid in pack.other_ids:
            continue

        kws = pack.pos_keywords.get(lid, [])
        if not kws:
            continue

        score = 0
        for kw in kws:
            k = str(kw).strip().lower()
            if not k:
                continue

            if len(k) <= 3:
                # 短词使用边界匹配，但允许更灵活的匹配方式
                if re.search(rf"\b{re.escape(k)}\b", t) is not None:
                    score += 2  # 短词匹配权重更高
            else:
                # 长词允许子串匹配，提高匹配率
                if k in t:
                    score += 1

        if score > 0:
            scores.append((lid, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    
    # 如果有得分大于0的标签，至少返回一个，即使低于min_score
    if scores:
        # 先返回得分大于等于min_score的标签
        valid_candidates = [lid for lid, score in scores if score >= min_score]
        if valid_candidates:
            return valid_candidates[:topk]
        # 如果没有足够分数的标签，返回得分最高的几个
        else:
            return [lid for lid, _ in scores[:topk]]
    
    # 如果没有任何匹配，返回所有非兜底标签（避免空列表）
    return [lid for lid in pack.allowed_ids if lid not in pack.other_ids][:topk]


# =========================
# 4) LLM 调用与校验（封闭输出契约）
# =========================
def validate_pred(pred: Dict[str, Any], pack: LabelPack, max_secondary: int) -> Tuple[bool, str]:
    """
    校验并规范化模型输出：
      - 必须包含 4 个字段
      - primary_label_id 必须在允许集合
      - secondary_label_ids 必须为 list，且去重/去 primary/截断到 max_secondary
      - confidence 强制转 float 并 clamp 到 [0,1]
      - evidence_spans 必须为 list，并截断长度与数量
    """
    need = ["primary_label_id", "secondary_label_ids", "confidence", "evidence_spans"]
    for k in need:
        if k not in pred:
            return False, f"缺少字段：{k}"

    primary = pred.get("primary_label_id")
    if primary not in pack.allowed_ids:
        return False, f"primary_label_id 不在允许集合：{primary}"

    secondary = pred.get("secondary_label_ids")
    if not isinstance(secondary, list):
        return False, "secondary_label_ids 必须是 list"

    cleaned: List[str] = []
    for x in secondary:
        if x == primary:
            continue
        if x in pack.allowed_ids and x not in cleaned:
            cleaned.append(x)
    pred["secondary_label_ids"] = cleaned[:max_secondary]

    try:
        conf = float(pred.get("confidence"))
    except Exception:
        return False, "confidence 不是数值"
    pred["confidence"] = max(0.0, min(1.0, conf))

    ev = pred.get("evidence_spans")
    if not isinstance(ev, list):
        return False, "evidence_spans 必须是 list"
    pred["evidence_spans"] = [str(x)[:200].strip() for x in ev[:6] if str(x).strip()]

    return True, "ok"


def llm_classify(
    client: Any,
    model: str,
    pack: LabelPack,
    label_cards_text: str,
    paper_id: str,
    text: str,
    candidates: List[str],
    max_secondary: int,
    max_retries: int,
    sleep_s: float
) -> Dict[str, Any]:
    """
    两阶段策略：
      - 第一阶段：提供候选作为“优先考虑集合”，允许选候选外（证据充分时）
      - 第二阶段：若第一阶段输出兜底类且 candidates 非空，则强制主标签在 candidates 内选择
    """
    text = truncate_text(text, TEXT_TRUNC_CHARS_DEFAULT)
    fallback = pack.fallback_id

    system_msg = (
        f"你是一个用于学术文献的主题分类助手。\n"
        "请充分发挥深度语言理解能力：\n"
        "1) 首先通读整个输入文本，理解其核心主题、研究对象、研究方法和应用场景\n"
        "2) 分析文本的整体内容和结构，把握其主要贡献和创新点\n"
        "3) 对照Label Cards中的scope、规则和关键词，综合判断最适合的分类\n"
        "4) 优先考虑文本的整体语义和上下文，而不是仅依赖个别关键词\n"
        "必须严格遵守：\n"
        "1) 只能从给定Label Cards中的 label_id 选择，不得编造；\n"
        "2) 仅输出严格JSON，且只能包含以下字段："
        "primary_label_id, secondary_label_ids, confidence, evidence_spans；\n"
        f"3) 允许多标签：输出1个主标签 + 最多{max_secondary}个次标签（与主标签不同）；\n"
        "4) evidence_spans 必须从输入文本中原样截取短片段（不要改写），用于支撑选择；\n"
        f"5) 兜底标签为 {fallback}：只有当文本信息严重不足、或确实无法匹配任何标签定义时才可使用。\n"
        "6) 请谨慎使用兜底标签：即使文本中没有明显的关键词匹配，只要主题与某个标签的scope相关，就应该选择该标签。\n"
    )

    def _force_contract(pred: Dict[str, Any]) -> Dict[str, Any]:
        """
        强制裁剪为 4 字段，避免模型额外输出字段影响后续处理一致性。
        """
        return {
            "primary_label_id": pred.get("primary_label_id", fallback),
            "secondary_label_ids": pred.get("secondary_label_ids", []),
            "confidence": pred.get("confidence", 0.0),
            "evidence_spans": pred.get("evidence_spans", []),
        }

    def _ask(force_in_candidates: bool) -> Dict[str, Any]:
        """
        单轮提问逻辑，包含重试与退避。
        force_in_candidates=True 时，强制 primary_label_id 必须来自 candidates 且不得使用 fallback。
        """
        cand_txt = ""
        if candidates:
            if force_in_candidates:
                cand_txt = (
                    "【候选标签集合】你必须从下列候选中选择 primary_label_id：\n"
                    + ", ".join(candidates) + "\n"
                    f"注意：本轮不得选择兜底标签 {fallback}。\n"
                    "请基于文本的整体内容和语义选择最适合的标签，而不仅仅依赖关键词匹配。\n"
                )
            else:
                cand_txt = (
                    "【候选标签集合】以下候选是基于关键词预筛选得到的参考集合。\n"
                    "请优先考虑文本的整体语义和上下文，而不是仅依赖关键词匹配。\n"
                    "你可以选择候选内或候选外的标签，但必须确保选择的标签与文本主题高度相关。\n"
                    f"只有当文本信息严重不足、或确实无法匹配任何标签定义时，才使用兜底标签 {fallback}。\n"
                    + ", ".join(candidates) + "\n"
                )

        user_msg = (
            f"{label_cards_text}\n\n"
            f"{cand_txt}\n"
            f"【paper_id】{paper_id}\n"
            "【输入文本(text=Title.[SEP]Abstract)】\n"
            f"{text}\n\n"
            "【输出JSON（严格遵守：仅4个字段）】\n"
            "{\n"
            '  "primary_label_id": "X?_XXXX",\n'
            '  "secondary_label_ids": ["X?_XXXX", "X?_XXXX"],\n'
            '  "confidence": 0.0,\n'
            '  "evidence_spans": ["...","..."]\n'
            "}\n"
        )

        last_err: Optional[str] = None

        for attempt in range(1, max_retries + 1):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0.0
                )

                content = (resp.choices[0].message.content or "").strip()

                pred = safe_json_loads(content)
                if pred is None:
                    raise ValueError(f"非JSON输出（前180字符）：{content[:180]}")

                ok, reason = validate_pred(pred, pack, max_secondary)
                if not ok:
                    raise ValueError(f"输出校验失败：{reason}；raw前180={content[:180]}")

                pred = _force_contract(pred)

                if sleep_s and sleep_s > 0:
                    time.sleep(sleep_s)

                return pred

            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                print(
                    f"[LLM_FAIL] paper_id={paper_id} attempt={attempt}/{max_retries} "
                    f"force_in_candidates={force_in_candidates} err={last_err}",
                    file=sys.stderr
                )
                # 简单线性退避：优先使用 sleep_s
                if sleep_s and sleep_s > 0:
                    time.sleep(sleep_s * attempt)
                else:
                    time.sleep(0.8 * attempt)

        print(
            f"[LLM_FALLBACK] paper_id={paper_id} fallback={fallback} last_err={last_err}",
            file=sys.stderr
        )
        return {
            "primary_label_id": fallback,
            "secondary_label_ids": [],
            "confidence": 0.0,
            "evidence_spans": []
        }

    pred1 = _ask(force_in_candidates=False)

    # 若第一轮输出 fallback 且有候选，执行第二轮强制候选内选择
    if pred1.get("primary_label_id") == fallback and candidates:
        return _ask(force_in_candidates=True)

    return pred1


# =========================
# 5) 端到端流程：四方向自动检测 + 缓存续跑 + 串/并行执行
# =========================
def parse_args() -> argparse.Namespace:
    """
    命令行参数解析。
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="*", default=["ALL"], help="要处理的方向；默认 ALL=自动检测四方向")
    ap.add_argument("--in_dir", type=str, default=str(IN_DIR_DEFAULT))
    ap.add_argument("--labels_dir", type=str, default=str(LABELS_DIR_DEFAULT))
    ap.add_argument("--out_dir", type=str, default=str(OUT_DIR_DEFAULT))
    ap.add_argument("--cache_dir", type=str, default=str(CACHE_DIR_DEFAULT))

    ap.add_argument("--model", type=str, default=MODEL_DEFAULT)
    ap.add_argument("--base_url", type=str, default=BASE_URL_DEFAULT)
    ap.add_argument("--api_key", type=str, default="", help="不填则读环境变量 DEEPSEEK_API_KEY")

    ap.add_argument("--workers", type=int, default=1, help="并发线程数（默认 1=串行）")
    ap.add_argument("--sleep", type=float, default=SLEEP_PER_CALL_DEFAULT, help="每条请求后 sleep 秒（防限流）")
    ap.add_argument("--limit", type=int, default=0, help="每方向仅处理前 N 条（调试用）")

    ap.add_argument("--max_secondary", type=int, default=MAX_SECONDARY_DEFAULT)
    ap.add_argument("--max_retries", type=int, default=MAX_RETRIES_DEFAULT)

    ap.add_argument("--cand_topk", type=int, default=CAND_TOPK_DEFAULT, help="关键词预路由 TopK 候选数")
    ap.add_argument("--cand_min_score", type=int, default=CAND_MIN_SCORE_DEFAULT, help="候选最小命中分数阈值")
    ap.add_argument("--report_every", type=int, default=REPORT_EVERY_DEFAULT, help="每 N 条汇报一次进度")

    return ap.parse_args()


def detect_domains(domains_arg: List[str]) -> List[str]:
    """
    将命令行 domains 参数解析为实际要处理的方向列表。
    """
    if len(domains_arg) == 1 and domains_arg[0].upper() == "ALL":
        return DOMAINS_ALL
    return domains_arg


def corpus_fp(in_dir: Path, domain: str) -> Path:
    """
    语料文件路径约定。
    """
    return in_dir / f"corpus_{domain}_clean.xlsx"


def labels_fp(labels_dir: Path, domain: str) -> Path:
    """
    标签 JSON 文件路径约定。
    """
    return labels_dir / f"{domain}.json"


def run_one_domain(
    domain: str,
    args: argparse.Namespace,
    make_client: Any,
) -> None:
    """
    单方向处理流程：
      1) 读取语料与标签
      2) 读取缓存，生成待处理任务列表
      3) 串行或并行执行 LLM 分类（并行时主线程统一写缓存）
      4) 合并结果并导出 Excel
    """
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    cache_dir = Path(args.cache_dir)
    lbl_dir = Path(args.labels_dir)

    cfp = corpus_fp(in_dir, domain)
    lfp = labels_fp(lbl_dir, domain)

    if not cfp.exists():
        print(f"[SKIP] {domain}: 缺少语料文件 {cfp}", file=sys.stderr)
        return
    if not lfp.exists():
        print(f"[SKIP] {domain}: 未检测到标签文件 {lfp}", file=sys.stderr)
        return

    # 目录确保存在（导出与缓存）
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 标签包与压缩 Label Cards 文本
    pack = load_label_pack(lfp)
    cards_text = build_label_cards_text(pack)

    # 读取输入语料
    df = pd.read_excel(cfp)
    if "paper_id" not in df.columns or "text" not in df.columns:
        print(f"[SKIP] {domain}: 输入缺少 paper_id/text 列", file=sys.stderr)
        return

    # 可选调试：仅取前 N 条
    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()

    # 缓存文件：每个方向一份 jsonl
    cache_file = cache_dir / f"{domain}_cache.jsonl"
    cache = load_jsonl_cache(cache_file)

    # 任务列表：使用 (paper_id + label_version + text_hash) 组成 cache_key
    # row_idx 用于后续输出对齐，避免 paper_id 重复导致 merge 错配
    tasks: List[Tuple[int, str, str, str]] = []  # (row_idx, cache_key, paper_id, text)
    cached_hits = 0

    for idx, row in df.iterrows():
        pid = str(row.get("paper_id", ""))
        txt = str(row.get("text", ""))
        key = f"{pid}||v={pack.version}||h={sha1(txt)}"

        if key in cache:
            cached_hits += 1
        else:
            tasks.append((int(idx), key, pid, txt))

    print(f"[INFO] {domain}: total={len(df)}, cached={cached_hits}, to_run={len(tasks)}", file=sys.stderr)

    def process_task(row_idx: int, key: str, pid: str, txt: str) -> Dict[str, Any]:
        """
        工作函数：执行一次 LLM 分类并返回缓存记录。
        """
        try:
            # 每个线程获取自身的 client（thread-local）
            client = make_client()

            # 关键词预路由
            cand = keyword_route_candidates(
                pack=pack,
                text=txt,
                topk=args.cand_topk,
                min_score=args.cand_min_score
            )

            # LLM 分类
            pred = llm_classify(
                client=client,
                model=args.model,
                pack=pack,
                label_cards_text=cards_text,
                paper_id=pid,
                text=txt,
                candidates=cand,
                max_secondary=args.max_secondary,
                max_retries=args.max_retries,
                sleep_s=args.sleep
            )

            # 构造正常记录
            return {
                "cache_key": key,
                "domain": domain,
                "paper_id": pid,
                "row_idx": int(row_idx),
                "labels_version": pack.version,
                "label_source_json": lfp.name,
                "text_hash": sha1(txt),
                "pred": pred
            }
        except Exception as e:
            # 构造异常兜底记录
            err_msg = f"{type(e).__name__}: {e}"
            print(f"[ERROR] {domain}: paper_id={pid} err={err_msg}", file=sys.stderr)
            return {
                "cache_key": key,
                "domain": domain,
                "paper_id": pid,
                "row_idx": int(row_idx),
                "labels_version": pack.version,
                "label_source_json": lfp.name,
                "text_hash": sha1(txt) if txt is not None else None,
                "pred": {
                    "primary_label_id": pack.fallback_id,
                    "secondary_label_ids": [],
                    "confidence": 0.0,
                    "evidence_spans": []
                },
                "error": err_msg
            }

    # new_records 仅保存本次新跑出的记录（包含正常与 fallback）
    new_records: Dict[str, Dict[str, Any]] = {}
    done = 0

    # 执行模式选择：workers<=1 为串行，其它为并行
    if args.workers <= 1:
        print(f"[MODE] {domain}: serial (workers={args.workers})", file=sys.stderr)

        for row_idx, key, pid, txt in tasks:
            rec = process_task(row_idx, key, pid, txt)
            new_records[key] = rec
            append_jsonl(cache_file, rec)  # 写入缓存

            done += 1
            if done % args.report_every == 0:
                print(f"[PROGRESS] {domain}: new_done={done}/{len(tasks)}", file=sys.stderr)

    else:
        print(f"[MODE] {domain}: parallel (workers={args.workers})", file=sys.stderr)

        # 并行执行
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            # 提交所有任务
            futures = [executor.submit(process_task, row_idx, key, pid, txt) for row_idx, key, pid, txt in tasks]
            
            # 处理完成的任务
            for future in as_completed(futures):
                rec = future.result()
                key = rec["cache_key"]
                new_records[key] = rec
                append_jsonl(cache_file, rec)  # 主线程统一写入，确保缓存稳定

                done += 1
                if done % args.report_every == 0:
                    print(f"[PROGRESS] {domain}: new_done={done}/{len(tasks)}", file=sys.stderr)

    # 结果合并与导出：
    # - 优先用 new_records（本次跑出的）
    # - 否则使用历史 cache（已有缓存命中）
    out_rows: List[Dict[str, Any]] = []

    for idx, row in df.iterrows():
        pid = str(row.get("paper_id", ""))
        txt = str(row.get("text", ""))
        key = f"{pid}||v={pack.version}||h={sha1(txt)}"

        # 获取记录，优先使用新记录，其次使用缓存
        rec = new_records.get(key) or cache.get(key)

        # 构造预测结果
        if not rec:
            pred = {
                "primary_label_id": pack.fallback_id,
                "secondary_label_ids": [],
                "confidence": 0.0,
                "evidence_spans": [],
                "rationale": "missing cache record"
            }
        else:
            pred = rec.get("pred", {}) or {}

        # 提取标签信息
        primary = pred.get("primary_label_id", pack.fallback_id)
        secondary = pred.get("secondary_label_ids", [])

        # 构造输出行
        out_rows.append({
            "row_idx": int(idx),
            "primary_label_id": primary,
            "primary_label_name_zh": pack.id2name.get(primary, ""),
            "secondary_label_ids": ";".join(secondary),
            "secondary_label_names_zh": ";".join([pack.id2name.get(x, "") for x in secondary]),
            "confidence": float(pred.get("confidence", 0.0)),
            "evidence_spans": " | ".join([str(x) for x in pred.get("evidence_spans", [])]),
            "rationale": str(pred.get("rationale", ""))[:400],
            "labels_version": pack.version,
            "label_source_json": lfp.name
        })

    # 使用 row_idx 对齐 merge，避免 paper_id 重复导致结果错配
    df_out = (
        df.reset_index()
        .rename(columns={"index": "row_idx"})
        .merge(pd.DataFrame(out_rows), on="row_idx", how="left")
        .drop(columns=["row_idx"])
    )

    # 导出：完整结果
    out_fp = out_dir / f"llm_labeled_{domain}.xlsx"
    df_out.to_excel(out_fp, index=False)

    # 导出：paper-domain-label map（用于知识库映射）
    map_fp = out_dir / f"paper_domain_label_map_{domain}.xlsx"
    df_map = df_out[[
        "paper_id", "domain",
        "primary_label_id", "secondary_label_ids",
        "confidence", "labels_version"
    ]].copy()
    df_map.to_excel(map_fp, index=False)

    print(f"[DONE] {domain}: {out_fp}", file=sys.stderr)
    print(f"[DONE] {domain}: {map_fp}", file=sys.stderr)


def main() -> None:
    """
    主入口：
      - 解析参数
      - 初始化 thread-local client 工厂
      - 按方向循环处理
    """
    args = parse_args()

    if OpenAI is None:
        print("[FATAL] 缺少 openai 包：请先 pip install openai", file=sys.stderr)
        raise SystemExit(1)

    api_key = (args.api_key or os.getenv("DEEPSEEK_API_KEY", "")).strip()
    if not api_key:
        print("[FATAL] 未提供API Key：请设置 DEEPSEEK_API_KEY 或传入 --api_key", file=sys.stderr)
        raise SystemExit(1)

    # thread-local：每个线程各自持有一个 client 实例，避免共享状态
    _tls = threading.local()

    def make_client():
        c = getattr(_tls, "client", None)
        if c is None:
            _tls.client = OpenAI(api_key=api_key, base_url=args.base_url)
        return _tls.client

    domains = detect_domains(args.domains)

    for d in domains:
        try:
            run_one_domain(d, args, make_client=make_client)
        except Exception as e:
            print(f"[ERROR] {d}: {e}", file=sys.stderr)

    print("[ALL DONE] end-to-end labeling finished.", file=sys.stderr)


if __name__ == "__main__":
    main()
