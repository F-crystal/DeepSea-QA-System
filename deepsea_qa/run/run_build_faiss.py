# -*- coding: utf-8 -*-
"""
run_build_faiss.py

作者：Accilia
创建时间：2026-02-22
用途说明：
一键运行创建 FAISS 索引

执行：
  python -m deepsea_qa.run.run_build_faiss
"""

from __future__ import annotations

# 忽略warning
import warnings
warnings.filterwarnings('ignore')

import os

from deepsea_qa.index.faiss_index import build_faiss_index


# ========== 1) 路径 ==========
from deepsea_qa.configs import paths
from deepsea_qa.configs.index_config import FAISSConfig

SQLITE_PATH = str(paths.SQLITE_PATH)  # 输入

# 输出目录
FAISS_INDEX_PATH = str(paths.FAISS_INDEX_PATH)
FAISS_META_PATH = str(paths.FAISS_META_PATH)
FAISS_IDMAP_PATH = str(paths.FAISS_IDMAP_PATH)

# SQLite 表字段
TABLE_NAME = "chunks"
CHUNK_ID_COL = "chunk_id"
TEXT_COL = "text"
# =====================================================


# 使用集中管理的 FAISS 配置
CFG = FAISSConfig()
# 如需覆盖默认值，可在此修改
# CFG.device = "cuda"  # 无 GPU 改 "cpu"



def main():
    assert os.path.exists(SQLITE_PATH), f"找不到 SQLite：{SQLITE_PATH}"

    print("=== Build FAISS Index ===")
    print(f"[SQLite] {SQLITE_PATH}")
    print(f"[FAISS ] {FAISS_INDEX_PATH}")
    print(f"[IDMAP ] {FAISS_IDMAP_PATH}")
    print(f"[Model ] {CFG.model_name} | device={CFG.device} | bs={CFG.batch_size}")

    # 确保输出目录存在
    os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)

    build_faiss_index(
        sqlite_path=SQLITE_PATH,
        out_faiss_path=FAISS_INDEX_PATH,
        out_meta_path=FAISS_META_PATH,
        out_idmap_path=FAISS_IDMAP_PATH,
        cfg=CFG,
        table_name=TABLE_NAME,
        chunk_id_col=CHUNK_ID_COL,
        text_col=TEXT_COL,
    )

    print("=== Done ===")


if __name__ == "__main__":
    main()