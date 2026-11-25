"""数据台账（data ledger）：记录「已训练数据」的存储结构 + 读写接口。

动机（与 resume 配套）：
- HF datasets 流式 shuffle 的 buffer 状态无法序列化，resume 无法精确恢复数据游标；
  真正保证「续训不重复、增量只喂新增」的机制是数据台账去重：训练前查台账，跳过已消费的分片/样本。
- 公开数据需溯源：哪个数据集/配置/分片/版本快照、何时训练多少步，用于合规与复现。

存储选型：SQLite（标准库 sqlite3，零额外依赖、事务安全、可索引去重、单文件、可并发读）。

两级粒度：
1. dataset + shard 级（核心，第一版）：记录 (dataset, config, split, version) 快照的
   消费进度与每个分片的消费状态。足够支撑「续训不重复同一 shard」与「公开数据溯源」。
2. sample 级哈希（预留，可选强化）：按内容哈希精确去重，需在数据 pipeline 对未增强的
   原始像素/文本做哈希（会侵入 image_transform），作为后续增强，不在第一版强制接入。

写入时机（约定）：
- 训练开始登记数据集快照 → mark_dataset()；
- 每完整消费一个分片 → mark_shard_consumed()；
- 每次训练会话结束 → record_run()；
- sample 级（可选）：每个 batch 对样本算 content_hash 后 record_sample() 去重。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone


SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
    dataset_id       TEXT PRIMARY KEY,             -- 规范化键 name||config||split
    name             TEXT NOT NULL,                -- 数据集名（HF id 或本地目录）
    config           TEXT NOT NULL DEFAULT '',     -- 数据集 config
    split            TEXT NOT NULL,                -- train / validation / test ...
    version          TEXT,                         -- 版本/commit/snapshot 标识（溯源）
    source_type      TEXT NOT NULL,                -- hf / parquet / imagefolder
    source_uri       TEXT,                         -- 源位置（HF id / parquet glob / 目录）
    sample_count     INTEGER,                      -- 快照总样本数（流式下可能未知，可空）
    consumed_shards  INTEGER NOT NULL DEFAULT 0,   -- 已完整消费分片数
    consumed_samples INTEGER NOT NULL DEFAULT 0,   -- 已消费样本数
    steps_trained    INTEGER NOT NULL DEFAULT 0,   -- 用该数据集累计训练步数
    status           TEXT NOT NULL DEFAULT 'in_progress',  -- in_progress / complete / superseded
    first_seen_at    TEXT NOT NULL,                -- 首次登记时间（ISO8601 UTC）
    last_trained_at  TEXT,                         -- 最近一次训练时间
    CHECK (status IN ('in_progress', 'complete', 'superseded'))
);

CREATE TABLE IF NOT EXISTS shards (
    shard_id         TEXT PRIMARY KEY,             -- sha256(dataset_id || shard_uri)
    dataset_id       TEXT NOT NULL REFERENCES datasets(dataset_id),
    shard_uri        TEXT NOT NULL,                -- 分片路径/URL
    shard_index      INTEGER,                      -- 分片序号
    status           TEXT NOT NULL DEFAULT 'consumed',  -- consumed / skipped
    sample_count     INTEGER,
    content_hash     TEXT,                         -- 分片内容 hash（版本漂移检测）
    consumed_at      TEXT,
    CHECK (status IN ('consumed', 'skipped'))
);

CREATE TABLE IF NOT EXISTS samples (
    content_hash     TEXT PRIMARY KEY,             -- sha256(image_hash || text_hash)
    image_hash       TEXT,                         -- 内容级图像哈希（预留）
    text_hash        TEXT,                         -- 归一化文本哈希（预留）
    dataset_id       TEXT,
    shard_id         TEXT,
    first_seen_at    TEXT NOT NULL,
    train_count      INTEGER NOT NULL DEFAULT 1    -- 重复出现次数（去重 + 统计）
);

CREATE TABLE IF NOT EXISTS runs (
    run_id           TEXT PRIMARY KEY,             -- 时间戳/uuid
    preset           TEXT,
    ckpt_path        TEXT,
    start_step       INTEGER,
    end_step         INTEGER,
    steps            INTEGER,
    started_at       TEXT NOT NULL,
    finished_at      TEXT,
    config_json      TEXT                          -- cfg 快照（复现用）
);

CREATE INDEX IF NOT EXISTS idx_shards_dataset  ON shards(dataset_id);
CREATE INDEX IF NOT EXISTS idx_shards_uri      ON shards(shard_uri);
CREATE INDEX IF NOT EXISTS idx_samples_dataset ON samples(dataset_id);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def text_hash(caption: str) -> str:
    """归一化文本哈希：strip + 小写 + 折叠空白，跨格式稳定。"""
    norm = " ".join(caption.strip().lower().split())
    return sha256_bytes(norm.encode("utf-8"))


def image_hash(pil_img, size=256) -> str:
    """内容级图像哈希（预留）：resize 到固定尺寸后对像素做哈希，不随机翻转，
    保证同一图片跨加载/跨编码稳定。

    注意：对 jpg/png 解码伪影不完全敏感；严格内容去重应改用原始字节，但 HF parquet
    内嵌图拿到的是解码后的 PIL 对象，此处取「像素级内容哈希」作为工程折中。
    """
    try:
        import numpy as np

        img = pil_img.convert("RGB").resize((size, size))
        arr = np.asarray(img, dtype="uint8")
        return sha256_bytes(arr.tobytes())
    except Exception:
        return ""


def content_hash(pil_img, caption, size=256) -> str:
    """样本级内容哈希 = sha256(image_hash || text_hash)，跨模态联合去重用。"""
    return sha256_bytes((image_hash(pil_img, size) + text_hash(caption)).encode("ascii"))


class Ledger:
    """数据台账：连接 SQLite、建表、登记/查询已训练数据。"""

    def __init__(self, path: str):
        self.conn = sqlite3.connect(path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)

    @staticmethod
    def dataset_id(name: str, config: str, split: str) -> str:
        return "||".join([name, config or "", split])

    def mark_dataset(self, *, name, config="", split="train", version=None,
                     source_type="hf", source_uri=None, sample_count=None) -> str:
        """登记/更新一个数据集快照，返回 dataset_id。"""
        dsid = self.dataset_id(name, config, split)
        self.conn.execute(
            """
            INSERT INTO datasets (dataset_id, name, config, split, version, source_type,
                                  source_uri, sample_count, first_seen_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(dataset_id) DO UPDATE SET
                version=excluded.version, source_uri=excluded.source_uri,
                sample_count=excluded.sample_count
            """,
            (dsid, name, config or "", split, version, source_type, source_uri,
             sample_count, now_iso()),
        )
        self.conn.commit()
        return dsid

    def mark_shard_consumed(self, dataset_id, shard_uri, shard_index=None,
                            sample_count=None, content_hash=None, skipped=False) -> str:
        """标记一个分片已消费（或跳过），并累加数据集的分片计数。"""
        sid = sha256_bytes(f"{dataset_id}||{shard_uri}".encode("utf-8"))
        self.conn.execute(
            """
            INSERT INTO shards (shard_id, dataset_id, shard_uri, shard_index, status,
                                sample_count, content_hash, consumed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(shard_id) DO UPDATE SET status=excluded.status
            """,
            (sid, dataset_id, shard_uri, shard_index,
             "skipped" if skipped else "consumed", sample_count, content_hash, now_iso()),
        )
        self.conn.execute(
            "UPDATE datasets SET consumed_shards = consumed_shards + 1, "
            "last_trained_at = ? WHERE dataset_id = ?",
            (now_iso(), dataset_id),
        )
        self.conn.commit()
        return sid

    def is_shard_consumed(self, shard_uri) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM shards WHERE shard_uri = ? LIMIT 1", (shard_uri,)
        ).fetchone()
        return row is not None

    def record_sample(self, pil_img, caption, dataset_id=None, shard_id=None) -> bool:
        """样本级记账（预留）：返回 True 表示新样本，False 表示已存在（去重命中）。"""
        ch = content_hash(pil_img, caption)
        cur = self.conn.execute(
            "INSERT OR IGNORE INTO samples (content_hash, image_hash, text_hash, "
            "dataset_id, shard_id, first_seen_at) VALUES (?, ?, ?, ?, ?, ?)",
            (ch, image_hash(pil_img), text_hash(caption), dataset_id, shard_id, now_iso()),
        )
        if cur.rowcount == 0:
            self.conn.execute(
                "UPDATE samples SET train_count = train_count + 1 WHERE content_hash = ?", (ch,)
            )
            self.conn.commit()
            return False
        self.conn.commit()
        return True

    def record_run(self, *, run_id, preset=None, ckpt_path=None, start_step=0,
                   end_step=0, steps=0, config=None):
        """记录一次训练会话，cfg 快照写入 config_json 用于复现。"""
        self.conn.execute(
            """
            INSERT INTO runs (run_id, preset, ckpt_path, start_step, end_step, steps,
                              started_at, finished_at, config_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (run_id, preset, ckpt_path, start_step, end_step, steps, now_iso(), now_iso(),
             json.dumps(config, ensure_ascii=False) if config is not None else None),
        )
        self.conn.commit()

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
