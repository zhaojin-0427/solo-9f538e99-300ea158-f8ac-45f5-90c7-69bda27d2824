"""质控抽检：从已交付快照按站点抽取化验记录，固化成抽检批次供人工复核。

规则
====
* 入参：快照号、该快照中固化的指标名、2~8 个不同站点；快照分片数须 <= 12。
* 每站选一条该指标非 NULL 的记录；同批分片不得重复（样本分散到不同来源）。
* 任何已入批的 ``(shard_id, row_no)`` 全局不得再选；建批在进程写锁 +
  ``BEGIN IMMEDIATE`` 事务内裁决，并由 ``qc_batch_items`` 上的
  ``UNIQUE(shard_id, row_no)`` 兜底，并发建批不会重复占行。
* 有多种可行组合时，把各站按站点名升序排列，比较各站来源 ``(shard_id, row_no)``
  组成的序列，取字典序最小者。
* 无可行组合则整批拒绝（409），不占任何行。
* 批次固化采样时刻、指标名与值、来源 ``(shard_id, row_no)``；后续上传、
  改名、删除指标乃至进程重启，都不改变已固化的批次。

说明
====
* ``row_no`` 为分片 Parquet 文件内 0 基原始行号（首行数据为 0）。
* 指标值为 NaN 的行不参与抽检：NaN 无法作为 REAL 固化进 SQLite，
  且对人工复核没有意义。
"""
from __future__ import annotations

import math
import sqlite3
from typing import Dict, List, Optional, Set, Tuple

import pyarrow.parquet as pq

from .config import INDICATOR_COLUMN_PREFIX, SAMPLED_AT_COLUMN, STATION_COLUMN
from .models import ConflictError, NotFound
from .query import get_snapshot_meta

# 一次抽检允许的最大快照分片数
MAX_SNAPSHOT_SHARDS = 12
# 每批站点数上下限
MIN_STATIONS = 2
MAX_STATIONS = 8

# 候选行：(shard_id, row_no, sampled_at 固化文本, value)
Candidate = Tuple[int, int, str, float]


# ---------------------------------------------------------------------------
# 入参校验
# ---------------------------------------------------------------------------

def _normalize_stations(raw_stations: List[str]) -> List[str]:
    stations: List[str] = []
    for raw in raw_stations:
        name = str(raw).strip()
        if not name:
            raise ConflictError("站点名不能为空或全为空白字符")
        stations.append(name)
    if not (MIN_STATIONS <= len(stations) <= MAX_STATIONS):
        raise ConflictError(
            f"站点数量须在 {MIN_STATIONS}~{MAX_STATIONS} 之间，实际为 {len(stations)}"
        )
    duplicated = sorted({s for s in stations if stations.count(s) > 1})
    if duplicated:
        raise ConflictError(f"站点必须互不相同，重复站点：{duplicated}")
    return stations


# ---------------------------------------------------------------------------
# 候选行收集：快照分片内、该站、该指标非 NULL、且未入过任何批次
# ---------------------------------------------------------------------------

def _format_timestamp(value) -> str:
    # 与查询输出一致：带微秒与 Z 后缀的 ISO-8601
    if value is None:
        return None
    return value.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _load_candidates(
    conn: sqlite3.Connection,
    shard_ids: List[int],
    indicator_id: int,
    stations: List[str],
    occupied: Set[Tuple[int, int]],
) -> Dict[str, List[Candidate]]:
    """逐分片收集候选行；每站每分片只保留行号最小的一条。

    同一站同一分片内，行号更小的候选在字典序上严格占优，且是否可行只取决于
    分片身份，与行号无关，因此只留最小行号不会改变最终答案。
    """
    canonical = f"{INDICATOR_COLUMN_PREFIX}{indicator_id}"
    wanted = set(stations)
    candidates: Dict[str, List[Candidate]] = {s: [] for s in stations}

    for shard_id in shard_ids:
        row = conn.execute(
            "SELECT file_path FROM shards WHERE shard_id = ?", (shard_id,)
        ).fetchone()
        if row is None:
            # 快照引用的分片理论上不会被删；防御性处理
            raise NotFound(f"快照引用的分片 {shard_id} 已丢失")
        file_path = row["file_path"]

        if canonical not in pq.read_schema(file_path).names:
            continue  # 旧分片没有该指标身份列，整片该指标为 NULL
        table = pq.read_table(
            file_path, columns=[STATION_COLUMN, SAMPLED_AT_COLUMN, canonical]
        )
        station_col = table.column(STATION_COLUMN).to_pylist()
        time_col = table.column(SAMPLED_AT_COLUMN).to_pylist()
        value_col = table.column(canonical).to_pylist()

        for row_no, (station, sampled_at, value) in enumerate(
            zip(station_col, time_col, value_col)
        ):
            if station not in wanted or value is None:
                continue
            if isinstance(value, float) and math.isnan(value):
                continue  # NaN 无法固化为 REAL，不参与抽检
            if (shard_id, row_no) in occupied:
                continue  # 已入过批的原始行不得再选
            candidates[station].append(
                (shard_id, row_no, _format_timestamp(sampled_at), float(value))
            )

    for station, cands in candidates.items():
        best_by_shard: Dict[int, Candidate] = {}
        for cand in cands:
            current = best_by_shard.get(cand[0])
            if current is None or cand[1] < current[1]:
                best_by_shard[cand[0]] = cand
        candidates[station] = sorted(best_by_shard.values(), key=lambda c: (c[0], c[1]))
    return candidates


# ---------------------------------------------------------------------------
# 选样：字典序最小且同批分片互不重复的组合
# ---------------------------------------------------------------------------

def _feasible(
    remaining: List[str],
    candidates: Dict[str, List[Candidate]],
    used_shards: Set[int],
) -> bool:
    """剩余站点能否在剩余分片上完成“每站一条、分片互不重复”的分配。

    二分图匹配（站点 -> 候选分片）增广路判定；规模极小（<=8 站、<=12 片）。
    """
    match: Dict[int, str] = {}  # shard_id -> station

    def augment(station: str, seen: Set[int]) -> bool:
        for cand in candidates[station]:
            shard_id = cand[0]
            if shard_id in used_shards or shard_id in seen:
                continue
            seen.add(shard_id)
            if shard_id not in match or augment(match[shard_id], seen):
                match[shard_id] = station
                return True
        return False

    for station in remaining:
        if not augment(station, set()):
            return False
    return True


def _select_lexicographically_minimal(
    stations_sorted: List[str],
    candidates: Dict[str, List[Candidate]],
) -> Optional[List[Candidate]]:
    """按站点名升序逐站深搜，候选按 (shard_id, row_no) 升序尝试。

    第一个完整可行解即为“各站来源序列”字典序最小者；
    每步用匹配可行性剪枝，无可行组合时返回 None。
    """
    assignment: List[Candidate] = []
    used_shards: Set[int] = set()

    def dfs(index: int) -> bool:
        if index == len(stations_sorted):
            return True
        station = stations_sorted[index]
        rest = stations_sorted[index + 1 :]
        for cand in candidates[station]:
            shard_id = cand[0]
            if shard_id in used_shards:
                continue
            used_shards.add(shard_id)
            assignment.append(cand)
            if _feasible(rest, candidates, used_shards) and dfs(index + 1):
                return True
            assignment.pop()
            used_shards.discard(shard_id)
        return False

    if not dfs(0):
        return None
    return list(assignment)


# ---------------------------------------------------------------------------
# 建批 / 查询
# ---------------------------------------------------------------------------

def create_batch(
    conn: sqlite3.Connection,
    snapshot_id: int,
    indicator_name: str,
    raw_stations: List[str],
) -> dict:
    """在调用方写事务内创建抽检批次。

    :raises NotFound: 快照不存在。
    :raises ConflictError: 站点/指标/分片数违规，或无可行组合（整批拒绝，不占行）。
    """
    stations = _normalize_stations(raw_stations)
    indicator_name = str(indicator_name).strip()
    if not indicator_name:
        raise ConflictError("指标名不能为空")

    meta = get_snapshot_meta(conn, snapshot_id)  # 不存在抛 404
    shard_ids = meta["shard_ids"]
    if len(shard_ids) > MAX_SNAPSHOT_SHARDS:
        raise ConflictError(
            f"快照 {snapshot_id} 含 {len(shard_ids)} 个分片，"
            f"超过抽检上限 {MAX_SNAPSHOT_SHARDS} 个，整批拒绝"
        )

    name_to_id = dict(zip(meta["indicator_names"], meta["indicator_ids"]))
    if indicator_name not in name_to_id:
        raise ConflictError(
            f"指标 {indicator_name!r} 不在快照 {snapshot_id} 固化的字段集合中；"
            f"该快照可用指标：{meta['indicator_names']}"
        )
    indicator_id = int(name_to_id[indicator_name])

    occupied = {
        (r["shard_id"], r["row_no"])
        for r in conn.execute("SELECT shard_id, row_no FROM qc_batch_items").fetchall()
    }
    candidates = _load_candidates(conn, shard_ids, indicator_id, stations, occupied)

    stations_sorted = sorted(stations)
    selection = _select_lexicographically_minimal(stations_sorted, candidates)
    if selection is None:
        shortage = [s for s in stations_sorted if not candidates[s]]
        raise ConflictError(
            "无可行组合：无法满足“每站一条非 NULL 记录、同批分片不重复、"
            "已入批行不再选”，整批拒绝，未占用任何行"
            + (f"；无任何候选行的站点：{shortage}" if shortage else "")
        )

    try:
        cur = conn.execute(
            "INSERT INTO qc_batches (snapshot_id, indicator_id, indicator_name) "
            "VALUES (?, ?, ?)",
            (snapshot_id, indicator_id, indicator_name),
        )
        batch_id = int(cur.lastrowid)
        for station, (shard_id, row_no, sampled_at, value) in zip(stations_sorted, selection):
            conn.execute(
                "INSERT INTO qc_batch_items "
                "(batch_id, station, shard_id, row_no, sampled_at, value) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (batch_id, station, shard_id, row_no, sampled_at, value),
            )
    except sqlite3.IntegrityError as exc:
        # 兜底：并发下 (shard_id, row_no) 唯一约束被击穿 -> 整批回滚
        raise ConflictError(f"占行冲突（可能并发建批），整批拒绝：{exc}") from exc

    created_at = conn.execute(
        "SELECT created_at FROM qc_batches WHERE batch_id = ?", (batch_id,)
    ).fetchone()["created_at"]

    return {
        "batch_id": batch_id,
        "snapshot_id": snapshot_id,
        "indicator_name": indicator_name,
        "created_at": created_at,
        "items": [
            {
                "station": station,
                "shard_id": shard_id,
                "row_no": row_no,
                "sampled_at": sampled_at,
                "value": value,
            }
            for station, (shard_id, row_no, sampled_at, value) in zip(
                stations_sorted, selection
            )
        ],
    }


def get_batch(conn: sqlite3.Connection, batch_id: int) -> dict:
    """读取单个已固化批次；不存在抛 404。"""
    batch = conn.execute(
        "SELECT batch_id, snapshot_id, indicator_name, created_at "
        "FROM qc_batches WHERE batch_id = ?",
        (batch_id,),
    ).fetchone()
    if batch is None:
        raise NotFound(f"抽检批次 {batch_id} 不存在")
    items = conn.execute(
        "SELECT station, shard_id, row_no, sampled_at, value FROM qc_batch_items "
        "WHERE batch_id = ? ORDER BY station",
        (batch_id,),
    ).fetchall()
    return {
        "batch_id": batch["batch_id"],
        "snapshot_id": batch["snapshot_id"],
        "indicator_name": batch["indicator_name"],
        "created_at": batch["created_at"],
        "items": [dict(item) for item in items],
    }


def list_batches(conn: sqlite3.Connection) -> List[dict]:
    """全部批次（含条目），按批次号升序。"""
    ids = [
        r["batch_id"]
        for r in conn.execute("SELECT batch_id FROM qc_batches ORDER BY batch_id").fetchall()
    ]
    return [get_batch(conn, batch_id) for batch_id in ids]
