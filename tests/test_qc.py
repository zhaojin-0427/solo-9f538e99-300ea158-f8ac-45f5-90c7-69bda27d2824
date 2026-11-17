"""抽检批次：选样规则、字典序最小、占行排斥、并发与固化语义。"""
from __future__ import annotations

import threading

import pyarrow as pa

from app.db import init_db
from app.main import app
from fastapi.testclient import TestClient


def upload_file(client, path, version: int):
    with open(path, "rb") as fh:
        return client.post(
            "/ingest/parquet",
            data={"schema_version": str(version)},
            files={"file": (path.name, fh.read(), "application/vnd.apache.parquet")},
        )


def make_indicator(client, name="硝酸盐") -> int:
    r = client.post("/schema/indicators", json={"name": name})
    assert r.status_code == 201, r.text
    return r.json()["new_version"]


def shard(client, make_parquet, name, stations, times, values, version):
    """上传一个单指标分片，返回 shard_id。"""
    f = make_parquet(name, {
        "station": stations,
        "sampled_at": times,
        "硝酸盐": values,
    }, types={"sampled_at": pa.string(), "硝酸盐": pa.float64()})
    r = upload_file(client, f, version)
    assert r.status_code == 200, r.text
    return r.json()["shard_id"]


def new_snapshot(client) -> int:
    body = client.get("/rows").json()
    assert body["snapshot_created"] is True
    return body["snapshot_id"]


def create_batch(client, snapshot_id, stations, indicator="硝酸盐"):
    return client.post("/qc/batches", json={
        "snapshot_id": snapshot_id,
        "indicator_name": indicator,
        "stations": stations,
    })


# ---------------------------------------------------------------------------
# 基本建批 + 固化：改名、删除、上传、重启都不影响旧批次
# ---------------------------------------------------------------------------

def test_batch_created_then_frozen_across_changes_and_restart(client, make_parquet):
    v = make_indicator(client)
    s1 = shard(client, make_parquet, "s1.parquet",
               ["A"], ["2026-01-01T08:00:00Z"], [1.5], v)
    s2 = shard(client, make_parquet, "s2.parquet",
               ["B"], ["2026-01-02T09:30:00Z"], [2.5], v)
    snap = new_snapshot(client)

    r = create_batch(client, snap, ["B", "A"])  # 入参顺序无关，结果按站点名升序
    assert r.status_code == 201, r.text
    batch = r.json()
    assert batch["snapshot_id"] == snap
    assert batch["indicator_name"] == "硝酸盐"
    assert [it["station"] for it in batch["items"]] == ["A", "B"]
    assert batch["items"][0] == {
        "station": "A", "shard_id": s1, "row_no": 0,
        "sampled_at": "2026-01-01T08:00:00.000000Z", "value": 1.5,
    }
    assert batch["items"][1] == {
        "station": "B", "shard_id": s2, "row_no": 0,
        "sampled_at": "2026-01-02T09:30:00.000000Z", "value": 2.5,
    }
    batch_id = batch["batch_id"]

    # 单批查询与列表查询一致
    assert client.get(f"/qc/batches/{batch_id}").json() == batch
    assert client.get("/qc/batches").json() == [batch]

    # 之后：改名、删除指标、新上传 —— 均不得影响已固化批次
    client.post("/schema/indicators/rename", json={"old_name": "硝酸盐", "new_name": "硝态氮"})
    cur = client.get("/schema").json()["current_version"]
    f = make_parquet("later.parquet", {
        "station": ["A"], "sampled_at": ["2026-01-03T00:00:00Z"], "硝态氮": [99.0],
    }, types={"sampled_at": pa.string(), "硝态氮": pa.float64()})
    assert upload_file(client, f, cur).status_code == 200
    client.delete("/schema/indicators/硝态氮")

    frozen = client.get(f"/qc/batches/{batch_id}").json()
    assert frozen == batch  # 指标名、值、采样时刻、来源全部保持建批时的样子

    # 重启：数据在 SQLite 中持久化，批次依然可读且不变
    del client
    init_db()
    with TestClient(app) as client2:
        revived = client2.get(f"/qc/batches/{batch_id}")
        assert revived.status_code == 200
        assert revived.json() == batch


# ---------------------------------------------------------------------------
# 字典序最小：按站点名升序比较来源 (shard_id, row_no) 序列
# ---------------------------------------------------------------------------

def test_lexicographically_minimal_source_sequence(client, make_parquet):
    v = make_indicator(client)
    # 分片 1：A 在第 0、1 行（验证同片取最小行号）
    shard(client, make_parquet, "s1.parquet",
           ["A", "A"], ["2026-02-01T00:00:00Z", "2026-02-01T01:00:00Z"],
           [10.0, 11.0], v)
    # 分片 2：A 在第 0 行、B 在第 1 行
    shard(client, make_parquet, "s2.parquet",
           ["A", "B"], ["2026-02-02T00:00:00Z", "2026-02-02T01:00:00Z"],
           [20.0, 21.0], v)
    # 分片 3：B 在第 0 行
    shard(client, make_parquet, "s3.parquet",
           ["B"], ["2026-02-03T00:00:00Z"], [30.0], v)
    snap = new_snapshot(client)

    r = create_batch(client, snap, ["A", "B"])
    assert r.status_code == 201, r.text
    items = {it["station"]: it for it in r.json()["items"]}
    # A 的最小来源是 (分片1, 行0)（不是行 1）；可行，直接选中
    assert (items["A"]["shard_id"], items["A"]["row_no"]) == (1, 0)
    assert items["A"]["value"] == 10.0
    # B 可选 (分片2, 行1) 与 (分片3, 行0)，字典序最小为分片 2
    assert (items["B"]["shard_id"], items["B"]["row_no"]) == (2, 1)
    assert items["B"]["value"] == 21.0


def test_shared_shard_forces_backtracking(client, make_parquet):
    """共享分片场景：A 不能贪心占最小分片，否则 B 无片可用。"""
    v = make_indicator(client)
    # 分片 1：A@0、B@1（A、B 共享分片 1）；分片 2：只有 A@0
    shard(client, make_parquet, "s1.parquet",
           ["A", "B"], ["2026-03-01T00:00:00Z", "2026-03-01T01:00:00Z"],
           [1.0, 2.0], v)
    shard(client, make_parquet, "s2.parquet",
           ["A"], ["2026-03-02T00:00:00Z"], [3.0], v)
    snap = new_snapshot(client)

    r = create_batch(client, snap, ["A", "B"])
    assert r.status_code == 201, r.text
    items = {it["station"]: it for it in r.json()["items"]}
    # A<-分片1 会让 B 无分片可用，故回溯为 A<-分片2、B<-分片1
    assert (items["A"]["shard_id"], items["A"]["row_no"]) == (2, 0)
    assert (items["B"]["shard_id"], items["B"]["row_no"]) == (1, 1)
    # 同批分片不得重复
    assert items["A"]["shard_id"] != items["B"]["shard_id"]


# ---------------------------------------------------------------------------
# 无可行组合：整批拒绝，不占任何行
# ---------------------------------------------------------------------------

def test_infeasible_rejected_and_rows_left_untouched(client, make_parquet):
    v = make_indicator(client)
    # 只有一个分片，A、B 都只在分片 1 -> 两站无法分片不重复
    shard(client, make_parquet, "s1.parquet",
           ["A", "B"], ["2026-04-01T00:00:00Z", "2026-04-01T01:00:00Z"],
           [1.0, 2.0], v)
    snap = new_snapshot(client)

    r = create_batch(client, snap, ["A", "B"])
    assert r.status_code == 409, r.text
    assert "无可行组合" in r.json()["detail"]
    assert client.get("/qc/batches").json() == []  # 整批拒绝，没有留下批次

    # 关键：失败的批次不占行。补一个分片后，分片 1 的行仍应可被选中。
    shard(client, make_parquet, "s2.parquet",
           ["A", "B"], ["2026-04-02T00:00:00Z", "2026-04-02T01:00:00Z"],
           [3.0, 4.0], v)
    snap2 = new_snapshot(client)
    r = create_batch(client, snap2, ["A", "B"])
    assert r.status_code == 201, r.text
    items = {it["station"]: it for it in r.json()["items"]}
    # 字典序最小：A<-分片1行0（若被失败批次占用则会落到分片2），B<-分片2行1
    assert (items["A"]["shard_id"], items["A"]["row_no"]) == (1, 0)
    assert (items["B"]["shard_id"], items["B"]["row_no"]) == (2, 1)


# ---------------------------------------------------------------------------
# 已入批的 (分片, 原始行号) 不得再选
# ---------------------------------------------------------------------------

def test_occupied_rows_never_reselected(client, make_parquet):
    v = make_indicator(client)
    shard(client, make_parquet, "s1.parquet",
           ["A", "B"], ["2026-05-01T00:00:00Z", "2026-05-01T01:00:00Z"],
           [1.0, 2.0], v)
    shard(client, make_parquet, "s2.parquet",
           ["A", "B"], ["2026-05-02T00:00:00Z", "2026-05-02T01:00:00Z"],
           [3.0, 4.0], v)
    snap = new_snapshot(client)

    b1 = create_batch(client, snap, ["A", "B"])
    assert b1.status_code == 201, b1.text
    got1 = {(it["station"], it["shard_id"], it["row_no"]) for it in b1.json()["items"]}
    assert got1 == {("A", 1, 0), ("B", 2, 1)}

    # 第二批：第一批占掉的 (1,0) 与 (2,1) 不得再选，只能交叉占行
    b2 = create_batch(client, snap, ["A", "B"])
    assert b2.status_code == 201, b2.text
    got2 = {(it["station"], it["shard_id"], it["row_no"]) for it in b2.json()["items"]}
    assert got2 == {("A", 2, 0), ("B", 1, 1)}

    # 行已抽完：第三批无可行组合，整批拒绝
    b3 = create_batch(client, snap, ["A", "B"])
    assert b3.status_code == 409
    assert len(client.get("/qc/batches").json()) == 2


def test_null_and_nan_rows_are_not_candidates(client, make_parquet):
    v = make_indicator(client)
    # 分片 1：A 的三行分别是 NULL、NaN、1.5 —— 只有第三行可抽
    shard(client, make_parquet, "s1.parquet",
           ["A", "A", "A"],
           ["2026-06-01T00:00:00Z", "2026-06-01T01:00:00Z", "2026-06-01T02:00:00Z"],
           [None, float("nan"), 1.5], v)
    shard(client, make_parquet, "s2.parquet",
           ["B"], ["2026-06-02T00:00:00Z"], [2.5], v)
    snap = new_snapshot(client)

    r = create_batch(client, snap, ["A", "B"])
    assert r.status_code == 201, r.text
    items = {it["station"]: it for it in r.json()["items"]}
    assert (items["A"]["shard_id"], items["A"]["row_no"]) == (1, 2)
    assert items["A"]["value"] == 1.5


# ---------------------------------------------------------------------------
# 入参校验
# ---------------------------------------------------------------------------

def test_validation_rules(client, make_parquet):
    v = make_indicator(client)
    shard(client, make_parquet, "s1.parquet",
           ["A", "B"], ["2026-07-01T00:00:00Z", "2026-07-01T01:00:00Z"],
           [1.0, 2.0], v)
    snap = new_snapshot(client)

    # 站点数越界（Pydantic 422）：1 个与 9 个
    assert create_batch(client, snap, ["A"]).status_code == 422
    assert create_batch(client, snap, [f"S{i}" for i in range(9)]).status_code == 422
    # 重复站点 / 空白站点名 -> 409
    r = create_batch(client, snap, ["A", "A"])
    assert r.status_code == 409 and "互不相同" in r.json()["detail"]
    r = create_batch(client, snap, ["A", "  "])
    assert r.status_code == 409 and "空" in r.json()["detail"]
    # 快照中不存在的指标名 -> 409，并报可用指标
    r = create_batch(client, snap, ["A", "B"], indicator="COD")
    assert r.status_code == 409 and "硝酸盐" in r.json()["detail"]
    # 快照不存在 -> 404；批次不存在 -> 404
    assert create_batch(client, 9999, ["A", "B"]).status_code == 404
    assert client.get("/qc/batches/9999").status_code == 404
    # 校验失败不占行、不留批次
    assert client.get("/qc/batches").json() == []


def test_snapshot_with_more_than_12_shards_rejected(client, make_parquet):
    v = make_indicator(client)
    for i in range(13):
        shard(client, make_parquet, f"m{i}.parquet",
               [f"S{i:02d}"], [f"2026-08-{i+1:02d}T00:00:00Z"], [float(i)], v)
    snap = new_snapshot(client)
    assert len(client.get(f"/snapshots/{snap}").json()["shard_ids"]) == 13

    r = create_batch(client, snap, ["S00", "S01"])
    assert r.status_code == 409, r.text
    assert "分片" in r.json()["detail"] and "12" in r.json()["detail"]
    assert client.get("/qc/batches").json() == []


def test_empty_snapshot_has_no_feasible_combination(client):
    make_indicator(client)          # 快照含指标但没有任何分片
    snap = new_snapshot(client)
    assert client.get(f"/snapshots/{snap}").json()["shard_ids"] == []
    r = create_batch(client, snap, ["A", "B"])
    assert r.status_code == 409
    assert "无可行组合" in r.json()["detail"]


# ---------------------------------------------------------------------------
# 并发建批：不能重复占行
# ---------------------------------------------------------------------------

def test_concurrent_batches_never_double_occupy(client, make_parquet):
    v = make_indicator(client)
    shard(client, make_parquet, "s1.parquet",
           ["A", "B"], ["2026-09-01T00:00:00Z", "2026-09-01T01:00:00Z"],
           [1.0, 2.0], v)
    shard(client, make_parquet, "s2.parquet",
           ["A", "B"], ["2026-09-02T00:00:00Z", "2026-09-02T01:00:00Z"],
           [3.0, 4.0], v)
    snap = new_snapshot(client)

    n_threads = 6
    barrier = threading.Barrier(n_threads)
    statuses: list[int] = []
    errors: list[Exception] = []

    def worker():
        try:
            barrier.wait()
            statuses.append(create_batch(client, snap, ["A", "B"]).status_code)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors

    # 候选行总共 4 行、每批占 2 行 -> 恰好 2 批成功，其余整批拒绝
    assert statuses.count(201) == 2
    assert statuses.count(409) == n_threads - 2

    # 所有成功批次的 (分片, 行号) 互不重复，且恰好瓜分全部 4 行
    batches = client.get("/qc/batches").json()
    assert len(batches) == 2
    occupied = [(it["shard_id"], it["row_no"]) for b in batches for it in b["items"]]
    assert len(occupied) == len(set(occupied)) == 4
    assert set(occupied) == {(1, 0), (1, 1), (2, 0), (2, 1)}


def test_qc_endpoints_in_openapi_docs(client):
    assert client.get("/docs").status_code == 200
    spec = client.get("/openapi.json").json()
    assert "/qc/batches" in spec["paths"]
    assert "/qc/batches/{batch_id}" in spec["paths"]
