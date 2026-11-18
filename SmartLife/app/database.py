"""SmartLife 的 SQLite 初始化和数据访问函数。"""

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from .schemas import AlertSetting, EventType, SoundEvent


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATABASE_PATH = DATA_DIR / "smartlife.db"
EVENT_TYPES: tuple[EventType, ...] = ("doorbell", "knock", "kettle", "alarm")


def _create_sound_events_table(
    connection: sqlite3.Connection, table_name: str = "sound_events"
) -> None:
    """创建当前版本的 sound_events 表；表名只由内部代码传入。"""
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL
                CHECK (event_type IN ('doorbell', 'knock', 'kettle', 'alarm')),
            occurred_at TEXT NOT NULL,
            confidence REAL CHECK (confidence >= 0 AND confidence <= 1),
            source TEXT NOT NULL
                CHECK (source IN ('simulated', 'acoustic'))
        )
        """
    )


def _migrate_sound_events_source(connection: sqlite3.Connection) -> bool:
    """把旧 source CHECK 安全迁移为 simulated/acoustic。

    返回 True 表示执行了迁移，False 表示表已是当前结构。
    `alert_settings` 不参与本函数，任何列、行和数据都不会被改动。
    """
    row = connection.execute(
        """
        SELECT sql
        FROM sqlite_master
        WHERE type = 'table' AND name = 'sound_events'
        """
    ).fetchone()
    if row is None:
        _create_sound_events_table(connection)
        return False

    schema_sql = str(row["sql"] or "")
    if "'simulated'" in schema_sql and "'acoustic'" in schema_sql:
        return False

    temp_table = "sound_events__source_migration"
    connection.execute("BEGIN IMMEDIATE")
    try:
        connection.execute(f"DROP TABLE IF EXISTS {temp_table}")
        _create_sound_events_table(connection, temp_table)

        before_count = int(
            connection.execute("SELECT COUNT(*) FROM sound_events").fetchone()[0]
        )
        connection.execute(
            f"""
            INSERT INTO {temp_table}
                (id, event_type, occurred_at, confidence, source)
            SELECT id, event_type, occurred_at, confidence, source
            FROM sound_events
            ORDER BY id
            """
        )
        after_count = int(
            connection.execute(f"SELECT COUNT(*) FROM {temp_table}").fetchone()[0]
        )
        if before_count != after_count:
            raise RuntimeError(
                "sound_events 迁移行数不一致："
                f"{before_count} -> {after_count}"
            )

        old_without_new = int(
            connection.execute(
                f"""
                SELECT COUNT(*)
                FROM sound_events AS old
                LEFT JOIN {temp_table} AS new ON new.id = old.id
                WHERE new.id IS NULL
                   OR old.event_type IS NOT new.event_type
                   OR old.occurred_at IS NOT new.occurred_at
                   OR old.confidence IS NOT new.confidence
                   OR old.source IS NOT new.source
                """
            ).fetchone()[0]
        )
        new_without_old = int(
            connection.execute(
                f"""
                SELECT COUNT(*)
                FROM {temp_table} AS new
                LEFT JOIN sound_events AS old ON old.id = new.id
                WHERE old.id IS NULL
                   OR new.event_type IS NOT old.event_type
                   OR new.occurred_at IS NOT old.occurred_at
                   OR new.confidence IS NOT old.confidence
                   OR new.source IS NOT old.source
                """
            ).fetchone()[0]
        )
        if old_without_new or new_without_old:
            raise RuntimeError(
                "sound_events 迁移内容核对失败："
                f"旧表差异 {old_without_new} 行，新表差异 {new_without_old} 行"
            )

        connection.execute("DROP TABLE sound_events")
        connection.execute(
            f"ALTER TABLE {temp_table} RENAME TO sound_events"
        )
        connection.execute(
            """
            CREATE INDEX idx_sound_events_occurred_at
            ON sound_events (occurred_at DESC, id DESC)
            """
        )
        connection.commit()
        return True
    except Exception:
        connection.rollback()
        raise


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    """打开一个数据库连接，并在结束时提交或回滚后关闭。"""
    connection = sqlite3.connect(DATABASE_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_database() -> None:
    """创建数据目录和表；重复启动不会覆盖已有事件或设置。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with get_connection() as connection:
        _migrate_sound_events_source(connection)
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_sound_events_occurred_at
            ON sound_events (occurred_at DESC, id DESC)
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS alert_settings (
                event_type TEXT PRIMARY KEY
                    CHECK (event_type IN ('doorbell', 'knock', 'kettle', 'alarm')),
                enabled INTEGER NOT NULL CHECK (enabled IN (0, 1)),
                updated_at TEXT NOT NULL
            )
            """
        )

        updated_at = datetime.now(timezone.utc).isoformat()
        for event_type in EVENT_TYPES:
            connection.execute(
                """
                INSERT OR IGNORE INTO alert_settings (event_type, enabled, updated_at)
                VALUES (?, 1, ?)
                """,
                (event_type, updated_at),
            )


def _event_from_row(row: sqlite3.Row) -> SoundEvent:
    return SoundEvent(
        id=row["id"],
        event_type=row["event_type"],
        occurred_at=datetime.fromisoformat(row["occurred_at"]),
        confidence=row["confidence"],
        source=row["source"],
    )


def create_simulated_event(
    event_type: EventType, confidence: float
) -> SoundEvent:
    """将服务端标记的模拟事件写入 SQLite 并返回记录。"""
    occurred_at = datetime.now(timezone.utc)
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO sound_events (event_type, occurred_at, confidence, source)
            VALUES (?, ?, ?, 'simulated')
            """,
            (event_type, occurred_at.isoformat(), confidence),
        )
        row = connection.execute(
            "SELECT * FROM sound_events WHERE id = ?",
            (cursor.lastrowid,),
        ).fetchone()

    if row is None:
        raise RuntimeError("新建的模拟事件无法从数据库读取。")
    return _event_from_row(row)


def create_acoustic_knock_event(occurred_at: datetime) -> SoundEvent:
    """把一次已确认的敲门簇写成正式事件。

    与 `create_simulated_event` 的差别只有三处，都是本阶段的硬要求：
    * `event_type` 固定为 'knock'，不接受调用方传入；
    * `source` 固定 'acoustic'；
    * `confidence` 固定 NULL —— 声学候选是规则证据，不是校准概率，
      用任何数字冒充置信度都会让前端误读。

    `occurred_at` 由调用方给出（候选首次确认的 UTC 时间），这里不重新取
    当前时间，避免数据库写入与候选确认之间的延迟改变事件时间。
    """
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO sound_events (event_type, occurred_at, confidence, source)
            VALUES ('knock', ?, NULL, 'acoustic')
            """,
            (occurred_at.isoformat(),),
        )
        row = connection.execute(
            "SELECT * FROM sound_events WHERE id = ?",
            (cursor.lastrowid,),
        ).fetchone()

    if row is None:
        raise RuntimeError("新建的声学敲门事件无法从数据库读取。")
    return _event_from_row(row)


def get_sound_events(limit: int, offset: int) -> list[SoundEvent]:
    """按发生时间倒序读取一页声音事件。"""
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT * FROM sound_events
            ORDER BY occurred_at DESC, id DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()
    return [_event_from_row(row) for row in rows]


def _setting_from_row(row: sqlite3.Row) -> AlertSetting:
    return AlertSetting(
        event_type=row["event_type"],
        enabled=bool(row["enabled"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


def get_alert_settings() -> list[AlertSetting]:
    """读取所有提醒开关。"""
    with get_connection() as connection:
        rows = connection.execute(
            "SELECT * FROM alert_settings ORDER BY event_type"
        ).fetchall()
    return [_setting_from_row(row) for row in rows]


def update_alert_setting(event_type: EventType, enabled: bool) -> AlertSetting:
    """更新一种提醒开关并返回数据库中的最新值。"""
    updated_at = datetime.now(timezone.utc)
    with get_connection() as connection:
        cursor = connection.execute(
            """
            UPDATE alert_settings
            SET enabled = ?, updated_at = ?
            WHERE event_type = ?
            """,
            (int(enabled), updated_at.isoformat(), event_type),
        )
        if cursor.rowcount != 1:
            raise RuntimeError(f"找不到提醒设置：{event_type}")
        row = connection.execute(
            "SELECT * FROM alert_settings WHERE event_type = ?",
            (event_type,),
        ).fetchone()

    if row is None:
        raise RuntimeError(f"更新后无法读取提醒设置：{event_type}")
    return _setting_from_row(row)
