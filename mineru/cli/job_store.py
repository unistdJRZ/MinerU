import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobStore:
    SCHEMA_VERSION = 2

    def __init__(self, db_path: str):
        self.db_path = os.path.abspath(db_path)
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS parse_jobs (
                    ocr_id TEXT PRIMARY KEY,
                    task_id TEXT,
                    source_ocr_id TEXT,
                    result_hash TEXT NOT NULL,
                    status TEXT NOT NULL,
                    unique_dir TEXT NOT NULL,
                    output_dir TEXT NOT NULL,
                    source_pdf_file_names_json TEXT NOT NULL,
                    response_pdf_file_names_json TEXT NOT NULL,
                    upload_names_json TEXT NOT NULL DEFAULT '[]',
                    uploads_json TEXT NOT NULL DEFAULT '[]',
                    file_suffixes_json TEXT NOT NULL DEFAULT '[]',
                    pdf_hashes_json TEXT NOT NULL,
                    lang_list_json TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    backend TEXT NOT NULL,
                    parse_method TEXT NOT NULL,
                    formula_enable INTEGER NOT NULL,
                    table_enable INTEGER NOT NULL,
                    image_analysis INTEGER NOT NULL DEFAULT 1,
                    server_url TEXT,
                    return_md INTEGER NOT NULL,
                    return_middle_json INTEGER NOT NULL,
                    return_model_output INTEGER NOT NULL,
                    return_content_list INTEGER NOT NULL,
                    return_images INTEGER NOT NULL,
                    return_original_file INTEGER NOT NULL DEFAULT 0,
                    start_page_id INTEGER NOT NULL,
                    end_page_id INTEGER NOT NULL,
                    content_json TEXT,
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    completed_at TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_parse_jobs_result_hash_status
                ON parse_jobs(result_hash, status)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_parse_jobs_updated_at
                ON parse_jobs(updated_at)
                """
            )
            self._migrate_schema(conn)
            conn.execute(f"PRAGMA user_version={self.SCHEMA_VERSION}")

    def _migrate_schema(self, conn: sqlite3.Connection) -> None:
        rows = conn.execute("PRAGMA table_info(parse_jobs)").fetchall()
        existing_columns = {row["name"] for row in rows}
        column_definitions = {
            "task_id": "TEXT",
            "upload_names_json": "TEXT NOT NULL DEFAULT '[]'",
            "uploads_json": "TEXT NOT NULL DEFAULT '[]'",
            "file_suffixes_json": "TEXT NOT NULL DEFAULT '[]'",
            "image_analysis": "INTEGER NOT NULL DEFAULT 1",
            "return_original_file": "INTEGER NOT NULL DEFAULT 0",
        }
        for column_name, definition in column_definitions.items():
            if column_name not in existing_columns:
                conn.execute(
                    f"ALTER TABLE parse_jobs ADD COLUMN {column_name} {definition}"
                )

    def mark_incomplete_jobs_failed(self, error_message: str) -> None:
        now = utc_now_iso()
        error_json = json.dumps({"error": error_message}, ensure_ascii=False)
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE parse_jobs
                SET status = ?,
                    error_message = ?,
                    content_json = ?,
                    updated_at = ?,
                    completed_at = COALESCE(completed_at, ?)
                WHERE status IN ('PENDING', 'RUNNING')
                """,
                ("FAIL", error_message, error_json, now, now),
            )

    def create_job(self, job: Dict[str, Any]) -> None:
        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO parse_jobs (
                    ocr_id,
                    task_id,
                    source_ocr_id,
                    result_hash,
                    status,
                    unique_dir,
                    output_dir,
                    source_pdf_file_names_json,
                    response_pdf_file_names_json,
                    upload_names_json,
                    uploads_json,
                    file_suffixes_json,
                    pdf_hashes_json,
                    lang_list_json,
                    config_json,
                    backend,
                    parse_method,
                    formula_enable,
                    table_enable,
                    image_analysis,
                    server_url,
                    return_md,
                    return_middle_json,
                    return_model_output,
                    return_content_list,
                    return_images,
                    return_original_file,
                    start_page_id,
                    end_page_id,
                    content_json,
                    error_message,
                    created_at,
                    updated_at,
                    completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ocr_id) DO UPDATE SET
                    task_id = excluded.task_id,
                    source_ocr_id = excluded.source_ocr_id,
                    result_hash = excluded.result_hash,
                    status = excluded.status,
                    unique_dir = excluded.unique_dir,
                    output_dir = excluded.output_dir,
                    source_pdf_file_names_json = excluded.source_pdf_file_names_json,
                    response_pdf_file_names_json = excluded.response_pdf_file_names_json,
                    upload_names_json = excluded.upload_names_json,
                    uploads_json = excluded.uploads_json,
                    file_suffixes_json = excluded.file_suffixes_json,
                    pdf_hashes_json = excluded.pdf_hashes_json,
                    lang_list_json = excluded.lang_list_json,
                    config_json = excluded.config_json,
                    backend = excluded.backend,
                    parse_method = excluded.parse_method,
                    formula_enable = excluded.formula_enable,
                    table_enable = excluded.table_enable,
                    image_analysis = excluded.image_analysis,
                    server_url = excluded.server_url,
                    return_md = excluded.return_md,
                    return_middle_json = excluded.return_middle_json,
                    return_model_output = excluded.return_model_output,
                    return_content_list = excluded.return_content_list,
                    return_images = excluded.return_images,
                    return_original_file = excluded.return_original_file,
                    start_page_id = excluded.start_page_id,
                    end_page_id = excluded.end_page_id,
                    content_json = excluded.content_json,
                    error_message = excluded.error_message,
                    created_at = excluded.created_at,
                    updated_at = excluded.updated_at,
                    completed_at = excluded.completed_at
                """,
                (
                    job["ocr_id"],
                    job.get("task_id", job["ocr_id"]),
                    job.get("source_ocr_id"),
                    job["result_hash"],
                    job["status"],
                    job["unique_dir"],
                    job["output_dir"],
                    json.dumps(job["source_pdf_file_names"], ensure_ascii=False),
                    json.dumps(job["response_pdf_file_names"], ensure_ascii=False),
                    json.dumps(job.get("upload_names", []), ensure_ascii=False),
                    json.dumps(job.get("uploads", []), ensure_ascii=False),
                    json.dumps(job.get("file_suffixes", []), ensure_ascii=False),
                    json.dumps(job["pdf_hashes"], ensure_ascii=False),
                    json.dumps(job["lang_list"], ensure_ascii=False),
                    json.dumps(job["config"], ensure_ascii=False, sort_keys=True),
                    job["backend"],
                    job["parse_method"],
                    int(job["formula_enable"]),
                    int(job["table_enable"]),
                    int(job.get("image_analysis", True)),
                    job.get("server_url"),
                    int(job["return_md"]),
                    int(job["return_middle_json"]),
                    int(job["return_model_output"]),
                    int(job["return_content_list"]),
                    int(job["return_images"]),
                    int(job.get("return_original_file", False)),
                    job["start_page_id"],
                    job["end_page_id"],
                    self._dumps_json(job.get("content")),
                    job.get("error_message"),
                    now,
                    now,
                    now if job["status"] in {"FINISHED", "FAIL"} else None,
                ),
            )

    def update_job(
        self,
        ocr_id: str,
        *,
        status: str,
        content: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
    ) -> None:
        now = utc_now_iso()
        completed_at = now if status in {"FINISHED", "FAIL"} else None
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE parse_jobs
                SET status = ?,
                    content_json = ?,
                    error_message = ?,
                    updated_at = ?,
                    completed_at = COALESCE(?, completed_at)
                WHERE ocr_id = ?
                """,
                (
                    status,
                    self._dumps_json(content),
                    error_message,
                    now,
                    completed_at,
                    ocr_id,
                ),
            )

    def get_job(self, ocr_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM parse_jobs WHERE ocr_id = ?",
                (ocr_id,),
            ).fetchone()
        return self._row_to_job(row) if row else None

    def find_finished_job_by_result_hash(self, result_hash: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM parse_jobs
                WHERE result_hash = ? AND status = 'FINISHED'
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (result_hash,),
            ).fetchone()
        return self._row_to_job(row) if row else None

    @staticmethod
    def _dumps_json(value: Optional[Dict[str, Any]]) -> Optional[str]:
        if value is None:
            return None
        return json.dumps(value, ensure_ascii=False)

    @staticmethod
    def _loads_json(value: Optional[str], default: Any) -> Any:
        if not value:
            return default
        return json.loads(value)

    @staticmethod
    def _row_value(row: sqlite3.Row, key: str, default: Any = None) -> Any:
        if key not in row.keys():
            return default
        value = row[key]
        return default if value is None else value

    def _row_to_job(self, row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "ocr_id": row["ocr_id"],
            "task_id": self._row_value(row, "task_id", row["ocr_id"]),
            "source_ocr_id": row["source_ocr_id"],
            "result_hash": row["result_hash"],
            "status": row["status"],
            "unique_dir": row["unique_dir"],
            "output_dir": row["output_dir"],
            "source_pdf_file_names": self._loads_json(row["source_pdf_file_names_json"], []),
            "response_pdf_file_names": self._loads_json(row["response_pdf_file_names_json"], []),
            "upload_names": self._loads_json(self._row_value(row, "upload_names_json", None), []),
            "uploads": self._loads_json(self._row_value(row, "uploads_json", None), []),
            "file_suffixes": self._loads_json(self._row_value(row, "file_suffixes_json", None), []),
            "pdf_hashes": self._loads_json(row["pdf_hashes_json"], []),
            "lang_list": self._loads_json(row["lang_list_json"], []),
            "config": self._loads_json(row["config_json"], {}),
            "backend": row["backend"],
            "parse_method": row["parse_method"],
            "formula_enable": bool(row["formula_enable"]),
            "table_enable": bool(row["table_enable"]),
            "image_analysis": bool(self._row_value(row, "image_analysis", 1)),
            "server_url": row["server_url"],
            "return_md": bool(row["return_md"]),
            "return_middle_json": bool(row["return_middle_json"]),
            "return_model_output": bool(row["return_model_output"]),
            "return_content_list": bool(row["return_content_list"]),
            "return_images": bool(row["return_images"]),
            "return_original_file": bool(self._row_value(row, "return_original_file", 0)),
            "start_page_id": row["start_page_id"],
            "end_page_id": row["end_page_id"],
            "content": self._loads_json(row["content_json"], None),
            "error_message": row["error_message"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "completed_at": row["completed_at"],
        }
