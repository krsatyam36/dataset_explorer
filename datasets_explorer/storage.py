import sqlite3
import json
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from .models import Dataset, SearchQuery, DatasetSource
from .config import DB_PATH


class Storage:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def add_tag(self, dataset_id: int, tag: str) -> None:
        self._conn.execute("INSERT OR IGNORE INTO tags (dataset_id, tag) VALUES (?, ?)", (dataset_id, tag.strip().lower()))
        self._conn.commit()

    def remove_tag(self, dataset_id: int, tag: str) -> None:
        self._conn.execute("DELETE FROM tags WHERE dataset_id=? AND tag=?", (dataset_id, tag.strip().lower()))
        self._conn.commit()

    def get_tags(self, dataset_id: int) -> list[str]:
        rows = self._conn.execute("SELECT tag FROM tags WHERE dataset_id=?", (dataset_id,)).fetchall()
        return [r["tag"] for r in rows]

    def _init_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS search_queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                raw_query TEXT NOT NULL,
                parsed_intent TEXT DEFAULT '',
                format_filters TEXT DEFAULT '[]',
                keywords TEXT DEFAULT '[]',
                started_at TEXT,
                completed_at TEXT,
                status TEXT DEFAULT 'running',
                datasets_found INTEGER DEFAULT 0,
                session_log TEXT DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_id INTEGER NOT NULL,
                tag TEXT NOT NULL,
                UNIQUE(dataset_id, tag),
                FOREIGN KEY(dataset_id) REFERENCES datasets(id)
            );
            CREATE TABLE IF NOT EXISTS datasets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                url TEXT UNIQUE NOT NULL,
                download_url TEXT,
                source TEXT DEFAULT 'generic',
                formats TEXT DEFAULT '[]',
                size_bytes INTEGER,
                size_human TEXT,
                license TEXT,
                description TEXT DEFAULT '',
                tags TEXT DEFAULT '[]',
                date_range_start TEXT,
                date_range_end TEXT,
                num_samples INTEGER,
                relevance_score REAL DEFAULT 0.0,
                relevance_reasoning TEXT DEFAULT '',
                query_id INTEGER REFERENCES search_queries(id),
                discovered_at TEXT,
                reviewed INTEGER DEFAULT 0,
                notes TEXT DEFAULT ''
            );

            CREATE INDEX IF NOT EXISTS idx_datasets_query ON datasets(query_id);
            CREATE INDEX IF NOT EXISTS idx_datasets_relevance ON datasets(relevance_score DESC);
            CREATE INDEX IF NOT EXISTS idx_datasets_source ON datasets(source);
        """)
        # Migration: add columns to old DBs that lack them
        cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(datasets)").fetchall()}
        if "download_url" not in cols:
            self._conn.execute("ALTER TABLE datasets ADD COLUMN download_url TEXT")
        # Trust Card v1 columns — license + provenance metadata
        for col, ddl in [
            ("license_spdx",          "ALTER TABLE datasets ADD COLUMN license_spdx TEXT"),
            ("license_commercial_ok", "ALTER TABLE datasets ADD COLUMN license_commercial_ok INTEGER"),
            ("doi",                   "ALTER TABLE datasets ADD COLUMN doi TEXT"),
            ("authors",               "ALTER TABLE datasets ADD COLUMN authors TEXT DEFAULT '[]'"),
            ("institution",           "ALTER TABLE datasets ADD COLUMN institution TEXT"),
            ("country",               "ALTER TABLE datasets ADD COLUMN country TEXT"),
        ]:
            if col not in cols:
                self._conn.execute(ddl)
        # Use WAL mode for better durability under Ctrl+C
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.commit()

    def save_query(self, query: SearchQuery) -> int:
        cur = self._conn.execute(
            """INSERT INTO search_queries
               (raw_query, parsed_intent, format_filters, keywords, started_at, status)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                query.raw_query,
                query.parsed_intent,
                json.dumps(query.format_filters),
                json.dumps(query.keywords),
                query.started_at.isoformat(),
                query.status,
            ),
        )
        self._conn.commit()
        return cur.lastrowid

    def update_query_status(
        self,
        query_id: int,
        status: str,
        completed_at: Optional[datetime] = None,
    ) -> None:
        self._conn.execute(
            "UPDATE search_queries SET status=?, completed_at=? WHERE id=?",
            (status, completed_at.isoformat() if completed_at else None, query_id),
        )
        self._conn.commit()

    def update_datasets_found(self, query_id: int, count: int) -> None:
        self._conn.execute(
            "UPDATE search_queries SET datasets_found=? WHERE id=?",
            (count, query_id),
        )
        self._conn.commit()

    def save_dataset(self, dataset: Dataset) -> int:
        cur = self._conn.execute(
            """INSERT INTO datasets
               (name, url, download_url, source, formats, size_bytes, size_human, license, description,
                tags, date_range_start, date_range_end, num_samples, relevance_score,
                relevance_reasoning, query_id, discovered_at, reviewed, notes,
                license_spdx, license_commercial_ok, doi, authors, institution, country)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(url) DO UPDATE SET
                 relevance_score=MAX(relevance_score, excluded.relevance_score),
                 description=CASE WHEN length(excluded.description) > length(description)
                              THEN excluded.description ELSE description END,
                 download_url=COALESCE(excluded.download_url, download_url),
                 license_spdx=COALESCE(excluded.license_spdx, license_spdx),
                 license_commercial_ok=COALESCE(excluded.license_commercial_ok, license_commercial_ok),
                 doi=COALESCE(excluded.doi, doi),
                 authors=CASE WHEN length(excluded.authors) > length(COALESCE(authors,'[]'))
                              THEN excluded.authors ELSE authors END,
                 institution=COALESCE(excluded.institution, institution),
                 country=COALESCE(excluded.country, country),
                 query_id=excluded.query_id""",
            (
                dataset.name,
                dataset.url,
                dataset.download_url,
                dataset.source if isinstance(dataset.source, str) else dataset.source.value,
                json.dumps(dataset.formats),
                dataset.size_bytes,
                dataset.size_human,
                dataset.license,
                dataset.description,
                json.dumps(dataset.tags),
                dataset.date_range_start,
                dataset.date_range_end,
                dataset.num_samples,
                dataset.relevance_score,
                dataset.relevance_reasoning,
                dataset.query_id,
                dataset.discovered_at.isoformat(),
                int(dataset.reviewed),
                dataset.notes,
                dataset.license_spdx,
                None if dataset.license_commercial_ok is None else int(dataset.license_commercial_ok),
                dataset.doi,
                json.dumps(dataset.authors or []),
                dataset.institution,
                dataset.country,
            ),
        )
        self._conn.commit()
        return cur.lastrowid or self._get_id_by_url(dataset.url)

    def _get_id_by_url(self, url: str) -> int:
        row = self._conn.execute("SELECT id FROM datasets WHERE url=?", (url,)).fetchone()
        return row["id"] if row else -1

    def get_datasets(
        self,
        query_id: Optional[int] = None,
        min_relevance: float = 0.0,
        formats: Optional[List[str]] = None,
        reviewed: Optional[bool] = None,
        limit: int = 100,
    ) -> List[Dataset]:
        where_clauses = ["relevance_score >= ?"]
        params: list = [min_relevance]

        if query_id is not None:
            where_clauses.append("query_id = ?")
            params.append(query_id)

        if reviewed is not None:
            where_clauses.append("reviewed = ?")
            params.append(int(reviewed))

        sql = f"""SELECT * FROM datasets WHERE {' AND '.join(where_clauses)}
                  ORDER BY relevance_score DESC LIMIT ?"""
        params.append(limit)

        rows = self._conn.execute(sql, params).fetchall()
        datasets = [self._row_to_dataset(r) for r in rows]

        if formats:
            fmt_set = set(f.lower() for f in formats)
            datasets = [d for d in datasets if any(f.lower() in fmt_set for f in d.formats)]

        return datasets

    def _row_to_dataset(self, row: sqlite3.Row) -> Dataset:
        keys = set(row.keys())
        def col(name, default=None):
            return row[name] if name in keys else default
        lc = col("license_commercial_ok")
        return Dataset(
            id=row["id"],
            name=row["name"],
            url=row["url"],
            download_url=col("download_url"),
            source=row["source"],
            formats=json.loads(row["formats"] or "[]"),
            size_bytes=row["size_bytes"],
            size_human=row["size_human"],
            license=row["license"],
            description=row["description"] or "",
            tags=json.loads(row["tags"] or "[]"),
            date_range_start=row["date_range_start"],
            date_range_end=row["date_range_end"],
            num_samples=row["num_samples"],
            relevance_score=row["relevance_score"],
            relevance_reasoning=row["relevance_reasoning"] or "",
            query_id=row["query_id"],
            discovered_at=datetime.fromisoformat(row["discovered_at"]),
            reviewed=bool(row["reviewed"]),
            notes=row["notes"] or "",
            license_spdx=col("license_spdx"),
            license_commercial_ok=None if lc is None else bool(lc),
            doi=col("doi"),
            authors=json.loads(col("authors") or "[]") if col("authors") else [],
            institution=col("institution"),
            country=col("country"),
        )

    def list_queries(self) -> List[SearchQuery]:
        rows = self._conn.execute(
            "SELECT * FROM search_queries ORDER BY id DESC"
        ).fetchall()
        return [
            SearchQuery(
                id=r["id"],
                raw_query=r["raw_query"],
                parsed_intent=r["parsed_intent"] or "",
                format_filters=json.loads(r["format_filters"] or "[]"),
                keywords=json.loads(r["keywords"] or "[]"),
                started_at=datetime.fromisoformat(r["started_at"]),
                completed_at=datetime.fromisoformat(r["completed_at"]) if r["completed_at"] else None,
                status=r["status"],
                datasets_found=r["datasets_found"],
            )
            for r in rows
        ]

    def get_all_urls(self) -> set[str]:
        """Return every URL we have ever stored across all queries.
        Used to skip rediscovering datasets we already found."""
        rows = self._conn.execute(
            "SELECT url, download_url FROM datasets"
        ).fetchall()
        urls: set[str] = set()
        for r in rows:
            if r["url"]:
                urls.add(r["url"])
            if r["download_url"]:
                urls.add(r["download_url"])
        return urls

    def get_all_urls_with_names(self) -> list[tuple[str, str]]:
        """Return (url, name) for every stored dataset — used to brief the agent
        about what's already in the global database."""
        rows = self._conn.execute(
            "SELECT url, name FROM datasets ORDER BY relevance_score DESC LIMIT 200"
        ).fetchall()
        return [(r["url"], r["name"]) for r in rows]

    def update_dataset_notes(self, dataset_id: int, notes: str) -> None:
        self._conn.execute("UPDATE datasets SET notes=? WHERE id=?", (notes, dataset_id))
        self._conn.commit()

    def mark_reviewed(self, dataset_id: int, reviewed: bool = True) -> None:
        self._conn.execute("UPDATE datasets SET reviewed=? WHERE id=?", (int(reviewed), dataset_id))
        self._conn.commit()
# feat/storage-tags: Add storage-tags validation logic
# feat/storage-tags: Add storage-tags error handling
# feat/storage-tags: Add storage-tags docstring examples
