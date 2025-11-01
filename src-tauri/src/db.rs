use anyhow::{Context, Result};
use rusqlite::{params, Connection};
use std::path::{Path, PathBuf};
use tauri::Manager;

pub fn db_path(app: &tauri::AppHandle) -> Result<PathBuf> {
    // Unified location to match Python worker venv/cache on Windows
    #[cfg(target_os = "windows")]
    {
        if let Ok(local) = std::env::var("LOCALAPPDATA") {
            let base = PathBuf::from(local).join("SampleMap");
            let unified = base.join("samplemap.sqlite");
            let _ = std::fs::create_dir_all(&base);
            // One-time migrate from previous app_data_dir if needed
            if !unified.exists() {
                // prior Tauri app_data_dir location
                if let Ok(old_base) = app.path().app_data_dir() { let old = old_base.join("synthmap.sqlite"); if old.exists() { let _ = std::fs::copy(&old, &unified); } }
                // migrate from old SynthMap folder/name
                let old_named = PathBuf::from(local).join("SynthMap").join("synthmap.sqlite");
                if !unified.exists() && old_named.exists() { let _ = std::fs::copy(&old_named, &unified); }
            }
            return Ok(unified);
        }
    }

    // Non-Windows: use ~/.local/share/samplemap/samplemap.sqlite
    #[cfg(not(target_os = "windows"))]
    {
        if let Ok(home) = std::env::var("HOME") {
            let base = PathBuf::from(home).join(".local").join("share").join("samplemap");
            let unified = base.join("samplemap.sqlite");
            let _ = std::fs::create_dir_all(&base);
            if !unified.exists() {
                if let Ok(old_base) = app.path().app_data_dir() { let old = old_base.join("synthmap.sqlite"); if old.exists() { let _ = std::fs::copy(&old, &unified); } }
                // migrate from old synthmap path
                let old_named = PathBuf::from(home).join(".local").join("share").join("synthmap").join("synthmap.sqlite");
                if !unified.exists() && old_named.exists() { let _ = std::fs::copy(&old_named, &unified); }
            }
            return Ok(unified);
        }
    }

    // Fallback to Tauri app_data_dir
    let base = app
        .path()
        .app_data_dir()
        .context("no app_data_dir path available")?;
    std::fs::create_dir_all(&base).ok();
    Ok(base.join("samplemap.sqlite"))
}

pub fn open_or_create(path: &Path) -> Result<Connection> {
    let mut conn = Connection::open(path).with_context(|| format!("open db at {}", path.display()))?;
    conn.pragma_update(None, "journal_mode", &"WAL")?;
    conn.pragma_update(None, "synchronous", &"NORMAL")?;
    migrate(&mut conn)?;
    Ok(conn)
}

fn migrate(conn: &mut Connection) -> Result<()> {
    conn.execute_batch(
        r#"
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT
        );

        CREATE TABLE IF NOT EXISTS files (
            id INTEGER PRIMARY KEY,
            path TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            duration REAL,
            mtime INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS embeddings (
            file_id INTEGER PRIMARY KEY,
            dim INTEGER NOT NULL,
            vec BLOB NOT NULL,
            FOREIGN KEY(file_id) REFERENCES files(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS coords (
            file_id INTEGER PRIMARY KEY,
            x REAL NOT NULL,
            y REAL NOT NULL,
            FOREIGN KEY(file_id) REFERENCES files(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_files_path ON files(path);
        CREATE INDEX IF NOT EXISTS idx_embeddings_file ON embeddings(file_id);
        CREATE INDEX IF NOT EXISTS idx_coords_file ON coords(file_id);
        "#,
    )?;

    conn.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES('schema_version','1')",
        [],
    )?;
    Ok(())
}

pub struct FileRow<'a> {
    pub path: &'a str,
    pub name: &'a str,
    pub size_bytes: i64,
    pub duration: Option<f64>,
    pub mtime: i64,
}

pub fn upsert_file(conn: &Connection, f: &FileRow) -> Result<()> {
    // Update only if new or modified by mtime
    conn.execute(
        r#"
        INSERT INTO files(path, name, size_bytes, duration, mtime)
        VALUES(?, ?, ?, ?, ?)
        ON CONFLICT(path) DO UPDATE SET
            name=excluded.name,
            size_bytes=excluded.size_bytes,
            duration=excluded.duration,
            mtime=excluded.mtime
        WHERE files.mtime <> excluded.mtime;
        "#,
        params![f.path, f.name, f.size_bytes, f.duration, f.mtime],
    )?;
    Ok(())
}

pub fn file_count(conn: &Connection) -> Result<i64> {
    let mut stmt = conn.prepare("SELECT COUNT(*) FROM files")?;
    let cnt: i64 = stmt.query_row([], |r| r.get(0))?;
    Ok(cnt)
}
