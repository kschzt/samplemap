use crate::db::{db_path, open_or_create, upsert_file, FileRow};
use crate::worker;
use anyhow::Result;
use hound::WavReader;
use rusqlite::Connection;
use std::{fs, path::Path, sync::Arc, thread, time::SystemTime};
use tauri::Manager;

use parking_lot::Mutex;
use uuid::Uuid;
use walkdir::WalkDir;

#[derive(Clone, serde::Serialize)]
pub struct ScanStatus {
    pub stage: String,
    pub processed: usize,
    pub total: usize,
    pub done: bool,
    pub error: Option<String>,
}

impl Default for ScanStatus {
    fn default() -> Self {
        Self { stage: "idle".into(), processed: 0, total: 0, done: false, error: None }
    }
}

pub struct ScanManager {
    pub jobs: Mutex<std::collections::HashMap<String, Arc<Mutex<ScanStatus>>>>,
}

impl Default for ScanManager {
    fn default() -> Self { Self { jobs: Mutex::new(Default::default()) } }
}

pub fn start_scan(app: tauri::AppHandle, root: String, mgr: Arc<ScanManager>) -> String {
    let job_id = Uuid::new_v4().to_string();
    let status = Arc::new(Mutex::new(ScanStatus::default()));
    mgr.jobs.lock().insert(job_id.clone(), status.clone());

    thread::spawn(move || {
        let res = do_scan(&app, &root, &status);
        if let Err(e) = res {
            let mut s = status.lock();
            s.error = Some(e.to_string());
            s.done = true;
        }
    });

    job_id
}

fn do_scan(app: &tauri::AppHandle, root: &str, status: &Arc<Mutex<ScanStatus>>) -> Result<()> {
    // Count WAVs
    {
        let mut s = status.lock();
        s.stage = "scanning".into();
        s.total = WalkDir::new(root)
            .follow_links(true)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
            .filter(|e| e.path().extension().and_then(|x| x.to_str()).map(|x| x.eq_ignore_ascii_case("wav")).unwrap_or(false))
            .count();
        s.processed = 0;
    }

    let dbfile = db_path(app)?;
    let mut conn = open_or_create(&dbfile)?;

    for entry in WalkDir::new(root).follow_links(true).into_iter().filter_map(|e| e.ok()) {
        let p = entry.path();
        if !entry.file_type().is_file() { continue; }
        if p.extension().and_then(|x| x.to_str()).map(|x| x.eq_ignore_ascii_case("wav")).unwrap_or(false) {
            let _ = upsert_one(&mut conn, p);
            let mut s = status.lock();
            s.processed += 1;
        }
    }

    {
        let mut s = status.lock();
        s.stage = "embedding".into();
    }
    // Run embeddings + umap via python worker
    match worker::run_pipeline(app, "all") {
        Ok(_) => {
            let mut s = status.lock();
            s.stage = "done".into();
            s.done = true;
        }
        Err(e) => {
            let mut s = status.lock();
            s.stage = "done".into();
            s.error = Some(format!("embedding failed: {e}"));
            s.done = true;
        }
    }
    Ok(())
}

fn upsert_one(conn: &mut Connection, path: &Path) -> Result<()> {
    let meta = fs::metadata(path)?;
    let size_bytes = meta.len() as i64;
    let mtime = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");

    let duration = wav_duration_seconds(path).ok();
    let row = FileRow { path: &path.to_string_lossy(), name, size_bytes, duration, mtime };
    upsert_file(conn, &row)
}

fn wav_duration_seconds(path: &Path) -> Result<f64> {
    let r = WavReader::open(path)?;
    let spec = r.spec();
    let samples_per_ch = r.duration() as f64; // per channel count
    let sr = spec.sample_rate as f64;
    Ok(samples_per_ch / sr)
}
