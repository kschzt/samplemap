#[cfg_attr(mobile, tauri::mobile_entry_point)]
use std::{path::PathBuf, process::Command};
use tauri::Manager;

mod playback;
mod db;
mod scan;
mod worker;

use std::sync::Arc;
use parking_lot::Mutex;

struct AppState {
    audio: playback::AudioHandle,
    scans: Arc<scan::ScanManager>,
}

impl AppState {
    fn new() -> anyhow::Result<Self> { Ok(Self { audio: playback::AudioHandle::new()?, scans: Arc::new(Default::default()) }) }
}

#[tauri::command]
fn play_file(state: tauri::State<AppState>, path: String) -> Result<(), String> {
    state.audio.play_path(PathBuf::from(path)).map_err(|e| e.to_string())
}

#[tauri::command]
fn stop_playback(state: tauri::State<AppState>) -> Result<(), String> {
    state.audio.stop();
    Ok(())
}

#[tauri::command]
fn reveal_in_explorer(path: String) -> Result<(), String> {
    // Windows-specific: open Explorer with the file selected
    Command::new("explorer")
        .args(["/select,", &path])
        .spawn()
        .map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
fn copy_to_clipboard(app: tauri::AppHandle, text: String) -> Result<(), String> {
    use tauri_plugin_clipboard_manager::ClipboardExt;
    let clipboard = app.clipboard();
    clipboard
        .write_text(text)
        .map_err(|e| format!("clipboard error: {e}"))
}

#[derive(serde::Serialize)]
struct FileEntry {
    path: String,
    name: String,
}

#[tauri::command]
fn list_wavs(root_path: String, limit: Option<usize>) -> Result<Vec<FileEntry>, String> {
    let mut out = Vec::new();
    let lim = limit.unwrap_or(1000);
    for entry in walkdir::WalkDir::new(&root_path)
        .follow_links(true)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
    {
        let p = entry.path();
        if let Some(ext) = p.extension().and_then(|s| s.to_str()) {
            if ext.eq_ignore_ascii_case("wav") {
                let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("").to_string();
                out.push(FileEntry { path: p.to_string_lossy().to_string(), name });
                if out.len() >= lim { break; }
            }
        }
    }
    Ok(out)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .setup(|app| {
            if cfg!(debug_assertions) {
                app.handle().plugin(
                    tauri_plugin_log::Builder::default()
                        .level(log::LevelFilter::Info)
                        .build(),
                )?;
            }
            // Clipboard plugin
            let _ = app.handle().plugin(tauri_plugin_clipboard_manager::init());
            let _ = app.handle().plugin(tauri_plugin_dialog::init());
            Ok(())
        })
        .manage(AppState::new().expect("audio init"))
        .invoke_handler(tauri::generate_handler![
            play_file,
            stop_playback,
            reveal_in_explorer,
            copy_to_clipboard,
            list_wavs,
            start_scan,
            scan_status,
            get_stats,
            get_coords,
            get_file_info
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct ScanStart { job_id: String }

#[tauri::command]
fn start_scan(app: tauri::AppHandle, state: tauri::State<AppState>, root_path: String) -> Result<ScanStart, String> {
    let id = scan::start_scan(app, root_path, state.scans.clone());
    Ok(ScanStart { job_id: id })
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct ScanStatusResp { stage: String, processed: usize, total: usize, done: bool, error: Option<String> }

#[tauri::command]
fn scan_status(state: tauri::State<AppState>, job_id: String) -> Result<ScanStatusResp, String> {
    let jobs = state.scans.jobs.lock();
    let st = jobs.get(&job_id).ok_or_else(|| "job not found".to_string())?.lock().clone();
    Ok(ScanStatusResp { stage: st.stage, processed: st.processed, total: st.total, done: st.done, error: st.error })
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct Stats { file_count: i64, embedding_count: i64, coord_count: i64, db_path: String }

#[tauri::command]
fn get_stats(app: tauri::AppHandle) -> Result<Stats, String> {
    let p = db::db_path(&app).map_err(|e| e.to_string())?;
    let conn = db::open_or_create(&p).map_err(|e| e.to_string())?;
    let files = db::file_count(&conn).map_err(|e| e.to_string())?;
    let emb: i64 = conn.prepare("SELECT COUNT(*) FROM embeddings").map_err(|e| e.to_string())?
        .query_row([], |r| r.get(0)).map_err(|e| e.to_string())?;
    let coords: i64 = conn.prepare("SELECT COUNT(*) FROM coords").map_err(|e| e.to_string())?
        .query_row([], |r| r.get(0)).map_err(|e| e.to_string())?;
    Ok(Stats { file_count: files, embedding_count: emb, coord_count: coords, db_path: p.to_string_lossy().to_string() })
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct Point { file_id: i64, x: f32, y: f32 }

#[tauri::command]
fn get_coords(app: tauri::AppHandle, offset: Option<i64>, limit: Option<i64>) -> Result<Vec<Point>, String> {
    let p = db::db_path(&app).map_err(|e| e.to_string())?;
    let conn = db::open_or_create(&p).map_err(|e| e.to_string())?;
    let off = offset.unwrap_or(0);
    let lim = limit.unwrap_or(10000);
    let mut stmt = conn.prepare("SELECT file_id, x, y FROM coords ORDER BY file_id LIMIT ? OFFSET ?").map_err(|e| e.to_string())?;
    let rows = stmt
        .query_map(rusqlite::params![lim, off], |r| {
            Ok(Point { file_id: r.get::<_, i64>(0)?, x: r.get::<_, f64>(1)? as f32, y: r.get::<_, f64>(2)? as f32 })
        })
        .map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for r in rows { out.push(r.map_err(|e| e.to_string())?); }
    Ok(out)
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct FileInfo { path: String, name: String, size_bytes: i64, duration: Option<f64> }

#[tauri::command]
fn get_file_info(app: tauri::AppHandle, file_id: i64) -> Result<FileInfo, String> {
    let p = db::db_path(&app).map_err(|e| e.to_string())?;
    let conn = db::open_or_create(&p).map_err(|e| e.to_string())?;
    let mut stmt = conn.prepare("SELECT path, name, size_bytes, duration FROM files WHERE id = ?").map_err(|e| e.to_string())?;
    let r = stmt.query_row(rusqlite::params![file_id], |r| {
        Ok(FileInfo { path: r.get(0)?, name: r.get(1)?, size_bytes: r.get(2)?, duration: r.get(3)? })
    }).map_err(|e| e.to_string())?;
    Ok(r)
}
