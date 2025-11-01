use crate::db;
use anyhow::{Context, Result};
use std::{path::{PathBuf}, process::Command};
use tauri::{AppHandle, Manager};

fn find_worker(app: &AppHandle) -> Result<PathBuf> {
    // Prefer bundled resource dir
    if let Ok(dir) = app.path().resource_dir() {
        let p = dir.join("python").join("worker.py");
        if p.exists() { return Ok(p); }
    }
    // Fallback: search upwards from current exe for /python/worker.py (dev mode)
    let mut exe_dir = std::env::current_exe()?.parent().map(|p| p.to_path_buf()).unwrap_or_default();
    for _ in 0..6 {
        let p = exe_dir.join("python").join("worker.py");
        if p.exists() { return Ok(p); }
        if let Some(parent) = exe_dir.parent() { exe_dir = parent.to_path_buf(); } else { break; }
    }
    // Fallbacks relative to current working directory in dev
    let here = std::env::current_dir()?;
    let candidates = [
        here.join("python").join("worker.py"),
        here.join("src-tauri").join("python").join("worker.py"),
        here.join("app").join("src-tauri").join("python").join("worker.py"),
        here.join("..").join("src-tauri").join("python").join("worker.py"),
    ];
    for c in candidates { if c.exists() { return Ok(c); } }
    anyhow::bail!("worker.py not found in resources or nearby filesystem")
}

fn find_python() -> String {
    // Prefer the Windows py launcher
    if cfg!(target_os = "windows") {
        return "py".to_string();
    }
    "python3".to_string()
}

pub fn run_pipeline(app: &AppHandle, stage: &str) -> Result<()> {
    let dbp = db::db_path(app)?;
    let worker = find_worker(app)?;
    let python = find_python();
    let args = if cfg!(target_os = "windows") && python == "py" {
        vec!["-3".into(), worker.to_string_lossy().to_string(), dbp.to_string_lossy().to_string(), stage.into()]
    } else {
        vec![worker.to_string_lossy().to_string(), dbp.to_string_lossy().to_string(), stage.into()]
    };
    let status = Command::new(python)
        .args(args)
        .status()
        .context("failed to spawn python worker")?;
    if !status.success() { anyhow::bail!("python worker exited with status {status}"); }
    Ok(())
}
