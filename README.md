# Sample Map

![Sample Map screenshot](./app.png)

Zoomable point‑cloud browser for WAV samples: scans a folder, embeds with CLAP (HTS‑AT), maps with UMAP, and renders a WebGL scatter; click any dot to instantly audition, copy its path, and reveal in Explorer.

Installation: install Rust and Node.js, then `npm i` and `npx tauri dev` (optional: install a CUDA build of PyTorch to accelerate embeddings).

Keyboard shortcuts:
- Left/Right (or Up/Down): navigate selection history and auto‑play
- Space: replay current selection
- Shift+Space: stop playback
- Left‑click: select + play
- Right‑drag: pan
- Wheel: zoom

Status: 0.1 — everything is working.

License: MIT.
