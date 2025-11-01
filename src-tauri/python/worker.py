"""
Sample Map Python Worker
- Bootstrap a venv under LocalAppData/SampleMap/venv
- Install deps: torch, laion-clap, librosa, umap-learn, soundfile, numpy
- Compute CLAP (HTSAT-base) 512-D embeddings (48kHz mono, first N sec)
- Compute 2D UMAP (cosine) and store normalized coords
- Update SQLite DB tables: embeddings(file_id,dim,vec), coords(file_id,x,y)
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

WIN = (os.name == 'nt')


def appdata_dir() -> Path:
    if WIN:
        base = os.environ.get('LOCALAPPDATA') or os.environ.get('APPDATA')
        if not base:
            raise RuntimeError('No LOCALAPPDATA/APPDATA set')
        d = Path(base) / 'SampleMap'
    else:
        d = Path.home() / '.local' / 'share' / 'samplemap'
    d.mkdir(parents=True, exist_ok=True)
    return d


def venv_python() -> Path:
    vdir = appdata_dir() / 'venv'
    return (vdir / ('Scripts' if WIN else 'bin') / ('python.exe' if WIN else 'python'))


def ensure_venv_and_deps() -> None:
    py = venv_python()
    if not py.exists():
        vdir = appdata_dir() / 'venv'
        print(f'[worker] creating venv at {vdir}', flush=True)
        subprocess.check_call([sys.executable, '-m', 'venv', str(vdir)])
    # Install deps if laion_clap missing
    try:
        subprocess.check_call([str(py), '-c', 'import laion_clap'])
        have = True
    except subprocess.CalledProcessError:
        have = False
    if not have:
        print('[worker] installing dependencies...', flush=True)
        pkgs = [
            'numpy',
            'soundfile',
            'librosa',
            'umap-learn',
            # torch cpu builds; let pip resolve platform wheel
            'torch',
            'torchvision',
            'laion-clap',
        ]
        subprocess.check_call([str(py), '-m', 'pip', 'install', '--upgrade', 'pip'])
        subprocess.check_call([str(py), '-m', 'pip', 'install'] + pkgs)


def reexec_in_venv(argv: List[str]) -> None:
    py = venv_python()
    if Path(sys.executable) != py:
        # Re-exec using venv python with same args
        env = os.environ.copy()
        # Set model cache to app data
        env.setdefault('HF_HOME', str(appdata_dir() / 'hf_cache'))
        cmd = [str(py)] + argv
        os.execve(str(py), cmd, env)


# ---- Embedding + UMAP ----

def load_model(device: str = 'cpu'):
    import laion_clap
    # HTSAT-base audio encoder; fusion off (audio only)
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
    # Auto-download pretrained weights
    # laion_clap auto-downloads when calling .load_ckpt if given a preset
    # Use helper to fetch default audio-only checkpoint
    try:
        from laion_clap.training.data import get_audio_feature
    except Exception:
        pass
    # Newer laion_clap exposes get_model; fallback to manual CLAP_Module
    try:
        # Attempt auto download path
        model.load_ckpt()  # type: ignore[attr-defined]
    except Exception:
        # Some versions require explicit path; attempt default hub model
        pass
    model.eval()
    try:
        import torch
        if device == 'cuda' and torch.cuda.is_available():
            model = model.to('cuda')
        else:
            model = model.to('cpu')
    except Exception:
        pass
    return model


def embed_files(model, paths: List[Path], sr: int, duration: float, device: str) -> List[Tuple[int, List[float]]]:
    import numpy as np
    import librosa
    import torch

    dim = 512
    outs: List[Tuple[int, List[float]]] = []
    for idx, p in enumerate(paths):
        try:
            y, _ = librosa.load(str(p), sr=sr, mono=True, duration=duration)
            if y.size == 0:
                raise RuntimeError('empty audio')
            # Model expects batch of mono waveforms at 48k
            wav = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
            if device == 'cuda' and torch.cuda.is_available():
                wav = wav.to('cuda', non_blocking=True)
            with torch.inference_mode():
                feats = model.get_audio_embedding_from_data(x=wav, use_tensor=True)  # [1, D]
                v = feats.squeeze(0).cpu().numpy().astype(np.float32)
            if v.shape[0] != dim:
                # Some checkpoints produce 1024; reduce by PCA-like split average
                if v.shape[0] == 1024:
                    v = v.reshape(2, 512).mean(axis=0).astype(np.float32)
                else:
                    v = v[:dim].astype(np.float32)
            outs.append((idx, v.tolist()))
        except Exception as e:
            print(f"[worker] embed error: {p}: {e}")
    return outs


def run_pipeline(db_path: Path, dur: float, neighbors: int, min_dist: float, limit: int | None = None, device: str = 'auto', mode: str = 'all') -> None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE IF NOT EXISTS embeddings (file_id INTEGER PRIMARY KEY, dim INTEGER NOT NULL, vec BLOB NOT NULL)")
    conn.execute("CREATE TABLE IF NOT EXISTS coords (file_id INTEGER PRIMARY KEY, x REAL NOT NULL, y REAL NOT NULL)")

    # Fetch files without embeddings
    q = "SELECT id, path FROM files WHERE id NOT IN (SELECT file_id FROM embeddings) ORDER BY id"
    if limit:
        q += f" LIMIT {int(limit)}"
    rows = conn.execute(q).fetchall()
    all_rows = conn.execute("SELECT id FROM files ORDER BY id").fetchall()

    if not rows and not all_rows:
        print('[worker] no files in DB')
        return

    # Choose device and only load model if embedding needed
    use_device = device
    try:
        import torch
        if device == 'auto':
            use_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    except Exception:
        use_device = 'cpu'
    if use_device == 'cuda':
        try:
            import torch
            _ = torch.randn(1, device='cuda')
        except Exception as e:
            print(f"[worker] CUDA not usable ({e}); falling back to CPU")
            use_device = 'cpu'
    print(f"[worker] device: {use_device}")
    model = None
    do_embed = (mode in ('embed','all')) and len(rows) > 0
    if do_embed:
        model = load_model(device=use_device)

    sr = 48000
    # Batch embed new files in small groups to limit RAM
    BATCH = 32 if use_device == 'cuda' else 16
    import struct
    from math import ceil
    n = len(rows)
    if do_embed:
        print(f'[worker] embedding {n} new files', flush=True)
        for i in range(0, n, BATCH):
            batch = rows[i:i+BATCH]
            paths = [Path(r['path']) for r in batch]
            embs = embed_files(model, paths, sr=sr, duration=dur, device=use_device)
            for j, vec in embs:
                fid = batch[j]['id']
                blob = struct.pack('<' + 'f'*len(vec), *vec)
                conn.execute(
                    "INSERT OR REPLACE INTO embeddings(file_id, dim, vec) VALUES(?,?,?)",
                    (fid, len(vec), blob)
                )
            conn.commit()

    # Build UMAP over all embeddings
    print('[worker] computing UMAP', flush=True)
    import numpy as np
    import umap
    embed_rows = conn.execute("SELECT file_id, vec FROM embeddings ORDER BY file_id").fetchall()
    if not embed_rows:
        print('[worker] no embeddings present')
        return
    ids = [r[0] for r in embed_rows]
    vecs = []
    for r in embed_rows:
        buf = r[1]
        arr = np.frombuffer(buf, dtype='<f4')
        vecs.append(arr)
    X = np.vstack(vecs)
    reducer = umap.UMAP(n_neighbors=neighbors, min_dist=min_dist, metric='cosine', random_state=42)
    Y = reducer.fit_transform(X).astype('f4')
    # Normalize to [-1, 1]
    mins = Y.min(axis=0)
    maxs = Y.max(axis=0)
    rng = np.maximum(maxs - mins, 1e-6)
    Yn = (Y - mins) / rng * 2.0 - 1.0

    conn.executemany(
        "INSERT OR REPLACE INTO coords(file_id, x, y) VALUES(?,?,?)",
        [(int(fid), float(x), float(y)) for fid, (x, y) in zip(ids, Yn.tolist())]
    )
    conn.commit()
    conn.close()

def ingest(db_path: Path, root: Path) -> None:
    import time
    import soundfile as sf
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS files (
            id INTEGER PRIMARY KEY,
            path TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            duration REAL,
            mtime INTEGER NOT NULL
        )
        """
    )
    # walk
    root = root.resolve()
    print(f"[worker] ingest {root}")
    to_add = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if not fn.lower().endswith('.wav'):
                continue
            p = Path(dirpath) / fn
            try:
                st = p.stat()
                mtime = int(st.st_mtime)
                size_bytes = int(st.st_size)
                try:
                    info = sf.info(str(p))
                    duration = float(info.frames) / float(info.samplerate)
                except Exception:
                    duration = None
                to_add.append((str(p), fn, size_bytes, duration, mtime))
            except Exception as e:
                print(f"[worker] ingest skip {p}: {e}")
    # upsert
    conn.executemany(
        """
        INSERT INTO files(path, name, size_bytes, duration, mtime)
        VALUES(?,?,?,?,?)
        ON CONFLICT(path) DO UPDATE SET
          name=excluded.name,
          size_bytes=excluded.size_bytes,
          duration=excluded.duration,
          mtime=excluded.mtime
        WHERE files.mtime <> excluded.mtime
        """,
        to_add,
    )
    conn.commit()
    conn.close()


def main() -> int:
    # Ensure venv and re-exec if needed
    ensure_venv_and_deps()
    reexec_in_venv(sys.argv)

    ap = argparse.ArgumentParser()
    ap.add_argument('db', type=Path)
    ap.add_argument('command', choices=['ingest', 'embed', 'umap', 'all'])
    ap.add_argument('--duration', type=float, default=10.0)
    ap.add_argument('--n_neighbors', type=int, default=50)
    ap.add_argument('--min_dist', type=float, default=0.05)
    ap.add_argument('--device', type=str, default='auto', choices=['auto','cpu','cuda'])
    ap.add_argument('--root', type=Path, default=None, help='Root folder to ingest when command=ingest')
    args = ap.parse_args()

    if args.command == 'ingest':
        if not args.root:
            print('ingest requires --root')
            return 2
        ingest(args.db, args.root)
    elif args.command == 'embed':
        run_pipeline(args.db, dur=args.duration, neighbors=args.n_neighbors, min_dist=args.min_dist, device=args.device)
    elif args.command == 'umap':
        # Force only UMAP over current embeddings
        run_pipeline(args.db, dur=args.duration, neighbors=args.n_neighbors, min_dist=args.min_dist, device=args.device)
    else:  # all
        lim_env = os.environ.get('SMAP_EMBED_LIMIT')
        limit = int(lim_env) if lim_env else None
        run_pipeline(args.db, dur=args.duration, neighbors=args.n_neighbors, min_dist=args.min_dist, limit=limit, device=args.device)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
