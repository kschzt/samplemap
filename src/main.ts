import './style.css'
import { open } from '@tauri-apps/plugin-dialog'
import { invoke } from '@tauri-apps/api/core'
import { Scatter } from './scatter'
import type { Point as MapPoint } from './scatter'

const app = document.querySelector<HTMLDivElement>('#app')!
const statusbar = document.querySelector<HTMLDivElement>('#statusbar')!

let currentPath: string | null = null

app.innerHTML = `
  <div class="toolbar">
    <button id="choose">Choose folder</button>
    <span id="chosen" style="color:#888"></span>
    <span style="flex:1"></span>
    <div id="stage" style="font-size:12px;color:#aaa;margin-right:8px;">Idle</div>
    <progress id="prog" value="0" max="1" style="width:200px"></progress>
    <span id="pct" style="font-size:12px;color:#888;width:48px;text-align:right;">0%</span>
    <button id="refresh" style="margin-left:12px;">Refresh</button>
  </div>
  <div class="content">
    <div class="canvaswrap">
      <canvas id="gl" class="webgl"></canvas>
      <div class="hud">Left-click: play • Right-drag: pan • Wheel: zoom</div>
    </div>
  </div>
`

statusbar.innerHTML = `
  <div id="selpath">Selected: —</div>
  <button id="reveal">Reveal in Explorer</button>
`

const chooseBtn = document.getElementById('choose') as HTMLButtonElement
const chosenSpan = document.getElementById('chosen') as HTMLSpanElement
const selPathEl = document.getElementById('selpath') as HTMLDivElement
const revealBtn = document.getElementById('reveal') as HTMLButtonElement
const stageEl = document.getElementById('stage') as HTMLDivElement
const progEl = document.getElementById('prog') as HTMLProgressElement
const pctEl = document.getElementById('pct') as HTMLSpanElement
const refreshBtn = document.getElementById('refresh') as HTMLButtonElement
const canvas = document.getElementById('gl') as HTMLCanvasElement
const scatter = new Scatter(canvas)

let idToPath = new Map<number, string>()

async function loadCoords() {
  const page = 50000
  let off = 0
  const pts: MapPoint[] = []
  while (true) {
    const rows = await invoke('get_coords', { offset: off, limit: page }) as { fileId: number, x: number, y: number }[]
    if (rows.length === 0) break
    for (const r of rows) pts.push({ id: r.fileId, x: r.x, y: r.y })
    off += rows.length
  }
  scatter.setPoints(pts)
  // Build id->path cache for quick play
  idToPath.clear()
}

// Load any cached map data on startup so the user doesn't need to refresh
void loadCoords()

chooseBtn.addEventListener('click', async () => {
  let dir: string | null = null
  try {
    const sel = await open({ directory: true, multiple: false, defaultPath: 'D:/samples_from_mars' })
    if (typeof sel === 'string') dir = sel
  } catch (e) {
    console.warn('folder dialog failed; falling back to default', e)
    dir = 'D:/samples_from_mars'
  }
  if (!dir) return
  chosenSpan.textContent = dir
  await loadCoords()
  // kick off scan job
  try {
    const { jobId } = await invoke('start_scan', { rootPath: dir }) as { jobId: string }
    pollScan(jobId)
  } catch (e) {
    console.error('start_scan failed', e)
  }
})

revealBtn.addEventListener('click', async () => {
  if (currentPath) await invoke('reveal_in_explorer', { path: currentPath })
})

window.addEventListener('keydown', async (e) => {
  if (e.code === 'Space') {
    e.preventDefault()
    await invoke('stop_playback')
  }
})

async function pollScan(jobId: string) {
  const t = setInterval(async () => {
    try {
      const s = await invoke('scan_status', { jobId }) as { stage: string, processed: number, total: number, done: boolean, error?: string }
      stageEl.textContent = `Stage: ${s.stage}`
      const frac = s.total > 0 ? s.processed / s.total : 0
      progEl.max = 1
      progEl.value = frac
      pctEl.textContent = `${Math.round(frac * 100)}%`
      if (s.done) {
        clearInterval(t)
        if (s.error) stageEl.textContent = `Error: ${s.error}`
        // refresh coords after done
        await loadCoords()
      }
    } catch (e) {
      console.warn('scan_status error', e)
      clearInterval(t)
    }
  }, 600)
}

refreshBtn.addEventListener('click', async () => {
  await loadCoords()
  try {
    const stats = await invoke('get_stats', {}) as { fileCount: number, embeddingCount: number, coordCount: number }
    stageEl.textContent = `Files: ${stats.fileCount} • Emb: ${stats.embeddingCount} • Coords: ${stats.coordCount}`
  } catch {}
})

// Ensure play commands are serialized to preserve order
let playQueue = Promise.resolve()
scatter.onPick = (res) => {
  if (!res) return
  playQueue = playQueue.then(async () => {
    try {
      let path = idToPath.get(res.id)
      if (!path) {
        const info = await invoke('get_file_info', { fileId: res.id }) as { path: string }
        path = info.path
        idToPath.set(res.id, path)
      }
      await invoke('play_file', { path })
      selPathEl.textContent = `Selected: ${path}`
      currentPath = path
      try { await invoke('copy_to_clipboard', { text: path }) } catch {}
    } catch (e) {
      console.error('pick play failed', e)
    }
  })
}
