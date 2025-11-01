export type Point = { id: number; x: number; y: number }

type PickResult = { id: number; x: number; y: number; distPx: number } | null

export class Scatter {
  private canvas: HTMLCanvasElement
  private gl: WebGL2RenderingContext
  private program: WebGLProgram
  private buf: WebGLBuffer
  private vao: WebGLVertexArrayObject
  private uScale: WebGLUniformLocation
  private uTranslate: WebGLUniformLocation
  private uPointSize: WebGLUniformLocation
  private uColor: WebGLUniformLocation

  private points: Point[] = []
  private data: Float32Array = new Float32Array()
  private needsUpload = false
  private idToIndex = new Map<number, number>()

  // view state
  private scale = 1
  private tx = 0
  private ty = 0
  private dragging = false
  private lastX = 0
  private lastY = 0
  private selectedIndex = -1

  // spatial hash for picking in world space
  private grid = new Map<string, number[]>() // key:"gx,gy" -> indices
  private gridSize = 0.05 // world units (coords in [-1,1])

  onPick: (res: PickResult) => void = () => {}

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas
    const gl = canvas.getContext('webgl2') as WebGL2RenderingContext
    if (!gl) throw new Error('WebGL2 not available')
    this.gl = gl
    const vs = `#version 300 es
    precision highp float;
    layout(location=0) in vec2 aPos;
    uniform vec2 uScale;
    uniform vec2 uTranslate;
    uniform float uPointSize;
    void main(){
      vec2 p = aPos * uScale + uTranslate;
      gl_Position = vec4(p, 0.0, 1.0);
      gl_PointSize = uPointSize;
    }`
    const fs = `#version 300 es
    precision highp float;
    uniform vec3 uColor;
    out vec4 o;
    void main(){
      vec2 d = gl_PointCoord - vec2(0.5);
      float r = dot(d,d);
      if(r>0.25) discard;
      o = vec4(uColor,1.0);
    }`
    const prog = this.createProgram(vs, fs)
    this.program = prog
    this.uScale = gl.getUniformLocation(prog, 'uScale')!
    this.uTranslate = gl.getUniformLocation(prog, 'uTranslate')!
    this.uPointSize = gl.getUniformLocation(prog, 'uPointSize')!
    this.uColor = gl.getUniformLocation(prog, 'uColor')!
    this.buf = gl.createBuffer()!
    this.vao = gl.createVertexArray()!
    gl.bindVertexArray(this.vao)
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buf)
    gl.enableVertexAttribArray(0)
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 8, 0)
    gl.bindVertexArray(null)

    this.resize()
    // ResizeObserver type erased for build settings
    // @ts-ignore
    new ResizeObserver(() => this.resize()).observe(canvas)
    this.attachEvents()
    requestAnimationFrame(() => this.draw())
  }

  setPoints(pts: Point[]) {
    this.points = pts
    this.data = new Float32Array(pts.length * 2)
    this.idToIndex.clear()
    for (let i = 0; i < pts.length; i++) {
      this.data[i * 2 + 0] = pts[i].x
      this.data[i * 2 + 1] = pts[i].y
      this.idToIndex.set(pts[i].id, i)
    }
    this.rebuildGrid()
    this.needsUpload = true
    this.invalidate()
  }

  appendPoints(pts: Point[]) {
    if (pts.length === 0) return
    this.points.push(...pts)
    const newData = new Float32Array(this.points.length * 2)
    newData.set(this.data)
    for (let i = 0; i < pts.length; i++) {
      const idx = (this.points.length - pts.length + i) * 2
      newData[idx + 0] = pts[i].x
      newData[idx + 1] = pts[i].y
      this.idToIndex.set(pts[i].id, this.points.length - pts.length + i)
    }
    this.data = newData
    this.addToGrid(this.points.length - pts.length)
    this.needsUpload = true
    this.invalidate()
  }

  private rebuildGrid() {
    this.grid.clear()
    for (let i = 0; i < this.points.length; i++) {
      const p = this.points[i]
      const gx = Math.floor(p.x / this.gridSize)
      const gy = Math.floor(p.y / this.gridSize)
      const key = `${gx},${gy}`
      const arr = this.grid.get(key)
      if (arr) arr.push(i)
      else this.grid.set(key, [i])
    }
  }

  private addToGrid(startIndex: number) {
    for (let i = startIndex; i < this.points.length; i++) {
      const p = this.points[i]
      const gx = Math.floor(p.x / this.gridSize)
      const gy = Math.floor(p.y / this.gridSize)
      const key = `${gx},${gy}`
      const arr = this.grid.get(key)
      if (arr) arr.push(i)
      else this.grid.set(key, [i])
    }
  }

  private attachEvents() {
    const c = this.canvas
    c.addEventListener('contextmenu', (e) => e.preventDefault())
    c.addEventListener('mousedown', (e) => {
      if (e.button === 2) {
        this.dragging = true
        this.lastX = e.clientX
        this.lastY = e.clientY
      }
    })
    window.addEventListener('mouseup', () => (this.dragging = false))
    window.addEventListener('mousemove', (e) => {
      if (!this.dragging) return
      const dx = e.clientX - this.lastX
      const dy = e.clientY - this.lastY
      this.lastX = e.clientX
      this.lastY = e.clientY
      // translate in NDC (pixels to NDC)
      const rect = this.canvas.getBoundingClientRect()
      const nx = (2 * dx) / rect.width
      const ny = (-2 * dy) / rect.height
      this.tx += nx
      this.ty += ny
      this.invalidate()
    })
    c.addEventListener('wheel', (e) => {
      e.preventDefault()
      const delta = Math.sign(e.deltaY)
      const zoom = Math.pow(1.1, -delta)
      // zoom to cursor
      const rect = c.getBoundingClientRect()
      const sx = ((e.clientX - rect.left) / rect.width) * 2 - 1
      const sy = -(((e.clientY - rect.top) / rect.height) * 2 - 1)
      // world before
      const wx = (sx - this.tx) / this.scale
      const wy = (sy - this.ty) / this.scale
      this.scale *= zoom
      // new translate to keep cursor focused
      this.tx = sx - wx * this.scale
      this.ty = sy - wy * this.scale
      this.invalidate()
    }, { passive: false })
    c.addEventListener('click', (e) => {
      const rect = c.getBoundingClientRect()
      const x = e.clientX - rect.left
      const y = e.clientY - rect.top
      const res = this.pick(x, y)
      this.onPick(res)
    })
  }

  setSelectedId(id: number | null) {
    if (id == null) { this.selectedIndex = -1; this.invalidate(); return }
    const idx = this.idToIndex.get(id)
    this.selectedIndex = (idx != null ? idx : -1)
    this.invalidate()
  }

  centerOnSelected(smooth = true) {
    if (this.selectedIndex < 0) return
    const p = this.points[this.selectedIndex]
    const targetTx = -p.x * this.scale
    const targetTy = -p.y * this.scale
    if (!smooth) {
      this.tx = targetTx; this.ty = targetTy; this.invalidate(); return
    }
    const startTx = this.tx, startTy = this.ty
    const dur = 150
    const t0 = performance.now()
    const step = () => {
      const t = Math.min(1, (performance.now() - t0) / dur)
      const ease = t * (2 - t)
      this.tx = startTx + (targetTx - startTx) * ease
      this.ty = startTy + (targetTy - startTy) * ease
      this.invalidate()
      if (t < 1) requestAnimationFrame(step)
    }
    requestAnimationFrame(step)
  }

  private pick(px: number, py: number): PickResult {
    // Use CSS pixel space for input and projection to avoid DPR mismatch
    const rect = this.canvas.getBoundingClientRect()
    // screen -> NDC (CSS pixels)
    const sx = (px / rect.width) * 2 - 1
    const sy = -((py / rect.height) * 2 - 1)
    // unproject to world
    const wx = (sx - this.tx) / this.scale
    const wy = (sy - this.ty) / this.scale
    // search nearby grid cells
    const gx = Math.floor(wx / this.gridSize)
    const gy = Math.floor(wy / this.gridSize)
    let best: PickResult = null
    // Expand search radius until at least one candidate is found
    const maxR = 48 // enough to cover entire view with gridSize ~0.05
    for (let r = 0; r <= maxR; r++) {
      for (let dy = -r; dy <= r; dy++) {
        for (let dx = -r; dx <= r; dx++) {
          // Only check the border of the square ring for r>0 to reduce repeats
          if (r > 0 && Math.abs(dx) !== r && Math.abs(dy) !== r) continue
          const arr = this.grid.get(`${gx + dx},${gy + dy}`)
          if (!arr) continue
          for (const i of arr) {
            const p = this.points[i]
            const sx2 = p.x * this.scale + this.tx
            const sy2 = p.y * this.scale + this.ty
            // NDC -> CSS pixels
            const px2 = ((sx2 + 1) * 0.5) * rect.width
            const py2 = ((-sy2 + 1) * 0.5) * rect.height
            const dxp = px2 - px
            const dyp = py2 - py
            const d = Math.hypot(dxp, dyp)
            if (!best || d < best.distPx) {
              best = { id: this.points[i].id, x: p.x, y: p.y, distPx: d }
            }
          }
        }
      }
      if (best) break
    }
    return best
  }

  private resize() {
    const dpr = Math.max(1, Math.min(2, window.devicePixelRatio || 1))
    const displayW = Math.floor(this.canvas.clientWidth * dpr)
    const displayH = Math.floor(this.canvas.clientHeight * dpr)
    if (this.canvas.width !== displayW || this.canvas.height !== displayH) {
      this.canvas.width = displayW
      this.canvas.height = displayH
      this.invalidate()
    }
  }

  private invalidate() {
    this.draw()
  }

  private draw() {
    const gl = this.gl
    gl.viewport(0, 0, this.canvas.width, this.canvas.height)
    gl.clearColor(0.06, 0.06, 0.06, 1)
    gl.clear(gl.COLOR_BUFFER_BIT)
    gl.useProgram(this.program)
    gl.uniform2f(this.uScale, this.scale, this.scale)
    gl.uniform2f(this.uTranslate, this.tx, this.ty)
    const ps = Math.max(1.5, Math.min(6.0, 3.0 / Math.sqrt(this.scale)))
    gl.uniform1f(this.uPointSize, ps)
    gl.uniform3f(this.uColor, 0.65, 0.75, 0.95)
    gl.bindVertexArray(this.vao)
    if (this.needsUpload) {
      gl.bindBuffer(gl.ARRAY_BUFFER, this.buf)
      gl.bufferData(gl.ARRAY_BUFFER, this.data, gl.STATIC_DRAW)
      this.needsUpload = false
    }
    gl.drawArrays(gl.POINTS, 0, this.points.length)
    // highlight selected point overlay
    if (this.selectedIndex >= 0 && this.selectedIndex < this.points.length) {
      gl.uniform1f(this.uPointSize, ps * 1.8)
      gl.uniform3f(this.uColor, 0.85, 0.95, 1.0)
      gl.drawArrays(gl.POINTS, this.selectedIndex, 1)
    }
    gl.bindVertexArray(null)
  }

  private createProgram(vsSrc: string, fsSrc: string) {
    const gl = this.gl
    const vs = gl.createShader(gl.VERTEX_SHADER)!
    gl.shaderSource(vs, vsSrc)
    gl.compileShader(vs)
    if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) {
      throw new Error(gl.getShaderInfoLog(vs) || 'vs compile failed')
    }
    const fs = gl.createShader(gl.FRAGMENT_SHADER)!
    gl.shaderSource(fs, fsSrc)
    gl.compileShader(fs)
    if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
      throw new Error(gl.getShaderInfoLog(fs) || 'fs compile failed')
    }
    const prog = gl.createProgram()!
    gl.attachShader(prog, vs)
    gl.attachShader(prog, fs)
    gl.linkProgram(prog)
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error(gl.getProgramInfoLog(prog) || 'program link failed')
    }
    gl.deleteShader(vs)
    gl.deleteShader(fs)
    return prog
  }
}
