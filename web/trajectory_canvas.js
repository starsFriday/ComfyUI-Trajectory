import { app } from '../../../scripts/app.js'

const STYLE_ID = 'comfyui-trajectory-editor'

function ensureStyles() {
  if (document.getElementById(STYLE_ID)) {
    return
  }
  const style = document.createElement('style')
  style.id = STYLE_ID
  style.textContent = `
    .trajectory-overlay {
      position: fixed;
      inset: 0;
      background: rgba(6, 9, 17, 0.85);
      display: flex;
      align-items: center;
      justify-content: center;
      z-index: 10000;
    }
    .trajectory-panel {
      background: #10131f;
      border-radius: 12px;
      padding: 20px;
      width: min(960px, 95vw);
      box-shadow: 0 12px 60px rgba(0,0,0,0.45);
      color: #f4f4f8;
      display: flex;
      flex-direction: column;
      gap: 16px;
      max-height: 90vh;
    }
    .trajectory-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
    }
    .trajectory-header h2 {
      margin: 0;
      font-size: 18px;
      font-weight: 600;
    }
    .trajectory-canvas-wrap {
      position: relative;
      width: 100%;
      flex: 1;
      border: 1px solid #29304c;
      border-radius: 10px;
      background: #05070f;
      overflow: auto;
    }
    .trajectory-canvas-wrap canvas {
      width: 100%;
      height: auto;
      display: block;
      cursor: crosshair;
    }
    .trajectory-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .trajectory-actions button,
    .trajectory-actions label {
      background: #1f273d;
      border: 1px solid #394158;
      color: #fefefe;
      padding: 8px 14px;
      border-radius: 8px;
      cursor: pointer;
      font-size: 13px;
      transition: background 0.2s ease;
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }
    .trajectory-actions button:hover,
    .trajectory-actions label:hover {
      background: #2c3550;
    }
    .trajectory-actions label input {
      display: none;
    }
    .trajectory-footer {
      display: flex;
      justify-content: flex-end;
      gap: 12px;
    }
    .trajectory-footer button {
      padding: 10px 18px;
      border-radius: 8px;
      font-size: 14px;
      border: none;
      cursor: pointer;
    }
    .trajectory-footer button.primary {
      background: #4d7cfe;
      color: white;
    }
    .trajectory-footer button.secondary {
      background: transparent;
      border: 1px solid #4d5a7c;
      color: #e5e8f2;
    }
    .trajectory-hint {
      font-size: 12px;
      opacity: 0.85;
    }
  `
  document.head.appendChild(style)
}

function showEditor(node, widget) {
  ensureStyles()

  const overlay = document.createElement('div')
  overlay.className = 'trajectory-overlay'

  const panel = document.createElement('div')
  panel.className = 'trajectory-panel'
  overlay.appendChild(panel)

  const header = document.createElement('div')
  header.className = 'trajectory-header'
  const title = document.createElement('h2')
  title.textContent = '绘制轨迹 (按住 Shift 开始绘制)'
  const hint = document.createElement('div')
  hint.className = 'trajectory-hint'
  hint.textContent = 'Shift + 鼠标左键 = 绘制，双击空白区域 = 快速清空当前笔画'
  header.appendChild(title)
  header.appendChild(hint)
  panel.appendChild(header)

  const canvasWrap = document.createElement('div')
  canvasWrap.className = 'trajectory-canvas-wrap'
  panel.appendChild(canvasWrap)

  const canvas = document.createElement('canvas')
  const ctx = canvas.getContext('2d')
  const DEFAULT_WIDTH = 960
  const DEFAULT_HEIGHT = 540
  canvasWrap.appendChild(canvas)

  const actions = document.createElement('div')
  actions.className = 'trajectory-actions'
  panel.appendChild(actions)

  const footer = document.createElement('div')
  footer.className = 'trajectory-footer'
  panel.appendChild(footer)

  const cancelBtn = document.createElement('button')
  cancelBtn.className = 'secondary'
  cancelBtn.textContent = '取消'
  footer.appendChild(cancelBtn)

  const applyBtn = document.createElement('button')
  applyBtn.className = 'primary'
  applyBtn.textContent = '应用路径'
  footer.appendChild(applyBtn)

  const fileLabel = document.createElement('label')
  fileLabel.textContent = '载入参考图'
  const fileInput = document.createElement('input')
  fileInput.type = 'file'
  fileInput.accept = 'image/*'
  fileLabel.appendChild(fileInput)
  actions.appendChild(fileLabel)

  const undoBtn = document.createElement('button')
  undoBtn.textContent = '撤销上一笔'
  actions.appendChild(undoBtn)

  const clearBtn = document.createElement('button')
  clearBtn.textContent = '清空'
  actions.appendChild(clearBtn)

  const downloadBtn = document.createElement('button')
  downloadBtn.textContent = '导出 PNG'
  actions.appendChild(downloadBtn)

  const infoLabel = document.createElement('div')
  infoLabel.className = 'trajectory-hint'
  panel.appendChild(infoLabel)

  let strokes = []
  let currentStroke = []
  let drawing = false
  let backgroundImage = null
  let cursorPos = null
  let referenceImageData = null
  let referenceImageSize = null

  try {
    if (widget.value) {
      const parsed = JSON.parse(widget.value)
      if (parsed && Array.isArray(parsed.strokes)) {
        strokes = parsed.strokes
      }
      if (parsed?.reference_image) {
        referenceImageData = parsed.reference_image
      }
      if (parsed?.reference_size) {
        referenceImageSize = parsed.reference_size
      }
    }
  } catch (err) {
    console.warn('无法解析 Trajectory JSON', err)
  }

  function updateInfo() {
    const totalPoints = strokes.reduce((acc, stroke) => acc + stroke.points.length, 0)
    const refText = referenceImageSize
      ? `，参考图：${referenceImageSize.width}×${referenceImageSize.height}`
      : ''
    infoLabel.textContent = `当前笔画：${strokes.length}，总采样点：${totalPoints}${refText}`
  }

  function drawAll() {
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    if (backgroundImage) {
      ctx.drawImage(backgroundImage, 0, 0, canvas.width, canvas.height)
    }
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'
    ctx.lineWidth = 4

    ctx.strokeStyle = 'rgba(111, 210, 255, 0.95)'
    ctx.shadowColor = 'rgba(111, 210, 255, 0.55)'
    ctx.shadowBlur = 8

    for (const stroke of strokes) {
      drawStroke(stroke.points)
    }

    if (currentStroke.length > 0) {
      ctx.strokeStyle = 'rgba(255, 180, 64, 0.95)'
      ctx.shadowColor = 'rgba(255, 180, 64, 0.6)'
      drawStroke(currentStroke)
    }

    if (cursorPos) {
      ctx.save()
      ctx.shadowColor = 'transparent'
      ctx.lineWidth = 1
      ctx.strokeStyle = 'rgba(255,255,255,0.8)'
      ctx.beginPath()
      ctx.arc(cursorPos.x * canvas.width, cursorPos.y * canvas.height, 6, 0, Math.PI * 2)
      ctx.stroke()
      ctx.restore()
    }
  }

  function drawStroke(points) {
    if (points.length < 2) return
    ctx.beginPath()
    const [startX, startY] = points[0]
    ctx.moveTo(startX * canvas.width, startY * canvas.height)
    for (let i = 1; i < points.length; i++) {
      const [x, y] = points[i]
      ctx.lineTo(x * canvas.width, y * canvas.height)
    }
    ctx.stroke()
  }

  function normalizePointer(ev) {
    const rect = canvas.getBoundingClientRect()
    const x = Math.min(Math.max((ev.clientX - rect.left) / rect.width, 0), 1)
    const y = Math.min(Math.max((ev.clientY - rect.top) / rect.height, 0), 1)
    return { x, y }
  }

  function addPoint(ev) {
    const pos = normalizePointer(ev)
    cursorPos = pos
    const last = currentStroke[currentStroke.length - 1]
    if (!last || Math.hypot(last[0] - pos.x, last[1] - pos.y) > 0.003) {
      currentStroke.push([pos.x, pos.y])
    }
  }

  function commitStroke() {
    if (currentStroke.length > 1) {
      strokes.push({ points: [...currentStroke] })
    }
    currentStroke = []
    updateInfo()
    drawAll()
  }

  canvas.addEventListener('pointerdown', (ev) => {
    const isTouch = ev.pointerType === 'touch'
    if (!isTouch && !ev.shiftKey) {
      hint.textContent = '👉 按住 Shift 再点击即可开始绘制轨迹'
      hint.style.color = '#ffc75f'
      drawAll()
      return
    }
    hint.textContent = 'Shift + 拖拽 = 绘制；松开 = 结束当前笔画'
    hint.style.color = '#f4f4f8'
    drawing = true
    currentStroke = []
    addPoint(ev)
    canvas.setPointerCapture(ev.pointerId)
    ev.preventDefault()
    drawAll()
  })

  canvas.addEventListener('pointermove', (ev) => {
    cursorPos = normalizePointer(ev)
    if (drawing) {
      addPoint(ev)
    }
    drawAll()
  })

  canvas.addEventListener('pointerup', (ev) => {
    if (!drawing) return
    drawing = false
    cursorPos = normalizePointer(ev)
    try {
      canvas.releasePointerCapture(ev.pointerId)
    } catch (err) {
      /* ignore */
    }
    commitStroke()
  })

  canvas.addEventListener('pointercancel', (ev) => {
    drawing = false
    currentStroke = []
    try {
      canvas.releasePointerCapture(ev.pointerId)
    } catch (err) {
      /* ignore */
    }
    cursorPos = null
    drawAll()
  })

  canvas.addEventListener('pointerleave', () => {
    cursorPos = null
    drawAll()
  })

  canvas.addEventListener('dblclick', () => {
    currentStroke = []
    drawAll()
  })

  undoBtn.addEventListener('click', () => {
    if (strokes.length) {
      strokes.pop()
      updateInfo()
      drawAll()
    }
  })

  clearBtn.addEventListener('click', () => {
    strokes = []
    currentStroke = []
    updateInfo()
    drawAll()
  })

  downloadBtn.addEventListener('click', () => {
    const link = document.createElement('a')
    link.download = 'trajectory-preview.png'
    link.href = canvas.toDataURL('image/png')
    link.click()
  })

  fileInput.addEventListener('change', (ev) => {
    const file = ev.target.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = () => {
      referenceImageData = reader.result
      const img = new Image()
      img.onload = () => {
        referenceImageSize = { width: img.naturalWidth, height: img.naturalHeight }
        backgroundImage = img
        setCanvasResolution(referenceImageSize.width, referenceImageSize.height)
      }
      img.src = referenceImageData
    }
    reader.readAsDataURL(file)
  })

  function setCanvasResolution(width, height) {
    const targetWidth = Math.max(16, Math.round(width || DEFAULT_WIDTH))
    const targetHeight = Math.max(16, Math.round(height || DEFAULT_HEIGHT))
    canvas.width = targetWidth
    canvas.height = targetHeight

    const maxPanelWidth = Math.min(window.innerWidth * 0.95, targetWidth + 120)
    panel.style.width = `${maxPanelWidth}px`

    const availableWidth = maxPanelWidth - 40
    const availableHeight = Math.min(window.innerHeight * 0.75, targetHeight)
    const scale = Math.min(1, availableWidth / targetWidth, availableHeight / targetHeight)

    const displayWidth = Math.max(120, targetWidth * scale)
    const displayHeight = Math.max(90, targetHeight * scale)

    canvas.style.width = `${displayWidth}px`
    canvas.style.height = `${displayHeight}px`
    canvasWrap.style.width = `${displayWidth}px`
    canvasWrap.style.height = `${displayHeight}px`
    canvasWrap.style.maxHeight = `${displayHeight}px`
    canvasWrap.style.overflow = 'hidden'

    updateInfo()
    drawAll()
  }

  function loadReferenceFromData(dataUrl, sizeHint) {
    if (!dataUrl) {
      backgroundImage = null
      referenceImageSize = null
      setCanvasResolution(DEFAULT_WIDTH, DEFAULT_HEIGHT)
      return
    }
    const img = new Image()
    img.onload = () => {
      backgroundImage = img
      const width = sizeHint?.width || img.naturalWidth || DEFAULT_WIDTH
      const height = sizeHint?.height || img.naturalHeight || DEFAULT_HEIGHT
      referenceImageSize = { width, height }
      setCanvasResolution(width, height)
    }
    img.src = dataUrl
  }

  if (referenceImageData) {
    const hintWidth = referenceImageSize?.width || DEFAULT_WIDTH
    const hintHeight = referenceImageSize?.height || DEFAULT_HEIGHT
    setCanvasResolution(hintWidth, hintHeight)
    loadReferenceFromData(referenceImageData, referenceImageSize)
  } else {
    setCanvasResolution(DEFAULT_WIDTH, DEFAULT_HEIGHT)
  }

  document.body.appendChild(overlay)
  updateInfo()
  drawAll()

  const removeOverlay = () => {
    window.removeEventListener('keydown', escHandler)
    if (overlay.parentNode) {
      document.body.removeChild(overlay)
    }
  }

  const escHandler = (ev) => {
    if (ev.key === 'Escape') {
      removeOverlay()
    }
  }

  window.addEventListener('keydown', escHandler)

  cancelBtn.addEventListener('click', removeOverlay)

  applyBtn.addEventListener('click', () => {
    const payload = {
      strokes,
      edited_at: new Date().toISOString(),
    }
    if (referenceImageData) {
      payload.reference_image = referenceImageData
      payload.reference_size =
        referenceImageSize || { width: canvas.width, height: canvas.height }
    }
    widget.value = JSON.stringify(payload)
    widget.callback?.(widget.value)
    node.setDirtyCanvas(true, true)
    removeOverlay()
  })
}

app.registerExtension({
  name: 'ComfyUI.Trajectory.Editor',
  nodeCreated(node) {
    if (node.comfyClass !== 'TrajectoryDrawer') {
      return
    }
    const widget = node.widgets?.find((w) => w.name === 'path_json')
    if (!widget) return

    if (widget.inputEl) {
      widget.inputEl.readOnly = true
      widget.inputEl.style.display = 'none'
    }

    if (node.__trajectoryButtonAdded) {
      return
    }
    node.__trajectoryButtonAdded = true

    const drawButton = node.addWidget(
      'button',
      '编辑轨迹 (Shift 绘制)',
      'edit',
      () => {}
    )
    drawButton.tooltip = '打开画布，按住 Shift + 拖拽即可绘制路径'
    drawButton.callback = () => showEditor(node, widget)
  },
})
