import { useEffect, useMemo, useRef, useState } from "react";

type Tool = "paint" | "erase";

export function MaskEditor(props: {
  imageDataUrl: string;
  value?: string;
  onChange: (maskDataUrl: string) => void;
  disabled?: boolean;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const onChangeRef = useRef(props.onChange);
  const drawingRef = useRef(false);
  const lastRef = useRef<{ x: number; y: number } | null>(null);
  const lastImageKeyRef = useRef<string>("");

  const [tool, setTool] = useState<Tool>("paint");
  const [brushSize, setBrushSize] = useState(32);
  const [ready, setReady] = useState(false);
  const [imageSize, setImageSize] = useState<{ w: number; h: number } | null>(null);

  const imageKey = useMemo(() => {
    const src = props.imageDataUrl || "";
    return `${src.length}:${src.slice(0, 28)}:${src.slice(-28)}`;
  }, [props.imageDataUrl]);

  useEffect(() => {
    onChangeRef.current = props.onChange;
  }, [props.onChange]);

  function getContext(): CanvasRenderingContext2D | null {
    const c = canvasRef.current;
    if (!c) return null;
    return c.getContext("2d");
  }

  function canvasPointFromEvent(e: React.PointerEvent): { x: number; y: number } | null {
    const c = canvasRef.current;
    if (!c) return null;
    const rect = c.getBoundingClientRect();
    if (!rect.width || !rect.height) return null;

    const sx = c.width / rect.width;
    const sy = c.height / rect.height;

    const x = (e.clientX - rect.left) * sx;
    const y = (e.clientY - rect.top) * sy;
    return { x, y };
  }

  function exportMaskDataUrl(): string {
    const c = canvasRef.current;
    if (!c) return "";
    const out = document.createElement("canvas");
    out.width = c.width;
    out.height = c.height;
    const ctx = out.getContext("2d");
    if (!ctx) return "";
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, out.width, out.height);
    ctx.drawImage(c, 0, 0);
    return out.toDataURL("image/png");
  }

  function clearCanvas() {
    const ctx = getContext();
    const c = canvasRef.current;
    if (!ctx || !c) return;
    ctx.clearRect(0, 0, c.width, c.height);
    props.onChange(exportMaskDataUrl());
  }

  useEffect(() => {
    let cancelled = false;
    setReady(false);
    setImageSize(null);

    const img = new Image();
    img.onload = () => {
      if (cancelled) return;
      const w = Math.max(1, Math.floor(img.naturalWidth || img.width || 1));
      const h = Math.max(1, Math.floor(img.naturalHeight || img.height || 1));
      setImageSize({ w, h });
      setReady(true);
    };
    img.onerror = () => {
      if (!cancelled) setReady(false);
    };
    img.src = props.imageDataUrl;

    return () => {
      cancelled = true;
    };
  }, [props.imageDataUrl]);

  useEffect(() => {
    const c = canvasRef.current;
    if (!c || !ready || !imageSize) return;
    c.width = imageSize.w;
    c.height = imageSize.h;
    const ctx = getContext();
    if (!ctx) return;
    ctx.clearRect(0, 0, c.width, c.height);

    // Auto-init a blank mask for this image if the parent has none.
    if (!props.value && lastImageKeyRef.current !== imageKey) {
      lastImageKeyRef.current = imageKey;
      onChangeRef.current(exportMaskDataUrl());
    }
  }, [ready, imageSize, imageKey, props.value]);

  return (
    <div className="maskEditor" aria-disabled={props.disabled ? "true" : "false"}>
      <div className="maskEditor__toolbar">
        <button
          className={`btn ${tool === "paint" ? "btn--primary" : ""}`}
          type="button"
          disabled={props.disabled}
          onClick={() => setTool("paint")}
        >
          畫筆（白=修補）
        </button>
        <button
          className={`btn ${tool === "erase" ? "btn--primary" : ""}`}
          type="button"
          disabled={props.disabled}
          onClick={() => setTool("erase")}
        >
          橡皮擦（黑=保留）
        </button>

        <label className="maskEditor__slider">
          <span className="field__label">Brush</span>
          <input
            className="control"
            type="range"
            min={2}
            max={140}
            step={1}
            value={brushSize}
            disabled={props.disabled}
            onChange={(e) => setBrushSize(Number(e.target.value))}
          />
          <span className="hint">{brushSize}px</span>
        </label>

        <button className="btn" type="button" disabled={props.disabled || !ready} onClick={clearCanvas}>
          清空
        </button>
        <span className="hint">提示：也可按住 Shift 或右鍵拖曳擦除</span>
      </div>

      <div
        className="maskEditor__stage"
        onContextMenu={(e) => e.preventDefault()}
        aria-label="Inpaint mask editor"
      >
        <img className="maskEditor__img" src={props.imageDataUrl} alt="init" />
        <canvas
          ref={canvasRef}
          className="maskEditor__canvas"
          onPointerDown={(e) => {
            if (props.disabled || !ready) return;
            const c = canvasRef.current;
            const ctx = getContext();
            if (!c || !ctx) return;
            const p = canvasPointFromEvent(e);
            if (!p) return;

            drawingRef.current = true;
            lastRef.current = p;
            c.setPointerCapture(e.pointerId);

            const isErase = tool === "erase" || e.button === 2 || e.shiftKey;
            ctx.globalCompositeOperation = isErase ? "destination-out" : "source-over";
            ctx.lineCap = "round";
            ctx.lineJoin = "round";
            ctx.lineWidth = brushSize;
            ctx.strokeStyle = "rgba(255,255,255,1)";

            ctx.beginPath();
            ctx.moveTo(p.x, p.y);
            ctx.lineTo(p.x + 0.01, p.y + 0.01);
            ctx.stroke();
          }}
          onPointerMove={(e) => {
            if (!drawingRef.current || props.disabled || !ready) return;
            const ctx = getContext();
            const p = canvasPointFromEvent(e);
            const last = lastRef.current;
            if (!ctx || !p || !last) return;

            ctx.beginPath();
            ctx.moveTo(last.x, last.y);
            ctx.lineTo(p.x, p.y);
            ctx.stroke();
            lastRef.current = p;
          }}
          onPointerUp={(e) => {
            if (!drawingRef.current) return;
            drawingRef.current = false;
            lastRef.current = null;
            try {
              canvasRef.current?.releasePointerCapture(e.pointerId);
            } catch {}
            props.onChange(exportMaskDataUrl());
          }}
          onPointerCancel={(e) => {
            if (!drawingRef.current) return;
            drawingRef.current = false;
            lastRef.current = null;
            try {
              canvasRef.current?.releasePointerCapture(e.pointerId);
            } catch {}
            props.onChange(exportMaskDataUrl());
          }}
        />
        {!ready ? <div className="maskEditor__loading">載入畫布中…</div> : null}
      </div>
    </div>
  );
}
