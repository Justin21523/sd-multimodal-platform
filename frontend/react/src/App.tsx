import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { apiDelete, apiGet, apiPatch, apiPost, apiPostEmpty, apiPostForm, resolveMediaUrl, withQuery } from "./api/client";
import { Card } from "./components/Card";
import { MaskEditor } from "./components/MaskEditor";
import { StatusPill } from "./components/StatusPill";
import { Tabs, type TabKey } from "./components/Tabs";

type ModelItem = {
  model_id: string;
  name?: string;
  type?: string;
  loaded?: boolean;
  active?: boolean;
  capabilities?: string[];
};

type ModelsResponse = {
  success: boolean;
  data: { current_model_id?: string | null; models: ModelItem[] };
};

type HealthResponse = {
  status?: string;
  data?: any;
};

type Txt2ImgResponse = {
  success: boolean;
  task_id?: string;
  message?: string;
  data?: any;
};

type Img2ImgResponse = {
  success: boolean;
  message?: string;
  data?: any;
};

type InpaintResponse = {
  success: boolean;
  message?: string;
  data?: any;
};

type QueueEnqueueResponse = {
  success: boolean;
  task_id?: string;
  message: string;
};

type QueueTaskStatus = {
  task_id: string;
  status: string;
  task_type?: string;
  progress_percent?: number;
  current_step?: string;
  result_data?: any;
  error_info?: any;
};

type QueueStats = {
  total_tasks?: number;
  pending_tasks?: number;
  running_tasks?: number;
  completed_tasks?: number;
  failed_tasks?: number;
  cancelled_tasks?: number;
  average_wait_time?: number;
  average_processing_time?: number;
  queue_throughput?: number;
  active_workers?: number;
  total_workers?: number;
  gpu_memory_usage?: number;
  current_hour_requests?: number;
  daily_requests?: number;
  rate_limit_violations?: number;
};

type QueueOverview = {
  queue_stats: QueueStats;
  worker_status: any;
  system_health: any;
};

type QueueTask = {
  task_id: string;
  status: string;
  task_type: string;
  priority: string;
  created_at: string;
  started_at?: string | null;
  completed_at?: string | null;
  progress_percent?: number;
  current_step?: string;
  total_steps?: number | null;
  result_data?: any;
  error_info?: any;
};

type QueueTaskList = {
  tasks: QueueTask[];
  total_count: number;
  page: number;
  page_size: number;
  has_next: boolean;
};

type Asset = {
  asset_id: string;
  filename: string;
  category: string;
  tags?: string[];
  description?: string;
  file_size?: number;
  created_at?: number;
  file_url?: string | null;
  thumbnail_url?: string | null;
  download_url?: string | null;
};

type AssetsListResponse = {
  success: boolean;
  data: {
    assets: Asset[];
    pagination: { limit: number; offset: number; total: number };
    filters: { category?: string | null; tags?: string[] | null };
  };
  timestamp: number;
};

type AssetCategoriesResponse = {
  success: boolean;
  data: { categories: Record<string, number>; total_assets: number };
  timestamp: number;
};

type HistoryListResponse = {
  success: boolean;
  data: { records: any[]; limit: number; offset: number; count: number };
  timestamp: number;
  message?: string;
};

type HistoryRerunResponse = {
  success: boolean;
  data?: { task_id?: string; task_type?: string };
  message?: string;
  timestamp?: number;
};

type PromptPreset = {
  id: string;
  name: string;
  prompt: string;
  negative_prompt: string;
  tags: string[];
  created_at: number;
  builtin?: boolean;
};

type PromptHistoryItem = {
  id: string;
  prompt: string;
  negative_prompt: string;
  model_id?: string | null;
  mode: "txt2img" | "img2img" | "inpaint";
  run_mode: "sync" | "async";
  width: number;
  height: number;
  steps: number;
  cfg_scale: number;
  seed: number;
  num_images: number;
  strength?: number;
  mask_blur?: number;
  inpainting_fill?: string;
  controlnet?: { type: ControlNetType; strength: number; preprocess: boolean } | null;
  init_asset_id?: string | null;
  mask_asset_id?: string | null;
  control_asset_id?: string | null;
  created_at: number;
};

const LS_PRESETS_KEY = "sdmm.promptPresets.v1";
const LS_HISTORY_KEY = "sdmm.promptHistory.v1";

const BUILTIN_PRESETS: PromptPreset[] = [
  {
    id: "builtin:blue-dashboard",
    name: "藍色系產品化 Dashboard（卡片式）",
    prompt:
      "一張藍色系、乾淨、產品感的卡片式 UI 儀表板介面，強烈視覺語言，現代化，光感，清晰資訊層級，grid cards，soft shadows，glassmorphism，優雅留白，8pt spacing，排版層級清楚，iconography，微互動，responsive layout，高對比，產品截圖風格",
    negative_prompt: "blurry, low quality, clutter, messy layout, unreadable text, watermark, logo",
    tags: ["blue", "uiux", "dashboard", "cards"],
    created_at: 0,
    builtin: true
  },
  {
    id: "builtin:blue-mobile",
    name: "藍色系 Mobile App（卡片/互動）",
    prompt:
      "藍色系行動 App UI 設計，卡片式資訊架構，清晰主次與留白，互動式元件（toggle, tabs, chips），現代化光感與柔和陰影，整體一致視覺語言，responsive，產品截圖風格",
    negative_prompt: "blurry, low quality, clutter, unreadable text, watermark, logo",
    tags: ["blue", "uiux", "mobile", "cards"],
    created_at: 0,
    builtin: true
  },
  {
    id: "builtin:blue-landing",
    name: "藍色系 SaaS Landing（Hero/卡片）",
    prompt:
      "藍色系 SaaS landing page，hero + feature cards，強烈視覺語言，現代化光感，清楚資訊層級，網格布局，產品化，乾淨留白，高對比，CTA 按鈕明確，產品截圖風格",
    negative_prompt: "blurry, low quality, clutter, unreadable text, watermark, logo",
    tags: ["blue", "uiux", "landing", "saas"],
    created_at: 0,
    builtin: true
  }
];

type ControlNetType = "canny" | "depth" | "openpose" | "scribble" | "mlsd" | "normal";
type ControlAssetCategory = "pose" | "depth" | "controlnet" | "reference" | "custom";

function suggestedControlCategory(controlType: ControlNetType): ControlAssetCategory {
  if (controlType === "scribble") return "controlnet";
  return "reference";
}

function makeLocalId(prefix: string): string {
  const rand = Math.random().toString(16).slice(2);
  return `${prefix}_${Date.now().toString(16)}_${rand}`;
}

function parseTags(raw: string): string[] {
  return raw
    .split(",")
    .map((t) => t.trim())
    .filter(Boolean)
    .slice(0, 12);
}

function clampInt(value: string, fallback: number): number {
  const n = Number.parseInt(value, 10);
  return Number.isFinite(n) ? n : fallback;
}

function clampFloat(value: string, fallback: number): number {
  const n = Number.parseFloat(value);
  return Number.isFinite(n) ? n : fallback;
}

function fileToDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error("Failed to read file"));
    reader.onload = () => resolve(String(reader.result));
    reader.readAsDataURL(file);
  });
}

function blobToDataUrl(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error("Failed to read blob"));
    reader.onload = () => resolve(String(reader.result));
    reader.readAsDataURL(blob);
  });
}

async function urlToDataUrl(url: string): Promise<string> {
  const resp = await fetch(url, { method: "GET" });
  if (!resp.ok) throw new Error(`Failed to fetch image: ${resp.status}`);
  const blob = await resp.blob();
  return blobToDataUrl(blob);
}

function isOutputsUrl(rawUrl: string): boolean {
  const s = String(rawUrl ?? "").trim();
  if (!s) return false;
  try {
    const u = new URL(s, typeof window !== "undefined" ? window.location.origin : "http://localhost");
    return u.pathname.startsWith("/outputs/");
  } catch {
    return s.startsWith("/outputs/") || s.startsWith("outputs/") || s.includes("/outputs/");
  }
}

function formatBytes(bytes?: number): string {
  if (!bytes || bytes <= 0) return "-";
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let i = 0;
  while (value >= 1024 && i < units.length - 1) {
    value /= 1024;
    i += 1;
  }
  return `${value.toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
}

function formatIso(iso?: string | null): string {
  if (!iso) return "-";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? String(iso) : d.toLocaleString();
}

function formatEpochSeconds(sec?: number): string {
  if (!sec) return "-";
  const d = new Date(sec * 1000);
  return Number.isNaN(d.getTime()) ? String(sec) : d.toLocaleString();
}

function formatApiError(e: any): string {
  const status = e?.status;
  const rawDetail = e?.detail;
  const nestedDetail = rawDetail?.detail ?? rawDetail;

  const message =
    typeof nestedDetail === "string"
      ? nestedDetail
      : typeof rawDetail === "string"
        ? rawDetail
        : typeof e?.message === "string"
          ? e.message
          : "";

  if (status && message) return `${status} · ${message}`;
  if (status) return String(status);
  if (message) return message;
  return "unknown";
}

export default function App() {
  const [tab, setTab] = useState<TabKey>("generate");

  const [models, setModels] = useState<ModelItem[]>([]);
  const [modelsLoading, setModelsLoading] = useState(false);
  const activeModelId = useMemo(
    () => models.find((m) => m.active)?.model_id ?? models[0]?.model_id,
    [models]
  );

  const [mode, setMode] = useState<"txt2img" | "img2img" | "inpaint">("txt2img");
  const [runMode, setRunMode] = useState<"sync" | "async">("sync");

  const [prompt, setPrompt] = useState("一張藍色系、乾淨、產品感的卡片式 UI 儀表板介面，強烈視覺語言，現代化，光感");
  const [negativePrompt, setNegativePrompt] = useState("blurry, low quality, jpeg artifacts");
  const [modelId, setModelId] = useState<string | undefined>(undefined);
  const [width, setWidth] = useState("1024");
  const [height, setHeight] = useState("1024");
  const [steps, setSteps] = useState("25");
  const [cfg, setCfg] = useState("7.5");
  const [seed, setSeed] = useState("");
  const [numImages, setNumImages] = useState("1");

  const [strength, setStrength] = useState("0.75");
  const [initImageDataUrl, setInitImageDataUrl] = useState<string>("");
  const [img2imgAssetChoices, setImg2imgAssetChoices] = useState<Asset[]>([]);
  const [img2imgAssetId, setImg2imgAssetId] = useState<string>("");
  const [img2imgAssetLoading, setImg2imgAssetLoading] = useState(false);
  const [maskImageDataUrl, setMaskImageDataUrl] = useState<string>("");
  const [maskBlur, setMaskBlur] = useState("4");
  const [inpaintFill, setInpaintFill] = useState<"original" | "latent_noise" | "latent_nothing" | "white">(
    "original"
  );
  const [maskAssetChoices, setMaskAssetChoices] = useState<Asset[]>([]);
  const [maskAssetId, setMaskAssetId] = useState<string>("");
  const [maskAssetLoading, setMaskAssetLoading] = useState(false);
  const [useControlNet, setUseControlNet] = useState(false);
  const [controlType, setControlType] = useState<ControlNetType>("openpose");
  const [controlStrength, setControlStrength] = useState("1.0");
  const [controlPreprocess, setControlPreprocess] = useState(true);
  const [controlImageDataUrl, setControlImageDataUrl] = useState<string>("");
  const [controlAssetCategory, setControlAssetCategory] = useState<ControlAssetCategory>(
    suggestedControlCategory("openpose")
  );
  const [controlCategoryAuto, setControlCategoryAuto] = useState(true);
  const [controlAssetChoices, setControlAssetChoices] = useState<Asset[]>([]);
  const [controlAssetId, setControlAssetId] = useState<string>("");
  const [controlAssetLoading, setControlAssetLoading] = useState(false);

  const [busy, setBusy] = useState(false);
  const [lastTaskId, setLastTaskId] = useState<string>("");
  const [taskCenterIds, setTaskCenterIds] = useState<string[]>([]);
  const [taskCenterStatusById, setTaskCenterStatusById] = useState<Record<string, QueueTaskStatus>>({});
  const taskCenterUserStreamRef = useRef<EventSource | null>(null);
  const lastTaskIdRef = useRef<string>("");
  const [status, setStatus] = useState<QueueTaskStatus | null>(null);
  const [images, setImages] = useState<string[]>([]);
  const [error, setError] = useState<string>("");
  const inpaintPrevInitRef = useRef<string>("");

  // Queue dashboard
  const [queueOverview, setQueueOverview] = useState<QueueOverview | null>(null);
  const [queueTasks, setQueueTasks] = useState<QueueTaskList | null>(null);
  const [queueSelectedTask, setQueueSelectedTask] = useState<QueueTask | null>(null);
  const [queueAutoRefresh, setQueueAutoRefresh] = useState(true);
  const [queueStatusFilter, setQueueStatusFilter] = useState<string>("");
  const [queueUserFilter, setQueueUserFilter] = useState<string>("local");
  const [queuePage, setQueuePage] = useState<number>(1);
  const [queuePageSize, setQueuePageSize] = useState<number>(20);
  const [queueLoading, setQueueLoading] = useState(false);
  const queueTasksStreamRef = useRef<EventSource | null>(null);
  const [queueTasksStreamOk, setQueueTasksStreamOk] = useState(false);
  const queueVisibleTaskIdsRef = useRef<Set<string>>(new Set());
  const queueSelectedTaskIdRef = useRef<string | null>(null);
  const queueRefreshTimerRef = useRef<number | null>(null);

  // Assets
  const [assetCategories, setAssetCategories] = useState<Record<string, number>>({});
  const [assetCategory, setAssetCategory] = useState<string>("");
  const [assetTagsFilter, setAssetTagsFilter] = useState<string>("");
  const [assetQuery, setAssetQuery] = useState<string>("");
  const [assets, setAssets] = useState<Asset[]>([]);
  const [assetsLoading, setAssetsLoading] = useState(false);
  const [assetsLimit, setAssetsLimit] = useState<number>(30);
  const [assetsOffset, setAssetsOffset] = useState<number>(0);
  const [selectedAssetIds, setSelectedAssetIds] = useState<string[]>([]);
  const [bulkTags, setBulkTags] = useState<string>("");
  const [bulkCategory, setBulkCategory] = useState<string>("");
  const [assetEditId, setAssetEditId] = useState<string>("");
  const [assetEditTags, setAssetEditTags] = useState<string>("");
  const [assetEditDescription, setAssetEditDescription] = useState<string>("");
  const [assetEditCategory, setAssetEditCategory] = useState<string>("");

  const [uploadCategory, setUploadCategory] = useState<string>("reference");
  const [uploadTags, setUploadTags] = useState<string>("");
  const [uploadDescription, setUploadDescription] = useState<string>("");
  const [uploadFiles, setUploadFiles] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);
  const [applyingAsset, setApplyingAsset] = useState(false);
  const [quickSaveCategory, setQuickSaveCategory] = useState<string>("reference");
  const [quickSaveTags, setQuickSaveTags] = useState<string>("");
  const [quickSaveDescription, setQuickSaveDescription] = useState<string>("");
  const [savingAsset, setSavingAsset] = useState(false);

  // Prompt/Style library (local-only)
  const [libraryView, setLibraryView] = useState<"presets" | "history">("presets");
  const [presetName, setPresetName] = useState("");
  const [presetTags, setPresetTags] = useState("");
  const [userPresets, setUserPresets] = useState<PromptPreset[]>([]);
  const [promptHistory, setPromptHistory] = useState<PromptHistoryItem[]>([]);
  const [serverHistory, setServerHistory] = useState<any[]>([]);
  const [serverHistoryLoading, setServerHistoryLoading] = useState(false);
  const [serverHistoryLimit, setServerHistoryLimit] = useState<number>(60);
  const [serverHistoryOffset, setServerHistoryOffset] = useState<number>(0);
  const [serverHistoryQuery, setServerHistoryQuery] = useState<string>("");
  const [serverHistoryTaskType, setServerHistoryTaskType] = useState<string>("");
  const [serverHistoryUserId, setServerHistoryUserId] = useState<string>("local");
  const [serverHistorySince, setServerHistorySince] = useState<string>("");
  const [serverHistoryUntil, setServerHistoryUntil] = useState<string>("");
  const [serverHistoryCleanupDays, setServerHistoryCleanupDays] = useState<string>("30");
  const allPresets = useMemo(() => [...BUILTIN_PRESETS, ...userPresets], [userPresets]);

  useEffect(() => {
    try {
      const raw = window.localStorage.getItem(LS_PRESETS_KEY);
      if (raw) {
        const parsed = JSON.parse(raw);
        if (Array.isArray(parsed)) {
          const cleaned = parsed
            .filter((p) => p && typeof p.id === "string" && typeof p.prompt === "string")
            .map((p) => ({
              id: String(p.id),
              name: typeof p.name === "string" && p.name.trim() ? p.name.trim() : "未命名",
              prompt: String(p.prompt ?? ""),
              negative_prompt: typeof p.negative_prompt === "string" ? p.negative_prompt : "",
              tags: Array.isArray(p.tags) ? p.tags.map((t: any) => String(t)).filter(Boolean) : [],
              created_at: typeof p.created_at === "number" ? p.created_at : Date.now()
            })) as PromptPreset[];
          setUserPresets(cleaned);
        }
      }
    } catch {
      // ignore
    }

    try {
      const raw = window.localStorage.getItem(LS_HISTORY_KEY);
      if (raw) {
        const parsed = JSON.parse(raw);
        if (Array.isArray(parsed)) {
          const cleaned = parsed
            .filter((h) => h && typeof h.id === "string" && typeof h.prompt === "string")
            .map((h) => ({
              id: String(h.id),
              prompt: String(h.prompt ?? ""),
              negative_prompt: typeof h.negative_prompt === "string" ? h.negative_prompt : "",
              model_id: typeof h.model_id === "string" ? h.model_id : null,
              mode: (h.mode === "txt2img" || h.mode === "img2img" || h.mode === "inpaint" ? h.mode : "txt2img") as any,
              run_mode: (h.run_mode === "sync" || h.run_mode === "async" ? h.run_mode : "sync") as any,
              width: typeof h.width === "number" ? h.width : 1024,
              height: typeof h.height === "number" ? h.height : 1024,
              steps: typeof h.steps === "number" ? h.steps : 25,
              cfg_scale: typeof h.cfg_scale === "number" ? h.cfg_scale : 7.5,
              seed: typeof h.seed === "number" ? h.seed : -1,
              num_images: typeof h.num_images === "number" ? h.num_images : 1,
              strength: typeof h.strength === "number" ? h.strength : undefined,
              mask_blur: typeof h.mask_blur === "number" ? h.mask_blur : undefined,
              inpainting_fill: typeof h.inpainting_fill === "string" ? h.inpainting_fill : undefined,
              controlnet:
                h.controlnet &&
                  typeof h.controlnet === "object" &&
                  (h.controlnet.type === "canny" ||
                    h.controlnet.type === "depth" ||
                    h.controlnet.type === "openpose" ||
                    h.controlnet.type === "scribble" ||
                    h.controlnet.type === "mlsd" ||
                    h.controlnet.type === "normal")
                  ? {
                      type: h.controlnet.type as ControlNetType,
                      strength: typeof h.controlnet.strength === "number" ? h.controlnet.strength : 1.0,
                      preprocess: typeof h.controlnet.preprocess === "boolean" ? h.controlnet.preprocess : true
                    }
                  : null,
              init_asset_id: typeof h.init_asset_id === "string" ? h.init_asset_id : null,
              mask_asset_id: typeof h.mask_asset_id === "string" ? h.mask_asset_id : null,
              control_asset_id: typeof h.control_asset_id === "string" ? h.control_asset_id : null,
              created_at: typeof h.created_at === "number" ? h.created_at : Date.now()
            })) as PromptHistoryItem[];
          setPromptHistory(cleaned.slice(0, 60));
        }
      }
    } catch {
      // ignore
    }
  }, []);

  useEffect(() => {
    try {
      window.localStorage.setItem(LS_PRESETS_KEY, JSON.stringify(userPresets));
    } catch {
      // ignore
    }
  }, [userPresets]);

  useEffect(() => {
    try {
      window.localStorage.setItem(LS_HISTORY_KEY, JSON.stringify(promptHistory.slice(0, 60)));
    } catch {
      // ignore
    }
  }, [promptHistory]);

  useEffect(() => {
    if (tab !== "generate" || libraryView !== "history") return;
    refreshServerHistoryOnce({ offset: 0 });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tab, libraryView]);

  useEffect(() => {
    let cancelled = false;
    async function loadModels() {
      setModelsLoading(true);
      try {
        const resp = await apiGet<ModelsResponse>("/api/v1/models");
        if (!cancelled && resp?.success) {
          setModels(resp.data?.models ?? []);
        }
      } catch (e: any) {
        if (!cancelled) setError(`模型列表讀取失敗：${formatApiError(e)}`);
      } finally {
        if (!cancelled) setModelsLoading(false);
      }
    }
    loadModels();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!modelId && activeModelId) setModelId(activeModelId);
  }, [activeModelId, modelId]);

  useEffect(() => {
    lastTaskIdRef.current = lastTaskId;
  }, [lastTaskId]);

  useEffect(() => {
    queueVisibleTaskIdsRef.current = new Set(queueTasks?.tasks.map((t) => t.task_id) ?? []);
  }, [queueTasks]);

  useEffect(() => {
    queueSelectedTaskIdRef.current = queueSelectedTask?.task_id ?? null;
  }, [queueSelectedTask]);

  useEffect(() => {
    if (runMode !== "async") {
      const existing = taskCenterUserStreamRef.current;
      if (existing) {
        try {
          existing.close();
        } catch {
          // ignore
        }
      }
      taskCenterUserStreamRef.current = null;
      return;
    }

    if (typeof window === "undefined" || !("EventSource" in window)) return;

    const userId = encodeURIComponent(queueUserFilter || "local");
    let cancelled = false;
    let es: EventSource | null = null;

    try {
      es = new EventSource(resolveMediaUrl(`/api/v1/queue/stream/user/${userId}`));
      taskCenterUserStreamRef.current = es;

      es.onmessage = (ev) => {
        if (cancelled) return;
        try {
          const parsed = JSON.parse(String(ev.data ?? "{}")) as QueueTaskStatus;
          if (!parsed || typeof parsed.task_id !== "string") return;

          setTaskCenterStatusById((prev) => ({ ...prev, [parsed.task_id]: parsed }));
          setQueueTasks((prev) => {
            if (!prev) return prev;
            const idx = prev.tasks.findIndex((t) => t.task_id === parsed.task_id);
            if (idx < 0) return prev;
            const current = prev.tasks[idx];
            const totalSteps = (parsed as any).total_steps;
            const nextTask: QueueTask = {
              ...current,
              status: typeof parsed.status === "string" ? parsed.status : current.status,
              progress_percent: typeof parsed.progress_percent === "number" ? parsed.progress_percent : current.progress_percent,
              current_step: typeof parsed.current_step === "string" ? parsed.current_step : current.current_step,
              result_data: parsed.result_data ?? current.result_data,
              error_info: parsed.error_info ?? current.error_info,
              total_steps: typeof totalSteps === "number" ? totalSteps : current.total_steps
            };
            const tasks = [...prev.tasks];
            tasks[idx] = nextTask;
            return { ...prev, tasks };
          });
          setQueueSelectedTask((prev) => {
            if (!prev) return prev;
            if (prev.task_id !== parsed.task_id) return prev;
            const totalSteps = (parsed as any).total_steps;
            return {
              ...prev,
              status: typeof parsed.status === "string" ? parsed.status : prev.status,
              progress_percent: typeof parsed.progress_percent === "number" ? parsed.progress_percent : prev.progress_percent,
              current_step: typeof parsed.current_step === "string" ? parsed.current_step : prev.current_step,
              result_data: parsed.result_data ?? prev.result_data,
              error_info: parsed.error_info ?? prev.error_info,
              total_steps: typeof totalSteps === "number" ? totalSteps : prev.total_steps
            };
          });

          const activeLast = lastTaskIdRef.current;
          if (activeLast && parsed.task_id === activeLast) {
            setStatus(parsed);

            if (parsed.status === "completed") {
              const rd = parsed.result_data ?? {};
              const imgs: any[] = Array.isArray(rd?.result?.images) ? rd.result.images : [];
              const urls = imgs
                .map((it) => (typeof it?.image_url === "string" ? it.image_url : null))
                .filter(Boolean) as string[];
              const resolved = Array.from(new Set(urls)).map(resolveMediaUrl);
              if (parsed.task_type === "upscale" || parsed.task_type === "face_restore") {
                setImages((prev) => Array.from(new Set([...resolved, ...prev])));
              } else {
                setImages(resolved);
              }
              setBusy(false);
              return;
            }

            if (parsed.status === "failed" || parsed.status === "cancelled" || parsed.status === "timeout") {
              setBusy(false);
            }
          }
        } catch {
          // ignore
        }
      };

      es.onerror = () => {
        if (cancelled) return;
        try {
          es?.close();
        } catch {
          // ignore
        }
        if (taskCenterUserStreamRef.current === es) taskCenterUserStreamRef.current = null;
      };
    } catch {
      // ignore
    }

    return () => {
      cancelled = true;
      if (es) {
        try {
          es.close();
        } catch {
          // ignore
        }
      }
      if (taskCenterUserStreamRef.current === es) taskCenterUserStreamRef.current = null;
    };
  }, [runMode, queueUserFilter]);

  useEffect(() => {
    if (tab !== "queue" || runMode !== "async" || !queueAutoRefresh) {
      const existing = queueTasksStreamRef.current;
      if (existing) {
        try {
          existing.close();
        } catch {
          // ignore
        }
      }
      queueTasksStreamRef.current = null;
      if (queueRefreshTimerRef.current !== null) {
        clearTimeout(queueRefreshTimerRef.current);
        queueRefreshTimerRef.current = null;
      }
      setQueueTasksStreamOk(false);
      return;
    }

    if (typeof window === "undefined" || !("EventSource" in window)) return;

    let cancelled = false;
    let es: EventSource | null = null;

    try {
      es = new EventSource(resolveMediaUrl("/api/v1/queue/stream/tasks"));
      queueTasksStreamRef.current = es;

      es.onopen = () => {
        if (cancelled) return;
        setQueueTasksStreamOk(true);
      };

      es.onmessage = (ev) => {
        if (cancelled) return;
        try {
          const parsed = JSON.parse(String(ev.data ?? "{}")) as QueueTaskStatus;
          if (!parsed || typeof parsed.task_id !== "string") return;

          const selectedId = queueSelectedTaskIdRef.current;
          const isVisible = queueVisibleTaskIdsRef.current.has(parsed.task_id);
          const isSelected = !!selectedId && selectedId === parsed.task_id;
          if (!isVisible && !isSelected && queueRefreshTimerRef.current === null) {
            queueRefreshTimerRef.current = window.setTimeout(() => {
              queueRefreshTimerRef.current = null;
              refreshQueueOnce();
            }, 300);
          }

          setQueueTasks((prev) => {
            if (!prev) return prev;
            const idx = prev.tasks.findIndex((t) => t.task_id === parsed.task_id);
            if (idx < 0) return prev;
            const current = prev.tasks[idx];
            const totalSteps = (parsed as any).total_steps;
            const nextTask: QueueTask = {
              ...current,
              status: typeof parsed.status === "string" ? parsed.status : current.status,
              progress_percent: typeof parsed.progress_percent === "number" ? parsed.progress_percent : current.progress_percent,
              current_step: typeof parsed.current_step === "string" ? parsed.current_step : current.current_step,
              result_data: parsed.result_data ?? current.result_data,
              error_info: parsed.error_info ?? current.error_info,
              total_steps: typeof totalSteps === "number" ? totalSteps : current.total_steps
            };
            const tasks = [...prev.tasks];
            tasks[idx] = nextTask;
            return { ...prev, tasks };
          });
          setQueueSelectedTask((prev) => {
            if (!prev) return prev;
            if (prev.task_id !== parsed.task_id) return prev;
            const totalSteps = (parsed as any).total_steps;
            return {
              ...prev,
              status: typeof parsed.status === "string" ? parsed.status : prev.status,
              progress_percent: typeof parsed.progress_percent === "number" ? parsed.progress_percent : prev.progress_percent,
              current_step: typeof parsed.current_step === "string" ? parsed.current_step : prev.current_step,
              result_data: parsed.result_data ?? prev.result_data,
              error_info: parsed.error_info ?? prev.error_info,
              total_steps: typeof totalSteps === "number" ? totalSteps : prev.total_steps
            };
          });
        } catch {
          // ignore
        }
      };

      es.onerror = () => {
        if (cancelled) return;
        setQueueTasksStreamOk(false);
        try {
          es?.close();
        } catch {
          // ignore
        }
        if (queueTasksStreamRef.current === es) queueTasksStreamRef.current = null;
      };
    } catch {
      // ignore
    }

    return () => {
      cancelled = true;
      if (queueRefreshTimerRef.current !== null) {
        clearTimeout(queueRefreshTimerRef.current);
        queueRefreshTimerRef.current = null;
      }
      setQueueTasksStreamOk(false);
      if (es) {
        try {
          es.close();
        } catch {
          // ignore
        }
      }
      if (queueTasksStreamRef.current === es) queueTasksStreamRef.current = null;
    };
  }, [tab, runMode, queueAutoRefresh, refreshQueueOnce]);

  useEffect(() => {
    const prev = inpaintPrevInitRef.current;
    inpaintPrevInitRef.current = initImageDataUrl;
    if (mode !== "inpaint") return;
    if (!initImageDataUrl) return;
    if (prev && prev !== initImageDataUrl) {
      setMaskAssetId("");
      setMaskImageDataUrl("");
    }
  }, [mode, initImageDataUrl]);

  useEffect(() => {
    if (!useControlNet) return;
    if (!controlCategoryAuto) return;
    const suggested = suggestedControlCategory(controlType);
    setControlAssetCategory(suggested);
    setControlPreprocess(suggested === "reference");
  }, [controlType, controlCategoryAuto, useControlNet]);

  useEffect(() => {
    if (tab !== "generate" || (mode !== "img2img" && mode !== "inpaint")) return;
    let cancelled = false;

    async function loadImg2ImgAssets() {
      setImg2imgAssetLoading(true);
      try {
        const resp = await apiGet<AssetsListResponse>(
          withQuery("/api/v1/assets/list", { category: "reference", limit: 20, offset: 0 })
        );
        if (cancelled) return;
        setImg2imgAssetChoices(resp.data?.assets ?? []);
      } catch (e: any) {
        if (!cancelled) setError(`資產載入失敗（reference）：${formatApiError(e)}`);
      } finally {
        if (!cancelled) setImg2imgAssetLoading(false);
      }
    }

    loadImg2ImgAssets();
    return () => {
      cancelled = true;
    };
  }, [tab, mode]);

  useEffect(() => {
    if (tab !== "generate" || mode !== "inpaint") return;
    let cancelled = false;

    async function loadMaskAssets() {
      setMaskAssetLoading(true);
      try {
        const resp = await apiGet<AssetsListResponse>(
          withQuery("/api/v1/assets/list", { category: "mask", limit: 20, offset: 0 })
        );
        if (cancelled) return;
        setMaskAssetChoices(resp.data?.assets ?? []);
      } catch (e: any) {
        if (!cancelled) setError(`資產載入失敗（mask）：${formatApiError(e)}`);
      } finally {
        if (!cancelled) setMaskAssetLoading(false);
      }
    }

    loadMaskAssets();
    return () => {
      cancelled = true;
    };
  }, [tab, mode]);

  useEffect(() => {
    if (tab !== "generate" || mode !== "img2img" || !useControlNet) return;
    let cancelled = false;

    async function loadControlAssets() {
      setControlAssetLoading(true);
      try {
        const resp = await apiGet<AssetsListResponse>(
          withQuery("/api/v1/assets/list", { category: controlAssetCategory, limit: 20, offset: 0 })
        );
        if (cancelled) return;
        setControlAssetChoices(resp.data?.assets ?? []);
      } catch (e: any) {
        if (!cancelled) setError(`資產載入失敗（ControlNet）：${formatApiError(e)}`);
      } finally {
        if (!cancelled) setControlAssetLoading(false);
      }
    }

    loadControlAssets();
    return () => {
      cancelled = true;
    };
  }, [tab, mode, useControlNet, controlAssetCategory]);

  useEffect(() => {
    if (tab !== "queue") return;
    let cancelled = false;

    async function refresh() {
      setQueueLoading(true);
      try {
        const overview = await apiGet<QueueOverview>("/api/v1/queue/status");
        const tasks = await apiGet<QueueTaskList>(
          withQuery("/api/v1/queue/tasks", {
            status: queueStatusFilter || undefined,
            user_id: queueUserFilter || undefined,
            page: queuePage,
            page_size: queuePageSize
          })
        );
        if (cancelled) return;
        setQueueOverview(overview);
        setQueueTasks(tasks);
        setQueueSelectedTask((prev) => {
          if (!prev) return prev;
          return tasks.tasks.find((t) => t.task_id === prev.task_id) ?? prev;
        });
      } catch (e: any) {
        if (!cancelled) setError(`佇列讀取失敗：${formatApiError(e)}`);
      } finally {
        if (!cancelled) setQueueLoading(false);
      }
    }

    refresh();

    if (!queueAutoRefresh) {
      return () => {
        cancelled = true;
      };
    }
    const intervalMs = runMode === "async" ? (queueTasksStreamOk ? 60000 : 10000) : 2000;
    const i = window.setInterval(refresh, intervalMs);
    return () => {
      cancelled = true;
      window.clearInterval(i);
    };
  }, [tab, queueAutoRefresh, queuePage, queuePageSize, queueStatusFilter, queueUserFilter, runMode, queueTasksStreamOk]);

  useEffect(() => {
    if (tab !== "assets") return;
    let cancelled = false;

    async function refresh() {
      setAssetsLoading(true);
      try {
        const [cats, list] = await Promise.all([
          apiGet<AssetCategoriesResponse>("/api/v1/assets/categories"),
          apiGet<AssetsListResponse>(
            withQuery("/api/v1/assets/list", {
              category: assetCategory || undefined,
              tags: assetTagsFilter || undefined,
              q: assetQuery || undefined,
              limit: assetsLimit,
              offset: assetsOffset
            })
          )
        ]);
        if (cancelled) return;
        setAssetCategories(cats.data?.categories ?? {});
        setAssets(list.data?.assets ?? []);
      } catch (e: any) {
        if (!cancelled) setError(`資產讀取失敗：${formatApiError(e)}`);
      } finally {
        if (!cancelled) setAssetsLoading(false);
      }
    }

    refresh();
    return () => {
      cancelled = true;
    };
  }, [tab, assetCategory, assetTagsFilter, assetQuery, assetsLimit, assetsOffset]);

  function addPreset(preset: { name: string; prompt: string; negative_prompt: string; tags: string[] }) {
    const name = preset.name.trim() || "未命名";
    const item: PromptPreset = {
      id: makeLocalId("preset"),
      name,
      prompt: preset.prompt,
      negative_prompt: preset.negative_prompt,
      tags: preset.tags,
      created_at: Date.now()
    };
    setUserPresets((prev) => [item, ...prev].slice(0, 60));
  }

  function addPresetFromCurrent() {
    const name = presetName.trim() || (prompt.trim() ? prompt.trim().slice(0, 32) : "未命名");
    addPreset({ name, prompt, negative_prompt: negativePrompt, tags: parseTags(presetTags) });
    setPresetName("");
    setPresetTags("");
    setLibraryView("presets");
  }

  function deletePreset(id: string) {
    setUserPresets((prev) => prev.filter((p) => p.id !== id));
  }

  function applyPresetToInputs(p: { prompt: string; negative_prompt: string }) {
    setPrompt(p.prompt);
    setNegativePrompt(p.negative_prompt);
    if (tab !== "generate") setTab("generate");
  }

  function historyKey(entry: Omit<PromptHistoryItem, "id" | "created_at">): string {
    return JSON.stringify({
      prompt: entry.prompt,
      negative_prompt: entry.negative_prompt,
      model_id: entry.model_id ?? null,
      mode: entry.mode,
      run_mode: entry.run_mode,
      width: entry.width,
      height: entry.height,
      steps: entry.steps,
      cfg_scale: entry.cfg_scale,
      seed: entry.seed,
      num_images: entry.num_images,
      strength: typeof entry.strength === "number" ? entry.strength : null,
      mask_blur: typeof entry.mask_blur === "number" ? entry.mask_blur : null,
      inpainting_fill: typeof entry.inpainting_fill === "string" ? entry.inpainting_fill : null,
      controlnet: entry.controlnet
        ? {
            type: entry.controlnet.type,
            strength: entry.controlnet.strength,
            preprocess: entry.controlnet.preprocess
          }
        : null,
      init_asset_id: entry.init_asset_id ?? null,
      mask_asset_id: entry.mask_asset_id ?? null,
      control_asset_id: entry.control_asset_id ?? null
    });
  }

  function historyKeyFromItem(item: PromptHistoryItem): string {
    const { id: _id, created_at: _createdAt, ...entry } = item;
    return historyKey(entry);
  }

  function recordHistory(entry: Omit<PromptHistoryItem, "id" | "created_at">) {
    const item: PromptHistoryItem = { ...entry, id: makeLocalId("hist"), created_at: Date.now() };
    const key = historyKey(entry);
    setPromptHistory((prev) => {
      const deduped = prev.filter((h) => historyKeyFromItem(h) !== key);
      return [item, ...deduped].slice(0, 60);
    });
  }

  function serverRecordToHistoryItem(rec: any): PromptHistoryItem | null {
    if (!rec || typeof rec !== "object") return null;
    const taskType = rec.task_type;
    if (taskType !== "txt2img" && taskType !== "img2img" && taskType !== "inpaint") return null;
    const input = rec.input_params && typeof rec.input_params === "object" ? rec.input_params : {};
    const assets = rec.source_assets && typeof rec.source_assets === "object" ? rec.source_assets : {};

    const createdAtMs = typeof rec.created_at === "string" ? Date.parse(rec.created_at) : NaN;
    const created_at = Number.isFinite(createdAtMs) ? createdAtMs : Date.now();

    const width = typeof input.width === "number" ? input.width : 1024;
    const height = typeof input.height === "number" ? input.height : 1024;
    const steps =
      typeof input.num_inference_steps === "number"
        ? input.num_inference_steps
        : typeof input.steps === "number"
          ? input.steps
          : 25;
    const cfg_scale =
      typeof input.guidance_scale === "number"
        ? input.guidance_scale
        : typeof input.cfg_scale === "number"
          ? input.cfg_scale
          : 7.5;
    const seed = typeof input.seed === "number" ? input.seed : -1;
    const num_images = typeof input.num_images === "number" ? input.num_images : 1;

    const cn = input.controlnet && typeof input.controlnet === "object" ? input.controlnet : null;
    const controlnet =
      cn && (cn.type === "canny" || cn.type === "depth" || cn.type === "openpose" || cn.type === "scribble" || cn.type === "mlsd" || cn.type === "normal")
        ? {
            type: cn.type as ControlNetType,
            strength: typeof cn.strength === "number" ? cn.strength : 1.0,
            preprocess: typeof cn.preprocess === "boolean" ? cn.preprocess : true
          }
        : null;

    const init_asset_id =
      typeof assets.init_asset_id === "string"
        ? assets.init_asset_id
        : typeof input.init_asset_id === "string"
          ? input.init_asset_id
          : null;
    const mask_asset_id =
      typeof assets.mask_asset_id === "string"
        ? assets.mask_asset_id
        : typeof input.mask_asset_id === "string"
          ? input.mask_asset_id
          : null;
    const control_asset_id =
      typeof assets.control_asset_id === "string"
        ? assets.control_asset_id
        : typeof input.control_asset_id === "string"
          ? input.control_asset_id
          : cn && typeof cn.asset_id === "string"
            ? cn.asset_id
            : null;

    return {
      id: `server:${String(rec.history_id ?? created_at)}`,
      prompt: typeof input.prompt === "string" ? input.prompt : "",
      negative_prompt: typeof input.negative_prompt === "string" ? input.negative_prompt : "",
      model_id: typeof input.model_id === "string" ? input.model_id : null,
      mode: taskType,
      run_mode: rec.run_mode === "sync" || rec.run_mode === "async" ? rec.run_mode : "async",
      width,
      height,
      steps,
      cfg_scale,
      seed,
      num_images,
      strength: typeof input.strength === "number" ? input.strength : undefined,
      mask_blur: typeof input.mask_blur === "number" ? input.mask_blur : undefined,
      inpainting_fill: typeof input.inpainting_fill === "string" ? input.inpainting_fill : undefined,
      controlnet,
      init_asset_id,
      mask_asset_id,
      control_asset_id,
      created_at
    };
  }

  async function rerunServerHistoryRecord(rec: any) {
    setError("");
    setStatus(null);
    setBusy(true);
    let enqueued = false;
    try {
      const id = rec?.history_id;
      if (!id || typeof id !== "string") throw new Error("缺少 history_id");
      const resp = await apiPost<HistoryRerunResponse>(`/api/v1/history/${encodeURIComponent(id)}/rerun`, {
        priority: "normal",
        user_id: queueUserFilter || "local"
      });
      if (!resp?.success || !resp?.data?.task_id) {
        throw new Error(resp?.message || "後端 rerun 失敗");
      }
      trackTask(resp.data.task_id);
      enqueued = true;
    } catch (e: any) {
      setError(`後端重跑失敗：${formatApiError(e)}`);
    } finally {
      if (!enqueued) setBusy(false);
    }
  }

  async function importServerHistoryOutputs(rec: any) {
    setError("");
    setSavingAsset(true);
    try {
      const id = rec?.history_id;
      if (!id || typeof id !== "string") throw new Error("缺少 history_id");
      const resp = await apiPost<any>("/api/v1/assets/import_from_history", {
        history_id: id,
        category: quickSaveCategory,
        tags: parseTags(quickSaveTags),
        description: quickSaveDescription
      });
      if (!resp?.success) throw new Error(resp?.message || "匯入失敗");
      await refreshAssetsOnce();
    } catch (e: any) {
      setError(`匯入資產失敗：${formatApiError(e)}`);
    } finally {
      setSavingAsset(false);
    }
  }

  function trackTask(taskId: string) {
    setLastTaskId(taskId);
    setTaskCenterIds((prev) => [taskId, ...prev.filter((id) => id !== taskId)].slice(0, 10));
  }

  function untrackTask(taskId: string) {
    setTaskCenterIds((prev) => prev.filter((id) => id !== taskId));
    setTaskCenterStatusById((prev) => {
      const next = { ...prev };
      delete next[taskId];
      return next;
    });
    if (lastTaskId === taskId) {
      setLastTaskId("");
      setStatus(null);
    }
  }

  function clearTaskCenter() {
    setTaskCenterIds([]);
    setTaskCenterStatusById({});
  }

  async function copyText(text: string) {
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(text);
        return;
      }
    } catch {
      // ignore and fallback below
    }
    const el = document.createElement("textarea");
    el.value = text;
    el.setAttribute("readonly", "true");
    el.style.position = "fixed";
    el.style.left = "-9999px";
    document.body.appendChild(el);
    el.select();
    document.execCommand("copy");
    document.body.removeChild(el);
  }

  async function refreshServerHistoryOnce(opts?: { offset?: number }) {
    setServerHistoryLoading(true);
    try {
      const limit = Math.max(1, Math.min(200, serverHistoryLimit));
      const offset = Math.max(0, opts?.offset ?? serverHistoryOffset);
      const q = serverHistoryQuery.trim() || undefined;
      const task_type = serverHistoryTaskType.trim() || undefined;
      const user_id = serverHistoryUserId.trim() || undefined;
      const since = serverHistorySince.trim() || undefined;
      const until = serverHistoryUntil.trim() || undefined;

      const resp = await apiGet<HistoryListResponse>(
        withQuery("/api/v1/history/list", { limit, offset, q, task_type, user_id, since, until })
      );
      if (!resp?.success) throw new Error(resp?.message || "後端 history 讀取失敗");
      setServerHistory(Array.isArray(resp.data?.records) ? resp.data.records : []);
      setServerHistoryOffset(offset);
    } catch (e: any) {
      setError(`後端 history 讀取失敗：${formatApiError(e)}`);
    } finally {
      setServerHistoryLoading(false);
    }
  }

  async function cleanupServerHistory() {
    setError("");
    setServerHistoryLoading(true);
    try {
      const days = clampInt(serverHistoryCleanupDays, 30);
      const ok = window.confirm(`確認要清理 ${days} 天以前的歷史紀錄？此操作會刪除檔案。`);
      if (!ok) return;

      const resp = await apiDelete<any>(withQuery("/api/v1/history/cleanup", { older_than_days: days }));
      if (!resp?.success) throw new Error(resp?.message || "清理失敗");
      await refreshServerHistoryOnce({ offset: 0 });
    } catch (e: any) {
      setError(`清理失敗：${formatApiError(e)}`);
    } finally {
      setServerHistoryLoading(false);
    }
  }

  async function loadAssetDataUrlById(assetId: string): Promise<string> {
    const resp = await apiGet<any>(`/api/v1/assets/${assetId}`);
    if (!resp?.success) throw new Error("資產讀取失敗");
    const asset = resp?.data;
    const src = asset?.file_url || asset?.download_url;
    if (!src) throw new Error("此資產缺少可用的 file_url/download_url");
    return urlToDataUrl(resolveMediaUrl(src));
  }

  async function applyHistoryItem(h: PromptHistoryItem, opts?: { loadAssets?: boolean }) {
    setPrompt(h.prompt);
    setNegativePrompt(h.negative_prompt);
    if (h.model_id) setModelId(h.model_id);
    setMode(h.mode);
    setRunMode(h.run_mode);
    setWidth(String(h.width));
    setHeight(String(h.height));
    setSteps(String(h.steps));
    setCfg(String(h.cfg_scale));
    setSeed(h.seed === -1 ? "" : String(h.seed));
    setNumImages(String(h.num_images));
    if (typeof h.strength === "number") setStrength(String(h.strength));
    if (typeof h.mask_blur === "number") setMaskBlur(String(h.mask_blur));
    if (typeof h.inpainting_fill === "string") setInpaintFill(h.inpainting_fill as any);

    const hasCn = !!h.controlnet && h.mode === "img2img";
    setUseControlNet(hasCn);
    if (h.controlnet) {
      setControlType(h.controlnet.type);
      setControlStrength(String(h.controlnet.strength));
      setControlPreprocess(!!h.controlnet.preprocess);
    }

    setImg2imgAssetId(h.init_asset_id ?? "");
    setMaskAssetId(h.mask_asset_id ?? "");
    setControlAssetId(h.control_asset_id ?? "");

    if (tab !== "generate") setTab("generate");

    if (opts?.loadAssets) {
      if (h.init_asset_id) {
        const dataUrl = await loadAssetDataUrlById(h.init_asset_id);
        setInitImageDataUrl(dataUrl);
      }
      if (h.mask_asset_id) {
        const dataUrl = await loadAssetDataUrlById(h.mask_asset_id);
        setMaskImageDataUrl(dataUrl);
      }
      if (h.control_asset_id) {
        const dataUrl = await loadAssetDataUrlById(h.control_asset_id);
        setControlImageDataUrl(dataUrl);
      }
    }
  }

  async function rerunHistoryItem(h: PromptHistoryItem) {
    setError("");
    setStatus(null);
    setBusy(true);

    try {
      await applyHistoryItem(h, { loadAssets: false });
      const { id: _id, created_at: _createdAt, ...historyEntry } = h;

      const baseParams = {
        prompt: h.prompt,
        negative_prompt: h.negative_prompt,
        model_id: h.model_id,
        width: h.width,
        height: h.height,
        num_inference_steps: h.steps,
        guidance_scale: h.cfg_scale,
        seed: h.seed,
        num_images: h.num_images
      };

      const resolveInit = async (): Promise<string> => {
        if (h.init_asset_id) {
          const dataUrl = await loadAssetDataUrlById(h.init_asset_id);
          setInitImageDataUrl(dataUrl);
          return dataUrl;
        }
        if (initImageDataUrl) return initImageDataUrl;
        throw new Error("缺少 init_image：請先在 Generate 分頁選擇圖片，或使用資產庫來源。");
      };

      const resolveMask = async (): Promise<string> => {
        if (h.mask_asset_id) {
          const dataUrl = await loadAssetDataUrlById(h.mask_asset_id);
          setMaskImageDataUrl(dataUrl);
          return dataUrl;
        }
        if (maskImageDataUrl) return maskImageDataUrl;
        throw new Error("缺少 mask_image：請先在 Generate 分頁提供 mask，或使用資產庫來源。");
      };

      const resolveControl = async (): Promise<string> => {
        if (h.control_asset_id) {
          const dataUrl = await loadAssetDataUrlById(h.control_asset_id);
          setControlImageDataUrl(dataUrl);
          return dataUrl;
        }
        if (controlImageDataUrl) return controlImageDataUrl;
        throw new Error("缺少 control image：請先在 Generate 分頁提供 control image，或使用資產庫來源。");
      };

      let initAssetIdLocal = typeof h.init_asset_id === "string" ? h.init_asset_id : "";
      let maskAssetIdLocal = typeof h.mask_asset_id === "string" ? h.mask_asset_id : "";
      let controlAssetIdLocal = typeof h.control_asset_id === "string" ? h.control_asset_id : "";

      if (h.run_mode === "async") {
        const taskType = h.mode;
        try {
          if ((h.mode === "img2img" || h.mode === "inpaint") && !initAssetIdLocal) {
            const dataUrl = await resolveInit();
            initAssetIdLocal = await uploadDataUrlAsAssetId(dataUrl, { category: "reference", filename: "init.png" });
            setImg2imgAssetId(initAssetIdLocal);
          }
          if (h.mode === "inpaint" && !maskAssetIdLocal) {
            const dataUrl = await resolveMask();
            maskAssetIdLocal = await uploadDataUrlAsAssetId(dataUrl, { category: "mask", filename: "mask.png" });
            setMaskAssetId(maskAssetIdLocal);
          }
          if (h.mode === "img2img" && h.controlnet && !controlAssetIdLocal) {
            const dataUrl = await resolveControl();
            const category = suggestedControlCategory(h.controlnet.type);
            controlAssetIdLocal = await uploadDataUrlAsAssetId(dataUrl, { category, filename: "control.png" });
            setControlAssetId(controlAssetIdLocal);
          }
        } catch (e: any) {
          setError(`資產上傳失敗：${formatApiError(e)}`);
          setBusy(false);
          return;
        }

        const initFromAsset = !!initAssetIdLocal;
        const maskFromAsset = !!maskAssetIdLocal;
        const controlFromAsset = !!controlAssetIdLocal;
        const parameters =
          h.mode === "img2img"
            ? {
                ...baseParams,
                ...(initFromAsset ? { init_asset_id: initAssetIdLocal } : { init_image: await resolveInit() }),
                strength: typeof h.strength === "number" ? h.strength : 0.75,
                ...(h.controlnet
                  ? {
                      controlnet: {
                        type: h.controlnet.type,
                        ...(controlFromAsset ? {} : { image: await resolveControl() }),
                        preprocess: !!h.controlnet.preprocess,
                        strength: typeof h.controlnet.strength === "number" ? h.controlnet.strength : 1.0
                      }
                    }
                  : {}),
                ...(h.controlnet && controlFromAsset ? { control_asset_id: controlAssetIdLocal } : {})
              }
            : h.mode === "inpaint"
              ? {
                  ...baseParams,
                  ...(initFromAsset ? { init_asset_id: initAssetIdLocal } : { init_image: await resolveInit() }),
                  ...(maskFromAsset ? { mask_asset_id: maskAssetIdLocal } : { mask_image: await resolveMask() }),
                  strength: typeof h.strength === "number" ? h.strength : 0.75,
                  mask_blur: typeof h.mask_blur === "number" ? h.mask_blur : 4,
                  inpainting_fill: typeof h.inpainting_fill === "string" ? h.inpainting_fill : "original"
                }
              : baseParams;

	        const resp = await apiPost<QueueEnqueueResponse>("/api/v1/queue/enqueue", {
	          task_type: taskType,
	          parameters,
	          priority: "normal",
	          user_id: queueUserFilter || "local"
	        });
        if (!resp.success || !resp.task_id) {
          setBusy(false);
	          setError(resp.message || "佇列提交失敗");
	          return;
	        }
	        trackTask(resp.task_id);
	        recordHistory({
            ...historyEntry,
            init_asset_id: initAssetIdLocal || null,
            mask_asset_id: h.mode === "inpaint" ? (maskAssetIdLocal || null) : null,
            control_asset_id: h.controlnet ? (controlAssetIdLocal || null) : null
          });
	        return;
	      }

      if (h.mode === "txt2img") {
        const resp = await apiPost<Txt2ImgResponse>("/api/v1/txt2img/", {
          ...baseParams,
          user_id: queueUserFilter || "local",
          save_images: true,
          return_base64: false
        });
        const imgs = resp?.data?.results?.images ?? [];
        const urls = imgs
          .map((it: any) => (typeof it?.url === "string" ? it.url : null))
          .filter(Boolean) as string[];
        setImages(urls.map(resolveMediaUrl));
        recordHistory(historyEntry);
        return;
      }

      if (h.mode === "img2img") {
        let initDataUrl: string | null = null;
        let controlDataUrl: string | null = null;

        if (!initAssetIdLocal) {
          initDataUrl = await resolveInit();
          if (initDataUrl.startsWith("data:")) {
            try {
              initAssetIdLocal = await uploadDataUrlAsAssetId(initDataUrl, { category: "reference", filename: "init.png" });
              setImg2imgAssetId(initAssetIdLocal);
            } catch (e) {
              console.warn("Failed to upload init image to assets; falling back to base64", e);
            }
          }
        }

        if (h.controlnet && !controlAssetIdLocal) {
          controlDataUrl = await resolveControl();
          if (controlDataUrl.startsWith("data:")) {
            try {
              const category = suggestedControlCategory(h.controlnet.type);
              controlAssetIdLocal = await uploadDataUrlAsAssetId(controlDataUrl, { category, filename: "control.png" });
              setControlAssetId(controlAssetIdLocal);
            } catch (e) {
              console.warn("Failed to upload control image to assets; falling back to base64", e);
            }
          }
        }

        const initFromAsset = !!initAssetIdLocal;
        const controlFromAsset = !!controlAssetIdLocal;
        const resp = await apiPost<Img2ImgResponse>("/api/v1/img2img/", {
          ...baseParams,
          user_id: queueUserFilter || "local",
          ...(initFromAsset ? { init_asset_id: initAssetIdLocal } : { init_image: initDataUrl ?? await resolveInit() }),
          strength: typeof h.strength === "number" ? h.strength : 0.75,
          ...(h.controlnet
            ? {
                controlnet: {
                  type: h.controlnet.type,
                  ...(controlFromAsset ? { asset_id: controlAssetIdLocal } : { image: controlDataUrl ?? await resolveControl() }),
                  preprocess: !!h.controlnet.preprocess,
                  strength: typeof h.controlnet.strength === "number" ? h.controlnet.strength : 1.0
                }
              }
            : {})
        });
        const urls = (resp?.data?.images ?? []).filter((u: any) => typeof u === "string") as string[];
        setImages(urls.map(resolveMediaUrl));
        recordHistory({
          ...historyEntry,
          init_asset_id: initAssetIdLocal || null,
          mask_asset_id: null,
          control_asset_id: h.controlnet ? (controlAssetIdLocal || null) : null
        });
        return;
      }

      let initDataUrl: string | null = null;
      let maskDataUrl: string | null = null;

      if (!initAssetIdLocal) {
        initDataUrl = await resolveInit();
        if (initDataUrl.startsWith("data:")) {
          try {
            initAssetIdLocal = await uploadDataUrlAsAssetId(initDataUrl, { category: "reference", filename: "init.png" });
            setImg2imgAssetId(initAssetIdLocal);
          } catch (e) {
            console.warn("Failed to upload init image to assets; falling back to base64", e);
          }
        }
      }

      if (!maskAssetIdLocal) {
        maskDataUrl = await resolveMask();
        if (maskDataUrl.startsWith("data:")) {
          try {
            maskAssetIdLocal = await uploadDataUrlAsAssetId(maskDataUrl, { category: "mask", filename: "mask.png" });
            setMaskAssetId(maskAssetIdLocal);
          } catch (e) {
            console.warn("Failed to upload mask image to assets; falling back to base64", e);
          }
        }
      }

      const initFromAsset = !!initAssetIdLocal;
      const maskFromAsset = !!maskAssetIdLocal;
      const resp = await apiPost<InpaintResponse>("/api/v1/img2img/inpaint", {
        ...baseParams,
        user_id: queueUserFilter || "local",
        ...(initFromAsset ? { init_asset_id: initAssetIdLocal } : { init_image: initDataUrl ?? await resolveInit() }),
        ...(maskFromAsset ? { mask_asset_id: maskAssetIdLocal } : { mask_image: maskDataUrl ?? await resolveMask() }),
        strength: typeof h.strength === "number" ? h.strength : 0.75,
        mask_blur: typeof h.mask_blur === "number" ? h.mask_blur : 4,
        inpainting_fill: typeof h.inpainting_fill === "string" ? h.inpainting_fill : "original"
      });
      const urls = (resp?.data?.images ?? []).filter((u: any) => typeof u === "string") as string[];
      setImages(urls.map(resolveMediaUrl));
      recordHistory({
        ...historyEntry,
        init_asset_id: initAssetIdLocal || null,
        mask_asset_id: maskAssetIdLocal || null,
        control_asset_id: null
      });
    } catch (e: any) {
      setError(`重跑失敗：${formatApiError(e)}`);
      setBusy(false);
    } finally {
      if (h.run_mode === "sync") setBusy(false);
    }
  }

  async function handleGenerate() {
    setError("");
    setImages([]);
    setStatus(null);
    setBusy(true);

    const seedValue = seed.trim() ? clampInt(seed, -1) : -1;
    const baseParams = {
      prompt,
      negative_prompt: negativePrompt,
      model_id: modelId,
      width: clampInt(width, 1024),
      height: clampInt(height, 1024),
      num_inference_steps: clampInt(steps, 25),
      guidance_scale: clampFloat(cfg, 7.5),
      seed: seedValue,
      num_images: clampInt(numImages, 1)
    };

    try {
      if (runMode === "async") {
        const taskType = mode;
        if (mode === "img2img" && !initImageDataUrl && !img2imgAssetId) {
          setError("請先選擇一張輸入圖片（img2img）");
          setBusy(false);
          return;
        }
        if (mode === "img2img" && useControlNet && !controlImageDataUrl && !controlAssetId) {
          setError("已開啟 ControlNet，但尚未提供 control image");
          setBusy(false);
          return;
        }
        if (mode === "inpaint" && !initImageDataUrl && !img2imgAssetId) {
          setError("請先選擇一張輸入圖片（inpaint）");
          setBusy(false);
          return;
        }
        if (mode === "inpaint" && !maskImageDataUrl && !maskAssetId) {
          setError("請先選擇一張 mask（inpaint）");
          setBusy(false);
          return;
        }

        let initAssetId = img2imgAssetId;
        let maskAssetIdLocal = maskAssetId;
        let controlAssetIdLocal = controlAssetId;

        try {
          if ((mode === "img2img" || mode === "inpaint") && !initAssetId) {
            initAssetId = await uploadDataUrlAsAssetId(initImageDataUrl, { category: "reference", filename: "init.png" });
            setImg2imgAssetId(initAssetId);
          }
          if (mode === "inpaint" && !maskAssetIdLocal) {
            maskAssetIdLocal = await uploadDataUrlAsAssetId(maskImageDataUrl, { category: "mask", filename: "mask.png" });
            setMaskAssetId(maskAssetIdLocal);
          }
          if (mode === "img2img" && useControlNet && !controlAssetIdLocal) {
            const category = controlAssetCategory || "reference";
            controlAssetIdLocal = await uploadDataUrlAsAssetId(controlImageDataUrl, { category, filename: "control.png" });
            setControlAssetId(controlAssetIdLocal);
          }
        } catch (e: any) {
          setError(`資產上傳失敗：${formatApiError(e)}`);
          setBusy(false);
          return;
        }

        const initFromAsset = !!initAssetId;
        const maskFromAsset = !!maskAssetIdLocal;
        const controlFromAsset = !!controlAssetIdLocal;
        const parameters =
          mode === "img2img"
            ? {
                ...baseParams,
                ...(initFromAsset ? { init_asset_id: initAssetId } : { init_image: initImageDataUrl }),
                strength: clampFloat(strength, 0.75),
                ...(useControlNet
                  ? {
                      controlnet: {
                        type: controlType,
                        ...(controlFromAsset ? {} : { image: controlImageDataUrl }),
                        preprocess: controlPreprocess,
                        strength: clampFloat(controlStrength, 1.0)
                      }
                    }
                  : {}),
                ...(useControlNet && controlFromAsset ? { control_asset_id: controlAssetIdLocal } : {})
              }
            : mode === "inpaint"
              ? {
                  ...baseParams,
                  ...(initFromAsset ? { init_asset_id: initAssetId } : { init_image: initImageDataUrl }),
                  ...(maskFromAsset ? { mask_asset_id: maskAssetIdLocal } : { mask_image: maskImageDataUrl }),
                  strength: clampFloat(strength, 0.75),
                  mask_blur: clampInt(maskBlur, 4),
                  inpainting_fill: inpaintFill
                }
              : baseParams;

	        const resp = await apiPost<QueueEnqueueResponse>("/api/v1/queue/enqueue", {
	          task_type: taskType,
	          parameters,
	          priority: "normal",
	          user_id: queueUserFilter || "local"
	        });
        if (!resp.success || !resp.task_id) {
          setBusy(false);
          setError(resp.message || "佇列提交失敗");
          return;
        }
        trackTask(resp.task_id);
        recordHistory({
          prompt,
          negative_prompt: negativePrompt,
          model_id: modelId ?? null,
          mode,
          run_mode: "async",
          width: baseParams.width,
          height: baseParams.height,
          steps: baseParams.num_inference_steps,
          cfg_scale: baseParams.guidance_scale,
          seed: baseParams.seed,
          num_images: baseParams.num_images,
          strength: mode === "txt2img" ? undefined : clampFloat(strength, 0.75),
          mask_blur: mode === "inpaint" ? clampInt(maskBlur, 4) : undefined,
          inpainting_fill: mode === "inpaint" ? inpaintFill : undefined,
	          controlnet:
	            mode === "img2img" && useControlNet
	              ? { type: controlType, strength: clampFloat(controlStrength, 1.0), preprocess: controlPreprocess }
	              : null,
	          init_asset_id: mode === "img2img" || mode === "inpaint" ? (initAssetId || null) : null,
	          mask_asset_id: mode === "inpaint" ? (maskAssetIdLocal || null) : null,
	          control_asset_id: mode === "img2img" && useControlNet ? (controlAssetIdLocal || null) : null
	        });
	        return;
	      }

      if (mode === "txt2img") {
        const resp = await apiPost<Txt2ImgResponse>("/api/v1/txt2img/", {
          ...baseParams,
          user_id: queueUserFilter || "local",
          save_images: true,
          return_base64: false
        });
        const imgs = resp?.data?.results?.images ?? [];
        const urls = imgs
          .map((it: any) => (typeof it?.url === "string" ? it.url : null))
          .filter(Boolean) as string[];
        setImages(urls.map(resolveMediaUrl));
        recordHistory({
          prompt,
          negative_prompt: negativePrompt,
          model_id: modelId ?? null,
          mode: "txt2img",
          run_mode: "sync",
          width: baseParams.width,
          height: baseParams.height,
          steps: baseParams.num_inference_steps,
          cfg_scale: baseParams.guidance_scale,
          seed: baseParams.seed,
          num_images: baseParams.num_images,
          controlnet: null,
          init_asset_id: null,
          mask_asset_id: null,
          control_asset_id: null
        });
      } else if (mode === "img2img") {
        if (!initImageDataUrl && !img2imgAssetId) {
          setError("請先選擇一張輸入圖片（img2img），或提供 init_asset_id");
          setBusy(false);
          return;
        }
        if (useControlNet && !controlImageDataUrl && !controlAssetId) {
          setError("已開啟 ControlNet，但尚未提供 control image（或 control asset）");
          setBusy(false);
          return;
        }
        let initAssetIdLocal = img2imgAssetId;
        let controlAssetIdLocal = controlAssetId;
        if (!initAssetIdLocal && initImageDataUrl && initImageDataUrl.startsWith("data:")) {
          try {
            initAssetIdLocal = await uploadDataUrlAsAssetId(initImageDataUrl, { category: "reference", filename: "init.png" });
            setImg2imgAssetId(initAssetIdLocal);
          } catch (e) {
            console.warn("Failed to upload init image to assets; falling back to base64", e);
          }
        }
        if (useControlNet && !controlAssetIdLocal && controlImageDataUrl && controlImageDataUrl.startsWith("data:")) {
          try {
            const category = controlAssetCategory || "reference";
            controlAssetIdLocal = await uploadDataUrlAsAssetId(controlImageDataUrl, { category, filename: "control.png" });
            setControlAssetId(controlAssetIdLocal);
          } catch (e) {
            console.warn("Failed to upload control image to assets; falling back to base64", e);
          }
        }
        const resp = await apiPost<Img2ImgResponse>("/api/v1/img2img/", {
          ...baseParams,
          user_id: queueUserFilter || "local",
          ...(initAssetIdLocal ? { init_asset_id: initAssetIdLocal } : { init_image: initImageDataUrl }),
          strength: clampFloat(strength, 0.75),
          ...(useControlNet
            ? {
                controlnet: {
                  type: controlType,
                  ...(controlAssetIdLocal ? { asset_id: controlAssetIdLocal } : { image: controlImageDataUrl }),
                  preprocess: controlPreprocess,
                  strength: clampFloat(controlStrength, 1.0)
                }
              }
            : {})
        });
        const urls = (resp?.data?.images ?? []).filter((u: any) => typeof u === "string") as string[];
        setImages(urls.map(resolveMediaUrl));
        recordHistory({
          prompt,
          negative_prompt: negativePrompt,
          model_id: modelId ?? null,
          mode: "img2img",
          run_mode: "sync",
          width: baseParams.width,
          height: baseParams.height,
          steps: baseParams.num_inference_steps,
          cfg_scale: baseParams.guidance_scale,
          seed: baseParams.seed,
          num_images: baseParams.num_images,
          strength: clampFloat(strength, 0.75),
          controlnet: useControlNet ? { type: controlType, strength: clampFloat(controlStrength, 1.0), preprocess: controlPreprocess } : null,
          init_asset_id: initAssetIdLocal || null,
          mask_asset_id: null,
          control_asset_id: useControlNet ? (controlAssetIdLocal || null) : null
        });
      } else {
        if (!initImageDataUrl && !img2imgAssetId) {
          setError("請先選擇一張輸入圖片（inpaint），或提供 init_asset_id");
          setBusy(false);
          return;
        }
        if (!maskImageDataUrl && !maskAssetId) {
          setError("請先選擇一張 mask（inpaint），或提供 mask_asset_id");
          setBusy(false);
          return;
        }

        let initAssetIdLocal = img2imgAssetId;
        let maskAssetIdLocal = maskAssetId;
        if (!initAssetIdLocal && initImageDataUrl && initImageDataUrl.startsWith("data:")) {
          try {
            initAssetIdLocal = await uploadDataUrlAsAssetId(initImageDataUrl, { category: "reference", filename: "init.png" });
            setImg2imgAssetId(initAssetIdLocal);
          } catch (e) {
            console.warn("Failed to upload init image to assets; falling back to base64", e);
          }
        }
        if (!maskAssetIdLocal && maskImageDataUrl && maskImageDataUrl.startsWith("data:")) {
          try {
            maskAssetIdLocal = await uploadDataUrlAsAssetId(maskImageDataUrl, { category: "mask", filename: "mask.png" });
            setMaskAssetId(maskAssetIdLocal);
          } catch (e) {
            console.warn("Failed to upload mask image to assets; falling back to base64", e);
          }
        }

        const resp = await apiPost<InpaintResponse>("/api/v1/img2img/inpaint", {
          ...baseParams,
          user_id: queueUserFilter || "local",
          ...(initAssetIdLocal ? { init_asset_id: initAssetIdLocal } : { init_image: initImageDataUrl }),
          ...(maskAssetIdLocal ? { mask_asset_id: maskAssetIdLocal } : { mask_image: maskImageDataUrl }),
          strength: clampFloat(strength, 0.75),
          mask_blur: clampInt(maskBlur, 4),
          inpainting_fill: inpaintFill
        });
        const urls = (resp?.data?.images ?? []).filter((u: any) => typeof u === "string") as string[];
        setImages(urls.map(resolveMediaUrl));
        recordHistory({
          prompt,
          negative_prompt: negativePrompt,
          model_id: modelId ?? null,
          mode: "inpaint",
          run_mode: "sync",
          width: baseParams.width,
          height: baseParams.height,
          steps: baseParams.num_inference_steps,
          cfg_scale: baseParams.guidance_scale,
          seed: baseParams.seed,
          num_images: baseParams.num_images,
          strength: clampFloat(strength, 0.75),
          mask_blur: clampInt(maskBlur, 4),
          inpainting_fill: inpaintFill,
          controlnet: null,
          init_asset_id: initAssetIdLocal || null,
          mask_asset_id: maskAssetIdLocal || null,
          control_asset_id: null
        });
      }
    } catch (e: any) {
      setError(`生成失敗：${formatApiError(e)}`);
      setBusy(false);
    } finally {
      if (runMode === "sync") setBusy(false);
    }
  }

  async function uploadImageUrlAsAssetId(imageUrl: string, opts?: { category?: string; tags?: string[]; description?: string }) {
    const fetchUrl = resolveMediaUrl(imageUrl);
    const resp = await fetch(fetchUrl, { method: "GET" });
    if (!resp.ok) throw new Error(`Failed to fetch image: ${resp.status}`);
    const blob = await resp.blob();

    let filename = "";
    try {
      const u = new URL(fetchUrl, typeof window !== "undefined" ? window.location.origin : "http://localhost");
      filename = u.pathname.split("/").pop() || "";
    } catch {
      filename = fetchUrl.split("?")[0].split("#")[0].split("/").pop() || "";
    }

    const mime = String(blob.type || "").toLowerCase();
    const extFromMime =
      mime === "image/png"
        ? ".png"
        : mime === "image/jpeg"
          ? ".jpg"
          : mime === "image/webp"
            ? ".webp"
            : mime === "image/gif"
              ? ".gif"
              : ".png";

    filename = filename.replace(/[^a-zA-Z0-9._-]+/g, "_").replace(/^_+/, "");
    if (!filename) filename = `image${extFromMime}`;
    if (!filename.includes(".")) filename = `${filename}${extFromMime}`;

    const file = new File([blob], filename, { type: blob.type || "application/octet-stream" });
    const form = new FormData();
    form.append("files", file, file.name);
    form.append("category", opts?.category || "reference");
    form.append("tags", (opts?.tags ?? []).join(","));
    if (opts?.description) form.append("descriptions", opts.description);

    const uploaded = await apiPostForm<any>("/api/v1/assets/upload", form);
    const first = uploaded?.data?.uploaded_assets?.[0];
    const assetId = first?.asset_id;
    if (!uploaded?.success || typeof assetId !== "string" || !assetId) {
      throw new Error(uploaded?.message || "匯入資產失敗");
    }
    return assetId;
  }

  async function uploadDataUrlAsAssetId(
    dataUrl: string,
    opts?: { category?: string; filename?: string; tags?: string[]; description?: string }
  ) {
    const src = String(dataUrl ?? "").trim();
    if (!src || !src.startsWith("data:")) throw new Error("Missing data URL");

    const resp = await fetch(src, { method: "GET" });
    if (!resp.ok) throw new Error(`Failed to fetch image: ${resp.status}`);
    const blob = await resp.blob();

    const mime = String(blob.type || "").toLowerCase();
    const extFromMime =
      mime === "image/png"
        ? ".png"
        : mime === "image/jpeg"
          ? ".jpg"
          : mime === "image/webp"
            ? ".webp"
            : mime === "image/gif"
              ? ".gif"
              : ".png";

    let filename = (opts?.filename || `image${extFromMime}`).trim();
    filename = filename.replace(/[^a-zA-Z0-9._-]+/g, "_").replace(/^_+/, "");
    if (!filename) filename = `image${extFromMime}`;
    if (!filename.includes(".")) filename = `${filename}${extFromMime}`;

    const file = new File([blob], filename, { type: blob.type || "application/octet-stream" });
    const form = new FormData();
    form.append("files", file, file.name);
    form.append("category", opts?.category || "reference");
    form.append("tags", (opts?.tags ?? []).join(","));
    if (opts?.description) form.append("descriptions", opts.description);

    const uploaded = await apiPostForm<any>("/api/v1/assets/upload", form);
    const first = uploaded?.data?.uploaded_assets?.[0];
    const assetId = first?.asset_id;
    if (!uploaded?.success || typeof assetId !== "string" || !assetId) {
      throw new Error(uploaded?.message || "匯入資產失敗");
    }
    return assetId;
  }

  async function handleUpscaleFromUrl(imageUrl: string, scale: number) {
    setError("");
    setStatus(null);
    setBusy(true);

    try {
	      if (runMode === "async") {
	        const userId = queueUserFilter || "local";
	        if (isOutputsUrl(imageUrl)) {
          const imported = await apiPost<any>("/api/v1/assets/import_from_output", {
            image_url: imageUrl,
            category: "reference",
            tags: [],
            description: ""
          });
          const assetId = imported?.data?.asset_id;
          if (!imported?.success || typeof assetId !== "string") {
            throw new Error(imported?.message || "匯入 outputs 失敗");
          }

          const resp = await apiPost<QueueEnqueueResponse>("/api/v1/queue/enqueue", {
            task_type: "upscale",
            parameters: { image_asset_id: assetId, scale, model: "RealESRGAN_x4plus" },
            priority: "normal",
            user_id: userId
          });
          if (!resp.success || !resp.task_id) {
            setBusy(false);
            setError(resp.message || "佇列提交失敗");
            return;
          }
	          trackTask(resp.task_id);
	          return;
	        }

	        const assetId = await uploadImageUrlAsAssetId(imageUrl, { category: "reference" });
	        const resp = await apiPost<QueueEnqueueResponse>("/api/v1/queue/enqueue", {
	          task_type: "upscale",
	          parameters: { image_asset_id: assetId, scale, model: "RealESRGAN_x4plus" },
	          priority: "normal",
	          user_id: userId
	        });
	        if (!resp.success || !resp.task_id) {
          setBusy(false);
          setError(resp.message || "佇列提交失敗");
          return;
        }
        trackTask(resp.task_id);
        return;
      }

      const userId = queueUserFilter || "local";
      let assetId = "";
      if (isOutputsUrl(imageUrl)) {
        const imported = await apiPost<any>("/api/v1/assets/import_from_output", {
          image_url: imageUrl,
          category: "reference",
          tags: [],
          description: ""
        });
        assetId = imported?.data?.asset_id;
        if (!imported?.success || typeof assetId !== "string") {
          throw new Error(imported?.message || "匯入 outputs 失敗");
        }
      } else {
        assetId = await uploadImageUrlAsAssetId(imageUrl, { category: "reference" });
      }
      const resp = await apiPost<any>("/api/v1/upscale/", {
        image_asset_id: assetId,
        scale,
        model: "RealESRGAN_x4plus",
        user_id: userId
      });
      const url = resp?.data?.result?.image_url;
      if (typeof url === "string") {
        setImages((prev) => Array.from(new Set([resolveMediaUrl(url), ...prev])));
      }
    } catch (e: any) {
      setError(`放大失敗：${formatApiError(e)}`);
      setBusy(false);
    } finally {
      if (runMode === "sync") setBusy(false);
    }
  }

  async function handleFaceRestoreFromUrl(imageUrl: string) {
    setError("");
    setStatus(null);
    setBusy(true);

    try {
	      if (runMode === "async") {
	        const userId = queueUserFilter || "local";
	        if (isOutputsUrl(imageUrl)) {
          const imported = await apiPost<any>("/api/v1/assets/import_from_output", {
            image_url: imageUrl,
            category: "reference",
            tags: [],
            description: ""
          });
          const assetId = imported?.data?.asset_id;
          if (!imported?.success || typeof assetId !== "string") {
            throw new Error(imported?.message || "匯入 outputs 失敗");
          }

          const resp = await apiPost<QueueEnqueueResponse>("/api/v1/queue/enqueue", {
            task_type: "face_restore",
            parameters: { image_asset_id: assetId, model: "GFPGAN_v1.4", upscale: 2 },
            priority: "normal",
            user_id: userId
          });
          if (!resp.success || !resp.task_id) {
            setBusy(false);
            setError(resp.message || "佇列提交失敗");
            return;
          }
	          trackTask(resp.task_id);
	          return;
	        }

	        const assetId = await uploadImageUrlAsAssetId(imageUrl, { category: "reference" });
	        const resp = await apiPost<QueueEnqueueResponse>("/api/v1/queue/enqueue", {
	          task_type: "face_restore",
	          parameters: { image_asset_id: assetId, model: "GFPGAN_v1.4", upscale: 2 },
	          priority: "normal",
	          user_id: userId
	        });
	        if (!resp.success || !resp.task_id) {
          setBusy(false);
          setError(resp.message || "佇列提交失敗");
          return;
        }
        trackTask(resp.task_id);
        return;
      }

      const userId = queueUserFilter || "local";
      let assetId = "";
      if (isOutputsUrl(imageUrl)) {
        const imported = await apiPost<any>("/api/v1/assets/import_from_output", {
          image_url: imageUrl,
          category: "reference",
          tags: [],
          description: ""
        });
        assetId = imported?.data?.asset_id;
        if (!imported?.success || typeof assetId !== "string") {
          throw new Error(imported?.message || "匯入 outputs 失敗");
        }
      } else {
        assetId = await uploadImageUrlAsAssetId(imageUrl, { category: "reference" });
      }
      const resp = await apiPost<any>("/api/v1/face_restore/", {
        image_asset_id: assetId,
        model: "GFPGAN_v1.4",
        upscale: 2,
        user_id: userId
      });
      const url = resp?.data?.result?.image_url;
      if (typeof url === "string") {
        setImages((prev) => Array.from(new Set([resolveMediaUrl(url), ...prev])));
      }
    } catch (e: any) {
      setError(`人臉修復失敗：${formatApiError(e)}`);
      setBusy(false);
    } finally {
      if (runMode === "sync") setBusy(false);
    }
  }

  async function applyAssetToGenerate(asset: Asset, role: "init" | "mask" | "control", forcedType?: ControlNetType) {
    setError("");
    const src = asset.file_url || asset.download_url;
    if (!src) {
      setError("此資產缺少可用的 file_url/download_url");
      return;
    }

    setApplyingAsset(true);
    try {
      const dataUrl = await urlToDataUrl(resolveMediaUrl(src));

      if (role === "init") {
        setInitImageDataUrl(dataUrl);
        setImg2imgAssetId(asset.asset_id);
        setTab("generate");
        if (mode === "txt2img") setMode("img2img");
        return;
      }

      if (role === "mask") {
        setMaskImageDataUrl(dataUrl);
        setMaskAssetId(asset.asset_id);
        setTab("generate");
        setMode("inpaint");
        return;
      }

      setControlImageDataUrl(dataUrl);
      setControlAssetId(asset.asset_id);
      if (asset.category === "pose" || asset.category === "depth" || asset.category === "controlnet" || asset.category === "custom") {
        setControlCategoryAuto(false);
        setControlAssetCategory(asset.category as ControlAssetCategory);
        setControlPreprocess(false);
      }
      setUseControlNet(true);
      setTab("generate");
      if (mode !== "img2img") setMode("img2img");
      if (forcedType) setControlType(forcedType);
    } catch (e: any) {
      setError(`資產套用失敗：${formatApiError(e)}`);
    } finally {
      setApplyingAsset(false);
    }
  }

  const refreshQueueOnce = useCallback(async () => {
    setError("");
    setQueueLoading(true);
    try {
      const overview = await apiGet<QueueOverview>("/api/v1/queue/status");
      const tasks = await apiGet<QueueTaskList>(
        withQuery("/api/v1/queue/tasks", {
          status: queueStatusFilter || undefined,
          user_id: queueUserFilter || undefined,
          page: queuePage,
          page_size: queuePageSize
        })
      );
      setQueueOverview(overview);
      setQueueTasks(tasks);
    } catch (e: any) {
      setError(`佇列讀取失敗：${formatApiError(e)}`);
    } finally {
      setQueueLoading(false);
    }
  }, [queuePage, queuePageSize, queueStatusFilter, queueUserFilter]);

  async function cancelQueueTask(taskId: string, opts?: { force?: boolean }) {
    setError("");
    try {
      const force = !!opts?.force;
      if (force) {
        const ok = window.confirm(
          "強制終止會 terminate Celery task（可能影響 GPU worker 穩定性）。\n\n確定要強制終止嗎？"
        );
        if (!ok) return;
      }
      await apiPostEmpty(
        withQuery(`/api/v1/queue/cancel/${taskId}`, {
          user_id: queueUserFilter || "local",
          force: force ? true : undefined
        })
      );
      await refreshQueueOnce();
    } catch (e: any) {
      setError(`取消任務失敗：${formatApiError(e)}`);
    }
  }

  async function rerunQueueTask(taskId: string) {
    setError("");
    try {
      const resp = await apiPost<any>(`/api/v1/queue/rerun/${taskId}`, {
        priority: "normal",
        user_id: queueUserFilter || "local",
        overrides: {}
      });
      const newTaskId = resp?.data?.task_id;
      if (!resp?.success || typeof newTaskId !== "string" || !newTaskId) {
        throw new Error(resp?.message || "重跑失敗");
      }
      trackTask(newTaskId);
      await refreshQueueOnce();
    } catch (e: any) {
      setError(`重跑失敗：${formatApiError(e)}`);
    }
  }

  async function retryQueueTask(taskId: string) {
    setError("");
    try {
      const resp = await apiPost<any>(`/api/v1/queue/retry/${taskId}`, {
        priority: "normal",
        user_id: queueUserFilter || "local",
        overrides: {}
      });
      const newTaskId = resp?.data?.task_id;
      if (!resp?.success || typeof newTaskId !== "string" || !newTaskId) {
        throw new Error(resp?.message || "重試失敗");
      }
      trackTask(newTaskId);
      await refreshQueueOnce();
    } catch (e: any) {
      setError(`重試失敗：${formatApiError(e)}`);
    }
  }

  async function refreshAssetsOnce() {
    setError("");
    setAssetsLoading(true);
    try {
      const [cats, list] = await Promise.all([
        apiGet<AssetCategoriesResponse>("/api/v1/assets/categories"),
        apiGet<AssetsListResponse>(
          withQuery("/api/v1/assets/list", {
            category: assetCategory || undefined,
            tags: assetTagsFilter || undefined,
            q: assetQuery || undefined,
            limit: assetsLimit,
            offset: assetsOffset
          })
        )
      ]);
      setAssetCategories(cats.data?.categories ?? {});
      setAssets(list.data?.assets ?? []);
    } catch (e: any) {
      setError(`資產讀取失敗：${formatApiError(e)}`);
    } finally {
      setAssetsLoading(false);
    }
  }

  async function handleUploadAssets() {
    setError("");
    if (!uploadFiles.length) {
      setError("請先選擇要上傳的圖片檔案");
      return;
    }

    setUploading(true);
    try {
      const form = new FormData();
      for (const file of uploadFiles) form.append("files", file);
      form.append("category", uploadCategory);
      form.append("tags", uploadTags);
      const desc = uploadDescription.trim();
      if (desc) {
        form.append("descriptions", uploadFiles.map(() => desc).join("|"));
      }

      await apiPostForm<any>("/api/v1/assets/upload", form);
      setUploadFiles([]);
      await refreshAssetsOnce();
    } catch (e: any) {
      setError(`上傳失敗：${formatApiError(e)}`);
    } finally {
      setUploading(false);
    }
  }

  async function deleteAsset(assetId: string) {
    setError("");
    try {
      await apiDelete<any>(`/api/v1/assets/${assetId}`);
      await refreshAssetsOnce();
    } catch (e: any) {
      setError(`刪除資產失敗：${formatApiError(e)}`);
    }
  }

  function toggleAssetSelected(assetId: string) {
    setSelectedAssetIds((prev) => {
      const set = new Set(prev);
      if (set.has(assetId)) set.delete(assetId);
      else set.add(assetId);
      return Array.from(set);
    });
  }

  function selectAllAssetsOnPage() {
    setSelectedAssetIds(assets.map((a) => a.asset_id));
  }

  function clearAssetSelection() {
    setSelectedAssetIds([]);
  }

  async function bulkDeleteAssets() {
    if (!selectedAssetIds.length) return;
    const ok = window.confirm(`確定要刪除 ${selectedAssetIds.length} 筆資產？此操作不可復原。`);
    if (!ok) return;

    setError("");
    setAssetsLoading(true);
    try {
      await apiPost<any>("/api/v1/assets/batch/delete", { asset_ids: selectedAssetIds });
      setSelectedAssetIds([]);
      await refreshAssetsOnce();
    } catch (e: any) {
      setError(`批次刪除失敗：${formatApiError(e)}`);
    } finally {
      setAssetsLoading(false);
    }
  }

  async function bulkUpdateAssets() {
    if (!selectedAssetIds.length) return;
    const tags = bulkTags.trim() ? parseTags(bulkTags) : null;
    const category = bulkCategory.trim() ? bulkCategory.trim() : null;
    if (!tags && !category) {
      setError("請先填寫要套用的 tags 或 category");
      return;
    }

    setError("");
    setAssetsLoading(true);
    try {
      await apiPost<any>("/api/v1/assets/batch/update", {
        items: selectedAssetIds.map((asset_id) => ({
          asset_id,
          tags: tags ?? undefined,
          category: category ?? undefined
        }))
      });
      await refreshAssetsOnce();
    } catch (e: any) {
      setError(`批次更新失敗：${formatApiError(e)}`);
    } finally {
      setAssetsLoading(false);
    }
  }

  function beginEditAsset(a: Asset) {
    setAssetEditId(a.asset_id);
    setAssetEditTags((a.tags ?? []).join(", "));
    setAssetEditDescription(a.description ?? "");
    setAssetEditCategory(a.category);
  }

  function cancelEditAsset() {
    setAssetEditId("");
    setAssetEditTags("");
    setAssetEditDescription("");
    setAssetEditCategory("");
  }

  async function saveEditAsset(assetId: string) {
    setError("");
    try {
      await apiPatch<any>(`/api/v1/assets/${assetId}`, {
        tags: parseTags(assetEditTags),
        description: assetEditDescription,
        category: assetEditCategory
      });
      cancelEditAsset();
      await refreshAssetsOnce();
    } catch (e: any) {
      setError(`更新資產失敗：${formatApiError(e)}`);
    }
  }

  async function saveImageToAssets(imageUrl: string) {
    setError("");
    setSavingAsset(true);
    try {
      const resp = await apiPost<any>("/api/v1/assets/import_from_output", {
        image_url: imageUrl,
        category: quickSaveCategory,
        tags: parseTags(quickSaveTags),
        description: quickSaveDescription
      });
      if (!resp?.success) throw new Error(resp?.message || "匯入失敗");
      await refreshAssetsOnce();
    } catch (e: any) {
      setError(`存成資產失敗：${formatApiError(e)}`);
    } finally {
      setSavingAsset(false);
    }
  }

  async function loadHealth() {
    setError("");
    try {
      const resp = await apiGet<HealthResponse>("/api/v1/health");
      alert(`Health: ${resp?.status ?? "unknown"}`);
    } catch (e: any) {
      setError(`健康檢查失敗：${formatApiError(e)}`);
    }
  }

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <div className="brand__logo">SD</div>
          <div className="brand__meta">
            <div className="brand__title">SD Multi‑Modal Platform</div>
            <div className="brand__subtitle">React UI · Blue System · Card‑First</div>
          </div>
        </div>
        <div className="topbar__right">
          {modelsLoading ? <StatusPill label="模型載入中…" tone="info" /> : <StatusPill label="就緒" tone="ok" />}
        </div>
      </header>

      <div className="container">
        <Tabs value={tab} onChange={setTab} />

        {error ? (
          <div className="alert" role="alert">
            {error}
          </div>
        ) : null}

        {tab === "generate" && (
          <div className="grid">
            <Card
              title="生成設定"
              right={
	              <div className="row row--tight">
	                  <select className="control" value={mode} onChange={(e) => setMode(e.target.value as any)}>
	                    <option value="txt2img">文字生圖</option>
	                    <option value="img2img">圖生圖</option>
	                    <option value="inpaint">修補（Inpaint）</option>
	                  </select>
	                  <select className="control" value={runMode} onChange={(e) => setRunMode(e.target.value as any)}>
	                    <option value="sync">同步</option>
	                    <option value="async">非同步（佇列）</option>
	                  </select>
	                </div>
              }
            >
              <div className="form">
                <label className="field">
                  <span className="field__label">Prompt</span>
                  <textarea className="control control--textarea" value={prompt} onChange={(e) => setPrompt(e.target.value)} />
                </label>

                <label className="field">
                  <span className="field__label">Negative Prompt</span>
                  <textarea
                    className="control control--textarea"
                    value={negativePrompt}
                    onChange={(e) => setNegativePrompt(e.target.value)}
                  />
                </label>

	                {mode === "img2img" && (
	                  <div className="grid2">
                    <label className="field">
                      <span className="field__label">輸入圖片</span>
                      <input
                        className="control"
                        type="file"
                        accept="image/*"
                        onChange={async (e) => {
                          const file = e.target.files?.[0];
                          if (!file) return;
                          setImg2imgAssetId("");
                          setInitImageDataUrl(await fileToDataUrl(file));
                        }}
                      />
                    </label>
                    <label className="field">
                      <span className="field__label">Strength</span>
                      <input
                        className="control"
                        type="range"
                        min="0"
                        max="1"
                        step="0.01"
                        value={strength}
                        onChange={(e) => setStrength(e.target.value)}
                      />
                      <div className="hint">{strength}</div>
                    </label>
	                  </div>
	                )}

	                {mode === "img2img" ? (
	                  <div className="field">
	                    <span className="field__label">ControlNet（可選）</span>
	                    <div className="row row--tight">
	                      <button
	                        className={`btn ${useControlNet ? "btn--primary" : ""}`}
	                        type="button"
	                        onClick={() => {
	                          setUseControlNet((v) => {
	                            const next = !v;
	                            if (next) {
	                              setControlCategoryAuto(true);
	                              const suggested = suggestedControlCategory(controlType);
	                              setControlAssetCategory(suggested);
	                              setControlPreprocess(suggested === "reference");
	                            } else {
	                              setControlAssetId("");
	                              setControlImageDataUrl("");
	                            }
	                            return next;
	                          });
	                        }}
	                      >
	                        ControlNet：{useControlNet ? "開" : "關"}
	                      </button>
	                      <span className="hint">
	                        {useControlNet
	                          ? "提示：reference 適合開啟自動前處理；pose/depth/controlnet 多半是已預處理條件圖，建議關閉自動前處理"
	                          : "需要姿態/深度等條件圖時再開啟"}
	                      </span>
	                    </div>

	                    {useControlNet ? (
	                      <div className="form" style={{ marginTop: 10 }}>
	                        <div className="grid2">
	                          <label className="field">
	                            <span className="field__label">Type</span>
	                            <select
	                              className="control"
	                              value={controlType}
	                              onChange={(e) => {
	                                setControlType(e.target.value as ControlNetType);
	                                setControlAssetId("");
	                              }}
	                            >
	                              <option value="openpose">openpose</option>
	                              <option value="depth">depth</option>
	                              <option value="canny">canny</option>
	                              <option value="scribble">scribble</option>
	                              <option value="mlsd">mlsd</option>
	                              <option value="normal">normal</option>
	                            </select>
	                          </label>
	                          <label className="field">
	                            <span className="field__label">Strength</span>
	                            <input
	                              className="control"
	                              type="range"
	                              min="0"
	                              max="2"
	                              step="0.05"
	                              value={controlStrength}
	                              onChange={(e) => setControlStrength(e.target.value)}
	                            />
	                            <div className="hint">{controlStrength}</div>
	                          </label>
	                        </div>

                          <div className="field">
                            <span className="field__label">Preprocess</span>
                            <div className="row row--tight">
                              <button
                                className={`btn ${controlPreprocess ? "btn--primary" : ""}`}
                                type="button"
                                onClick={() => setControlPreprocess((v) => !v)}
                              >
                                自動前處理：{controlPreprocess ? "開" : "關"}
                              </button>
                              <span className="hint">
                                {controlPreprocess ? "需要 controlnet-aux/annotators" : "直接使用條件圖（已預處理時用）"}
                              </span>
                            </div>
                          </div>

	                        <div className="grid2">
	                          <label className="field">
	                            <span className="field__label">Control Image（上傳）</span>
	                            <input
	                              className="control"
	                              type="file"
	                              accept="image/*"
	                              onChange={async (e) => {
	                                const file = e.target.files?.[0];
	                                if (!file) return;
	                                setControlAssetId("");
	                                setControlImageDataUrl(await fileToDataUrl(file));
	                              }}
	                            />
	                          </label>
	                          <label className="field">
	                            <span className="field__label">Control Image（資產庫）</span>
	                            <div className="row row--tight">
	                              <select
	                                className="control"
	                                value={controlAssetCategory}
	                                disabled={controlAssetLoading}
	                                onChange={(e) => {
	                                  const next = e.target.value as ControlAssetCategory;
	                                  setControlCategoryAuto(false);
	                                  setControlAssetCategory(next);
	                                  setControlPreprocess(next === "reference");
	                                  setControlAssetId("");
	                                }}
	                              >
	                                <option value="reference">reference</option>
	                                <option value="pose">pose</option>
	                                <option value="depth">depth</option>
	                                <option value="controlnet">controlnet</option>
	                                <option value="custom">custom</option>
	                              </select>
	                              <select
	                                className="control"
	                                value={controlAssetId}
	                                disabled={controlAssetLoading}
	                                onChange={async (e) => {
	                                  const id = e.target.value;
	                                  setControlAssetId(id);
	                                  if (!id) return;
	                                  const asset = controlAssetChoices.find((a) => a.asset_id === id);
	                                  const src = asset?.file_url || asset?.download_url;
	                                  if (!src) {
	                                    setError("此 ControlNet 資產缺少可用的 file_url/download_url");
	                                    return;
	                                  }
	                                  try {
	                                    const dataUrl = await urlToDataUrl(resolveMediaUrl(src));
	                                    setControlImageDataUrl(dataUrl);
	                                  } catch (err: any) {
	                                    setError(`ControlNet 資產讀取失敗：${err?.message ?? String(err)}`);
	                                  }
	                                }}
	                              >
	                                <option value="">（不選擇）</option>
	                                {controlAssetChoices.map((a) => (
	                                  <option key={a.asset_id} value={a.asset_id}>
	                                    {a.filename}
	                                  </option>
	                                ))}
	                              </select>
	                              <button
	                                className="btn"
	                                type="button"
	                                disabled={controlAssetLoading}
	                                onClick={async () => {
	                                  setError("");
	                                  setControlAssetLoading(true);
	                                  try {
	                                    const resp = await apiGet<AssetsListResponse>(
	                                      withQuery("/api/v1/assets/list", {
	                                        category: controlAssetCategory,
	                                        limit: 20,
	                                        offset: 0
	                                      })
	                                    );
	                                    setControlAssetChoices(resp.data?.assets ?? []);
	                                  } catch (e: any) {
	                                    setError(`資產載入失敗（ControlNet）：${formatApiError(e)}`);
	                                  } finally {
	                                    setControlAssetLoading(false);
	                                  }
	                                }}
	                              >
	                                重新整理
	                              </button>
	                            </div>
	                            <div className="hint">選擇後會自動載入成 controlnet.image（base64）</div>
	                          </label>
	                        </div>
	                      </div>
	                    ) : null}
	                  </div>
	                ) : null}

	                {mode === "inpaint" && (
	                  <div className="grid2">
	                    <label className="field">
	                      <span className="field__label">輸入圖片（init）</span>
	                      <input
	                        className="control"
	                        type="file"
	                        accept="image/*"
	                        onChange={async (e) => {
	                          const file = e.target.files?.[0];
	                          if (!file) return;
	                          setImg2imgAssetId("");
	                          setInitImageDataUrl(await fileToDataUrl(file));
	                        }}
	                      />
	                    </label>
	                    <label className="field">
	                      <span className="field__label">Mask（白=修補 / 黑=保留）</span>
	                      <input
	                        className="control"
	                        type="file"
	                        accept="image/*"
	                        onChange={async (e) => {
	                          const file = e.target.files?.[0];
	                          if (!file) return;
	                          setMaskAssetId("");
	                          setMaskImageDataUrl(await fileToDataUrl(file));
	                        }}
	                      />
	                    </label>
	                  </div>
	                )}

                  {mode === "inpaint" && initImageDataUrl ? (
                    <div className="field">
                      <span className="field__label">Mask Editor（繪製）</span>
                      <MaskEditor
                        imageDataUrl={initImageDataUrl}
                        value={maskImageDataUrl}
                        onChange={(v) => {
                          setMaskAssetId("");
                          setMaskImageDataUrl(v);
                        }}
                      />
                      <div className="hint">若你用畫筆繪製，會直接更新 mask_image（並可再搭配下方資產庫選擇覆蓋）。</div>
                    </div>
                  ) : null}

	                {mode === "inpaint" ? (
	                  <div className="grid2">
	                    <label className="field">
	                      <span className="field__label">Strength</span>
	                      <input
	                        className="control"
	                        type="range"
	                        min="0"
	                        max="1"
	                        step="0.01"
	                        value={strength}
	                        onChange={(e) => setStrength(e.target.value)}
	                      />
	                      <div className="hint">{strength}</div>
	                    </label>
	                    <label className="field">
	                      <span className="field__label">Mask Blur</span>
	                      <input
	                        className="control"
	                        type="range"
	                        min="0"
	                        max="20"
	                        step="1"
	                        value={maskBlur}
	                        onChange={(e) => setMaskBlur(e.target.value)}
	                      />
	                      <div className="hint">{maskBlur}</div>
	                    </label>
	                  </div>
	                ) : null}

	                {mode === "inpaint" ? (
	                  <label className="field">
	                    <span className="field__label">Fill Method</span>
	                    <select className="control" value={inpaintFill} onChange={(e) => setInpaintFill(e.target.value as any)}>
	                      <option value="original">original</option>
	                      <option value="latent_noise">latent_noise</option>
	                      <option value="latent_nothing">latent_nothing</option>
	                      <option value="white">white</option>
	                    </select>
	                  </label>
	                ) : null}

	                {mode === "img2img" || mode === "inpaint" ? (
	                  <label className="field">
	                    <span className="field__label">從資產庫選擇（reference）</span>
	                    <div className="row row--tight">
	                      <select
                        className="control"
                        value={img2imgAssetId}
                        disabled={img2imgAssetLoading}
                        onChange={async (e) => {
                          const id = e.target.value;
                          setImg2imgAssetId(id);
                          if (!id) return;
                          const asset = img2imgAssetChoices.find((a) => a.asset_id === id);
                          const src = asset?.file_url || asset?.download_url;
                          if (!src) {
                            setError("此資產缺少可用的 file_url/download_url");
                            return;
                          }
                          try {
                            const dataUrl = await urlToDataUrl(resolveMediaUrl(src));
                            setInitImageDataUrl(dataUrl);
                          } catch (err: any) {
                            setError(`資產讀取失敗：${err?.message ?? String(err)}`);
                          }
                        }}
                      >
                        <option value="">（不選擇）</option>
                        {img2imgAssetChoices.map((a) => (
                          <option key={a.asset_id} value={a.asset_id}>
                            {a.filename}
                          </option>
                        ))}
	                      </select>
	                      <button
                        className="btn"
                        type="button"
                        disabled={img2imgAssetLoading}
                        onClick={async () => {
                          setError("");
                          setImg2imgAssetLoading(true);
                          try {
                            const resp = await apiGet<AssetsListResponse>(
                              withQuery("/api/v1/assets/list", { category: "reference", limit: 20, offset: 0 })
                            );
                            setImg2imgAssetChoices(resp.data?.assets ?? []);
                          } catch (e: any) {
                            setError(`資產載入失敗（reference）：${formatApiError(e)}`);
                          } finally {
                            setImg2imgAssetLoading(false);
                          }
                        }}
                      >
	                        重新整理
	                      </button>
	                    </div>
	                    <div className="hint">選擇後會自動載入成 init_image（base64）</div>
	                  </label>
	                ) : null}

	                {mode === "inpaint" ? (
	                  <label className="field">
	                    <span className="field__label">從資產庫選擇（mask）</span>
	                    <div className="row row--tight">
	                      <select
	                        className="control"
	                        value={maskAssetId}
	                        disabled={maskAssetLoading}
	                        onChange={async (e) => {
	                          const id = e.target.value;
	                          setMaskAssetId(id);
	                          if (!id) return;
	                          const asset = maskAssetChoices.find((a) => a.asset_id === id);
	                          const src = asset?.file_url || asset?.download_url;
	                          if (!src) {
	                            setError("此 mask 資產缺少可用的 file_url/download_url");
	                            return;
	                          }
	                          try {
	                            const dataUrl = await urlToDataUrl(resolveMediaUrl(src));
	                            setMaskImageDataUrl(dataUrl);
	                          } catch (err: any) {
	                            setError(`mask 資產讀取失敗：${err?.message ?? String(err)}`);
	                          }
	                        }}
	                      >
	                        <option value="">（不選擇）</option>
	                        {maskAssetChoices.map((a) => (
	                          <option key={a.asset_id} value={a.asset_id}>
	                            {a.filename}
	                          </option>
	                        ))}
	                      </select>
	                      <button
	                        className="btn"
	                        type="button"
	                        disabled={maskAssetLoading}
	                        onClick={async () => {
	                          setError("");
	                          setMaskAssetLoading(true);
	                          try {
	                            const resp = await apiGet<AssetsListResponse>(
	                              withQuery("/api/v1/assets/list", { category: "mask", limit: 20, offset: 0 })
	                            );
	                            setMaskAssetChoices(resp.data?.assets ?? []);
	                          } catch (e: any) {
	                            setError(`資產載入失敗（mask）：${formatApiError(e)}`);
	                          } finally {
	                            setMaskAssetLoading(false);
	                          }
	                        }}
	                      >
	                        重新整理
	                      </button>
	                    </div>
	                    <div className="hint">選擇後會自動載入成 mask_image（base64）</div>
	                  </label>
	                ) : null}

                <div className="grid2">
                  <label className="field">
                    <span className="field__label">Model</span>
                    <select className="control" value={modelId} onChange={(e) => setModelId(e.target.value)}>
                      {models.map((m) => (
                        <option key={m.model_id} value={m.model_id}>
                          {m.model_id}
                          {m.active ? " (active)" : ""}
                        </option>
                      ))}
                    </select>
                  </label>

                  <label className="field">
                    <span className="field__label">Seed（空白=隨機）</span>
                    <input className="control" value={seed} onChange={(e) => setSeed(e.target.value)} placeholder="-1" />
                  </label>
                </div>

                <div className="grid3">
                  <label className="field">
                    <span className="field__label">W</span>
                    <input className="control" value={width} onChange={(e) => setWidth(e.target.value)} />
                  </label>
                  <label className="field">
                    <span className="field__label">H</span>
                    <input className="control" value={height} onChange={(e) => setHeight(e.target.value)} />
                  </label>
                  <label className="field">
                    <span className="field__label">Images</span>
                    <input className="control" value={numImages} onChange={(e) => setNumImages(e.target.value)} />
                  </label>
                </div>

                <div className="grid2">
                  <label className="field">
                    <span className="field__label">Steps</span>
                    <input className="control" value={steps} onChange={(e) => setSteps(e.target.value)} />
                  </label>
                  <label className="field">
                    <span className="field__label">CFG</span>
                    <input className="control" value={cfg} onChange={(e) => setCfg(e.target.value)} />
                  </label>
                </div>

                <div className="row">
                  <button className="btn btn--primary" type="button" disabled={busy} onClick={handleGenerate}>
                    {busy ? "處理中…" : "開始生成"}
                  </button>
                  <button className="btn" type="button" disabled={busy} onClick={() => (setImages([]), setStatus(null), setLastTaskId(""))}>
                    清空
                  </button>
                  {runMode === "async" && lastTaskId ? <StatusPill label={`Task: ${lastTaskId}`} tone="info" /> : null}
                </div>
              </div>
            </Card>

            <Card
              title="結果"
              right={
                status ? (
                  <StatusPill
                    label={`${status.status}${status.progress_percent ? ` · ${status.progress_percent}%` : ""}`}
                    tone={status.status === "completed" ? "ok" : status.status === "failed" ? "bad" : "info"}
                  />
                ) : (
                  <StatusPill label={images.length ? `${images.length} 張` : "尚無"} tone="info" />
                )
              }
            >
	              {initImageDataUrl && (mode === "img2img" || mode === "inpaint") ? (
	                <div className="preview">
	                  <div className="preview__label">Input</div>
	                  <img className="preview__img" src={initImageDataUrl} alt="input" />
	                </div>
	              ) : null}

	              {mode === "img2img" && useControlNet && controlImageDataUrl ? (
	                <div className="preview">
	                  <div className="preview__label">Control</div>
	                  <img className="preview__img" src={controlImageDataUrl} alt="control" />
	                </div>
	              ) : null}

	              {mode === "inpaint" && maskImageDataUrl ? (
	                <div className="preview">
	                  <div className="preview__label">Mask</div>
	                  <img className="preview__img" src={maskImageDataUrl} alt="mask" />
	                </div>
	              ) : null}

              <div className="row row--tight" style={{ marginBottom: 10 }}>
                <select className="control" value={quickSaveCategory} onChange={(e) => setQuickSaveCategory(e.target.value)} disabled={savingAsset}>
                  <option value="reference">reference</option>
                  <option value="mask">mask</option>
                  <option value="pose">pose</option>
                  <option value="depth">depth</option>
                  <option value="controlnet">controlnet</option>
                  <option value="custom">custom</option>
                </select>
                <input
                  className="control"
                  value={quickSaveTags}
                  onChange={(e) => setQuickSaveTags(e.target.value)}
                  placeholder="存成資產 tags（逗號）"
                  disabled={savingAsset}
                />
                <input
                  className="control"
                  value={quickSaveDescription}
                  onChange={(e) => setQuickSaveDescription(e.target.value)}
                  placeholder="描述（可選）"
                  disabled={savingAsset}
                />
                {savingAsset ? <StatusPill label="存檔中…" tone="info" /> : <StatusPill label="一鍵存成資產" tone="info" />}
              </div>

              {images.length ? (
                <div className="images">
                  {images.map((u) => (
                    <div key={u} className="imageCard">
                      <a className="image" href={u} target="_blank" rel="noreferrer">
                        <img src={u} alt="generated" loading="lazy" />
                      </a>
                      <div className="row row--tight imageCard__actions">
                        <button className="btn" type="button" disabled={busy} onClick={() => handleUpscaleFromUrl(u, 2)}>
                          放大 ×2
                        </button>
                        <button className="btn" type="button" disabled={busy} onClick={() => handleUpscaleFromUrl(u, 4)}>
                          放大 ×4
                        </button>
                        <button className="btn" type="button" disabled={busy} onClick={() => handleFaceRestoreFromUrl(u)}>
                          人臉修復
                        </button>
                        <button className="btn" type="button" disabled={savingAsset} onClick={() => saveImageToAssets(u)}>
                          存成資產
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="empty">生成結果會顯示在這裡（支援 /outputs 靜態路徑）</div>
              )}
            </Card>

            <Card
              title="Task Center"
              right={<StatusPill label={`${taskCenterIds.length} 追蹤`} tone="info" />}
            >
              {runMode !== "async" ? (
                <div className="empty">切換到「非同步（佇列）」後才會顯示任務中心。</div>
              ) : (
                <>
                  <div className="row row--tight" style={{ marginBottom: 10 }}>
                    <button
                      className="btn"
                      type="button"
                      onClick={() => {
                        clearTaskCenter();
                        setLastTaskId("");
                        setStatus(null);
                      }}
                    >
                      清除追蹤
                    </button>
                  </div>

                  {taskCenterIds.length ? (
                    <div className="tasklist">
                      {taskCenterIds.map((id) => {
                        const s = taskCenterStatusById[id] || (id === lastTaskId ? status : null);
                        const st = s?.status ?? "pending";
                        const tone = st === "completed" ? "ok" : st === "failed" ? "bad" : "info";
                        const cancellable = st === "pending" || st === "running";
                        const retryable = st === "failed" || st === "timeout" || st === "cancelled";
                        const rerunnable = st === "completed" || retryable;
                        return (
                          <div key={id} className="task">
                            <div className="task__top">
                              <button className="task__id" type="button" onClick={() => trackTask(id)}>
                                {id}
                              </button>
                              <StatusPill label={`${s?.task_type ?? "-"} · ${st}`} tone={tone as any} />
                            </div>
                            <div className="task__meta">
                              <div>Step: {s?.current_step ?? "-"}</div>
                              <div>{typeof s?.progress_percent === "number" ? `${s.progress_percent}%` : "-"}</div>
                            </div>
                            <div className="task__bar">
                              <div className="task__barFill" style={{ width: `${s?.progress_percent ?? 0}%` }} />
                            </div>
                            <div className="row row--tight">
                              {cancellable ? (
                                <button className="btn" type="button" onClick={() => cancelQueueTask(id)}>
                                  取消
                                </button>
                              ) : null}
                              {cancellable && s?.status === "running" ? (
                                <button className="btn btn--danger" type="button" onClick={() => cancelQueueTask(id, { force: true })}>
                                  強制終止
                                </button>
                              ) : null}
                              {retryable ? (
                                <button className="btn" type="button" onClick={() => retryQueueTask(id)}>
                                  重試
                                </button>
                              ) : null}
                              {rerunnable ? (
                                <button className="btn" type="button" onClick={() => rerunQueueTask(id)}>
                                  重跑
                                </button>
                              ) : null}
                              <button className="btn" type="button" onClick={() => untrackTask(id)}>
                                移除
                              </button>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  ) : (
                    <div className="empty">尚無追蹤任務（提交佇列後會自動加入）。</div>
                  )}
                </>
              )}
            </Card>

            <Card
              className="card--full"
              title="Prompt / Style 卡片庫"
              right={<StatusPill label={`${userPresets.length} 收藏 · ${promptHistory.length} 本機歷史 · ${serverHistory.length} 後端`} tone="info" />}
            >
              <div className="row row--tight">
                <button
                  className={`btn ${libraryView === "presets" ? "btn--primary" : ""}`}
                  type="button"
                  onClick={() => setLibraryView("presets")}
                >
                  收藏
                </button>
                <button
                  className={`btn ${libraryView === "history" ? "btn--primary" : ""}`}
                  type="button"
                  onClick={() => setLibraryView("history")}
                >
                  歷史
                </button>
                <span className="hint">提示：本機 localStorage + 後端 /api/v1/history（可重跑）。</span>
              </div>

              {libraryView === "presets" ? (
                <>
                  <div className="form" style={{ marginTop: 12 }}>
                    <div className="grid2">
                      <label className="field">
                        <span className="field__label">名稱（可選）</span>
                        <input
                          className="control"
                          value={presetName}
                          onChange={(e) => setPresetName(e.target.value)}
                          placeholder="例如：藍色產品化 Dashboard"
                        />
                      </label>
                      <label className="field">
                        <span className="field__label">Tags（逗號分隔，可選）</span>
                        <input
                          className="control"
                          value={presetTags}
                          onChange={(e) => setPresetTags(e.target.value)}
                          placeholder="blue, uiux, cards"
                        />
                      </label>
                    </div>
                    <div className="row">
                      <button className="btn btn--primary" type="button" disabled={!prompt.trim()} onClick={addPresetFromCurrent}>
                        收藏目前 Prompt
                      </button>
                      <button className="btn" type="button" disabled={!userPresets.length} onClick={() => setUserPresets([])}>
                        清空收藏
                      </button>
                    </div>
                  </div>

                  <div className="presetGrid">
                    {allPresets.map((p) => (
                      <div key={p.id} className="preset">
                        <div className="preset__top">
                          <div className="preset__title">{p.name}</div>
                          {p.builtin ? <StatusPill label="Built‑in" tone="info" /> : null}
                        </div>
                        {p.tags?.length ? <div className="preset__tags">{p.tags.join(", ")}</div> : null}
                        <div className="preset__text">{p.prompt}</div>
                        <div className="row row--tight">
                          <button className="btn btn--primary" type="button" onClick={() => applyPresetToInputs(p)}>
                            套用
                          </button>
                          {!p.builtin ? (
                            <button className="btn" type="button" onClick={() => deletePreset(p.id)}>
                              刪除
                            </button>
                          ) : null}
                        </div>
                      </div>
                    ))}
                  </div>
                </>
              ) : (
                <>
                  <div className="form" style={{ marginTop: 12 }}>
                    <div className="grid3">
                      <label className="field">
                        <span className="field__label">搜尋（q）</span>
                        <input
                          className="control"
                          value={serverHistoryQuery}
                          onChange={(e) => setServerHistoryQuery(e.target.value)}
                          placeholder="prompt / model / task_id…"
                        />
                      </label>
                      <label className="field">
                        <span className="field__label">task_type</span>
                        <select className="control" value={serverHistoryTaskType} onChange={(e) => setServerHistoryTaskType(e.target.value)}>
                          <option value="">全部</option>
                          <option value="txt2img">txt2img</option>
                          <option value="img2img">img2img</option>
                          <option value="inpaint">inpaint</option>
                          <option value="upscale">upscale</option>
                          <option value="face_restore">face_restore</option>
                        </select>
                      </label>
                      <label className="field">
                        <span className="field__label">user_id</span>
                        <input
                          className="control"
                          value={serverHistoryUserId}
                          onChange={(e) => setServerHistoryUserId(e.target.value)}
                          placeholder="local"
                        />
                      </label>
                    </div>

                    <div className="grid3">
                      <label className="field">
                        <span className="field__label">since（ISO，可空）</span>
                        <input
                          className="control"
                          value={serverHistorySince}
                          onChange={(e) => setServerHistorySince(e.target.value)}
                          placeholder="2025-01-01T00:00:00Z"
                        />
                      </label>
                      <label className="field">
                        <span className="field__label">until（ISO，可空）</span>
                        <input
                          className="control"
                          value={serverHistoryUntil}
                          onChange={(e) => setServerHistoryUntil(e.target.value)}
                          placeholder="2025-01-31T23:59:59Z"
                        />
                      </label>
                      <label className="field">
                        <span className="field__label">limit</span>
                        <input
                          className="control"
                          value={String(serverHistoryLimit)}
                          onChange={(e) => setServerHistoryLimit(clampInt(e.target.value, 60))}
                        />
                      </label>
                    </div>

                    <div className="row row--tight">
                      <button
                        className="btn btn--primary"
                        type="button"
                        disabled={serverHistoryLoading}
                        onClick={() => refreshServerHistoryOnce({ offset: 0 })}
                      >
                        套用
                      </button>
                      <button className="btn" type="button" disabled={serverHistoryLoading} onClick={() => refreshServerHistoryOnce()}>
                        後端：重新整理
                      </button>
                      <button
                        className="btn"
                        type="button"
                        disabled={serverHistoryLoading || serverHistoryOffset <= 0}
                        onClick={() => refreshServerHistoryOnce({ offset: Math.max(0, serverHistoryOffset - serverHistoryLimit) })}
                      >
                        上一頁
                      </button>
                      <button
                        className="btn"
                        type="button"
                        disabled={serverHistoryLoading || serverHistory.length < serverHistoryLimit}
                        onClick={() => refreshServerHistoryOnce({ offset: serverHistoryOffset + serverHistoryLimit })}
                      >
                        下一頁
                      </button>
                      <button className="btn" type="button" disabled={!promptHistory.length} onClick={() => setPromptHistory([])}>
                        清空本機
                      </button>
                      {serverHistoryLoading ? (
                        <StatusPill label="後端載入中…" tone="info" />
                      ) : (
                        <StatusPill label={`後端 ${serverHistory.length} 筆 · offset=${serverHistoryOffset}`} tone="info" />
                      )}
                    </div>

                    <div className="row row--tight">
                      <label className="field" style={{ maxWidth: 220 }}>
                        <span className="field__label">清理（older_than_days）</span>
                        <input
                          className="control"
                          value={serverHistoryCleanupDays}
                          onChange={(e) => setServerHistoryCleanupDays(e.target.value)}
                        />
                      </label>
                      <button className="btn" type="button" disabled={serverHistoryLoading} onClick={cleanupServerHistory}>
                        清理後端歷史
                      </button>
                    </div>
                  </div>

                  {promptHistory.length ? (
                    <>
                      <div className="hint" style={{ marginTop: 10 }}>
                        本機（localStorage）
                      </div>
                      <div className="historyList">
                        {promptHistory.map((h) => (
                          <div key={h.id} className="history">
                            <div className="history__meta">
                              <span>{new Date(h.created_at).toLocaleString()}</span>
                              <span>
                                {h.mode} · {h.run_mode}
                              </span>
                              {h.model_id ? <span>model={h.model_id}</span> : null}
                              <span>
                                {h.width}×{h.height} · steps={h.steps} · cfg={h.cfg_scale}
                                {typeof h.strength === "number" ? ` · strength=${h.strength}` : ""}
                              </span>
                              <span>seed={h.seed}</span>
                              {h.controlnet ? <span>CN={h.controlnet.type}</span> : null}
                            </div>
                            <div className="history__prompt">{h.prompt}</div>
                            {h.negative_prompt ? <div className="history__neg">{h.negative_prompt}</div> : null}
                            <div className="row row--tight">
                              <button
                                className="btn btn--primary"
                                type="button"
                                onClick={async () => {
                                  setError("");
                                  try {
                                    await applyHistoryItem(h, { loadAssets: true });
                                  } catch (e: any) {
                                    setError(`套用失敗：${formatApiError(e)}`);
                                  }
                                }}
                              >
                                套用
                              </button>
                              <button className="btn" type="button" disabled={busy} onClick={() => rerunHistoryItem(h)}>
                                套用＋重跑
                              </button>
                              <button
                                className="btn"
                                type="button"
                                onClick={async () => {
                                  const { id: _id, created_at: _createdAt, ...entry } = h;
                                  await copyText(JSON.stringify(entry, null, 2));
                                }}
                              >
                                複製 JSON
                              </button>
                              <button
                                className="btn"
                                type="button"
                                onClick={() =>
                                  addPreset({
                                    name: h.prompt.trim() ? h.prompt.trim().slice(0, 32) : "未命名",
                                    prompt: h.prompt,
                                    negative_prompt: h.negative_prompt,
                                    tags: []
                                  })
                                }
                              >
                                收藏
                              </button>
                            </div>
                          </div>
                        ))}
                      </div>
                    </>
                  ) : null}

                  {serverHistory.length ? (
                    <>
                      <div className="hint" style={{ marginTop: 10 }}>
                        後端（History v1）
                      </div>
                      <div className="historyList">
                        {serverHistory
                          .map((rec) => ({ rec, h: serverRecordToHistoryItem(rec) }))
                          .filter((x) => !!x.h)
                          .map(({ rec, h }) => (
                            <div key={String(rec.history_id)} className="history">
                              <div className="history__meta">
                                <span>{new Date(h!.created_at).toLocaleString()}</span>
                                <span>
                                  {h!.mode} · {h!.run_mode} · server
                                </span>
                                {h!.model_id ? <span>model={h!.model_id}</span> : null}
                                <span>
                                  {h!.width}×{h!.height} · steps={h!.steps} · cfg={h!.cfg_scale}
                                  {typeof h!.strength === "number" ? ` · strength=${h!.strength}` : ""}
                                </span>
                                <span>seed={h!.seed}</span>
                                {h!.controlnet ? <span>CN={h!.controlnet.type}</span> : null}
                              </div>
                              <div className="history__prompt">{h!.prompt}</div>
                              {h!.negative_prompt ? <div className="history__neg">{h!.negative_prompt}</div> : null}
                              <div className="row row--tight">
                                <button
                                  className="btn btn--primary"
                                  type="button"
                                  onClick={async () => {
                                    setError("");
                                    try {
                                      await applyHistoryItem(h!, { loadAssets: true });
                                    } catch (e: any) {
                                      setError(`套用失敗：${formatApiError(e)}`);
                                    }
                                  }}
                                >
                                  套用
                                </button>
                                <button
                                  className="btn"
                                  type="button"
                                  disabled={busy}
                                  onClick={async () => {
                                    setError("");
                                    try {
                                      await applyHistoryItem(h!, { loadAssets: false });
                                      await rerunServerHistoryRecord(rec);
                                    } catch (e: any) {
                                      setError(`後端重跑失敗：${formatApiError(e)}`);
                                    }
                                  }}
                                >
                                  套用＋重跑（後端）
                                </button>
                                <button
                                  className="btn"
                                  type="button"
                                  onClick={async () => {
                                    const input = rec?.input_params && typeof rec.input_params === "object" ? rec.input_params : {};
                                    await copyText(JSON.stringify(input, null, 2));
                                  }}
                                >
                                  複製 JSON
                                </button>
                                <button
                                  className="btn"
                                  type="button"
                                  disabled={savingAsset}
                                  onClick={async () => {
                                    await importServerHistoryOutputs(rec);
                                  }}
                                >
                                  匯入 outputs → 資產
                                </button>
                              </div>
                            </div>
                          ))}
                      </div>
                    </>
                  ) : (
                    <div className="empty">尚無後端歷史（需要完成一次生成後才會寫入）。</div>
                  )}
                </>
              )}
            </Card>
          </div>
        )}

        {tab === "queue" && (
          <div className="grid">
            <Card
              title="佇列概覽"
              right={
                queueLoading ? (
                  <StatusPill label="更新中…" tone="info" />
                ) : queueOverview ? (
                  <StatusPill label="已連線" tone="ok" />
                ) : (
                  <StatusPill label="未知" tone="warn" />
                )
              }
            >
              <div className="row">
                <button className="btn btn--primary" type="button" onClick={refreshQueueOnce} disabled={queueLoading}>
                  重新整理
                </button>
                <button className="btn" type="button" onClick={() => setQueueAutoRefresh((v) => !v)}>
                  {queueAutoRefresh ? "自動更新：開" : "自動更新：關"}
                </button>
                <button className="btn" type="button" onClick={() => setLastTaskId("")}>
                  停止輪詢（單一任務）
                </button>
                <button className="btn" type="button" onClick={loadHealth}>
                  健康檢查
                </button>
              </div>

              {queueOverview ? (
                <div className="stats">
                  <div className="stat">
                    <div className="stat__k">Pending</div>
                    <div className="stat__v">{queueOverview.queue_stats?.pending_tasks ?? "-"}</div>
                  </div>
                  <div className="stat">
                    <div className="stat__k">Running</div>
                    <div className="stat__v">{queueOverview.queue_stats?.running_tasks ?? "-"}</div>
                  </div>
                  <div className="stat">
                    <div className="stat__k">Completed</div>
                    <div className="stat__v">{queueOverview.queue_stats?.completed_tasks ?? "-"}</div>
                  </div>
                  <div className="stat">
                    <div className="stat__k">Failed</div>
                    <div className="stat__v">{queueOverview.queue_stats?.failed_tasks ?? "-"}</div>
                  </div>
                </div>
              ) : (
                <div className="empty">
                  佇列資訊尚不可用（需要 Redis + Celery worker；若只跑同步模式可忽略）
                </div>
              )}

              {lastTaskId ? (
                <div className="mono">
                  <div>Polling Task: {lastTaskId}</div>
                  {status ? JSON.stringify(status, null, 2) : "等待狀態更新…"}
                </div>
              ) : null}
            </Card>

            <Card
              title="任務列表"
              right={
                queueTasks ? (
                  <StatusPill label={`${queueTasks.total_count} 筆`} tone="info" />
                ) : (
                  <StatusPill label="尚無" tone="info" />
                )
              }
            >
              <div className="row row--tight">
                <select className="control" value={queueStatusFilter} onChange={(e) => setQueueStatusFilter(e.target.value)}>
                  <option value="">全部狀態</option>
                  <option value="pending">pending</option>
                  <option value="running">running</option>
                  <option value="completed">completed</option>
                  <option value="failed">failed</option>
                  <option value="cancelled">cancelled</option>
                </select>
                <input
                  className="control"
                  value={queueUserFilter}
                  onChange={(e) => setQueueUserFilter(e.target.value)}
                  placeholder="user_id（可空白）"
                />
                <select className="control" value={queuePageSize} onChange={(e) => setQueuePageSize(Number(e.target.value))}>
                  <option value={10}>10</option>
                  <option value={20}>20</option>
                  <option value={50}>50</option>
                </select>
              </div>

              {queueTasks?.tasks?.length ? (
                <div className="tasklist">
                  {queueTasks.tasks.map((t) => {
                    const tone = t.status === "completed" ? "ok" : t.status === "failed" ? "bad" : "info";
                    const cancellable = t.status === "pending" || t.status === "running";
                    const retryable = t.status === "failed" || t.status === "timeout" || t.status === "cancelled";
                    const rerunnable = t.status === "completed" || retryable;
                    return (
                      <div key={t.task_id} className="task">
                        <div className="task__top">
                          <button className="task__id" type="button" onClick={() => setQueueSelectedTask(t)}>
                            {t.task_id}
                          </button>
                          <StatusPill label={`${t.task_type} · ${t.status}`} tone={tone as any} />
                        </div>
                        <div className="task__meta">
                          <div>Created: {formatIso(t.created_at)}</div>
                          <div>Step: {t.current_step ?? "-"}</div>
                        </div>
                        <div className="task__bar">
                          <div className="task__barFill" style={{ width: `${t.progress_percent ?? 0}%` }} />
                        </div>
                        <div className="row row--tight">
                          <button className="btn" type="button" onClick={() => trackTask(t.task_id)}>
                            追蹤
                          </button>
                          {cancellable ? (
                            <button className="btn" type="button" onClick={() => cancelQueueTask(t.task_id)}>
                              取消
                            </button>
                          ) : null}
                          {cancellable && t.status === "running" ? (
                            <button className="btn btn--danger" type="button" onClick={() => cancelQueueTask(t.task_id, { force: true })}>
                              強制終止
                            </button>
                          ) : null}
                          {retryable ? (
                            <button className="btn" type="button" onClick={() => retryQueueTask(t.task_id)}>
                              重試
                            </button>
                          ) : null}
                          {rerunnable ? (
                            <button className="btn" type="button" onClick={() => rerunQueueTask(t.task_id)}>
                              重跑
                            </button>
                          ) : null}
                        </div>
                      </div>
                    );
                  })}
                </div>
              ) : (
                <div className="empty">沒有任務（或佇列尚未啟用 / Redis 未連線）</div>
              )}

              <div className="row">
                <button className="btn" type="button" onClick={() => setQueuePage((p) => Math.max(1, p - 1))} disabled={queuePage <= 1}>
                  上一頁
                </button>
                <StatusPill label={`Page ${queuePage}`} tone="info" />
                <button
                  className="btn"
                  type="button"
                  onClick={() => setQueuePage((p) => p + 1)}
                  disabled={!queueTasks?.has_next}
                >
                  下一頁
                </button>
              </div>

              {queueSelectedTask ? (
                <div className="mono">{JSON.stringify(queueSelectedTask, null, 2)}</div>
              ) : (
                <div className="empty">點選任務 ID 可查看詳細資料</div>
              )}
            </Card>
          </div>
        )}

        {tab === "assets" && (
          <div className="grid">
            <Card
              title="上傳資產"
              right={uploading ? <StatusPill label="上傳中…" tone="info" /> : <StatusPill label="就緒" tone="ok" />}
            >
              <div className="form">
                <div className="grid2">
                  <label className="field">
                    <span className="field__label">Category</span>
                    <select className="control" value={uploadCategory} onChange={(e) => setUploadCategory(e.target.value)}>
                      <option value="reference">reference</option>
                      <option value="mask">mask</option>
                      <option value="pose">pose</option>
                      <option value="depth">depth</option>
                      <option value="controlnet">controlnet</option>
                      <option value="custom">custom</option>
                    </select>
                  </label>
                  <label className="field">
                    <span className="field__label">Tags（逗號分隔）</span>
                    <input className="control" value={uploadTags} onChange={(e) => setUploadTags(e.target.value)} placeholder="product,blue,ui" />
                  </label>
                </div>

                <label className="field">
                  <span className="field__label">Description（可選，套用到全部檔案）</span>
                  <input
                    className="control"
                    value={uploadDescription}
                    onChange={(e) => setUploadDescription(e.target.value)}
                    placeholder="例如：品牌參考圖 / mask / 姿態"
                  />
                </label>

                <label className="field">
                  <span className="field__label">Files（可多選）</span>
                  <input
                    className="control"
                    type="file"
                    accept="image/*"
                    multiple
                    onChange={(e) => setUploadFiles(Array.from(e.target.files ?? []))}
                  />
                  <div className="hint">{uploadFiles.length ? `${uploadFiles.length} 個檔案待上傳` : "尚未選擇檔案"}</div>
                </label>

                <div className="row">
                  <button className="btn btn--primary" type="button" disabled={uploading} onClick={handleUploadAssets}>
                    上傳
                  </button>
                  <button className="btn" type="button" disabled={uploading} onClick={() => setUploadFiles([])}>
                    清空
                  </button>
                </div>
              </div>
            </Card>

            <Card
              title="資產庫"
              right={
                assetsLoading ? (
                  <StatusPill label="載入中…" tone="info" />
                ) : (
                  <StatusPill label={`${selectedAssetIds.length} 已選取 · ${assets.length} 顯示`} tone="info" />
                )
              }
            >
              <div className="row row--tight">
                <select className="control" value={assetCategory} onChange={(e) => (setAssetsOffset(0), setAssetCategory(e.target.value))}>
                  <option value="">全部分類</option>
                  {Object.keys(assetCategories)
                    .sort()
                    .map((k) => (
                      <option key={k} value={k}>
                        {k} ({assetCategories[k]})
                      </option>
                    ))}
                </select>
                <input
                  className="control"
                  value={assetTagsFilter}
                  onChange={(e) => (setAssetsOffset(0), setAssetTagsFilter(e.target.value))}
                  placeholder="tags 過濾（逗號分隔）"
                />
                <input
                  className="control"
                  value={assetQuery}
                  onChange={(e) => (setAssetsOffset(0), setAssetQuery(e.target.value))}
                  placeholder="搜尋（檔名/描述/tags）"
                />
                <select className="control" value={assetsLimit} onChange={(e) => setAssetsLimit(Number(e.target.value))}>
                  <option value={12}>12</option>
                  <option value={30}>30</option>
                  <option value={60}>60</option>
                </select>
                <button className="btn" type="button" onClick={refreshAssetsOnce} disabled={assetsLoading}>
                  重新整理
                </button>
	              </div>

                <div className="row row--tight" style={{ marginTop: 10 }}>
                  <button className="btn" type="button" disabled={assetsLoading || !assets.length} onClick={selectAllAssetsOnPage}>
                    全選本頁
                  </button>
                  <button className="btn" type="button" disabled={!selectedAssetIds.length} onClick={clearAssetSelection}>
                    清除選取
                  </button>
                  <input
                    className="control"
                    value={bulkTags}
                    onChange={(e) => setBulkTags(e.target.value)}
                    placeholder="批次 tags（逗號）"
                  />
                  <select className="control" value={bulkCategory} onChange={(e) => setBulkCategory(e.target.value)}>
                    <option value="">（不改分類）</option>
                    <option value="reference">reference</option>
                    <option value="mask">mask</option>
                    <option value="pose">pose</option>
                    <option value="depth">depth</option>
                    <option value="controlnet">controlnet</option>
                    <option value="custom">custom</option>
                  </select>
                  <button className="btn" type="button" disabled={!selectedAssetIds.length || assetsLoading} onClick={bulkUpdateAssets}>
                    批次更新
                  </button>
                  <button className="btn" type="button" disabled={!selectedAssetIds.length || assetsLoading} onClick={bulkDeleteAssets}>
                    批次刪除
                  </button>
                </div>

	              <div className="hint">提示：可在卡片上點「設為輸入 / 設為 mask / 設為控制」直接帶入生成頁面。</div>

	              {assets.length ? (
                <div className="assetGrid">
                  {assets.map((a) => {
                    const thumb = a.thumbnail_url || a.file_url || a.download_url || "";
                    return (
                      <div key={a.asset_id} className="asset">
                        {thumb ? (
                          <a href={resolveMediaUrl(thumb)} target="_blank" rel="noreferrer" className="asset__thumb">
                            <img src={resolveMediaUrl(thumb)} alt={a.filename} loading="lazy" />
                          </a>
                        ) : (
                          <div className="asset__thumb asset__thumb--empty">No preview</div>
                        )}
                        <div className="asset__body">
                          <div className="row row--tight" style={{ justifyContent: "space-between" }}>
                            <label className="row row--tight" style={{ gap: 8 }}>
                              <input
                                type="checkbox"
                                checked={selectedAssetIds.includes(a.asset_id)}
                                onChange={() => toggleAssetSelected(a.asset_id)}
                              />
                              <span className="asset__name">{a.filename}</span>
                            </label>
                            <button className="btn" type="button" onClick={() => beginEditAsset(a)}>
                              編輯
                            </button>
                          </div>
                          <div className="asset__meta">
                            <span>{a.category}</span>
                            <span>{formatBytes(a.file_size)}</span>
                            <span>{formatEpochSeconds(a.created_at)}</span>
                          </div>
                          {assetEditId === a.asset_id ? (
                            <div className="form" style={{ marginTop: 8 }}>
                              <div className="grid2">
                                <label className="field">
                                  <span className="field__label">Category</span>
                                  <select className="control" value={assetEditCategory} onChange={(e) => setAssetEditCategory(e.target.value)}>
                                    <option value="reference">reference</option>
                                    <option value="mask">mask</option>
                                    <option value="pose">pose</option>
                                    <option value="depth">depth</option>
                                    <option value="controlnet">controlnet</option>
                                    <option value="custom">custom</option>
                                  </select>
                                </label>
                                <label className="field">
                                  <span className="field__label">Tags（逗號）</span>
                                  <input className="control" value={assetEditTags} onChange={(e) => setAssetEditTags(e.target.value)} />
                                </label>
                              </div>
                              <label className="field">
                                <span className="field__label">Description</span>
                                <input
                                  className="control"
                                  value={assetEditDescription}
                                  onChange={(e) => setAssetEditDescription(e.target.value)}
                                  placeholder="可選"
                                />
                              </label>
                              <div className="row row--tight">
                                <button className="btn btn--primary" type="button" onClick={() => saveEditAsset(a.asset_id)}>
                                  儲存
                                </button>
                                <button className="btn" type="button" onClick={cancelEditAsset}>
                                  取消
                                </button>
                              </div>
                            </div>
                          ) : (
                            <>
                              {a.tags?.length ? <div className="asset__tags">{a.tags.join(", ")}</div> : null}
                              {a.description ? <div className="hint">{a.description}</div> : null}
                            </>
                          )}
                          <div className="row row--tight">
                            {a.category === "reference" ? (
                              <button className="btn" type="button" disabled={applyingAsset} onClick={() => applyAssetToGenerate(a, "init")}>
                                設為輸入
                              </button>
                            ) : null}
                            {a.category === "mask" ? (
                              <button className="btn" type="button" disabled={applyingAsset} onClick={() => applyAssetToGenerate(a, "mask")}>
                                設為 mask
                              </button>
                            ) : null}
                            {a.category === "pose" ? (
                              <button
                                className="btn"
                                type="button"
                                disabled={applyingAsset}
                                onClick={() => applyAssetToGenerate(a, "control", "openpose")}
                              >
                                設為控制（openpose）
                              </button>
                            ) : null}
                            {a.category === "depth" ? (
                              <button className="btn" type="button" disabled={applyingAsset} onClick={() => applyAssetToGenerate(a, "control", "depth")}>
                                設為控制（depth）
                              </button>
                            ) : null}
                            {a.category === "controlnet" || a.category === "custom" ? (
                              <button className="btn" type="button" disabled={applyingAsset} onClick={() => applyAssetToGenerate(a, "control")}>
                                設為控制
                              </button>
                            ) : null}
                            {a.download_url ? (
                              <a className="btn" href={resolveMediaUrl(a.download_url)} target="_blank" rel="noreferrer">
                                下載
                              </a>
                            ) : null}
                            <button className="btn" type="button" onClick={() => deleteAsset(a.asset_id)}>
                              刪除
                            </button>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              ) : (
                <div className="empty">目前沒有資產（可先上傳幾張 reference/mask）</div>
              )}

              <div className="row">
                <button className="btn" type="button" onClick={() => setAssetsOffset((o) => Math.max(0, o - assetsLimit))} disabled={assetsOffset <= 0}>
                  上一頁
                </button>
                <StatusPill label={`Offset ${assetsOffset}`} tone="info" />
                <button className="btn" type="button" onClick={() => setAssetsOffset((o) => o + assetsLimit)} disabled={assets.length < assetsLimit}>
                  下一頁
                </button>
              </div>
            </Card>
          </div>
        )}

        {tab === "system" && (
          <div className="grid">
            <Card title="模型">
              <div className="mono">{JSON.stringify(models, null, 2)}</div>
            </Card>
            <Card title="提示">
              <ul className="list">
                <li>前端預設呼叫：<code>http://localhost:8000</code>（可用 <code>VITE_API_BASE_URL</code> 覆蓋）</li>
                <li>同步：<code>POST /api/v1/txt2img/</code> / <code>POST /api/v1/img2img/</code></li>
                <li>Inpaint：同步 <code>POST /api/v1/img2img/inpaint</code>；非同步 <code>POST /api/v1/queue/enqueue</code>（<code>task_type=inpaint</code>）</li>
                <li>
                  ControlNet：同步 <code>POST /api/v1/img2img/</code> 夾帶 <code>controlnet</code>；非同步 <code>POST /api/v1/queue/enqueue</code>（<code>task_type=img2img</code> 夾帶{" "}
                  <code>controlnet</code>；可用 <code>controlnet.preprocess=false</code> 直接吃已預處理條件圖）
                </li>
                <li>同步後處理：<code>POST /api/v1/upscale/</code> / <code>POST /api/v1/face_restore/</code>（使用 base64 圖片）</li>
                <li>非同步：<code>POST /api/v1/queue/enqueue</code>（需要 Redis + Celery worker）</li>
                <li>
                  非同步後處理：<code>task_type=upscale</code> / <code>task_type=face_restore</code>（參數支援{" "}
                  <code>image_asset_id</code>（preferred）/ <code>image_path</code>（限制在 ASSETS/OUTPUT）/ <code>image</code> base64（legacy））
                </li>
                <li>資產：<code>POST /api/v1/assets/upload</code>（檔案會存到 <code>/mnt/data/.../assets</code>，並透過 <code>/assets/*</code> 提供靜態存取）</li>
              </ul>
            </Card>
          </div>
        )}
      </div>
    </div>
  );
}
