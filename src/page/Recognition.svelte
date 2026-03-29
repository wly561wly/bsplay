<script lang="ts">
  import { onDestroy, onMount } from "svelte";
  import { invoke, open } from "../lib/invoker";
  import { readFile } from "@tauri-apps/plugin-fs";

  type RecognitionFileResult = {
    file_name: string;
    status: string;
    start_time_sec: number | null;
    end_time_sec: number | null;
    victory: boolean;
    ocr_preview: string | null;
    output_file: string | null;
    result_file: string | null;
    ocr_fields?: Record<string, string>;
    error: string | null;
  };

  type RecognitionBatchResult = {
    processed: number;
    skipped: number;
    source_dir: string;
    output_dir: string;
    templates_dir: string;
    interval_sec: number;
    results: RecognitionFileResult[];
  };

  type BuildDatasetSample = {
    id: string;
    image_path: string;
    video_path?: string;
    timestamp_sec?: number;
    phase: string;
    confidence: number;
    class_id?: number;
    roi?: [number, number, number, number];
    label_status: "pending" | "correct" | "wrong";
  };

  type PhaseLabel = "banpick" | "loading" | "gaming" | "victory_or_defeat" | "ending" | "other";

  const phaseOptions: Array<{ value: PhaseLabel; label: string }> = [
    { value: "banpick", label: "banpick" },
    { value: "loading", label: "loading" },
    { value: "gaming", label: "gaming" },
    { value: "victory_or_defeat", label: "victory/defeat" },
    { value: "ending", label: "ending" },
    { value: "other", label: "other" },
  ];

  type BuildDatasetResult = {
    source_dir: string;
    output_dir: string;
    phase_model_mode: string;
    frame_extract_workers: number;
    total_videos: number;
    total_samples: number;
    positives: number;
    negatives: number;
    samples: BuildDatasetSample[];
  };

  type ExportDatasetResult = {
    dataset_dir: string;
    images_count: number;
    labels_count: number;
    train_images_count: number;
    val_images_count: number;
    estimated_accuracy: number;
    bbox_model_mode: string;
    detected_box_labels: number;
    fallback_roi_labels: number;
    exported_at: string;
  };

  type TrainModelResult = {
    success: boolean;
    dataset_dir: string;
    run_dir: string;
    exported_onnx_path: string;
    replaced_model_path: string;
    report_file: string;
    train_stdout_tail: string;
    train_stderr_tail: string;
  };

  type PageMode = "recognition" | "dataset";

  let sourceDir = "";
  let outputDir = "";
  let frameIntervalSec = 1;
  let enableOcr = true;
  let dumpFrames = true;
  let mode: PageMode = "recognition";
  let running = false;
  let error = "";
  let result: RecognitionBatchResult | null = null;

  let datasetOutputDir = "";
  let modelPath = "";
  let highConfidenceThreshold = 0.85;
  let lowConfidenceThreshold = 0.35;
  let maxSamplesPerVideo = 60;
  let extractWorkers = 0;
  let skipExistingVideos = true;
  let detectionConfThreshold = 0.25;
  let detectionIouThreshold = 0.45;
  let datasetRunning = false;
  let datasetError = "";
  let datasetResult: BuildDatasetResult | null = null;
  let exportResult: ExportDatasetResult | null = null;
  let exporting = false;
  let trainRunning = false;
  let trainError = "";
  let trainResult: TrainModelResult | null = null;
  let trainingDatasetDir = "";
  let trainingOutputDir = "";
  let targetModelPath = "";
  let trainEpochs = 30;
  let trainImgsz = 640;
  let pythonBin = "python";
  let sampleFilter: "all" | "pending" | "correct" | "wrong" = "pending";
  let bulkPhase: PhaseLabel = "gaming";
  let activeSampleIndex = 0;
  let page = 1;
  let pageSize = 50;
  let previewImageUrl = "";
  let previewImageError = "";
  let previewImagePath = "";

  function normalizePhase(phase: string): PhaseLabel {
    const v = phase.trim().toLowerCase();
    if (v === "ban-pick") return "banpick";
    if (v === "game") return "gaming";
    if (v === "vectory" || v === "victory" || v === "defeat") return "victory_or_defeat";
    if (v === "unknown") return "other";
    if (v === "banpick" || v === "loading" || v === "gaming" || v === "victory_or_defeat" || v === "ending" || v === "other") {
      return v;
    }
    return "other";
  }

  $: filteredSamples = datasetResult
    ? datasetResult.samples.filter((s) => sampleFilter === "all" || s.label_status === sampleFilter)
    : [];

  $: totalPages = Math.max(1, Math.ceil(filteredSamples.length / pageSize));
  $: if (page > totalPages) {
    page = totalPages;
  }
  $: start = (page - 1) * pageSize;
  $: pagedSamples = filteredSamples.slice(start, start + pageSize);

  $: if (activeSampleIndex >= filteredSamples.length) {
    activeSampleIndex = Math.max(0, filteredSamples.length - 1);
  }
  $: activeSample = filteredSamples[activeSampleIndex] ?? null;
  $: void refreshActiveSamplePreview(activeSample);

  onDestroy(() => {
    if (previewImageUrl.startsWith("blob:")) {
      URL.revokeObjectURL(previewImageUrl);
    }
  });

  function normalizeLocalPath(path: string): string {
    const p = path.trim();
    if (!p) {
      return p;
    }

    const isWindows = navigator.userAgent.toLowerCase().includes("windows");
    if (isWindows && p.startsWith("/mnt/")) {
      const drive = p.slice(5, 6).toUpperCase();
      const rest = p.slice(6).replace(/\//g, "/");
      return `${drive}:${rest}`;
    }

    return p;
  }

  function mimeByExt(path: string): string {
    const ext = path.split(".").pop()?.toLowerCase() ?? "";
    if (ext === "png") return "image/png";
    if (ext === "jpg" || ext === "jpeg") return "image/jpeg";
    if (ext === "webp") return "image/webp";
    if (ext === "bmp") return "image/bmp";
    if (ext === "gif") return "image/gif";
    return "application/octet-stream";
  }

  async function refreshActiveSamplePreview(sample: BuildDatasetSample | null) {
    if (!sample) {
      if (previewImageUrl.startsWith("blob:")) {
        URL.revokeObjectURL(previewImageUrl);
      }
      previewImageUrl = "";
      previewImageError = "";
      previewImagePath = "";
      return;
    }

    const normalizedPath = normalizeLocalPath(sample.image_path);
    if (normalizedPath === previewImagePath) {
      return;
    }

    previewImagePath = normalizedPath;
    previewImageError = "";

    try {
      const bytes = await readFile(normalizedPath);
      const blob = new Blob([bytes], { type: mimeByExt(normalizedPath) });
      const nextUrl = URL.createObjectURL(blob);
      if (previewImageUrl.startsWith("blob:")) {
        URL.revokeObjectURL(previewImageUrl);
      }
      previewImageUrl = nextUrl;
    } catch (e) {
      if (previewImageUrl.startsWith("blob:")) {
        URL.revokeObjectURL(previewImageUrl);
      }
      previewImageUrl = "";
      previewImageError = `预览加载失败: ${String(e)}`;
    }
  }

  onMount(async () => {
    const isWindows = navigator.userAgent.toLowerCase().includes("windows");
    const baseDir = isWindows ? "D:/Desktop/happy/bili-shadowreplay" : "/mnt/d/Desktop/happy/bili-shadowreplay";

    sourceDir = `${baseDir}/test_videos/recog_video`;
    outputDir = `${baseDir}/test_videos/recognition_output`;
    datasetOutputDir = `${baseDir}/test_videos/recognition_dataset`;
    modelPath = `${baseDir}/models/yolo_game.onnx`;
    trainingDatasetDir = `${datasetOutputDir}/yolo_dataset`;
    trainingOutputDir = `${datasetOutputDir}/training_runs`;
    targetModelPath = modelPath;

    const onKeyDown = (ev: KeyboardEvent) => {
      if (mode !== "dataset" || filteredSamples.length === 0) {
        return;
      }

      if (ev.key === "ArrowDown" || ev.key === "j") {
        ev.preventDefault();
        activeSampleIndex = Math.min(filteredSamples.length - 1, activeSampleIndex + 1);
      }
      if (ev.key === "ArrowUp" || ev.key === "k") {
        ev.preventDefault();
        activeSampleIndex = Math.max(0, activeSampleIndex - 1);
      }
      if (ev.key === "a") {
        const sample = filteredSamples[activeSampleIndex];
        if (sample) {
          markSampleStatus(sample.id, "correct");
        }
      }
      if (ev.key === "d") {
        const sample = filteredSamples[activeSampleIndex];
        if (sample) {
          markSampleStatus(sample.id, "wrong");
        }
      }
      if (ev.key === "o") {
        const sample = filteredSamples[activeSampleIndex];
        if (sample) {
          open(normalizeLocalPath(sample.image_path));
        }
      }
      if (["1", "2", "3", "4", "5", "6"].includes(ev.key)) {
        const sample = filteredSamples[activeSampleIndex];
        if (!sample) {
          return;
        }

        const map: Record<string, PhaseLabel> = {
          "1": "banpick",
          "2": "loading",
          "3": "gaming",
          "4": "victory_or_defeat",
          "5": "ending",
          "6": "other",
        };
        updateSamplePhase(sample.id, map[ev.key]);
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  });

  async function startRecognition() {
    if (!sourceDir.trim()) {
      error = "请输入待识别视频目录";
      return;
    }

    if (!outputDir.trim()) {
      error = "请输入输出目录";
      return;
    }

    running = true;
    error = "";
    result = null;
    try {
      result = await invoke<RecognitionBatchResult>("recognize_and_mark_videos", {
        sourceDir,
        outputDir,
        frameIntervalSec,
        enableOcr,
        dumpFrames,
      });
    } catch (e) {
      error = `识别失败: ${String(e)}`;
    } finally {
      running = false;
    }
  }

  async function buildDataset() {
    if (!sourceDir.trim()) {
      datasetError = "请输入视频目录";
      return;
    }
    if (!datasetOutputDir.trim()) {
      datasetError = "请输入训练集输出目录";
      return;
    }

    datasetRunning = true;
    datasetError = "";
    datasetResult = null;
    exportResult = null;

    try {
      datasetResult = await invoke<BuildDatasetResult>("build_recognition_dataset", {
        sourceDir,
        outputDir: datasetOutputDir,
        modelPath,
        highConfidenceThreshold,
        lowConfidenceThreshold,
        maxSamplesPerVideo,
        extractWorkers,
        skipExistingVideos,
      });
      activeSampleIndex = 0;
    } catch (e) {
      datasetError = `构建训练集失败: ${String(e)}`;
    } finally {
      datasetRunning = false;
    }
  }

  async function markSampleStatus(sampleId: string, status: "correct" | "wrong") {
    if (!datasetResult) {
      return;
    }

    const sample = datasetResult.samples.find((s) => s.id === sampleId);
    if (!sample) {
      return;
    }

    sample.label_status = status;
    datasetResult = { ...datasetResult, samples: [...datasetResult.samples] };

    try {
      await invoke("update_dataset_sample_label", {
        outputDir: datasetOutputDir,
        sampleId,
        status,
      });
    } catch (e) {
      datasetError = `更新样本状态失败: ${String(e)}`;
    }
  }

  async function exportDataset() {
    if (!datasetResult) {
      return;
    }

    exporting = true;
    datasetError = "";
    exportResult = null;

    try {
      exportResult = await invoke<ExportDatasetResult>("export_recognition_dataset", {
        outputDir: datasetOutputDir,
        detectionModelPath: modelPath,
        detectionConfThreshold,
        detectionIouThreshold,
      });
      if (exportResult) {
        trainingDatasetDir = exportResult.dataset_dir;
        if (!trainingOutputDir.trim()) {
          trainingOutputDir = `${datasetOutputDir}/training_runs`;
        }
      }
    } catch (e) {
      datasetError = `导出训练集失败: ${String(e)}`;
    } finally {
      exporting = false;
    }
  }

  async function updateSamplePhase(sampleId: string, phase: PhaseLabel) {
    if (!datasetResult) {
      return;
    }

    const sample = datasetResult.samples.find((s) => s.id === sampleId);
    if (!sample) {
      return;
    }

    sample.phase = phase;
    datasetResult = { ...datasetResult, samples: [...datasetResult.samples] };

    try {
      await invoke("update_dataset_sample_phase", {
        outputDir: datasetOutputDir,
        sampleId,
        phase,
      });
    } catch (e) {
      datasetError = `更新样本阶段失败: ${String(e)}`;
    }
  }

  async function trainAndUpdateModel() {
    if (!trainingDatasetDir.trim()) {
      trainError = "请输入训练数据目录（包含 dataset.yaml）";
      return;
    }
    if (!trainingOutputDir.trim()) {
      trainError = "请输入训练输出目录";
      return;
    }
    if (!targetModelPath.trim()) {
      trainError = "请输入目标模型路径";
      return;
    }

    trainRunning = true;
    trainError = "";
    trainResult = null;
    try {
      trainResult = await invoke<TrainModelResult>("train_and_update_recognition_model", {
        datasetDir: trainingDatasetDir,
        outputDir: trainingOutputDir,
        targetModelPath,
        epochs: trainEpochs,
        imgsz: trainImgsz,
        pythonBin,
      });
      modelPath = targetModelPath;
    } catch (e) {
      trainError = `训练或更新模型失败: ${String(e)}`;
    } finally {
      trainRunning = false;
    }
  }

  async function bulkMarkCurrentPage(status: "correct" | "wrong") {
    if (!datasetResult || pagedSamples.length === 0) {
      return;
    }

    const ids = pagedSamples.map((s) => s.id);
    for (const s of datasetResult.samples) {
      if (ids.includes(s.id)) {
        s.label_status = status;
      }
    }
    datasetResult = { ...datasetResult, samples: [...datasetResult.samples] };

    try {
      await invoke<number>("update_dataset_sample_labels_bulk", {
        outputDir: datasetOutputDir,
        sampleIds: ids,
        status,
      });
    } catch (e) {
      datasetError = `批量更新样本状态失败: ${String(e)}`;
    }
  }

  async function bulkSetPhaseCurrentPage(phase: PhaseLabel) {
    if (!datasetResult || pagedSamples.length === 0) {
      return;
    }

    const ids = pagedSamples.map((s) => s.id);
    for (const s of datasetResult.samples) {
      if (ids.includes(s.id)) {
        s.phase = phase;
      }
    }
    datasetResult = { ...datasetResult, samples: [...datasetResult.samples] };

    try {
      await invoke<number>("update_dataset_sample_phases_bulk", {
        outputDir: datasetOutputDir,
        sampleIds: ids,
        phase,
      });
    } catch (e) {
      datasetError = `批量更新样本阶段失败: ${String(e)}`;
    }
  }

  function onSamplePhaseChange(sampleId: string, ev: Event) {
    const value = (ev.currentTarget as HTMLSelectElement | null)?.value ?? "other";
    updateSamplePhase(sampleId, normalizePhase(value));
  }
</script>

<div class="p-4 h-full overflow-y-auto">
  <h1 class="text-2xl font-bold dark:text-white">识别和标记</h1>
  <p class="mt-2 dark:text-gray-400">识别流程：YOLO 目标检测 + 阶段分类 + OCR 提取，支持自动构建训练集并人工校验。</p>

  <div class="mt-5 inline-flex rounded-lg border border-gray-300 dark:border-gray-700 overflow-hidden">
    <button
      class="px-4 py-2 text-sm"
      class:bg-blue-600={mode === "recognition"}
      class:text-white={mode === "recognition"}
      class:bg-white={mode !== "recognition"}
      class:dark:bg-gray-900={mode !== "recognition"}
      class:dark:text-gray-200={mode !== "recognition"}
      on:click={() => (mode = "recognition")}
    >
      识别批处理
    </button>
    <button
      class="px-4 py-2 text-sm"
      class:bg-blue-600={mode === "dataset"}
      class:text-white={mode === "dataset"}
      class:bg-white={mode !== "dataset"}
      class:dark:bg-gray-900={mode !== "dataset"}
      class:dark:text-gray-200={mode !== "dataset"}
      on:click={() => (mode = "dataset")}
    >
      训练集构建与校验
    </button>
  </div>

  <div class="mt-6 grid grid-cols-1 lg:grid-cols-2 gap-4">
    <div>
      <label class="block text-sm font-medium dark:text-gray-200 mb-1">待识别目录</label>
      <input
        class="w-full rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-3 py-2 dark:text-gray-100"
        bind:value={sourceDir}
        placeholder="例如: /path/to/videos"
      />
    </div>
    {#if mode === "recognition"}
      <div>
        <label class="block text-sm font-medium dark:text-gray-200 mb-1">输出目录</label>
        <input
          class="w-full rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-3 py-2 dark:text-gray-100"
          bind:value={outputDir}
          placeholder="例如: /path/to/recognition_completed"
        />
      </div>
    {:else}
      <div>
        <label class="block text-sm font-medium dark:text-gray-200 mb-1">训练集输出目录</label>
        <input
          class="w-full rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-3 py-2 dark:text-gray-100"
          bind:value={datasetOutputDir}
          placeholder="例如: /path/to/dataset"
        />
      </div>
    {/if}
  </div>

  {#if mode === "recognition"}
    <div class="mt-4 flex flex-wrap items-center gap-4">
      <div>
        <label class="block text-sm font-medium dark:text-gray-200 mb-1">抽帧间隔(秒)</label>
        <input
          type="number"
          min="1"
          max="10"
          class="w-32 rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-3 py-2 dark:text-gray-100"
          bind:value={frameIntervalSec}
        />
      </div>
      <label class="inline-flex items-center gap-2 mt-6 dark:text-gray-200">
        <input type="checkbox" bind:checked={enableOcr} />
        <span>启用 OCR 预览</span>
      </label>
      <label class="inline-flex items-center gap-2 mt-6 dark:text-gray-200">
        <input type="checkbox" bind:checked={dumpFrames} />
        <span>导出抽帧图片</span>
      </label>
      <button
        class="mt-6 bg-blue-500 hover:bg-blue-700 disabled:bg-gray-400 text-white font-bold py-2 px-4 rounded"
        on:click={startRecognition}
        disabled={running}
      >
        {#if running}识别中...{:else}开始识别{/if}
      </button>
    </div>
  {:else}
    <div class="mt-4 grid grid-cols-1 lg:grid-cols-4 gap-3">
      <div>
        <label class="block text-sm font-medium dark:text-gray-200 mb-1">YOLO ONNX 路径</label>
        <input
          class="w-full rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-3 py-2 dark:text-gray-100"
          bind:value={modelPath}
          placeholder="例如: /path/to/yolo_game.onnx"
        />
      </div>
      <div>
        <label class="block text-sm font-medium dark:text-gray-200 mb-1">高置信阈值</label>
        <input
          type="number"
          min="0.5"
          max="0.99"
          step="0.01"
          class="w-full rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-3 py-2 dark:text-gray-100"
          bind:value={highConfidenceThreshold}
        />
      </div>
      <div>
        <label class="block text-sm font-medium dark:text-gray-200 mb-1">低置信阈值</label>
        <input
          type="number"
          min="0.01"
          max="0.7"
          step="0.01"
          class="w-full rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-3 py-2 dark:text-gray-100"
          bind:value={lowConfidenceThreshold}
        />
      </div>
      <div>
        <label class="block text-sm font-medium dark:text-gray-200 mb-1">每视频最大样本数</label>
        <input
          type="number"
          min="10"
          max="500"
          class="w-full rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-3 py-2 dark:text-gray-100"
          bind:value={maxSamplesPerVideo}
        />
      </div>
      <div>
        <label class="block text-sm font-medium dark:text-gray-200 mb-1">抽帧并发数（0=自动）</label>
        <input
          type="number"
          min="0"
          max="8"
          class="w-full rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-3 py-2 dark:text-gray-100"
          bind:value={extractWorkers}
        />
      </div>
      <div>
        <label class="block text-sm font-medium dark:text-gray-200 mb-1">导出检测置信阈值</label>
        <input
          type="number"
          min="0.01"
          max="0.95"
          step="0.01"
          class="w-full rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-3 py-2 dark:text-gray-100"
          bind:value={detectionConfThreshold}
        />
      </div>
      <div>
        <label class="block text-sm font-medium dark:text-gray-200 mb-1">导出检测 NMS IoU</label>
        <input
          type="number"
          min="0.1"
          max="0.9"
          step="0.01"
          class="w-full rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-3 py-2 dark:text-gray-100"
          bind:value={detectionIouThreshold}
        />
      </div>
    </div>

    <div class="mt-3">
      <label class="inline-flex items-center gap-2 text-sm dark:text-gray-200">
        <input type="checkbox" bind:checked={skipExistingVideos} />
        <span>跳过已构建完成的视频/样本（复用 staging/manifest）</span>
      </label>
    </div>

    <div class="mt-4 flex flex-wrap items-center gap-3">
      <button
        class="bg-blue-500 hover:bg-blue-700 disabled:bg-gray-400 text-white font-bold py-2 px-4 rounded"
        on:click={buildDataset}
        disabled={datasetRunning}
      >
        {#if datasetRunning}构建中...{:else}开始构建训练集{/if}
      </button>
      <button
        class="bg-emerald-500 hover:bg-emerald-700 disabled:bg-gray-400 text-white font-bold py-2 px-4 rounded"
        on:click={exportDataset}
        disabled={exporting || !datasetResult}
      >
        {#if exporting}导出中...{:else}一键导出 YOLO 数据集{/if}
      </button>
      {#if exportResult}
        <span class="text-xs text-emerald-700 dark:text-emerald-300 break-all">导出目录: {exportResult.dataset_dir}</span>
      {/if}
    </div>

    <div class="mt-6 grid grid-cols-1 lg:grid-cols-3 gap-3">
      <div>
        <label class="block text-sm font-medium dark:text-gray-200 mb-1">训练数据目录</label>
        <input class="w-full rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-3 py-2 dark:text-gray-100" bind:value={trainingDatasetDir} />
      </div>
      <div>
        <label class="block text-sm font-medium dark:text-gray-200 mb-1">训练输出目录</label>
        <input class="w-full rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-3 py-2 dark:text-gray-100" bind:value={trainingOutputDir} />
      </div>
      <div>
        <label class="block text-sm font-medium dark:text-gray-200 mb-1">替换模型路径</label>
        <input class="w-full rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-3 py-2 dark:text-gray-100" bind:value={targetModelPath} />
      </div>
      <div>
        <label class="block text-sm font-medium dark:text-gray-200 mb-1">Epochs</label>
        <input type="number" min="1" max="300" class="w-full rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-3 py-2 dark:text-gray-100" bind:value={trainEpochs} />
      </div>
      <div>
        <label class="block text-sm font-medium dark:text-gray-200 mb-1">Imgsz</label>
        <input type="number" min="320" max="1280" class="w-full rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-3 py-2 dark:text-gray-100" bind:value={trainImgsz} />
      </div>
      <div>
        <label class="block text-sm font-medium dark:text-gray-200 mb-1">Python 可执行文件</label>
        <input class="w-full rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-3 py-2 dark:text-gray-100" bind:value={pythonBin} placeholder="python 或 python3" />
      </div>
    </div>

    <div class="mt-3">
      <button
        class="bg-indigo-600 hover:bg-indigo-700 disabled:bg-gray-400 text-white font-bold py-2 px-4 rounded"
        on:click={trainAndUpdateModel}
        disabled={trainRunning}
      >
        {#if trainRunning}训练并更新中...{:else}训练并自动替换模型{/if}
      </button>
    </div>
  {/if}

  {#if mode === "recognition" && error}
    <div class="mt-4 p-3 rounded border border-red-300 bg-red-50 text-red-700 dark:bg-red-900/20 dark:border-red-800 dark:text-red-300">
      {error}
    </div>
  {/if}

  {#if mode === "dataset" && datasetError}
    <div class="mt-4 p-3 rounded border border-red-300 bg-red-50 text-red-700 dark:bg-red-900/20 dark:border-red-800 dark:text-red-300">
      {datasetError}
    </div>
  {/if}

  {#if mode === "dataset" && trainError}
    <div class="mt-4 p-3 rounded border border-red-300 bg-red-50 text-red-700 dark:bg-red-900/20 dark:border-red-800 dark:text-red-300">
      {trainError}
    </div>
  {/if}

  {#if mode === "recognition" && result}
    <div class="mt-6 p-4 rounded border dark:border-gray-700 bg-gray-50 dark:bg-gray-800">
      <div class="dark:text-gray-100 font-semibold">批处理结果</div>
      <div class="mt-2 text-sm dark:text-gray-300">
        已处理: {result.processed}，已跳过: {result.skipped}，模板目录: {result.templates_dir}
      </div>
    </div>

    <div class="mt-4 space-y-3">
      {#each result.results as item}
        <div class="p-3 rounded border dark:border-gray-700 bg-white dark:bg-gray-900">
          <div class="font-medium dark:text-gray-100">{item.file_name}</div>
          <div class="mt-1 text-sm dark:text-gray-300">
            状态: {item.status}，开始: {item.start_time_sec ?? "-"}s，结束: {item.end_time_sec ?? "-"}s，胜利: {item.victory ? "是" : "否"}
          </div>
          {#if item.ocr_preview}
            <div class="mt-2 text-xs whitespace-pre-wrap dark:text-gray-400">OCR: {item.ocr_preview}</div>
          {/if}
          {#if item.error}
            <div class="mt-2 text-xs text-red-600 dark:text-red-400">错误: {item.error}</div>
          {/if}
        </div>
      {/each}
    </div>
  {/if}

  {#if mode === "dataset" && datasetResult}
    <div class="mt-6 p-4 rounded border dark:border-gray-700 bg-gray-50 dark:bg-gray-800">
      <div class="dark:text-gray-100 font-semibold">训练集构建结果</div>
      <div class="mt-2 text-sm dark:text-gray-300">
        视频数: {datasetResult.total_videos}，样本总数: {datasetResult.total_samples}，正样本: {datasetResult.positives}，负样本: {datasetResult.negatives}，阶段模型: {datasetResult.phase_model_mode}，抽帧并发: {datasetResult.frame_extract_workers}
      </div>
    </div>

    <div class="mt-4 grid grid-cols-1 lg:grid-cols-2 gap-3">
      <div class="rounded border dark:border-gray-700 bg-white dark:bg-gray-900 p-3">
        <div class="text-sm font-medium dark:text-gray-200">当前样本预览</div>
        {#if activeSample}
          <div class="mt-2 text-xs dark:text-gray-400 break-all">{activeSample.image_path}</div>
          {#if previewImageUrl}
            <img src={previewImageUrl} alt="sample preview" class="mt-2 w-full max-h-[460px] object-contain rounded border dark:border-gray-700 bg-black/20" />
          {:else if previewImageError}
            <div class="mt-2 text-xs text-rose-600 dark:text-rose-400">{previewImageError}</div>
          {:else}
            <div class="mt-2 text-xs dark:text-gray-400">正在加载预览...</div>
          {/if}
        {:else}
          <div class="mt-2 text-xs dark:text-gray-400">暂无可预览样本</div>
        {/if}
      </div>
      <div class="rounded border dark:border-gray-700 bg-white dark:bg-gray-900 p-3 text-xs dark:text-gray-300">
        <div>说明：</div>
        <div class="mt-1">1. 键盘上下/jk 切换样本时，左侧预览会自动切换。</div>
        <div class="mt-1">2. 打开按钮会调用系统外部打开（已兼容 /mnt/d 到 Windows 路径）。</div>
        <div class="mt-1">3. 标注完成后可直接点“训练并自动替换模型”，无需再次构建/导出。</div>
      </div>
    </div>

    {#if exportResult}
      <div class="mt-4 p-3 rounded border border-emerald-300 bg-emerald-50 text-emerald-700 dark:bg-emerald-900/20 dark:border-emerald-800 dark:text-emerald-300">
        导出完成: {exportResult.dataset_dir}，images: {exportResult.images_count}，labels: {exportResult.labels_count}，train: {exportResult.train_images_count}，val: {exportResult.val_images_count}，估计准确率: {(exportResult.estimated_accuracy * 100).toFixed(1)}%，框模式: {exportResult.bbox_model_mode}，检测框: {exportResult.detected_box_labels}，ROI回退: {exportResult.fallback_roi_labels}
      </div>
    {/if}

    {#if trainResult}
      <div class="mt-4 p-3 rounded border border-indigo-300 bg-indigo-50 text-indigo-700 dark:bg-indigo-900/20 dark:border-indigo-800 dark:text-indigo-300">
        训练完成并替换模型：{trainResult.replaced_model_path}
        <div class="mt-1 text-xs break-all">run: {trainResult.run_dir}</div>
        <div class="mt-1 text-xs break-all">report: {trainResult.report_file}</div>
      </div>
    {/if}

    <div class="mt-4 flex flex-wrap items-center gap-3 text-sm dark:text-gray-300">
      <label class="inline-flex items-center gap-2">
        <span>筛选:</span>
        <select
          class="rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-2 py-1"
          bind:value={sampleFilter}
        >
          <option value="pending">待校验</option>
          <option value="all">全部</option>
          <option value="correct">正确</option>
          <option value="wrong">错误</option>
        </select>
      </label>
      <label class="inline-flex items-center gap-2">
        <span>每页:</span>
        <select
          class="rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-2 py-1"
          bind:value={pageSize}
        >
          <option value={20}>20</option>
          <option value={50}>50</option>
          <option value={100}>100</option>
        </select>
      </label>
      <button
        class="bg-emerald-600 hover:bg-emerald-700 text-white text-sm py-1 px-3 rounded"
        on:click={() => bulkMarkCurrentPage("correct")}
        disabled={pagedSamples.length === 0}
      >
        本页全标正确
      </button>
      <button
        class="bg-rose-600 hover:bg-rose-700 text-white text-sm py-1 px-3 rounded"
        on:click={() => bulkMarkCurrentPage("wrong")}
        disabled={pagedSamples.length === 0}
      >
        本页全标错误
      </button>
      <label class="inline-flex items-center gap-2">
        <span>本页阶段:</span>
        <select
          class="rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-2 py-1"
          bind:value={bulkPhase}
        >
          {#each phaseOptions as phaseOpt}
            <option value={phaseOpt.value}>{phaseOpt.label}</option>
          {/each}
        </select>
      </label>
      <button
        class="bg-indigo-600 hover:bg-indigo-700 text-white text-sm py-1 px-3 rounded"
        on:click={() => bulkSetPhaseCurrentPage(bulkPhase)}
        disabled={pagedSamples.length === 0}
      >
        本页批量改阶段
      </button>
      <button
        class="bg-slate-600 hover:bg-slate-700 text-white text-sm py-1 px-3 rounded"
        on:click={() => (page = Math.max(1, page - 1))}
        disabled={page <= 1}
      >
        上一页
      </button>
      <span>第 {page} / {totalPages} 页</span>
      <button
        class="bg-slate-600 hover:bg-slate-700 text-white text-sm py-1 px-3 rounded"
        on:click={() => (page = Math.min(totalPages, page + 1))}
        disabled={page >= totalPages}
      >
        下一页
      </button>
      <span>快捷键: ↑/↓ 或 j/k 切换，a=正确，d=错误，1~6 标注阶段，o=打开样本图</span>
    </div>

    <div class="mt-4 space-y-3">
      {#each pagedSamples as sample, index}
        <div class="p-3 rounded border dark:border-gray-700 bg-white dark:bg-gray-900" class:ring-2={index + start === activeSampleIndex} class:ring-blue-500={index + start === activeSampleIndex}>
          <div class="flex items-center justify-between gap-3">
            <div>
              <div class="font-medium dark:text-gray-100 break-all">{sample.image_path}</div>
              <div class="mt-1 text-xs dark:text-gray-400">
                阶段: {sample.phase}，置信度: {sample.confidence.toFixed(3)}，状态: {sample.label_status}，时间点: {sample.timestamp_sec ?? 0}s，类别: {sample.class_id ?? 0}
              </div>
            </div>
            <div class="flex items-center gap-2">
              <select
                class="rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-2 py-1 text-xs dark:text-gray-100"
                value={normalizePhase(sample.phase)}
                on:change={(ev) => onSamplePhaseChange(sample.id, ev)}
              >
                {#each phaseOptions as phaseOpt}
                  <option value={phaseOpt.value}>{phaseOpt.label}</option>
                {/each}
              </select>
              <button
                class="bg-slate-500 hover:bg-slate-700 text-white text-sm py-1 px-3 rounded"
                on:click={() => open(normalizeLocalPath(sample.image_path))}
              >
                打开
              </button>
              <button
                class="bg-emerald-500 hover:bg-emerald-700 text-white text-sm py-1 px-3 rounded"
                on:click={() => markSampleStatus(sample.id, "correct")}
              >
                正确
              </button>
              <button
                class="bg-rose-500 hover:bg-rose-700 text-white text-sm py-1 px-3 rounded"
                on:click={() => markSampleStatus(sample.id, "wrong")}
              >
                错误
              </button>
            </div>
          </div>
        </div>
      {/each}
    </div>
  {/if}
</div>
