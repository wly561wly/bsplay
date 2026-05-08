# Tauri Invoker

## 概述

`invoker.ts` 是对 Tauri 命令调用的封装层，位于 `src/lib/invoker.ts`。它提供了类型安全的前后端通信接口，简化了 Tauri 命令的调用。

## 基本用法

### 导入

```typescript
import { invoke } from '@tauri-apps/api/core';
// 或使用封装的 invoker
import { invokeCommand } from '$lib/invoker';
```

### 调用 Tauri 命令

```typescript
// 基本调用
const result = await invoke('command_name', { arg1: value1, arg2: value2 });

// 使用类型安全的封装
const result = await invokeCommand<ReturnType>('command_name', {
  arg1: value1,
  arg2: value2
});
```

## Tauri 命令系统

### 后端命令定义

在 Rust 后端，使用 `#[tauri::command]` 宏定义命令：

```rust
#[tauri::command]
async fn get_rooms(state: State<'_, AppState>) -> Result<Vec<Room>, String> {
    // 实现逻辑
    Ok(rooms)
}

// 在 main.rs 中注册
fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            get_rooms,
            // ... 其他命令
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

### 前端调用

```typescript
import { invoke } from '@tauri-apps/api/core';

interface Room {
  id: string;
  name: string;
  platform: string;
  url: string;
}

async function loadRooms(): Promise<Room[]> {
  try {
    const rooms = await invoke<Room[]>('get_rooms');
    return rooms;
  } catch (error) {
    console.error('Failed to load rooms:', error);
    throw error;
  }
}
```

## 常用命令封装

### 识别与训练集命令

```typescript
// 批量识别（输出识别 JSON）
export async function recognizeAndMarkVideos(args: {
  sourceDir?: string;
  outputDir?: string;
  frameIntervalSec?: number;
  enableOcr?: boolean;
  dumpFrames?: boolean;
}) {
  return await invoke('recognize_and_mark_videos', args);
}

// 构建训练集样本（staging/manifest.json + 抽帧样本）
export async function buildRecognitionDataset(args: {
  sourceDir: string;
  outputDir: string;
  modelPath?: string;
  highConfidenceThreshold?: number;
  lowConfidenceThreshold?: number;
  maxSamplesPerVideo?: number;
  extractWorkers?: number;      // 0=自动，根据 CPU/负载保守并发
  skipExistingVideos?: boolean; // 默认 true，复用已有 staging/manifest
}) {
  return await invoke('build_recognition_dataset', args);
}

// build_recognition_dataset.phase_model_mode 可能值：
// - heuristic
// - onnx-active
// - onnx-failed-fallback-heuristic
// build_recognition_dataset.frame_extract_workers: 实际使用的抽帧并发数

// 更新样本校验状态（correct/wrong/pending）
export async function updateDatasetSampleLabel(args: {
  outputDir: string;
  sampleId: string;
  status: 'correct' | 'wrong' | 'pending';
}) {
  return await invoke('update_dataset_sample_label', args);
}

// 批量更新样本校验状态
export async function updateDatasetSampleLabelsBulk(args: {
  outputDir: string;
  sampleIds: string[];
  status: 'correct' | 'wrong' | 'pending';
}) {
  return await invoke('update_dataset_sample_labels_bulk', args);
}

// 更新单个样本阶段（6类）
export async function updateDatasetSamplePhase(args: {
  outputDir: string;
  sampleId: string;
  phase: 'banpick' | 'loading' | 'gaming' | 'victory_or_defeat' | 'ending' | 'other';
}) {
  return await invoke('update_dataset_sample_phase', args);
}

// 批量更新样本阶段（6类）
export async function updateDatasetSamplePhasesBulk(args: {
  outputDir: string;
  sampleIds: string[];
  phase: 'banpick' | 'loading' | 'gaming' | 'victory_or_defeat' | 'ending' | 'other';
}) {
  return await invoke('update_dataset_sample_phases_bulk', args);
}

// 导出 YOLO 数据集目录（images/train + images/val + labels）
export async function exportRecognitionDataset(args: {
  outputDir: string;
  detectionModelPath?: string;
  detectionConfThreshold?: number;
  detectionIouThreshold?: number;
}) {
  return await invoke('export_recognition_dataset', args);
}

// export_recognition_dataset 返回新增字段：
// - train_images_count / val_images_count
// - estimated_accuracy（基于人工校验 correct/wrong 的估计值）
// - bbox_model_mode（onnx-active / roi-fallback / onnx-failed-fallback-roi）
// - detected_box_labels（使用检测框写标注的数量）
// - fallback_roi_labels（回退ROI写标注的数量）
// - dataset.yaml 的类别固定为 6 类：banpick/loading/gaming/victory_or_defeat/ending/other

// 后台启动训练任务（可暂停，支持继续训练或重新训练）
export async function startRecognitionTrainingTask(args: {
  datasetDir: string;        // 包含 dataset.yaml
  outputDir: string;         // 训练输出目录
  targetModelPath: string;   // 目标 onnx 文件路径
  epochs?: number;
  imgsz?: number;
  pythonBin?: string;        // 默认 python
  resumeTraining?: boolean;  // true=继续训练（基于 last.pt）
  resetRun?: boolean;        // true=重新训练（清空旧 run）
}) {
  return await invoke('start_recognition_training_task', args);
}

// 查询训练状态（轮询）
export async function getRecognitionTrainingStatus() {
  return await invoke('get_recognition_training_status');
}

// 暂停训练（终止当前训练进程，保留已有 checkpoint）
export async function stopRecognitionTraining() {
  return await invoke('stop_recognition_training');
}
```

### 录制相关命令

```typescript
// 开始录制
export async function startRecording(roomId: string): Promise<string> {
  return await invoke<string>('start_recording', { roomId });
}

// 停止录制
export async function stopRecording(recordingId: string): Promise<void> {
  await invoke('stop_recording', { recordingId });
}

// 获取录制状态
export async function getRecordingStatus(recordingId: string) {
  return await invoke('get_recording_status', { recordingId });
}
```

### 直播间管理命令

```typescript
// 添加直播间
export async function addRoom(platform: string, url: string) {
  return await invoke('add_room', { platform, url });
}

// 删除直播间
export async function deleteRoom(roomId: string) {
  await invoke('delete_room', { roomId });
}

// 更新直播间配置
export async function updateRoomConfig(roomId: string, config: RoomConfig) {
  await invoke('update_room_config', { roomId, config });
}
```

### 切片相关命令

```typescript
// 生成切片
export async function createClip(
  recordingId: string,
  startTime: number,
  endTime: number
) {
  return await invoke('create_clip', {
    recordingId,
    startTime,
    endTime
  });
}

// 获取切片列表
export async function getClips(recordingId: string) {
  return await invoke('get_clips', { recordingId });
}
```

## 错误处理

### 后端错误返回

Rust 命令应该返回 `Result<T, String>`：

```rust
#[tauri::command]
async fn risky_operation() -> Result<String, String> {
    match perform_operation() {
        Ok(result) => Ok(result),
        Err(e) => Err(format!("Operation failed: {}", e))
    }
}
```

### 前端错误处理

```typescript
async function safeInvoke<T>(command: string, args?: any): Promise<T | null> {
  try {
    return await invoke<T>(command, args);
  } catch (error) {
    console.error(`Command ${command} failed:`, error);
    // 可以在这里添加用户通知
    showNotification('error', `操作失败: ${error}`);
    return null;
  }
}
```

## 事件系统

除了命令调用，Tauri 还支持事件系统用于后端主动推送数据。

### 后端发送事件

```rust
use tauri::Manager;

#[tauri::command]
async fn start_monitoring(app: tauri::AppHandle) {
    tokio::spawn(async move {
        loop {
            // 监控逻辑
            let status = get_status();

            // 发送事件到前端
            app.emit_all("status-update", status).unwrap();

            tokio::time::sleep(Duration::from_secs(1)).await;
        }
    });
}
```

### 前端监听事件

```typescript
import { listen } from '@tauri-apps/api/event';

// 监听事件
const unlisten = await listen('status-update', (event) => {
  console.log('Status update:', event.payload);
  // 更新 UI 或 store
});

// 组件销毁时取消监听
onDestroy(() => {
  unlisten();
});
```

## 类型定义

### 共享类型定义

在 `src/lib/interface.ts` 中定义前后端共享的类型：

```typescript
export interface Room {
  id: string;
  name: string;
  platform: 'bilibili' | 'douyin' | 'huya' | 'kuaishou' | 'tiktok';
  url: string;
  status: 'online' | 'offline';
  config: RoomConfig;
}

export interface RoomConfig {
  autoRecord: boolean;
  quality: 'high' | 'medium' | 'low';
  savePath: string;
}

export interface Recording {
  id: string;
  roomId: string;
  startTime: number;
  endTime?: number;
  status: 'recording' | 'stopped' | 'error';
  filePath: string;
}

export interface Clip {
  id: string;
  recordingId: string;
  startTime: number;
  endTime: number;
  title: string;
  filePath: string;
}
```

### 命令参数类型

```typescript
// 为每个命令定义参数类型
export interface StartRecordingParams {
  roomId: string;
}

## 识别和标记命令

前端“识别和标记”页面通过 `invoke` 调用后端命令 `recognize_and_mark_videos`。

```typescript
type RecognitionBatchResult = {
  processed: number;
  skipped: number;
  source_dir: string;
  output_dir: string;
  templates_dir: string;
  interval_sec: number;
  results: Array<{
    file_name: string;
    status: string;
    start_time_sec: number | null;
    end_time_sec: number | null;
    victory: boolean;
    ocr_preview: string | null;
    output_file: string | null;
    result_file: string | null;
    error: string | null;
  }>;
};

const result = await invoke<RecognitionBatchResult>("recognize_and_mark_videos", {
  sourceDir: "/path/to/videos",
  outputDir: "/path/to/recognition_completed",
  frameIntervalSec: 1,
  enableOcr: true,
});
```

该命令的第一阶段目标是“低存储占用的模板匹配流程”：仅复用单帧临时图片，不落地整段抽帧结果。

export interface CreateClipParams {
  recordingId: string;
  startTime: number;
  endTime: number;
  title?: string;
}

// 使用类型安全的调用
export async function startRecording(params: StartRecordingParams) {
  return await invoke<string>('start_recording', params);
}
```

## 最佳实践

1. **类型安全**: 始终为命令调用和返回值定义类型
2. **错误处理**: 使用 try-catch 包装所有 invoke 调用
3. **加载状态**: 在调用期间显示加载指示器
4. **超时处理**: 对长时间运行的命令设置超时
5. **重试机制**: 对网络相关的命令实现重试逻辑

## 示例：完整的命令封装

```typescript
import { invoke } from '@tauri-apps/api/core';
import { writable } from 'svelte/store';

// 加载状态 store
export const isLoading = writable(false);

// 通用命令调用封装
async function safeInvoke<T>(
  command: string,
  args?: any,
  options?: {
    showLoading?: boolean;
    timeout?: number;
  }
): Promise<T> {
  const { showLoading = true, timeout = 30000 } = options || {};

  if (showLoading) {
    isLoading.set(true);
  }

  try {
    // 创建超时 Promise
    const timeoutPromise = new Promise((_, reject) =>
      setTimeout(() => reject(new Error('Command timeout')), timeout)
    );

    // 执行命令
    const result = await Promise.race([
      invoke<T>(command, args),
      timeoutPromise
    ]);

    return result as T;
  } catch (error) {
    console.error(`Command ${command} failed:`, error);
    throw error;
  } finally {
    if (showLoading) {
      isLoading.set(false);
    }
  }
}

// 使用封装的命令
export async function startRecording(roomId: string) {
  return await safeInvoke<string>('start_recording', { roomId }, {
    showLoading: true,
    timeout: 10000
  });
}
```

## 调试

### 启用 Tauri 调试日志

在开发模式下，Tauri 会输出详细的日志：

```typescript
// 在 tauri.conf.json 中配置
{
  "build": {
    "devPath": "http://localhost:5173",
    "beforeDevCommand": "yarn dev"
  },
  "tauri": {
    "bundle": {
      "active": true
    }
  }
}
```

### 前端调试

```typescript
// 添加调试日志
if (import.meta.env.DEV) {
  const originalInvoke = invoke;
  window.invoke = async (command: string, args?: any) => {
    console.log(`[Invoke] ${command}`, args);
    const result = await originalInvoke(command, args);
    console.log(`[Result] ${command}`, result);
    return result;
  };
}
```