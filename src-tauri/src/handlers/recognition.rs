use crate::ffmpeg;
use crate::state::State;
use crate::state_type;
use chrono::Utc;
use image::imageops::FilterType;
use image::{DynamicImage, GrayImage};
use imageproc::template_matching::{find_extremes, match_template, MatchTemplateMethod};
use ndarray::Array4;
use ort::{session::Session, value::Tensor};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use tokio::sync::Semaphore;

#[cfg(feature = "gui")]
use tauri::State as TauriState;

#[cfg(target_os = "windows")]
const CREATE_NO_WINDOW: u32 = 0x08000000;
#[cfg(target_os = "windows")]
#[allow(unused_imports)]
use std::os::windows::process::CommandExt;

const DEFAULT_INTERVAL_SEC: u32 = 1;
const MAX_INTERVAL_SEC: u32 = 10;
const RESIZE_WIDTH: u32 = 960;
const HASH_SIDE: u32 = 8;
const FEATURE_PREFILTER_THRESHOLD: f32 = 0.92;
const START_THRESHOLD: f32 = 0.83;
const END_THRESHOLD: f32 = 0.84;
const VICTORY_THRESHOLD: f32 = 0.88;
const DETECTION_INPUT_SIZE: usize = 640;
const DETECTION_CLASS_COUNT: usize = 6;

const PHASE_LABELS: [&str; 6] = [
    "banpick",
    "loading",
    "gaming",
    "victory_or_defeat",
    "ending",
    "other",
];

#[derive(Debug, Clone)]
struct TemplateFeature {
    category: String,
    name: String,
    gray: GrayImage,
    signature: [f32; 64],
}

#[derive(Debug, Clone)]
struct TemplateLibrary {
    templates: Vec<TemplateFeature>,
    category_centroids: HashMap<String, [f32; 64]>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RecognitionFileResult {
    pub file_name: String,
    pub status: String,
    pub start_time_sec: Option<u64>,
    pub end_time_sec: Option<u64>,
    pub victory: bool,
    pub ocr_preview: Option<String>,
    pub output_file: Option<String>,
    pub result_file: Option<String>,
    pub ocr_fields: HashMap<String, String>,
    pub template_scores: HashMap<String, f32>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RecognitionBatchResult {
    pub processed: usize,
    pub skipped: usize,
    pub source_dir: String,
    pub output_dir: String,
    pub templates_dir: String,
    pub interval_sec: u32,
    pub results: Vec<RecognitionFileResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildDatasetSample {
    pub id: String,
    pub image_path: String,
    #[serde(default)]
    pub video_path: String,
    #[serde(default)]
    pub timestamp_sec: u64,
    pub phase: String,
    pub confidence: f32,
    #[serde(default)]
    pub class_id: u32,
    #[serde(default)]
    pub roi: [f32; 4],
    pub label_status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildDatasetResult {
    pub source_dir: String,
    pub output_dir: String,
    pub phase_model_mode: String,
    pub frame_extract_workers: usize,
    pub total_videos: usize,
    pub total_samples: usize,
    pub positives: usize,
    pub negatives: usize,
    pub samples: Vec<BuildDatasetSample>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DatasetManifest {
    source_dir: String,
    output_dir: String,
    samples: Vec<BuildDatasetSample>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ExportDatasetResult {
    pub dataset_dir: String,
    pub images_count: usize,
    pub labels_count: usize,
    pub train_images_count: usize,
    pub val_images_count: usize,
    pub estimated_accuracy: f32,
    pub bbox_model_mode: String,
    pub detected_box_labels: usize,
    pub fallback_roi_labels: usize,
    pub exported_at: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct TrainModelResult {
    pub success: bool,
    pub dataset_dir: String,
    pub run_dir: String,
    pub exported_onnx_path: String,
    pub replaced_model_path: String,
    pub report_file: String,
    pub train_stdout_tail: String,
    pub train_stderr_tail: String,
}

#[derive(Debug, Clone)]
struct PhaseInfo {
    class_id: u32,
    roi: [f32; 4],
}

#[derive(Debug, Clone, Copy)]
struct DetectionBox {
    class_id: u32,
    confidence: f32,
    roi: [f32; 4],
}

fn normalize_category(raw: &str) -> String {
    let v = raw.to_ascii_lowercase();
    match v.as_str() {
        "ban-pick" => "banpick".to_string(),
        _ => v,
    }
}

fn normalize_phase_name(raw: &str) -> String {
    match raw.trim().to_ascii_lowercase().as_str() {
        "ban-pick" => "banpick".to_string(),
        "game" => "gaming".to_string(),
        "gaming" => "gaming".to_string(),
        "vectory" | "victory" | "defeat" => "victory_or_defeat".to_string(),
        "unknown" | "transition" => "other".to_string(),
        v => v.to_string(),
    }
}

fn is_supported_phase_label(phase: &str) -> bool {
    PHASE_LABELS.contains(&phase)
}

fn build_uniform_sample_points(duration_sec: u64, max_samples: usize) -> Vec<u64> {
    if max_samples == 0 {
        return Vec::new();
    }

    if duration_sec == 0 {
        return vec![0];
    }

    let target = max_samples.min((duration_sec + 1) as usize);
    if target <= 1 {
        return vec![duration_sec / 2];
    }

    let mut points = Vec::with_capacity(target);
    for i in 0..target {
        let ratio = i as f64 / (target.saturating_sub(1)) as f64;
        let sec = (ratio * duration_sec as f64).round() as u64;
        if points.last().copied() != Some(sec) {
            points.push(sec);
        }
    }

    points
}

fn recommended_extract_workers(max_samples_per_video: usize) -> usize {
    let cpu = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);

    // 保守取 CPU 一半，避免 ffmpeg 子进程过多导致系统卡顿。
    let cpu_bound = (cpu / 2).clamp(1, 6);
    let memory_bound = if max_samples_per_video >= 200 {
        2
    } else if max_samples_per_video >= 120 {
        3
    } else {
        4
    };

    cpu_bound.min(memory_bound)
}

fn build_signature(gray: &GrayImage) -> [f32; 64] {
    let small = image::imageops::resize(gray, HASH_SIDE, HASH_SIDE, FilterType::Triangle);
    let mut out = [0.0_f32; 64];
    let mut sum = 0.0_f32;

    for (i, px) in small.pixels().enumerate() {
        let v = px.0[0] as f32 / 255.0;
        out[i] = v;
        sum += v;
    }

    let mean = sum / out.len() as f32;
    let mut norm = 0.0_f32;
    for v in &mut out {
        *v -= mean;
        norm += *v * *v;
    }

    let denom = norm.sqrt();
    if denom > 0.0 {
        for v in &mut out {
            *v /= denom;
        }
    }

    out
}

fn cosine_similarity(a: &[f32; 64], b: &[f32; 64]) -> f32 {
    let mut dot = 0.0_f32;
    for i in 0..64 {
        dot += a[i] * b[i];
    }
    dot
}

fn is_supported_video(path: &Path) -> bool {
    let Some(ext) = path.extension().and_then(OsStr::to_str) else {
        return false;
    };

    matches!(
        ext.to_ascii_lowercase().as_str(),
        "mp4" | "mkv" | "flv" | "mov" | "avi" | "webm"
    )
}

fn normalize_gray(img: DynamicImage) -> GrayImage {
    let gray = img.to_luma8();
    if gray.width() <= RESIZE_WIDTH {
        return gray;
    }

    let new_h = ((gray.height() as f32 / gray.width() as f32) * RESIZE_WIDTH as f32)
        .max(1.0)
        .round() as u32;
    image::imageops::resize(&gray, RESIZE_WIDTH, new_h, FilterType::Triangle)
}

fn find_templates_root() -> Option<PathBuf> {
    let candidates = [
        PathBuf::from("src-tauri/templates"),
        PathBuf::from("templates"),
    ];

    candidates.into_iter().find(|p| p.exists() && p.is_dir())
}

fn load_templates(root: &Path) -> Result<TemplateLibrary, String> {
    let mut list: Vec<TemplateFeature> = Vec::new();

    let categories = std::fs::read_dir(root)
        .map_err(|e| format!("读取模板目录失败: {} - {}", root.display(), e))?;

    for cat_entry in categories {
        let cat_entry = cat_entry.map_err(|e| format!("读取模板分类失败: {e}"))?;
        let cat_path = cat_entry.path();
        if !cat_path.is_dir() {
            continue;
        }

        let category = normalize_category(
            cat_path
            .file_name()
            .and_then(OsStr::to_str)
            .unwrap_or("unknown"),
        );

        let files = std::fs::read_dir(&cat_path)
            .map_err(|e| format!("读取模板分类内容失败: {} - {}", cat_path.display(), e))?;

        for file in files {
            let file = file.map_err(|e| format!("读取模板文件失败: {e}"))?;
            let path = file.path();
            if !path.is_file() {
                continue;
            }

            let ext = path
                .extension()
                .and_then(OsStr::to_str)
                .unwrap_or("")
                .to_ascii_lowercase();
            if !matches!(ext.as_str(), "png" | "jpg" | "jpeg" | "bmp" | "webp") {
                continue;
            }

            let img = image::open(&path)
                .map_err(|e| format!("读取模板图片失败: {} - {}", path.display(), e))?;
            let gray = normalize_gray(img);
            if gray.width() < 8 || gray.height() < 8 {
                continue;
            }
            let signature = build_signature(&gray);

            list.push(TemplateFeature {
                category: category.clone(),
                name: path
                    .file_name()
                    .and_then(OsStr::to_str)
                    .unwrap_or("template")
                    .to_string(),
                gray,
                signature,
            });
        }
    }

    if list.is_empty() {
        return Err(format!(
            "模板目录为空或没有可用图片: {}",
            root.display()
        ));
    }

    let mut grouped: HashMap<String, Vec<[f32; 64]>> = HashMap::new();
    for tpl in &list {
        grouped
            .entry(tpl.category.clone())
            .or_default()
            .push(tpl.signature);
    }

    let mut category_centroids: HashMap<String, [f32; 64]> = HashMap::new();
    for (cat, sigs) in grouped {
        let mut c = [0.0_f32; 64];
        for sig in &sigs {
            for i in 0..64 {
                c[i] += sig[i];
            }
        }
        let n = sigs.len() as f32;
        if n > 0.0 {
            for v in &mut c {
                *v /= n;
            }
        }
        let mut norm = 0.0_f32;
        for v in &c {
            norm += v * v;
        }
        let denom = norm.sqrt();
        if denom > 0.0 {
            for v in &mut c {
                *v /= denom;
            }
        }
        category_centroids.insert(cat, c);
    }

    Ok(TemplateLibrary {
        templates: list,
        category_centroids,
    })
}

async fn extract_frame_to_png(video: &Path, second: u64, output_png: &Path) -> Result<(), String> {
    let mut cmd = tokio::process::Command::new(ffmpeg::ffmpeg_path());
    #[cfg(target_os = "windows")]
    cmd.creation_flags(CREATE_NO_WINDOW);

    let status = cmd
        .args(["-hide_banner", "-loglevel", "error"])
        .args(["-ss", &second.to_string()])
        .args(["-i", &video.to_string_lossy()])
        .args(["-frames:v", "1"])
        .args(["-vf", "scale=960:-1"])
        .args(["-y", &output_png.to_string_lossy()])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .await
        .map_err(|e| format!("抽帧失败: {} - {}", video.display(), e))?;

    if !status.success() {
        return Err(format!(
            "抽帧命令执行失败: {} @ {}s",
            video.display(), second
        ));
    }

    Ok(())
}

fn match_score(frame: &GrayImage, tpl: &GrayImage) -> f32 {
    if frame.width() < tpl.width() || frame.height() < tpl.height() {
        return 0.0;
    }

    let response = match_template(frame, tpl, MatchTemplateMethod::CrossCorrelationNormalized);
    let extremes = find_extremes(&response);
    extremes.max_value
}

async fn ocr_from_image(image_path: &Path) -> Result<String, String> {
    let output = tokio::process::Command::new("tesseract")
        .arg(image_path)
        .arg("stdout")
        .args(["-l", "chi_sim+eng"])
        .args(["--psm", "6"])
        .output()
        .await
        .map_err(|e| format!("执行 tesseract 失败: {e}"))?;

    if !output.status.success() {
        return Err(format!(
            "tesseract 返回错误: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    let text = String::from_utf8_lossy(&output.stdout).trim().to_string();
    Ok(text)
}

async fn ocr_from_image_with_config(
    image_path: &Path,
    psm: u8,
    lang: &str,
    whitelist: Option<&str>,
) -> Result<String, String> {
    let mut cmd = tokio::process::Command::new("tesseract");
    cmd.arg(image_path)
        .arg("stdout")
        .args(["-l", lang])
        .args(["--psm", &psm.to_string()]);

    if let Some(wl) = whitelist {
        cmd.args(["-c", &format!("tessedit_char_whitelist={wl}")]);
    }

    let output = cmd
        .output()
        .await
        .map_err(|e| format!("执行 tesseract 失败: {e}"))?;

    if !output.status.success() {
        return Err(format!(
            "tesseract 返回错误: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn crop_roi_by_normalized(src: &DynamicImage, roi: [f32; 4]) -> DynamicImage {
    let w = src.width() as f32;
    let h = src.height() as f32;

    let cx = roi[0].clamp(0.0, 1.0) * w;
    let cy = roi[1].clamp(0.0, 1.0) * h;
    let rw = roi[2].clamp(0.0, 1.0) * w;
    let rh = roi[3].clamp(0.0, 1.0) * h;

    let x = (cx - rw / 2.0).max(0.0) as u32;
    let y = (cy - rh / 2.0).max(0.0) as u32;
    let cw = rw.min(w - x as f32).max(1.0) as u32;
    let ch = rh.min(h - y as f32).max(1.0) as u32;

    src.crop_imm(x, y, cw, ch)
}

async fn extract_structured_ocr_fields(frame_path: &Path) -> Result<HashMap<String, String>, String> {
    let mut fields = HashMap::new();
    let src = image::open(frame_path)
        .map_err(|e| format!("读取 OCR 帧失败: {} - {}", frame_path.display(), e))?;

    let rois: [(&str, [f32; 4], u8, &str, Option<&str>); 3] = [
        ("time", [0.50, 0.065, 0.20, 0.06], 7, "eng", Some("0123456789:")),
        ("score", [0.50, 0.11, 0.30, 0.07], 7, "eng", Some("0123456789:-")),
        ("player_id", [0.16, 0.90, 0.28, 0.08], 7, "chi_sim+eng", None),
    ];

    for (name, roi, psm, lang, whitelist) in rois {
        let crop = crop_roi_by_normalized(&src, roi);
        let temp_path = std::env::temp_dir().join(format!(
            "bsr_recognition_ocr_roi_{}_{}_{}.png",
            std::process::id(),
            name,
            Utc::now().timestamp_millis()
        ));
        crop
            .save(&temp_path)
            .map_err(|e| format!("保存 OCR ROI 失败: {}", e))?;

        let text = ocr_from_image_with_config(&temp_path, psm, lang, whitelist)
            .await
            .unwrap_or_default();
        let _ = std::fs::remove_file(&temp_path);

        let cleaned = text
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .trim()
            .to_string();
        if !cleaned.is_empty() {
            fields.insert(name.to_string(), cleaned);
        }
    }

    Ok(fields)
}

fn trim_preview(text: &str) -> String {
    let compact = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .take(6)
        .collect::<Vec<_>>()
        .join("\n");

    if compact.chars().count() > 240 {
        compact.chars().take(240).collect::<String>()
    } else {
        compact
    }
}

fn should_skip_file(name: &str) -> bool {
    name.starts_with("[completed]")
}

fn ensure_dir(path: &Path) -> Result<(), String> {
    if path.exists() {
        return Ok(());
    }
    std::fs::create_dir_all(path)
        .map_err(|e| format!("创建目录失败: {} - {}", path.display(), e))
}

fn should_skip_by_json(output_dir: &Path, file_name: &str) -> Option<PathBuf> {
    let completed_name = format!("[completed]{}", file_name);
    let json_path = output_dir.join(format!("{}.recognition.json", completed_name));
    if json_path.exists() {
        Some(json_path)
    } else {
        None
    }
}

fn infer_phase_by_time(sec: u64, duration_sec: u64) -> (&'static str, u64) {
    let victory_start = duration_sec.saturating_sub(180);
    let ending_start = duration_sec.saturating_sub(40);
    if sec <= 300 {
        if sec <= 120 {
            ("banpick", 2)
        } else {
            ("loading", 2)
        }
    } else if sec >= ending_start {
        ("ending", 1)
    } else if sec >= victory_start {
        ("victory_or_defeat", 1)
    } else {
        ("gaming", 8)
    }
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max_v = logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, |a, b| a.max(b));
    let exps: Vec<f32> = logits.iter().map(|v| (*v - max_v).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum <= 0.0 {
        return vec![0.0; logits.len()];
    }
    exps.into_iter().map(|v| v / sum).collect()
}

fn create_phase_classifier(model_path: &Path) -> Result<Session, String> {
    Session::builder()
        .map_err(|e| format!("创建 ONNX Session 失败: {e}"))?
        .with_intra_threads(1)
        .map_err(|e| format!("设置 ONNX 线程失败: {e}"))?
        .commit_from_file(model_path)
        .map_err(|e| format!("加载 ONNX 模型失败: {} - {}", model_path.display(), e))
}

fn classify_phase_with_onnx(session: &mut Session, img: &DynamicImage) -> Result<(String, f32), String> {
    // 约定输入为 224x224 RGB, NCHW
    let resized = img.resize_exact(224, 224, FilterType::Triangle).to_rgb8();
    let mut input = Array4::<f32>::zeros((1, 3, 224, 224));
    for (x, y, px) in resized.enumerate_pixels() {
        let xi = x as usize;
        let yi = y as usize;
        input[[0, 0, yi, xi]] = px[0] as f32 / 255.0;
        input[[0, 1, yi, xi]] = px[1] as f32 / 255.0;
        input[[0, 2, yi, xi]] = px[2] as f32 / 255.0;
    }

    let input_tensor = Tensor::from_array(input)
        .map_err(|e| format!("构建 ONNX 输入张量失败: {e}"))?;
    let outputs = session
        .run(ort::inputs![input_tensor])
        .map_err(|e| format!("执行 ONNX 推理失败: {e}"))?;
    let (shape, data) = outputs[0]
        .try_extract_tensor::<f32>()
        .map_err(|e| format!("解析 ONNX 输出失败: {e}"))?;
    let flat: Vec<f32> = data.to_vec();
    if flat.is_empty() {
        return Err("ONNX 输出为空".to_string());
    }

    let dims: Vec<usize> = shape
        .iter()
        .map(|&d| if d <= 0 { 0 } else { d as usize })
        .collect();

    let classes = if dims.len() >= 2 {
        *dims.last().unwrap_or(&flat.len())
    } else {
        flat.len()
    };
    if classes == 0 {
        return Err("ONNX 输出类别数为 0".to_string());
    }

    let logits: Vec<f32> = flat.into_iter().take(classes).collect();
    let probs = softmax(&logits);
    let (best_idx, best_prob) = probs
        .iter()
        .copied()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .ok_or_else(|| "ONNX 输出概率解析失败".to_string())?;

    let labels = [
        "banpick",
        "loading",
        "gaming",
        "victory_or_defeat",
        "ending",
        "other",
    ];
    let phase = normalize_phase_name(labels.get(best_idx).unwrap_or(&"other"));
    Ok((phase, best_prob))
}

fn prob_from_raw(v: f32) -> f32 {
    if (0.0..=1.0).contains(&v) {
        v
    } else {
        1.0 / (1.0 + (-v).exp())
    }
}

fn clamp_roi(roi: [f32; 4]) -> [f32; 4] {
    [
        roi[0].clamp(0.0, 1.0),
        roi[1].clamp(0.0, 1.0),
        roi[2].clamp(0.0, 1.0),
        roi[3].clamp(0.0, 1.0),
    ]
}

fn roi_iou(a: [f32; 4], b: [f32; 4]) -> f32 {
    let (ax1, ay1, ax2, ay2) = (a[0] - a[2] * 0.5, a[1] - a[3] * 0.5, a[0] + a[2] * 0.5, a[1] + a[3] * 0.5);
    let (bx1, by1, bx2, by2) = (b[0] - b[2] * 0.5, b[1] - b[3] * 0.5, b[0] + b[2] * 0.5, b[1] + b[3] * 0.5);

    let inter_x1 = ax1.max(bx1);
    let inter_y1 = ay1.max(by1);
    let inter_x2 = ax2.min(bx2);
    let inter_y2 = ay2.min(by2);

    let inter_w = (inter_x2 - inter_x1).max(0.0);
    let inter_h = (inter_y2 - inter_y1).max(0.0);
    let inter = inter_w * inter_h;
    if inter <= 0.0 {
        return 0.0;
    }

    let area_a = (ax2 - ax1).max(0.0) * (ay2 - ay1).max(0.0);
    let area_b = (bx2 - bx1).max(0.0) * (by2 - by1).max(0.0);
    let union = area_a + area_b - inter;
    if union <= 0.0 {
        0.0
    } else {
        inter / union
    }
}

fn nms_by_class(mut dets: Vec<DetectionBox>, iou_threshold: f32, max_det: usize) -> Vec<DetectionBox> {
    dets.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut kept: Vec<DetectionBox> = Vec::new();
    for d in dets {
        let mut suppressed = false;
        for k in &kept {
            if d.class_id == k.class_id && roi_iou(d.roi, k.roi) > iou_threshold {
                suppressed = true;
                break;
            }
        }
        if !suppressed {
            kept.push(d);
            if kept.len() >= max_det {
                break;
            }
        }
    }
    kept
}

fn create_detection_model(model_path: &Path) -> Result<Session, String> {
    create_phase_classifier(model_path)
}

fn detect_boxes_with_onnx(
    session: &mut Session,
    img: &DynamicImage,
    conf_threshold: f32,
    iou_threshold: f32,
) -> Result<Vec<DetectionBox>, String> {
    let resized = img
        .resize_exact(
            DETECTION_INPUT_SIZE as u32,
            DETECTION_INPUT_SIZE as u32,
            FilterType::Triangle,
        )
        .to_rgb8();
    let mut input = Array4::<f32>::zeros((1, 3, DETECTION_INPUT_SIZE, DETECTION_INPUT_SIZE));
    for (x, y, px) in resized.enumerate_pixels() {
        let xi = x as usize;
        let yi = y as usize;
        input[[0, 0, yi, xi]] = px[0] as f32 / 255.0;
        input[[0, 1, yi, xi]] = px[1] as f32 / 255.0;
        input[[0, 2, yi, xi]] = px[2] as f32 / 255.0;
    }

    let input_tensor = Tensor::from_array(input)
        .map_err(|e| format!("构建检测模型输入张量失败: {e}"))?;
    let outputs = session
        .run(ort::inputs![input_tensor])
        .map_err(|e| format!("执行检测模型推理失败: {e}"))?;
    let (shape, data) = outputs[0]
        .try_extract_tensor::<f32>()
        .map_err(|e| format!("解析检测模型输出失败: {e}"))?;

    let dims: Vec<usize> = shape
        .iter()
        .map(|&d| if d <= 0 { 0 } else { d as usize })
        .collect();
    if dims.len() != 3 {
        return Err(format!("检测模型输出维度不支持: {:?}", dims));
    }
    let flat: Vec<f32> = data.to_vec();
    if flat.is_empty() {
        return Ok(Vec::new());
    }

    let (attrs, count, attrs_first) = if dims[1] >= 6 && dims[2] > dims[1] {
        (dims[1], dims[2], true)
    } else if dims[2] >= 6 {
        (dims[2], dims[1], false)
    } else {
        return Err(format!("检测模型输出形状不支持: {:?}", dims));
    };

    if attrs * count > flat.len() {
        return Err("检测模型输出长度异常".to_string());
    }

    let get_val = |attr: usize, idx: usize| -> f32 {
        if attrs_first {
            flat[attr * count + idx]
        } else {
            flat[idx * attrs + attr]
        }
    };

    let mut dets = Vec::new();
    for i in 0..count {
        let cx_raw = get_val(0, i);
        let cy_raw = get_val(1, i);
        let w_raw = get_val(2, i);
        let h_raw = get_val(3, i);

        let (cls_start, obj_conf) = if attrs == DETECTION_CLASS_COUNT + 5 {
            (5, prob_from_raw(get_val(4, i)))
        } else {
            (4, 1.0)
        };
        if cls_start >= attrs {
            continue;
        }

        let mut best_cls = 0usize;
        let mut best_cls_prob = 0.0_f32;
        for cls_idx in cls_start..attrs {
            let p = prob_from_raw(get_val(cls_idx, i));
            if p > best_cls_prob {
                best_cls_prob = p;
                best_cls = cls_idx - cls_start;
            }
        }

        let score = obj_conf * best_cls_prob;
        if score < conf_threshold {
            continue;
        }

        let mut cx = cx_raw;
        let mut cy = cy_raw;
        let mut w = w_raw;
        let mut h = h_raw;
        if cx.abs() > 1.5 || cy.abs() > 1.5 || w.abs() > 1.5 || h.abs() > 1.5 {
            let size = DETECTION_INPUT_SIZE as f32;
            cx /= size;
            cy /= size;
            w /= size;
            h /= size;
        }
        if w <= 0.0 || h <= 0.0 {
            continue;
        }

        let roi = clamp_roi([cx, cy, w, h]);
        if roi[2] <= 0.0 || roi[3] <= 0.0 {
            continue;
        }

        dets.push(DetectionBox {
            class_id: best_cls as u32,
            confidence: score,
            roi,
        });
    }

    Ok(nms_by_class(dets, iou_threshold.clamp(0.1, 0.9), 100))
}

fn phase_info(phase: &str) -> PhaseInfo {
    let phase = normalize_phase_name(phase);
    match phase.as_str() {
        "banpick" => PhaseInfo {
            class_id: 0,
            roi: [0.50, 0.18, 0.74, 0.22],
        },
        "loading" => PhaseInfo {
            class_id: 1,
            roi: [0.50, 0.50, 0.90, 0.90],
        },
        "gaming" => PhaseInfo {
            class_id: 2,
            roi: [0.50, 0.08, 0.32, 0.12],
        },
        "victory_or_defeat" => PhaseInfo {
            class_id: 3,
            roi: [0.50, 0.50, 0.62, 0.24],
        },
        "ending" => PhaseInfo {
            class_id: 4,
            roi: [0.50, 0.50, 0.66, 0.28],
        },
        _ => PhaseInfo {
            class_id: 5,
            roi: [0.50, 0.50, 1.00, 1.00],
        },
    }
}

fn deterministic_val_split(sample_id: &str) -> bool {
    let digest = md5::compute(sample_id.as_bytes());
    // 大约 10% 进入 val
    digest[0] % 10 == 0
}

fn yolo_line(class_id: u32, roi: [f32; 4]) -> String {
    format!(
        "{} {:.6} {:.6} {:.6} {:.6}",
        class_id,
        roi[0].clamp(0.0, 1.0),
        roi[1].clamp(0.0, 1.0),
        roi[2].clamp(0.0, 1.0),
        roi[3].clamp(0.0, 1.0)
    )
}

fn write_dataset_yaml(dataset_dir: &Path) -> Result<(), String> {
    let yaml = [
        "path: .",
        "train: images/train",
        "val: images/val",
        "names:",
        "  0: banpick",
        "  1: loading",
        "  2: gaming",
        "  3: victory_or_defeat",
        "  4: ending",
        "  5: other",
        "",
    ]
    .join("\n");

    std::fs::write(dataset_dir.join("dataset.yaml"), yaml)
        .map_err(|e| format!("写入 dataset.yaml 失败: {}", e))
}

fn trim_tail(s: &str, max_chars: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= max_chars {
        return s.to_string();
    }
    chars[chars.len().saturating_sub(max_chars)..].iter().collect()
}

fn rebalance_negative_ratio(samples: &mut [BuildDatasetSample]) {
    let positives = samples
        .iter()
        .filter(|s| s.label_status == "correct")
        .count();
    if positives == 0 {
        return;
    }

    let target_neg = positives.saturating_mul(2);
    let mut wrong_indices: Vec<usize> = samples
        .iter()
        .enumerate()
        .filter_map(|(i, s)| if s.label_status == "wrong" { Some(i) } else { None })
        .collect();

    if wrong_indices.len() < target_neg {
        let mut pending_indices: Vec<usize> = samples
            .iter()
            .enumerate()
            .filter_map(|(i, s)| if s.label_status == "pending" { Some(i) } else { None })
            .collect();
        pending_indices.sort_by(|a, b| {
            samples[*a]
                .confidence
                .partial_cmp(&samples[*b].confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let need = target_neg.saturating_sub(wrong_indices.len());
        for idx in pending_indices.into_iter().take(need) {
            samples[idx].label_status = "wrong".to_string();
        }
        return;
    }

    if wrong_indices.len() > target_neg {
        wrong_indices.sort_by(|a, b| {
            samples[*b]
                .confidence
                .partial_cmp(&samples[*a].confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let extra = wrong_indices.len().saturating_sub(target_neg);
        for idx in wrong_indices.into_iter().take(extra) {
            samples[idx].label_status = "pending".to_string();
        }
    }
}

fn ensure_parent_dir(path: &Path) -> Result<(), String> {
    let Some(parent) = path.parent() else {
        return Ok(());
    };
    ensure_dir(parent)
}

fn load_manifest(manifest_path: &Path) -> Result<DatasetManifest, String> {
    let content = std::fs::read_to_string(manifest_path)
        .map_err(|e| format!("读取样本清单失败: {} - {}", manifest_path.display(), e))?;
    serde_json::from_str::<DatasetManifest>(&content)
        .map_err(|e| format!("解析样本清单失败: {} - {}", manifest_path.display(), e))
}

fn save_manifest(manifest_path: &Path, manifest: &DatasetManifest) -> Result<(), String> {
    ensure_parent_dir(manifest_path)?;
    let text = serde_json::to_string_pretty(manifest)
        .map_err(|e| format!("序列化样本清单失败: {e}"))?;
    std::fs::write(manifest_path, text)
        .map_err(|e| format!("写入样本清单失败: {} - {}", manifest_path.display(), e))
}

fn pick_candidate_categories(
    frame_signature: &[f32; 64],
    centroids: &HashMap<String, [f32; 64]>,
) -> Vec<String> {
    let mut scored = Vec::new();
    for (cat, c) in centroids {
        let s = cosine_similarity(frame_signature, c);
        scored.push((cat.clone(), s));
    }

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut out: Vec<String> = scored
        .iter()
        .filter(|(_, s)| *s >= FEATURE_PREFILTER_THRESHOLD)
        .map(|(cat, _)| cat.clone())
        .collect();

    if out.is_empty() {
        for (cat, _) in scored.into_iter().take(2) {
            out.push(cat);
        }
    }

    out
}

fn collect_videos(source_dir: &Path, output_dir: &Path) -> Result<Vec<PathBuf>, String> {
    let mut stack = vec![source_dir.to_path_buf()];
    let mut all = Vec::new();

    while let Some(dir) = stack.pop() {
        let iter = std::fs::read_dir(&dir)
            .map_err(|e| format!("读取目录失败: {} - {}", dir.display(), e))?;

        for entry in iter {
            let entry = entry.map_err(|e| format!("读取目录项失败: {e}"))?;
            let path = entry.path();
            if path.is_dir() {
                if path == output_dir {
                    continue;
                }
                stack.push(path);
                continue;
            }

            if path.is_file() && is_supported_video(&path) {
                all.push(path);
            }
        }
    }

    all.sort();
    Ok(all)
}

async fn recognize_single_video(
    video_path: &Path,
    library: &TemplateLibrary,
    interval_sec: u32,
    enable_ocr: bool,
    output_dir: &Path,
    dump_frames: bool,
) -> RecognitionFileResult {
    let file_name = video_path
        .file_name()
        .and_then(OsStr::to_str)
        .unwrap_or("unknown")
        .to_string();

    let mut result = RecognitionFileResult {
        file_name: file_name.clone(),
        status: "processed".to_string(),
        start_time_sec: None,
        end_time_sec: None,
        victory: false,
        ocr_preview: None,
        output_file: None,
        result_file: None,
        ocr_fields: HashMap::new(),
        template_scores: HashMap::new(),
        error: None,
    };

    let metadata = match ffmpeg::extract_video_metadata(video_path).await {
        Ok(v) => v,
        Err(e) => {
            result.status = "failed".to_string();
            result.error = Some(format!("读取视频元数据失败: {e}"));
            return result;
        }
    };

    let duration_sec = metadata.duration.max(0.0).ceil() as u64;
    let interval_sec = interval_sec.max(1) as u64;

    let temp_frame = std::env::temp_dir().join(format!(
        "bsr_recognition_frame_{}_{}.png",
        std::process::id(),
        Utc::now().timestamp_millis()
    ));

    let frame_dump_dir = if dump_frames {
        let stem = video_path
            .file_stem()
            .and_then(OsStr::to_str)
            .unwrap_or("video");
        let d = output_dir.join("frames").join(stem);
        if let Err(e) = ensure_dir(&d) {
            log::warn!("创建抽帧导出目录失败: {e}");
            None
        } else {
            Some(d)
        }
    } else {
        None
    };

    for sec in (0..=duration_sec).step_by(interval_sec as usize) {
        if let Err(e) = extract_frame_to_png(video_path, sec, &temp_frame).await {
            log::debug!("抽帧跳过: {e}");
            continue;
        }

        let frame = match image::open(&temp_frame) {
            Ok(v) => normalize_gray(v),
            Err(e) => {
                log::debug!("读取抽帧图片失败: {} - {}", temp_frame.display(), e);
                continue;
            }
        };

        if let Some(d) = &frame_dump_dir {
            let out = d.join(format!("{:06}.png", sec));
            if let Err(e) = std::fs::copy(&temp_frame, out) {
                log::debug!("导出抽帧失败: {e}");
            }
        }

        let frame_signature = build_signature(&frame);
        let candidates = pick_candidate_categories(&frame_signature, &library.category_centroids);

        let mut frame_best_by_category: HashMap<String, f32> = HashMap::new();

        for tpl in &library.templates {
            if !candidates.iter().any(|c| c == &tpl.category) {
                continue;
            }

            let score = match_score(&frame, &tpl.gray);
            let key = tpl.category.clone();
            let cur = frame_best_by_category.entry(key).or_insert(0.0);
            if score > *cur {
                *cur = score;
            }

            let name_key = format!("{}::{}", tpl.category, tpl.name);
            let cur_tpl = result.template_scores.entry(name_key).or_insert(0.0);
            if score > *cur_tpl {
                *cur_tpl = score;
            }
        }

        let loading_score = frame_best_by_category.get("loading").copied().unwrap_or(0.0);
        let banpick_score = frame_best_by_category.get("banpick").copied().unwrap_or(0.0);
        let ending_score = frame_best_by_category.get("ending").copied().unwrap_or(0.0);
        let victory_score = frame_best_by_category.get("vectory").copied().unwrap_or(0.0);

        if result.start_time_sec.is_none()
            && (loading_score >= START_THRESHOLD || banpick_score >= START_THRESHOLD)
        {
            result.start_time_sec = Some(sec);
        }

        if result.end_time_sec.is_none() && ending_score >= END_THRESHOLD {
            if let Some(start) = result.start_time_sec {
                if sec >= start {
                    result.end_time_sec = Some(sec);
                }
            } else {
                result.end_time_sec = Some(sec);
            }
        }

        if !result.victory && victory_score >= VICTORY_THRESHOLD {
            result.victory = true;
        }

        if result.start_time_sec.is_some() && result.end_time_sec.is_some() && result.victory {
            break;
        }
    }

    if enable_ocr {
        if let Some(end_sec) = result.end_time_sec {
            if extract_frame_to_png(video_path, end_sec, &temp_frame).await.is_ok() {
                match ocr_from_image(&temp_frame).await {
                    Ok(text) => {
                        let preview = trim_preview(&text);
                        if !preview.is_empty() {
                            result.ocr_preview = Some(preview);
                        }

                        if let Ok(fields) = extract_structured_ocr_fields(&temp_frame).await {
                            result.ocr_fields = fields;
                        }
                    }
                    Err(e) => {
                        log::warn!("OCR失败: {} - {}", video_path.display(), e);
                    }
                }
            }
        }
    }

    let _ = std::fs::remove_file(&temp_frame);

    let completed_name = format!("[completed]{}", file_name);
    let sidecar_path = output_dir.join(format!("{}.recognition.json", completed_name));
    result.result_file = Some(sidecar_path.to_string_lossy().to_string());
    match serde_json::to_string_pretty(&result) {
        Ok(json) => {
            if let Err(e) = std::fs::write(&sidecar_path, json) {
                result.status = "failed".to_string();
                result.error = Some(format!("写入识别结果失败: {} - {}", sidecar_path.display(), e));
                return result;
            }
        }
        Err(e) => {
            result.status = "failed".to_string();
            result.error = Some(format!("序列化识别结果失败: {e}"));
            return result;
        }
    }

    result
}

#[cfg_attr(feature = "gui", tauri::command)]
pub async fn recognize_and_mark_videos(
    state: state_type!(),
    source_dir: Option<String>,
    output_dir: Option<String>,
    frame_interval_sec: Option<u32>,
    enable_ocr: Option<bool>,
    dump_frames: Option<bool>,
) -> Result<RecognitionBatchResult, String> {
    let cfg = state.config.read().await;
    let source = source_dir.unwrap_or_else(|| cfg.output.clone());
    let output = output_dir.unwrap_or_else(|| format!("{}/recognition_completed", source));
    drop(cfg);

    let interval_sec = frame_interval_sec
        .unwrap_or(DEFAULT_INTERVAL_SEC)
        .clamp(1, MAX_INTERVAL_SEC);
    let enable_ocr = enable_ocr.unwrap_or(true);
    let dump_frames = dump_frames.unwrap_or(false);

    let source_path = PathBuf::from(&source);
    let output_path = PathBuf::from(&output);

    if !source_path.exists() || !source_path.is_dir() {
        return Err(format!("待识别目录不存在或不可用: {}", source_path.display()));
    }
    ensure_dir(&output_path)?;

    let templates_root =
        find_templates_root().ok_or_else(|| "未找到模板目录（尝试 src-tauri/templates 与 templates）".to_string())?;
    let library = load_templates(&templates_root)?;

    let videos = collect_videos(&source_path, &output_path)?;
    let mut results = Vec::new();
    let mut processed = 0usize;
    let mut skipped = 0usize;

    for video in videos {
        let name = video
            .file_name()
            .and_then(OsStr::to_str)
            .unwrap_or("unknown")
            .to_string();

        if should_skip_file(&name) {
            skipped += 1;
            results.push(RecognitionFileResult {
                file_name: name,
                status: "skipped_marked".to_string(),
                start_time_sec: None,
                end_time_sec: None,
                victory: false,
                ocr_preview: None,
                output_file: None,
                result_file: None,
                ocr_fields: HashMap::new(),
                template_scores: HashMap::new(),
                error: None,
            });
            continue;
        }

        if let Some(existing_json) = should_skip_by_json(&output_path, &name) {
            skipped += 1;
            results.push(RecognitionFileResult {
                file_name: name,
                status: "skipped_json_exists".to_string(),
                start_time_sec: None,
                end_time_sec: None,
                victory: false,
                ocr_preview: None,
                output_file: None,
                result_file: Some(existing_json.to_string_lossy().to_string()),
                ocr_fields: HashMap::new(),
                template_scores: HashMap::new(),
                error: None,
            });
            continue;
        }

        let item = recognize_single_video(
            &video,
            &library,
            interval_sec,
            enable_ocr,
            &output_path,
            dump_frames,
        )
        .await;
        if item.status != "failed" {
            processed += 1;
        }
        results.push(item);
    }

    Ok(RecognitionBatchResult {
        processed,
        skipped,
        source_dir: source,
        output_dir: output,
        templates_dir: templates_root.to_string_lossy().to_string(),
        interval_sec,
        results,
    })
}

#[cfg_attr(feature = "gui", tauri::command)]
pub async fn build_recognition_dataset(
    source_dir: String,
    output_dir: String,
    model_path: Option<String>,
    high_confidence_threshold: Option<f32>,
    low_confidence_threshold: Option<f32>,
    max_samples_per_video: Option<u32>,
    extract_workers: Option<u32>,
    skip_existing_videos: Option<bool>,
) -> Result<BuildDatasetResult, String> {
    let model_path = model_path.unwrap_or_default();
    let high_threshold = high_confidence_threshold.unwrap_or(0.85).clamp(0.5, 0.99);
    let low_threshold = low_confidence_threshold.unwrap_or(0.35).clamp(0.01, 0.7);
    let max_per_video = max_samples_per_video.unwrap_or(60).clamp(10, 500) as usize;
    let skip_existing = skip_existing_videos.unwrap_or(true);
    let frame_extract_workers = match extract_workers.unwrap_or(0) {
        0 => recommended_extract_workers(max_per_video),
        v => (v as usize).clamp(1, 8),
    };

    let mut phase_model_mode = "heuristic".to_string();
    let mut classifier = if !model_path.is_empty() && Path::new(&model_path).exists() {
        match create_phase_classifier(Path::new(&model_path)) {
            Ok(s) => {
                phase_model_mode = "onnx-active".to_string();
                Some(s)
            }
            Err(e) => {
                log::warn!("ONNX 阶段分类模型加载失败，回退到 heuristic: {e}");
                phase_model_mode = "onnx-failed-fallback-heuristic".to_string();
                None
            }
        }
    } else {
        None
    };

    let source_path = PathBuf::from(&source_dir);
    let output_path = PathBuf::from(&output_dir);
    if !source_path.exists() || !source_path.is_dir() {
        return Err(format!("视频目录不存在或不可用: {}", source_path.display()));
    }

    ensure_dir(&output_path)?;
    let staging_dir = output_path.join("staging");
    let images_dir = staging_dir.join("images");
    ensure_dir(&images_dir)?;
    let manifest_path = staging_dir.join("manifest.json");

    let mut samples: Vec<BuildDatasetSample> = Vec::new();
    let mut known_sample_ids: HashSet<String> = HashSet::new();
    let mut completed_video_paths: HashSet<String> = HashSet::new();

    if skip_existing && manifest_path.exists() {
        if let Ok(existing_manifest) = load_manifest(&manifest_path) {
            for s in existing_manifest.samples {
                if !s.video_path.is_empty() {
                    completed_video_paths.insert(s.video_path.clone());
                }
                known_sample_ids.insert(s.id.clone());
                samples.push(s);
            }
        }
    }

    let videos = collect_videos(&source_path, &output_path)?;
    let mut newly_added = 0usize;

    for video in &videos {
        let video_path_text = video.to_string_lossy().to_string();
        if skip_existing && completed_video_paths.contains(&video_path_text) {
            continue;
        }

        let stem = video
            .file_stem()
            .and_then(OsStr::to_str)
            .unwrap_or("video")
            .to_string();
        let video_img_dir = images_dir.join(&stem);
        ensure_dir(&video_img_dir)?;

        let metadata = match ffmpeg::extract_video_metadata(video).await {
            Ok(v) => v,
            Err(e) => {
                log::warn!("跳过视频(元数据失败): {} - {}", video.display(), e);
                continue;
            }
        };

        let duration_sec = metadata.duration.max(0.0).ceil() as u64;
        let sample_points = build_uniform_sample_points(duration_sec, max_per_video);
        let mut video_added = 0usize;

        let sem = Arc::new(Semaphore::new(frame_extract_workers));
        let mut extract_tasks = Vec::new();

        for sec in &sample_points {
            let sample_id = format!("{:x}", md5::compute(format!("{}:{}", video.display(), sec)));
            if known_sample_ids.contains(&sample_id) {
                continue;
            }

            let frame_path = video_img_dir.join(format!("{}_{}.png", stem, sec));
            if skip_existing && frame_path.exists() {
                continue;
            }

            let sem_cloned = Arc::clone(&sem);
            let video_cloned = video.clone();
            let frame_path_cloned = frame_path.clone();
            let sec_val = *sec;

            extract_tasks.push(tokio::spawn(async move {
                let permit = sem_cloned.acquire_owned().await.ok();
                let ok = extract_frame_to_png(&video_cloned, sec_val, &frame_path_cloned)
                    .await
                    .is_ok();
                drop(permit);
                (sec_val, ok)
            }));
        }

        let mut extracted_secs: HashSet<u64> = HashSet::new();
        for task in extract_tasks {
            if let Ok((sec, ok)) = task.await {
                if ok {
                    extracted_secs.insert(sec);
                }
            }
        }

        for sec in &sample_points {
            let sec = *sec;
            let sample_id = format!("{:x}", md5::compute(format!("{}:{}", video.display(), sec)));
            if known_sample_ids.contains(&sample_id) {
                continue;
            }

            let (phase_guess, _step_guess) = infer_phase_by_time(sec, duration_sec);
            let mut phase_now = normalize_phase_name(phase_guess);
            let mut phase_confidence = match phase_guess {
                "banpick" | "loading" => 0.90,
                "victory_or_defeat" => 0.92,
                "ending" => 0.90,
                _ => 0.60,
            };

            let frame_path = video_img_dir.join(format!("{}_{}.png", stem, sec));
            let mut frame_extracted = false;
            if frame_path.exists() || extracted_secs.contains(&sec) {
                frame_extracted = true;
                if let Some(c) = classifier.as_mut() {
                    if let Ok(img) = image::open(&frame_path) {
                        if let Ok((p, conf)) = classify_phase_with_onnx(c, &img) {
                            phase_now = normalize_phase_name(&p);
                            phase_confidence = conf;
                        }
                    }
                }
            }

            if frame_extracted {
                let p = phase_info(&phase_now);
                let default_status = if phase_confidence >= high_threshold {
                    "correct"
                } else if phase_confidence <= low_threshold {
                    "wrong"
                } else {
                    "pending"
                };

                samples.push(BuildDatasetSample {
                    id: sample_id,
                    image_path: frame_path.to_string_lossy().to_string(),
                    video_path: video.to_string_lossy().to_string(),
                    timestamp_sec: sec,
                    phase: phase_now,
                    confidence: phase_confidence,
                    class_id: p.class_id,
                    roi: p.roi,
                    label_status: default_status.to_string(),
                });
                known_sample_ids.insert(sample_id);
                newly_added += 1;
                video_added += 1;
            }
        }

        if video_added > 0 {
            completed_video_paths.insert(video_path_text);
        }
    }

    if !skip_existing {
        rebalance_negative_ratio(&mut samples);
    }

    let positives = samples.iter().filter(|s| s.label_status == "correct").count();
    let negatives = samples.iter().filter(|s| s.label_status == "wrong").count();

    let manifest = DatasetManifest {
        source_dir: source_dir.clone(),
        output_dir: output_dir.clone(),
        samples: samples.clone(),
    };
    save_manifest(&manifest_path, &manifest)?;

    Ok(BuildDatasetResult {
        source_dir,
        output_dir,
        phase_model_mode,
        frame_extract_workers,
        total_videos: videos.len(),
        total_samples: samples.len(),
        positives,
        negatives,
        samples,
    })
}

#[cfg_attr(feature = "gui", tauri::command)]
pub async fn update_dataset_sample_label(
    output_dir: String,
    sample_id: String,
    status: String,
) -> Result<bool, String> {
    if !matches!(status.as_str(), "correct" | "wrong" | "pending") {
        return Err(format!("不支持的样本状态: {status}"));
    }

    let manifest_path = PathBuf::from(output_dir).join("staging").join("manifest.json");
    let mut manifest = load_manifest(&manifest_path)?;

    let mut found = false;
    for sample in &mut manifest.samples {
        if sample.id == sample_id {
            sample.label_status = status.clone();
            found = true;
            break;
        }
    }

    if !found {
        return Ok(false);
    }

    save_manifest(&manifest_path, &manifest)?;
    Ok(true)
}

#[cfg_attr(feature = "gui", tauri::command)]
pub async fn update_dataset_sample_phase(
    output_dir: String,
    sample_id: String,
    phase: String,
) -> Result<bool, String> {
    let normalized = normalize_phase_name(&phase);
    if !is_supported_phase_label(&normalized) {
        return Err(format!("不支持的阶段标签: {phase}"));
    }

    let manifest_path = PathBuf::from(output_dir).join("staging").join("manifest.json");
    let mut manifest = load_manifest(&manifest_path)?;

    let mut found = false;
    for sample in &mut manifest.samples {
        if sample.id == sample_id {
            let p = phase_info(&normalized);
            sample.phase = normalized.clone();
            sample.class_id = p.class_id;
            sample.roi = p.roi;
            found = true;
            break;
        }
    }

    if !found {
        return Ok(false);
    }

    save_manifest(&manifest_path, &manifest)?;
    Ok(true)
}

#[cfg_attr(feature = "gui", tauri::command)]
pub async fn update_dataset_sample_labels_bulk(
    output_dir: String,
    sample_ids: Vec<String>,
    status: String,
) -> Result<usize, String> {
    if !matches!(status.as_str(), "correct" | "wrong" | "pending") {
        return Err(format!("不支持的样本状态: {status}"));
    }

    if sample_ids.is_empty() {
        return Ok(0);
    }

    let manifest_path = PathBuf::from(output_dir).join("staging").join("manifest.json");
    let mut manifest = load_manifest(&manifest_path)?;
    let target: std::collections::HashSet<String> = sample_ids.into_iter().collect();

    let mut changed = 0usize;
    for sample in &mut manifest.samples {
        if target.contains(&sample.id) && sample.label_status != status {
            sample.label_status = status.clone();
            changed += 1;
        }
    }

    if changed > 0 {
        save_manifest(&manifest_path, &manifest)?;
    }

    Ok(changed)
}

#[cfg_attr(feature = "gui", tauri::command)]
pub async fn update_dataset_sample_phases_bulk(
    output_dir: String,
    sample_ids: Vec<String>,
    phase: String,
) -> Result<usize, String> {
    if sample_ids.is_empty() {
        return Ok(0);
    }

    let normalized = normalize_phase_name(&phase);
    if !is_supported_phase_label(&normalized) {
        return Err(format!("不支持的阶段标签: {phase}"));
    }

    let manifest_path = PathBuf::from(output_dir).join("staging").join("manifest.json");
    let mut manifest = load_manifest(&manifest_path)?;
    let target: std::collections::HashSet<String> = sample_ids.into_iter().collect();
    let p = phase_info(&normalized);

    let mut changed = 0usize;
    for sample in &mut manifest.samples {
        if target.contains(&sample.id)
            && (sample.phase != normalized || sample.class_id != p.class_id || sample.roi != p.roi)
        {
            sample.phase = normalized.clone();
            sample.class_id = p.class_id;
            sample.roi = p.roi;
            changed += 1;
        }
    }

    if changed > 0 {
        save_manifest(&manifest_path, &manifest)?;
    }

    Ok(changed)
}

#[cfg_attr(feature = "gui", tauri::command)]
pub async fn export_recognition_dataset(
    output_dir: String,
    detection_model_path: Option<String>,
    detection_conf_threshold: Option<f32>,
    detection_iou_threshold: Option<f32>,
) -> Result<ExportDatasetResult, String> {
    let det_conf = detection_conf_threshold.unwrap_or(0.25).clamp(0.01, 0.95);
    let det_iou = detection_iou_threshold.unwrap_or(0.45).clamp(0.1, 0.9);
    let output_root = PathBuf::from(&output_dir);
    let manifest_path = output_root.join("staging").join("manifest.json");
    let manifest = load_manifest(&manifest_path)?;

    let mut bbox_model_mode = "roi-fallback".to_string();
    let mut detector = if let Some(p) = detection_model_path.as_ref().map(|v| v.trim()).filter(|v| !v.is_empty()) {
        let model_path = PathBuf::from(p);
        if model_path.exists() {
            match create_detection_model(&model_path) {
                Ok(s) => {
                    bbox_model_mode = "onnx-active".to_string();
                    Some(s)
                }
                Err(e) => {
                    log::warn!("加载检测模型失败，使用ROI回退: {e}");
                    bbox_model_mode = "onnx-failed-fallback-roi".to_string();
                    None
                }
            }
        } else {
            log::warn!("检测模型路径不存在，使用ROI回退: {}", model_path.display());
            None
        }
    } else {
        None
    };

    let dataset_dir = output_root.join("yolo_dataset");
    let train_images = dataset_dir.join("images").join("train");
    let val_images = dataset_dir.join("images").join("val");
    let train_labels = dataset_dir.join("labels").join("train");
    let val_labels = dataset_dir.join("labels").join("val");
    ensure_dir(&train_images)?;
    ensure_dir(&val_images)?;
    ensure_dir(&train_labels)?;
    ensure_dir(&val_labels)?;

    let mut images_count = 0usize;
    let mut labels_count = 0usize;
    let mut train_images_count = 0usize;
    let mut val_images_count = 0usize;
    let mut detected_box_labels = 0usize;
    let mut fallback_roi_labels = 0usize;

    let positives = manifest.samples.iter().filter(|s| s.label_status == "correct").count() as f32;
    let negatives = manifest.samples.iter().filter(|s| s.label_status == "wrong").count() as f32;
    let estimated_accuracy = if (positives + negatives) > 0.0 {
        positives / (positives + negatives)
    } else {
        0.0
    };

    for sample in &manifest.samples {
        if sample.label_status == "wrong" {
            continue;
        }

        let src = PathBuf::from(&sample.image_path);
        if !src.exists() || !src.is_file() {
            continue;
        }

        let file_name = src
            .file_name()
            .and_then(OsStr::to_str)
            .unwrap_or("sample.png")
            .to_string();
        let is_val = deterministic_val_split(&sample.id);
        let dst_img = if is_val {
            val_images.join(&file_name)
        } else {
            train_images.join(&file_name)
        };
        std::fs::copy(&src, &dst_img)
            .map_err(|e| format!("复制样本图片失败: {} - {}", src.display(), e))?;
        images_count += 1;
        if is_val {
            val_images_count += 1;
        } else {
            train_images_count += 1;
        }

        let label_name = Path::new(&file_name)
            .file_stem()
            .and_then(OsStr::to_str)
            .unwrap_or("sample")
            .to_string();
        let dst_label = if is_val {
            val_labels.join(format!("{}.txt", label_name))
        } else {
            train_labels.join(format!("{}.txt", label_name))
        };

        let mut final_class_id = sample.class_id;
        let mut final_roi = sample.roi;

        if let Some(d) = detector.as_mut() {
            if let Ok(img) = image::open(&src) {
                match detect_boxes_with_onnx(d, &img, det_conf, det_iou) {
                    Ok(mut dets) => {
                        dets.sort_by(|a, b| {
                            b.confidence
                                .partial_cmp(&a.confidence)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        });
                        let selected = dets
                            .iter()
                            .find(|x| x.class_id == sample.class_id)
                            .copied()
                            .or_else(|| dets.first().copied());

                        if let Some(best) = selected {
                            final_class_id = best.class_id;
                            final_roi = best.roi;
                            detected_box_labels += 1;
                        } else {
                            fallback_roi_labels += 1;
                        }
                    }
                    Err(e) => {
                        log::debug!("检测框推理失败，回退ROI: {} - {}", src.display(), e);
                        fallback_roi_labels += 1;
                    }
                }
            } else {
                fallback_roi_labels += 1;
            }
        } else {
            fallback_roi_labels += 1;
        }

        let label_line = yolo_line(final_class_id, final_roi);
        std::fs::write(&dst_label, format!("{}\n", label_line))
            .map_err(|e| format!("写入标签文件失败: {} - {}", dst_label.display(), e))?;
        labels_count += 1;
    }

    write_dataset_yaml(&dataset_dir)?;

    Ok(ExportDatasetResult {
        dataset_dir: dataset_dir.to_string_lossy().to_string(),
        images_count,
        labels_count,
        train_images_count,
        val_images_count,
        estimated_accuracy,
        bbox_model_mode,
        detected_box_labels,
        fallback_roi_labels,
        exported_at: Utc::now().to_rfc3339(),
    })
}

#[cfg_attr(feature = "gui", tauri::command)]
pub async fn train_and_update_recognition_model(
    dataset_dir: String,
    output_dir: String,
    target_model_path: String,
    epochs: Option<u32>,
    imgsz: Option<u32>,
    python_bin: Option<String>,
) -> Result<TrainModelResult, String> {
    let epochs = epochs.unwrap_or(30).clamp(1, 300);
    let imgsz = imgsz.unwrap_or(640).clamp(320, 1280);
    let python_bin = python_bin.unwrap_or_else(|| "python".to_string());

    let dataset_path = PathBuf::from(&dataset_dir);
    let dataset_yaml = dataset_path.join("dataset.yaml");
    if !dataset_yaml.exists() {
        return Err(format!("未找到 dataset.yaml: {}", dataset_yaml.display()));
    }

    let run_root = PathBuf::from(&output_dir);
    ensure_dir(&run_root)?;
    let run_name = "recognition_yolo";

    let mut train_cmd = tokio::process::Command::new(&python_bin);
    #[cfg(target_os = "windows")]
    train_cmd.creation_flags(CREATE_NO_WINDOW);

    let train_output = train_cmd
        .args([
            "-m",
            "ultralytics",
            "yolo",
            "detect",
            "train",
            &format!("data={}", dataset_yaml.to_string_lossy()),
            "model=yolov8n.pt",
            &format!("epochs={epochs}"),
            &format!("imgsz={imgsz}"),
            &format!("project={}", run_root.to_string_lossy()),
            &format!("name={run_name}"),
            "exist_ok=True",
        ])
        .output()
        .await
        .map_err(|e| format!("执行训练命令失败: {e}"))?;

    let train_stdout = String::from_utf8_lossy(&train_output.stdout).to_string();
    let train_stderr = String::from_utf8_lossy(&train_output.stderr).to_string();
    if !train_output.status.success() {
        return Err(format!(
            "训练失败:\n{}\n{}",
            trim_tail(&train_stdout, 2000),
            trim_tail(&train_stderr, 2000)
        ));
    }

    let run_dir = run_root.join(run_name);
    let best_pt = run_dir.join("weights").join("best.pt");
    if !best_pt.exists() {
        return Err(format!("训练完成但未找到 best.pt: {}", best_pt.display()));
    }

    let mut export_cmd = tokio::process::Command::new(&python_bin);
    #[cfg(target_os = "windows")]
    export_cmd.creation_flags(CREATE_NO_WINDOW);

    let export_output = export_cmd
        .args([
            "-m",
            "ultralytics",
            "yolo",
            "export",
            &format!("model={}", best_pt.to_string_lossy()),
            "format=onnx",
            &format!("imgsz={imgsz}"),
        ])
        .output()
        .await
        .map_err(|e| format!("执行导出命令失败: {e}"))?;

    let export_stdout = String::from_utf8_lossy(&export_output.stdout).to_string();
    let export_stderr = String::from_utf8_lossy(&export_output.stderr).to_string();
    if !export_output.status.success() {
        return Err(format!(
            "导出 ONNX 失败:\n{}\n{}",
            trim_tail(&export_stdout, 2000),
            trim_tail(&export_stderr, 2000)
        ));
    }

    let onnx_candidates = [
        run_dir.join("weights").join("best.onnx"),
        run_dir.join("best.onnx"),
    ];
    let exported_onnx = onnx_candidates
        .into_iter()
        .find(|p| p.exists())
        .ok_or_else(|| "导出完成但未找到 best.onnx".to_string())?;

    let target_path = PathBuf::from(&target_model_path);
    ensure_parent_dir(&target_path)?;
    std::fs::copy(&exported_onnx, &target_path).map_err(|e| {
        format!(
            "替换推理模型失败: {} -> {} ({})",
            exported_onnx.display(),
            target_path.display(),
            e
        )
    })?;

    let report_path = run_dir.join("training_report.json");
    let report = serde_json::json!({
        "created_at": Utc::now().to_rfc3339(),
        "dataset_yaml": dataset_yaml,
        "epochs": epochs,
        "imgsz": imgsz,
        "python_bin": python_bin,
        "run_dir": run_dir,
        "exported_onnx": exported_onnx,
        "target_model_path": target_path,
        "train_stdout_tail": trim_tail(&train_stdout, 2000),
        "train_stderr_tail": trim_tail(&train_stderr, 2000),
        "export_stdout_tail": trim_tail(&export_stdout, 2000),
        "export_stderr_tail": trim_tail(&export_stderr, 2000)
    });
    std::fs::write(&report_path, serde_json::to_string_pretty(&report).unwrap_or_default())
        .map_err(|e| format!("写入训练报告失败: {}", e))?;

    Ok(TrainModelResult {
        success: true,
        dataset_dir,
        run_dir: run_dir.to_string_lossy().to_string(),
        exported_onnx_path: exported_onnx.to_string_lossy().to_string(),
        replaced_model_path: target_path.to_string_lossy().to_string(),
        report_file: report_path.to_string_lossy().to_string(),
        train_stdout_tail: trim_tail(&train_stdout, 1200),
        train_stderr_tail: trim_tail(&train_stderr, 1200),
    })
}
