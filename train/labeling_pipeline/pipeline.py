from __future__ import annotations

import hashlib
import json
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

CLASS_NAMES = [
    "banpick",
    "loading",
    "gaming",
    "victory_or_defeat",
    "ending",
    "other",
]
PHASE_TO_CLASS_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# cx, cy, w, h (normalized)
PHASE_DEFAULT_ROI = {
    "banpick": [0.50, 0.18, 0.74, 0.22],
    "loading": [0.50, 0.50, 0.90, 0.90],
    "gaming": [0.50, 0.08, 0.32, 0.12],
    "victory_or_defeat": [0.50, 0.50, 0.62, 0.24],
    "ending": [0.50, 0.50, 0.66, 0.28],
    "other": [0.50, 0.50, 1.00, 1.00],
}

VALID_VIDEO_EXTS = {".mp4", ".mkv", ".flv", ".mov", ".avi", ".webm"}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def list_videos(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VALID_VIDEO_EXTS])


def hash_sample_id(video_path: Path, sec: int) -> str:
    raw = f"{video_path.as_posix()}:{sec}".encode("utf-8")
    return hashlib.md5(raw).hexdigest()


def load_manifest(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {
            "source_dir": "",
            "output_dir": "",
            "samples": [],
        }
    return json.loads(path.read_text(encoding="utf-8"))


def save_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def ffprobe_duration_sec(video_path: Path) -> Optional[float]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
        if not out:
            return None
        return max(0.0, float(out))
    except Exception:
        return None


def uniform_jitter_points(duration_sec: float, sample_count: int, seed: int) -> List[int]:
    if sample_count <= 0:
        return []
    if duration_sec <= 0:
        return [0]

    rng = random.Random(seed)
    seg = duration_sec / sample_count
    points = []
    for i in range(sample_count):
        start = i * seg
        end = min(duration_sec, (i + 1) * seg)
        if end <= start:
            sec = int(round(start))
        else:
            sec = int(round(rng.uniform(start, end)))
        points.append(max(0, sec))
    return sorted(set(points))


def ffmpeg_extract_frame(video_path: Path, sec: int, output_path: Path, width: int = 960) -> None:
    ensure_dir(output_path.parent)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        str(sec),
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-vf",
        f"scale={width}:-1",
        "-y",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def build_sample(
    sample_id: str,
    image_path: Path,
    video_path: Path,
    timestamp_sec: int,
    phase: str,
    label_status: str = "pending",
    use_as_negative: bool = False,
    confidence: float = 0.6,
) -> Dict[str, Any]:
    phase_norm = phase.strip().lower()
    if phase_norm not in PHASE_TO_CLASS_ID:
        phase_norm = "other"

    class_id = PHASE_TO_CLASS_ID[phase_norm]
    roi = PHASE_DEFAULT_ROI.get(phase_norm, PHASE_DEFAULT_ROI["other"])

    return {
        "id": sample_id,
        "image_path": str(image_path),
        "video_path": str(video_path),
        "timestamp_sec": int(timestamp_sec),
        "phase": phase_norm,
        "class_id": int(class_id),
        "roi": [float(v) for v in roi],
        "label_status": label_status,
        "use_as_negative": bool(use_as_negative),
        "confidence": float(confidence),
    }


def build_samples_from_videos(
    video_root: Path,
    staging_dir: Path,
    manifest_path: Path,
    samples_per_video: int,
    default_phase: str,
    seed: int,
) -> Tuple[int, int, int]:
    ensure_dir(staging_dir)
    frame_dir = staging_dir / "images"
    ensure_dir(frame_dir)

    manifest = load_manifest(manifest_path)
    manifest["source_dir"] = str(video_root)
    manifest["output_dir"] = str(staging_dir)
    existing_ids = {s["id"] for s in manifest.get("samples", []) if "id" in s}

    videos = list_videos(video_root)
    new_samples: List[Dict[str, Any]] = []

    for video in videos:
        dur = ffprobe_duration_sec(video)
        if dur is None:
            dur = 1800.0
        pts = uniform_jitter_points(dur, samples_per_video, seed=seed ^ hash(video.as_posix()))

        out_dir = frame_dir / video.stem
        ensure_dir(out_dir)

        for sec in pts:
            sample_id = hash_sample_id(video, sec)
            if sample_id in existing_ids:
                continue

            frame_path = out_dir / f"{video.stem}_{sec}.png"
            try:
                if not frame_path.exists():
                    ffmpeg_extract_frame(video, sec, frame_path)
            except Exception:
                continue

            sample = build_sample(
                sample_id=sample_id,
                image_path=frame_path,
                video_path=video,
                timestamp_sec=sec,
                phase=default_phase,
                label_status="pending",
            )
            new_samples.append(sample)
            existing_ids.add(sample_id)

    manifest.setdefault("samples", []).extend(new_samples)
    save_manifest(manifest_path, manifest)

    return len(videos), len(new_samples), len(manifest.get("samples", []))


def normalize_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    # Backward compatibility for old manifest fields.
    s = dict(sample)
    if "phase" not in s:
        s["phase"] = "other"
    if "class_id" not in s:
        s["class_id"] = PHASE_TO_CLASS_ID.get(s["phase"], PHASE_TO_CLASS_ID["other"])
    if "roi" not in s:
        s["roi"] = PHASE_DEFAULT_ROI.get(s["phase"], PHASE_DEFAULT_ROI["other"])
    if "label_status" not in s:
        s["label_status"] = "pending"
    if "use_as_negative" not in s:
        s["use_as_negative"] = False
    return s


def copy_organized_outputs(
    manifest_path: Path,
    output_root: Path,
    collect_negative: bool,
) -> Dict[str, int]:
    manifest = load_manifest(manifest_path)
    samples = [normalize_sample(s) for s in manifest.get("samples", [])]

    stats = {
        "copied_positive": 0,
        "copied_negative": 0,
        "skipped_missing": 0,
    }

    for s in samples:
        src = Path(s.get("image_path", ""))
        if not src.exists():
            stats["skipped_missing"] += 1
            continue

        phase = str(s["phase"])
        status = str(s["label_status"])
        dst_dir = output_root / "organized" / phase / status
        ensure_dir(dst_dir)
        (dst_dir / src.name).write_bytes(src.read_bytes())
        stats["copied_positive"] += 1

        if collect_negative and bool(s.get("use_as_negative", False)):
            neg_dir = output_root / "negatives"
            ensure_dir(neg_dir)
            (neg_dir / src.name).write_bytes(src.read_bytes())
            stats["copied_negative"] += 1

    return stats
