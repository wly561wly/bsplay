from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from labeling_pipeline.pipeline import (  # noqa: E402
    CLASS_NAMES,
    PHASE_DEFAULT_ROI,
    PHASE_TO_CLASS_ID,
    build_samples_from_videos,
    copy_organized_outputs,
    load_manifest,
    normalize_sample,
    save_manifest,
)

st.set_page_config(page_title="Dataset Labeling Pipeline", layout="wide")

st.title("视频抽帧与快速标注流水线")
st.caption("功能: 均匀随机抽帧 -> 快速分类与有效性标注 -> 可选负样本收集 -> 自动归档")


# Defaults aligned with current repository layout.
DATASET_ROOT = ROOT / "test_videos" / "recognition_dataset"
DEFAULT_VIDEO_ROOT = DATASET_ROOT / "recog_video"
DEFAULT_STAGING = DATASET_ROOT / "staging_v2"
DEFAULT_MANIFEST = DEFAULT_STAGING / "manifest.json"
DEFAULT_EXPORT = DATASET_ROOT / "labeled_export"


with st.sidebar:
    st.header("配置")

    video_root_str = st.text_input("视频目录", str(DEFAULT_VIDEO_ROOT))
    staging_dir_str = st.text_input("抽帧输出目录", str(DEFAULT_STAGING))
    manifest_path_str = st.text_input("manifest 路径", str(DEFAULT_MANIFEST))
    export_root_str = st.text_input("归档输出目录", str(DEFAULT_EXPORT))

    samples_per_video = st.number_input("每个视频抽帧数", min_value=1, max_value=500, value=50)
    random_seed = st.number_input("随机种子", min_value=0, max_value=999999, value=2026)
    default_phase = st.selectbox("新样本默认分类", options=CLASS_NAMES, index=CLASS_NAMES.index("other"))

    st.divider()
    st.subheader("抽帧")
    do_sample = st.button("1) 从视频生成抽帧样本", use_container_width=True)

    st.divider()
    st.subheader("归档")
    collect_negative = st.checkbox("导出时收集负样本", value=True)
    do_export = st.button("2) 自动归档到分类文件夹", use_container_width=True)


video_root = Path(video_root_str)
staging_dir = Path(staging_dir_str)
manifest_path = Path(manifest_path_str)
export_root = Path(export_root_str)

if do_sample:
    with st.spinner("正在抽帧并更新 manifest..."):
        v_count, n_new, total = build_samples_from_videos(
            video_root=video_root,
            staging_dir=staging_dir,
            manifest_path=manifest_path,
            samples_per_video=int(samples_per_video),
            default_phase=default_phase,
            seed=int(random_seed),
        )
    st.success(f"完成: 视频 {v_count} 个, 新增样本 {n_new} 个, 总样本 {total} 个")

manifest = load_manifest(manifest_path)
samples = [normalize_sample(s) for s in manifest.get("samples", [])]

if not samples:
    st.warning("当前没有样本。请先点击侧边栏的 `从视频生成抽帧样本`。")
    st.stop()


# Persist cursor in session state.
if "sample_idx" not in st.session_state:
    st.session_state.sample_idx = 0

max_idx = len(samples) - 1
st.session_state.sample_idx = max(0, min(st.session_state.sample_idx, max_idx))

col_nav1, col_nav2, col_nav3 = st.columns([1, 2, 1])
with col_nav1:
    if st.button("上一张", use_container_width=True):
        st.session_state.sample_idx = max(0, st.session_state.sample_idx - 1)
with col_nav2:
    st.write(f"样本进度: {st.session_state.sample_idx + 1} / {len(samples)}")
with col_nav3:
    if st.button("下一张", use_container_width=True):
        st.session_state.sample_idx = min(max_idx, st.session_state.sample_idx + 1)

idx = st.session_state.sample_idx
sample = samples[idx]
img_path = Path(sample["image_path"])

left, right = st.columns([2, 1])

with left:
    st.subheader("图像预览 (含 GT 框)")
    if not img_path.exists():
        st.error(f"图像不存在: {img_path}")
    else:
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        w, h = img.size

        cx, cy, bw, bh = [float(x) for x in sample["roi"]]
        x1 = (cx - bw / 2.0) * w
        y1 = (cy - bh / 2.0) * h
        x2 = (cx + bw / 2.0) * w
        y2 = (cy + bh / 2.0) * h
        draw.rectangle([(x1, y1), (x2, y2)], outline="lime", width=3)

        st.image(img, caption=f"{img_path.name}", use_container_width=True)

with right:
    st.subheader("快速标注")
    st.text(f"视频: {Path(sample.get('video_path', '')).name}")
    st.text(f"时间点(秒): {sample.get('timestamp_sec', -1)}")

    phase = st.selectbox("分类", options=CLASS_NAMES, index=CLASS_NAMES.index(sample["phase"]))
    label_status = st.radio("有效性", options=["correct", "wrong", "pending"], index=["correct", "wrong", "pending"].index(sample["label_status"]))
    use_as_negative = st.checkbox("作为负样本", value=bool(sample.get("use_as_negative", False)))

    st.markdown("GT 框修正 (YOLO 归一化)")
    cx = st.slider("cx", min_value=0.0, max_value=1.0, value=float(sample["roi"][0]), step=0.001)
    cy = st.slider("cy", min_value=0.0, max_value=1.0, value=float(sample["roi"][1]), step=0.001)
    bw = st.slider("w", min_value=0.01, max_value=1.0, value=float(sample["roi"][2]), step=0.001)
    bh = st.slider("h", min_value=0.01, max_value=1.0, value=float(sample["roi"][3]), step=0.001)

    c1, c2 = st.columns(2)
    with c1:
        save_one = st.button("保存当前", use_container_width=True)
    with c2:
        save_next = st.button("保存并下一张", use_container_width=True)

    if save_one or save_next:
        sample["phase"] = phase
        sample["class_id"] = PHASE_TO_CLASS_ID.get(phase, PHASE_TO_CLASS_ID["other"])
        sample["label_status"] = label_status
        sample["use_as_negative"] = bool(use_as_negative)
        sample["roi"] = [float(cx), float(cy), float(bw), float(bh)]
        samples[idx] = sample

        manifest["samples"] = samples
        save_manifest(manifest_path, manifest)
        st.success("已保存")

        if save_next:
            st.session_state.sample_idx = min(max_idx, st.session_state.sample_idx + 1)
            st.rerun()

st.divider()

summary_df = pd.DataFrame(samples)
if not summary_df.empty:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("总样本", len(summary_df))
    with c2:
        st.metric("correct", int((summary_df["label_status"] == "correct").sum()))
    with c3:
        st.metric("wrong", int((summary_df["label_status"] == "wrong").sum()))

    with st.expander("分类统计"):
        st.dataframe(summary_df.groupby(["phase", "label_status"], dropna=False).size().reset_index(name="count"), use_container_width=True)

if do_export:
    with st.spinner("正在归档图片..."):
        stats = copy_organized_outputs(
            manifest_path=manifest_path,
            output_root=export_root,
            collect_negative=bool(collect_negative),
        )
    st.success(
        "归档完成: "
        f"正样本复制 {stats['copied_positive']} 张, "
        f"负样本复制 {stats['copied_negative']} 张, "
        f"缺失跳过 {stats['skipped_missing']} 张"
    )

st.info(
    "建议流程: 1) 先抽帧 2) 逐图标注与修框 3) 归档导出 4) 在训练 Notebook 里复用归档结果继续训练。"
)
