# Training Notebook (YOLO)

This folder contains a notebook-first training workflow for the recognition model.
It is designed to run in WSL and to be compatible with the app's dataset format.

## What it does

- Extract frames from videos and build a staging manifest.
- Provide a lightweight labeling/verification UI inside the notebook.
- Export a YOLO dataset (images/labels + dataset.yaml).
- Train and export ONNX, then copy it to models/yolo_game.onnx.

## Quick start (WSL)

1. Create a virtualenv and install deps:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r train/requirements.txt
   ```

2. Start Jupyter (VS Code or CLI):

   ```bash
   jupyter notebook
   ```

3. Open: train/recognition_training.ipynb

## Web labeling pipeline (recommended for fast review)

For high-throughput frame review and labeling, use the Streamlit app:

1. Install dependencies and launch:

   ```bash
   cd train/labeling_pipeline
   pip install -r requirements.txt
   streamlit run app.py
   ```

2. In the web UI:

- Generate uniformly distributed random samples from mp4 videos.
- Label class (`phase`) and validity (`correct/wrong/pending`).
- Optionally mark samples for negative collection.
- Adjust GT box (`cx, cy, w, h`) and preview immediately.
- Export organized folders automatically.

3. Export structure:

- `organized/<phase>/<label_status>/*.png`
- `negatives/*.png` (when enabled)

## Default paths

- Input videos: test_videos/recognition_dataset/recog_video
- Staging: test_videos/recognition_dataset/staging
- YOLO export: test_videos/recognition_dataset/yolo_dataset
- Model output: models/yolo_game.onnx

You can override all paths in the notebook.

## Notes

- This notebook keeps the class list fixed to 6 classes:
  banpick, loading, gaming, victory_or_defeat, ending, other
- The staging manifest uses the same fields as the backend:
  id, image_path, video_path, timestamp_sec, phase, confidence,
  class_id, roi, label_status
