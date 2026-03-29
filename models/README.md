# Models Directory

This directory stores local ONNX models used by recognition.

## Recommended files

- `yolo_game.onnx`: phase classifier and/or detector model used by the recognition pipeline.

## If the model file is missing

The backend automatically falls back:

- dataset build: `phase_model_mode = heuristic`
- dataset export: `bbox_model_mode = roi-fallback`

So recognition/export still works, but with lower quality labels.

## Create model with built-in pipeline

1. Build and verify dataset in UI (`Recognition -> dataset mode`).
2. Export YOLO dataset.
3. Click `训练并自动替换模型`.
4. Set target path to `models/yolo_game.onnx`.

## Manual training (optional)

```bash
python -m ultralytics yolo detect train data=/path/to/dataset.yaml model=yolov8n.pt epochs=30 imgsz=640
python -m ultralytics yolo export model=/path/to/best.pt format=onnx imgsz=640
```

Then copy exported `best.onnx` to `models/yolo_game.onnx`.
