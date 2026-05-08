# Labeling Pipeline (Web)

这个子项目提供一个可视化流水线，用于:

1. 从指定 mp4 目录进行均匀随机抽帧
2. 快速标注分类 (`phase`) 与有效性 (`correct/wrong/pending`)
3. 可选负样本收集 (`use_as_negative`)
4. 在线修正 GT 框 (YOLO: cx, cy, w, h)
5. 自动归档到分类文件夹

## 目录

- `app.py`: Streamlit 标注界面
- `pipeline.py`: 抽帧、manifest、归档核心逻辑
- `requirements.txt`: 依赖

## 使用方法

在仓库根目录执行:

```powershell
cd train\labeling_pipeline
pip install -r requirements.txt
streamlit run app.py
```

浏览器打开后:

1. 在左侧配置 `视频目录`、`抽帧输出目录`、`manifest 路径`
2. 点击 `1) 从视频生成抽帧样本`
3. 中间逐图标注并保存 (支持保存并下一张)
4. 勾选是否收集负样本，点击 `2) 自动归档到分类文件夹`

## 归档输出结构

默认输出目录下会生成:

- `organized/<phase>/<label_status>/*.png`
- `negatives/*.png` (仅在勾选收集负样本时)

示例:

- `organized/gaming/correct/xxx.png`
- `organized/loading/wrong/yyy.png`
- `negatives/zzz.png`

## 备注

- 抽帧优先使用 `ffprobe` 获取真实时长，失败时会回退到 1800 秒近似时长。
- 采样策略是每段均匀切分后段内随机，兼顾覆盖度与随机性。
- `manifest.json` 为主数据源，支持多次追加样本，不会重复生成同一个样本 ID。
