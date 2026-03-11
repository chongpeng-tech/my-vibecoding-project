# CCPD-ALPR（精度优先）

这是一个完整的车牌检测与识别系统，包含：

1. `YOLO Pose` 车牌定位 + 四角点回归  
2. 基于四角点透视矫正后的 `CRNN + CTC` 识别  
3. 可视化验证 UI（单图 / 视频 / 批量目录）

默认数据路径使用：`F:\ccpd_b`。  
你的机器上 `conda run` 在长日志任务中可能出现编码问题，因此本文统一使用环境内 `python.exe` 直接执行。

## 1. 环境安装（biye_sheji）

在目录 `F:\Code\my-vibecoding-project\ccpd_alpr` 执行：

```powershell
$env:ENV_PY="C:\Users\86187\.conda\envs\biye_sheji\python.exe"
& $env:ENV_PY -m pip install -U pip
& $env:ENV_PY -m pip install -r requirements.txt
& $env:ENV_PY -m pip install -e .
```

推荐安装 CUDA 版 PyTorch（RTX 3060）：

```powershell
& $env:ENV_PY -m pip uninstall -y torch torchvision torchaudio
& $env:ENV_PY -m pip install --force-reinstall --no-cache-dir torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

## 2. 数据准备

解析 CCPD 文件名标签，自动生成：
- 检测训练集（YOLO Pose 标签）
- OCR 索引文件
- `train/val/test = 0.9/0.05/0.05`

```powershell
& $env:ENV_PY scripts/prepare_ccpd.py --dataset-root F:\ccpd_b --output-root data
```

## 3. 训练检测模型（YOLO Pose）

```powershell
& $env:ENV_PY scripts/train_detector.py --data data/detector/ccpd_pose.yaml --model yolov8x-pose.pt --epochs 300 --imgsz 960 --batch 16 --device 0 --project detector
```

默认权重输出：
`runs/pose/detector/yolov8x_pose_ccpd/weights/best.pt`

## 4. 训练识别模型（CRNN + CTC）

```powershell
& $env:ENV_PY scripts/train_recognizer.py --train-index data/index/train.jsonl --val-index data/index/val.jsonl --epochs 80 --batch-size 256 --device cuda --amp
```

默认权重输出：
`runs/recognizer/crnn_ctc/best.pt`

从已有权重继续训练：

```powershell
& $env:ENV_PY scripts/train_recognizer.py --train-index data/index/train.jsonl --val-index data/index/val.jsonl --output-dir runs/recognizer/crnn_ctc --epochs 80 --batch-size 256 --device cuda --amp --resume runs/recognizer/crnn_ctc/last.pt
```

## 5. 命令行推理与评估

端到端评估：

```powershell
& $env:ENV_PY scripts/evaluate_end2end.py --test-index data/index/test.jsonl --detector-weights runs/pose/detector/yolov8x_pose_ccpd/weights/best.pt --recognizer-weights runs/recognizer/crnn_ctc/best.pt --device cuda:0
```

批量推理：

```powershell
& $env:ENV_PY scripts/infer.py --source F:\ccpd_b --detector-weights runs/pose/detector/yolov8x_pose_ccpd/weights/best.pt --recognizer-weights runs/recognizer/crnn_ctc/best.pt --device cuda:0 --output-dir runs/infer_ccpd
```

## 6. 精美 UI 验证系统

启动 UI：

```powershell
& $env:ENV_PY app.py --host 127.0.0.1 --port 7860 --device cuda:0
```

打开后可直接使用：
- 单图验证：查看检测框、四角点、车牌矫正图、识别文本
- 视频验证：输出带标注视频，并导出 `CSV/JSONL`
- 批量验证：输入目录批量处理，自动生成可视化与结构化结果

如需从外网访问可加 `--share`。

## 7. 精度优先建议

- 检测优先大模型：`yolov8x-pose.pt`
- 使用较高输入分辨率：`imgsz=960/1152/1280`
- 增加训练轮次：检测 `300+`，识别 `80~120+`
- 推理时适当降低 `conf`（如 `0.20~0.30`）避免漏检
- 对目标域（非 CCPD）补充少量标注并微调，可显著提升泛化

