from __future__ import annotations

import argparse
import datetime as dt
import threading
from pathlib import Path
from typing import Any

import cv2
import gradio as gr
import numpy as np
import pandas as pd
import torch

from ccpd_alpr.service import ALPRService
from ccpd_alpr.utils import ensure_dir

APP_TITLE = "CCPD ALPR Studio"

DEFAULT_DETECTOR = Path("runs/pose/detector/yolov8x_pose_ccpd/weights/best.pt")
DEFAULT_RECOGNIZER = Path("runs/recognizer/crnn_ctc/best.pt")

CUSTOM_CSS = """
:root {
  --bg-main: radial-gradient(circle at 5% -10%, #f6fbff 0%, #ecf4fc 40%, #f6f7f8 100%);
  --panel: rgba(255, 255, 255, 0.9);
  --hero: linear-gradient(130deg, #113c59 0%, #1f6d8f 45%, #52a0a6 100%);
  --line: rgba(18, 56, 86, 0.12);
}
body, .gradio-container { background: var(--bg-main) !important; }
.hero {
  background: var(--hero);
  border-radius: 20px;
  padding: 26px 30px;
  margin-bottom: 14px;
  color: #fff;
  box-shadow: 0 14px 32px rgba(16, 55, 88, 0.25);
}
.hero h1 { margin: 0 0 8px 0; font-size: 30px; }
.hero p { margin: 0; font-size: 14px; opacity: 0.95; }
.glass {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 16px;
  box-shadow: 0 8px 24px rgba(38, 70, 95, 0.08);
}
"""

HERO_HTML = """
<div class="hero">
  <h1>CCPD ALPR Studio</h1>
  <p>车牌定位 + 四角点 + OCR 端到端验证平台。支持单图、视频、批量目录，并可导出结构化结果。</p>
</div>
"""


class ServiceManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._service: ALPRService | None = None
        self._signature: tuple[str, str, str, bool, float, str] | None = None

    def _signature_of(
        self,
        detector_weights: str,
        recognizer_weights: str,
        device: str,
        province_refine: bool,
        province_refine_threshold: float,
        province_classifier_weights: str,
    ) -> tuple[str, str, str, bool, float, str]:
        return (
            Path(detector_weights).expanduser().resolve().as_posix(),
            Path(recognizer_weights).expanduser().resolve().as_posix(),
            device.strip(),
            bool(province_refine),
            round(float(province_refine_threshold), 4),
            Path(province_classifier_weights).expanduser().resolve().as_posix(),
        )

    def get_service(
        self,
        detector_weights: str,
        recognizer_weights: str,
        device: str,
        province_refine: bool,
        province_refine_threshold: float,
        province_classifier_weights: str,
    ) -> ALPRService:
        sig = self._signature_of(
            detector_weights,
            recognizer_weights,
            device,
            province_refine,
            province_refine_threshold,
            province_classifier_weights,
        )
        with self._lock:
            if self._service is None or self._signature != sig:
                self._service = ALPRService(
                    detector_weights=Path(detector_weights).expanduser(),
                    recognizer_weights=Path(recognizer_weights).expanduser(),
                    device=device,
                    province_refine=province_refine,
                    province_refine_threshold=province_refine_threshold,
                    province_classifier_weights=Path(province_classifier_weights).expanduser(),
                )
                self._signature = sig
            return self._service


SERVICE_MANAGER = ServiceManager()


def _now_tag() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _to_bgr(image_rgb: np.ndarray | None) -> np.ndarray:
    if image_rgb is None:
        raise ValueError("未上传图片")
    if image_rgb.ndim == 2:
        return cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)


def _to_rgb(image_bgr: np.ndarray | None) -> np.ndarray | None:
    if image_bgr is None:
        return None
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _safe_path(raw: str) -> Path:
    return Path(raw).expanduser().resolve()


def _normalize_video_input(video_input: Any) -> str | None:
    if video_input is None:
        return None
    if isinstance(video_input, str):
        return video_input
    if isinstance(video_input, dict):
        for key in ("path", "video", "name"):
            value = video_input.get(key)
            if isinstance(value, str) and value:
                return value
    return None


def check_models(
    detector_weights: str,
    recognizer_weights: str,
    device: str,
    province_refine: bool,
    province_refine_threshold: float,
    province_classifier_weights: str,
) -> str:
    det = _safe_path(detector_weights)
    rec = _safe_path(recognizer_weights)
    gpu = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu else "CPU"
    return "\n".join(
        [
            "### 模型状态",
            f"- 检测模型: `{'OK' if det.exists() else '缺失'}` `{det.as_posix()}`",
            f"- 识别模型: `{'OK' if rec.exists() else '缺失'}` `{rec.as_posix()}`",
            f"- 推理设备: `{device}` (当前可用: `{gpu_name}`)",
            f"- 省份纠错: `{'开启' if province_refine else '关闭'}` 阈值 `{province_refine_threshold:.3f}`",
            f"- 省份分类器: `{Path(province_classifier_weights).expanduser().resolve().as_posix()}`",
        ]
    )


def run_single_image(
    image_rgb: np.ndarray | None,
    detector_weights: str,
    recognizer_weights: str,
    device: str,
    imgsz: int,
    conf: float,
    iou: float,
    max_det: int,
    province_refine: bool,
    province_refine_threshold: float,
    province_classifier_weights: str,
) -> tuple[np.ndarray | None, np.ndarray | None, str, pd.DataFrame]:
    if image_rgb is None:
        return None, None, "请先上传图片。", pd.DataFrame()
    try:
        service = SERVICE_MANAGER.get_service(
            detector_weights,
            recognizer_weights,
            device,
            province_refine,
            province_refine_threshold,
            province_classifier_weights,
        )
        pred = service.predict_image(
            _to_bgr(image_rgb),
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            max_det=max_det,
        )
    except Exception as exc:
        return None, None, f"推理失败: {exc}", pd.DataFrame()

    summary = [
        "### 单图结果",
        f"- 检测状态: `{'成功' if pred.detected else '失败'}`",
        f"- 检测置信度: `{pred.det_conf:.4f}`",
        f"- 识别文本: `{pred.plate_text if pred.plate_text else '(空)'}`",
        f"- OCR 置信度: `{pred.ocr_conf:.4f}`",
        f"- 省份纠错是否触发: `{'是' if pred.province_refined else '否'}`",
        f"- 省份候选: `{pred.province_candidate if pred.province_candidate else '(无)'}`",
        f"- 省份纠错置信差值: `{pred.province_conf:.4f}`",
    ]
    if pred.error:
        summary.append(f"- 备注: `{pred.error}`")

    table = pd.DataFrame(
        [
            {
                "detected": pred.detected,
                "det_conf": round(pred.det_conf, 6),
                "plate_text": pred.plate_text,
                "ocr_conf": round(pred.ocr_conf, 6),
                "province_refined": pred.province_refined,
                "province_candidate": pred.province_candidate,
                "province_conf": round(pred.province_conf, 6),
                "bbox_xyxy": None if pred.bbox_xyxy is None else [round(v, 2) for v in pred.bbox_xyxy],
                "error": pred.error,
            }
        ]
    )
    return _to_rgb(pred.annotated_bgr), _to_rgb(pred.plate_crop_bgr), "\n".join(summary), table


def run_video(
    video_input: Any,
    detector_weights: str,
    recognizer_weights: str,
    device: str,
    imgsz: int,
    conf: float,
    iou: float,
    max_det: int,
    frame_step: int,
    province_refine: bool,
    province_refine_threshold: float,
    province_classifier_weights: str,
) -> tuple[str | None, str, str | None, str | None]:
    video_path = _normalize_video_input(video_input)
    if not video_path:
        return None, "请先上传视频。", None, None
    try:
        service = SERVICE_MANAGER.get_service(
            detector_weights,
            recognizer_weights,
            device,
            province_refine,
            province_refine_threshold,
            province_classifier_weights,
        )
        output_dir = ensure_dir(Path("runs/ui/video") / _now_tag())
        summary = service.infer_video(
            video_path=video_path,
            output_dir=output_dir,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            max_det=max_det,
            frame_step=frame_step,
        )
    except Exception as exc:
        return None, f"视频推理失败: {exc}", None, None

    md = "\n".join(
        [
            "### 视频结果",
            f"- 总帧数: `{summary['num_frames']}`",
            f"- 检测到车牌帧数: `{summary['num_detected']}`",
            f"- 检测率: `{summary['detect_rate']:.4f}`",
            f"- 输出视频: `{summary['video_out']}`",
        ]
    )
    return summary["video_out"], md, summary["csv_path"], summary["jsonl_path"]


def run_batch(
    directory_path: str,
    detector_weights: str,
    recognizer_weights: str,
    device: str,
    imgsz: int,
    conf: float,
    iou: float,
    max_det: int,
    preview_count: int,
    province_refine: bool,
    province_refine_threshold: float,
    province_classifier_weights: str,
) -> tuple[list[str], str, pd.DataFrame, str | None, str | None]:
    if not directory_path.strip():
        return [], "请填写待处理目录路径。", pd.DataFrame(), None, None
    try:
        source_dir = _safe_path(directory_path)
        service = SERVICE_MANAGER.get_service(
            detector_weights,
            recognizer_weights,
            device,
            province_refine,
            province_refine_threshold,
            province_classifier_weights,
        )
        output_dir = ensure_dir(Path("runs/ui/batch") / _now_tag())
        summary = service.infer_directory(
            source_dir=source_dir,
            output_dir=output_dir,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            max_det=max_det,
        )
    except Exception as exc:
        return [], f"批量推理失败: {exc}", pd.DataFrame(), None, None

    vis_dir = Path(summary["vis_dir"])
    preview = sorted(str(p) for p in vis_dir.glob("*") if p.is_file())[: max(1, preview_count)]
    df = pd.DataFrame.from_records(summary["records"])
    md = "\n".join(
        [
            "### 批量结果",
            f"- 总图片数: `{summary['num_images']}`",
            f"- 检测成功数: `{summary['num_detected']}`",
            f"- 检测率: `{summary['detect_rate']:.4f}`",
            f"- 可视化目录: `{summary['vis_dir']}`",
        ]
    )
    return preview, md, df.head(500), summary["csv_path"], summary["jsonl_path"]


def build_demo(args: argparse.Namespace) -> gr.Blocks:
    with gr.Blocks(title=APP_TITLE) as demo:
        gr.HTML(HERO_HTML)
        with gr.Row():
            with gr.Column(scale=3, elem_classes=["glass"]):
                with gr.Tab("单图验证"):
                    in_image = gr.Image(label="上传图片", type="numpy")
                    run_image_btn = gr.Button("开始检测", variant="primary")
                    out_image = gr.Image(label="检测可视化", type="numpy")
                    out_crop = gr.Image(label="透视矫正车牌", type="numpy")
                    out_single_md = gr.Markdown()
                    out_single_df = gr.Dataframe(label="结构化结果", interactive=False)

                with gr.Tab("视频验证"):
                    in_video = gr.Video(label="上传视频", format="mp4")
                    frame_step = gr.Slider(label="抽帧步长", minimum=1, maximum=10, value=1, step=1)
                    run_video_btn = gr.Button("开始处理视频", variant="primary")
                    out_video = gr.Video(label="输出视频", format="mp4")
                    out_video_md = gr.Markdown()
                    out_video_csv = gr.File(label="导出 CSV")
                    out_video_jsonl = gr.File(label="导出 JSONL")

                with gr.Tab("批量验证"):
                    in_dir = gr.Textbox(
                        label="图片目录路径",
                        value="F:\\ccpd_b",
                        placeholder="例如: F:\\ccpd_b",
                    )
                    preview_count = gr.Slider(label="预览图数量", minimum=4, maximum=40, value=16, step=2)
                    run_batch_btn = gr.Button("开始批量推理", variant="primary")
                    out_gallery = gr.Gallery(label="可视化预览", columns=4, height=420)
                    out_batch_md = gr.Markdown()
                    out_batch_df = gr.Dataframe(label="批量结果表", interactive=False)
                    out_batch_csv = gr.File(label="导出 CSV")
                    out_batch_jsonl = gr.File(label="导出 JSONL")

            with gr.Column(scale=2, elem_classes=["glass"]):
                gr.Markdown("### 模型与参数")
                detector_weights = gr.Textbox(
                    label="检测模型权重路径",
                    value=args.detector_weights,
                    placeholder="例如: runs/pose/detector_demo/yolov8n_pose_demo/weights/best.pt",
                )
                recognizer_weights = gr.Textbox(
                    label="识别模型权重路径",
                    value=args.recognizer_weights,
                    placeholder="例如: runs/recognizer/crnn_ctc/best.pt",
                )
                device = gr.Textbox(label="推理设备", value=args.device, placeholder="cuda:0 或 cpu")
                imgsz = gr.Slider(label="检测分辨率", minimum=320, maximum=1280, value=640, step=32)
                conf = gr.Slider(label="检测阈值 conf", minimum=0.05, maximum=0.95, value=0.25, step=0.01)
                iou = gr.Slider(label="NMS 阈值 iou", minimum=0.10, maximum=0.95, value=0.60, step=0.01)
                max_det = gr.Slider(label="最大检测数量", minimum=1, maximum=6, value=1, step=1)
                province_refine = gr.Checkbox(label="启用省份位纠错", value=args.province_refine)
                province_refine_threshold = gr.Slider(
                    label="省份纠错阈值（越大越保守）",
                    minimum=0.05,
                    maximum=0.60,
                    value=args.province_refine_threshold,
                    step=0.005,
                )
                province_classifier_weights = gr.Textbox(
                    label="省份分类器权重路径",
                    value=args.province_classifier_weights,
                    placeholder="例如: runs/province_classifier/best.pt",
                )
                check_btn = gr.Button("检查模型状态")
                status_md = gr.Markdown()
                gr.Markdown(
                    "建议：非皖车牌多时保持“省份位纠错”开启。阈值可在 `0.20~0.35` 调节；偏低更激进，偏高更保守。"
                )

        check_btn.click(
            fn=check_models,
            inputs=[
                detector_weights,
                recognizer_weights,
                device,
                province_refine,
                province_refine_threshold,
                province_classifier_weights,
            ],
            outputs=[status_md],
        )
        run_image_btn.click(
            fn=run_single_image,
            inputs=[
                in_image,
                detector_weights,
                recognizer_weights,
                device,
                imgsz,
                conf,
                iou,
                max_det,
                province_refine,
                province_refine_threshold,
                province_classifier_weights,
            ],
            outputs=[out_image, out_crop, out_single_md, out_single_df],
        )
        run_video_btn.click(
            fn=run_video,
            inputs=[
                in_video,
                detector_weights,
                recognizer_weights,
                device,
                imgsz,
                conf,
                iou,
                max_det,
                frame_step,
                province_refine,
                province_refine_threshold,
                province_classifier_weights,
            ],
            outputs=[out_video, out_video_md, out_video_csv, out_video_jsonl],
        )
        run_batch_btn.click(
            fn=run_batch,
            inputs=[
                in_dir,
                detector_weights,
                recognizer_weights,
                device,
                imgsz,
                conf,
                iou,
                max_det,
                preview_count,
                province_refine,
                province_refine_threshold,
                province_classifier_weights,
            ],
            outputs=[out_gallery, out_batch_md, out_batch_df, out_batch_csv, out_batch_jsonl],
        )
    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch CCPD ALPR Gradio UI.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", default=False)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--detector-weights",
        type=str,
        default=Path("runs/pose/detector_demo/yolov8n_pose_demo/weights/best.pt").as_posix(),
    )
    parser.add_argument("--recognizer-weights", type=str, default=DEFAULT_RECOGNIZER.as_posix())
    parser.add_argument("--province-refine", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--province-refine-threshold", type=float, default=0.26)
    parser.add_argument("--province-classifier-weights", type=str, default="runs/province_classifier/best.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    demo = build_demo(args)
    demo.queue(default_concurrency_limit=1).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=True,
        theme=gr.themes.Soft(),
        css=CUSTOM_CSS,
    )


if __name__ == "__main__":
    main()
