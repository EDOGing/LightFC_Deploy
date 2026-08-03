"""Local CPU-only browser demo for LightFC single-object video tracking."""

from __future__ import annotations

import argparse
import csv
from collections import deque
from dataclasses import dataclass, field
import math
from pathlib import Path
import threading
import time
import traceback
from typing import Any, Optional
import uuid
import webbrowser

import cv2
from flask import Flask, Response, jsonify, request, send_file
import numpy as np
import onnxruntime as ort
import torch

from lib.models import LightFC
from lib.test.utils.hann import hann2d
from lib.utils.box_ops import clip_box
from lib.utils.load import load_yaml


ROOT = Path(__file__).resolve().parent
INDEX_HTML = ROOT / "web" / "index.html"
DEFAULT_CHECKPOINT = ROOT / "lightfc_ep0400.pth.tar"
DEFAULT_ONNX = ROOT / "lightfc_ep0400.onnx"
DEFAULT_ONNX_BACKBONE = ROOT / "lightfc_ep0400_backbone.onnx"
DEFAULT_ONNX_TRACKING = ROOT / "lightfc_ep0400_tracking.onnx"
DEFAULT_CONFIG = (
    ROOT
    / "experiments"
    / "lightfc"
    / "mobilnetv2_p_pwcorr_se_scf_sc_iab_sc_adj_concat_repn33_se_conv33_center_wiou.yaml"
)


def sample_target(
    image: np.ndarray, target_box: list[float], search_area_factor: float, output_size: int
) -> tuple[np.ndarray, float]:
    """Extract the square search/template crop used by official LightFC testing."""
    x, y, width, height = target_box
    crop_size = math.ceil(math.sqrt(width * height) * search_area_factor)
    if crop_size < 1:
        raise ValueError("目标框太小，无法生成跟踪区域。")
    x1 = round(x + 0.5 * width - 0.5 * crop_size)
    y1 = round(y + 0.5 * height - 0.5 * crop_size)
    x2, y2 = x1 + crop_size, y1 + crop_size
    left, top = max(0, -x1), max(0, -y1)
    right = max(x2 - image.shape[1] + 1, 0)
    bottom = max(y2 - image.shape[0] + 1, 0)
    crop = image[y1 + top : y2 - bottom, x1 + left : x2 - right]
    crop = cv2.copyMakeBorder(crop, top, bottom, left, right, cv2.BORDER_CONSTANT)
    crop = cv2.resize(crop, (output_size, output_size))
    return crop, output_size / crop_size


@dataclass
class VideoInfo:
    path: Path
    first_frame: Path
    width: int
    height: int
    fps: float
    frame_count: int


@dataclass
class JobInfo:
    video_id: str
    output_dir: Path
    backend: str = "pytorch"
    state: str = "queued"
    current_frame: int = 0
    total_frames: int = 0
    message: str = "等待开始"
    error: Optional[str] = None
    elapsed_seconds: float = 0.0
    tracking_fps: float = 0.0
    output_video: Optional[Path] = None
    output_csv: Optional[Path] = None
    preview_jpeg: Optional[bytes] = None
    preview_frame_index: int = -1
    cancel_event: threading.Event = field(default_factory=threading.Event)

    def public(self) -> dict[str, Any]:
        progress = min(1.0, self.current_frame / self.total_frames) if self.total_frames else 0.0
        eta = None
        if self.state == "running" and self.current_frame > 1 and progress > 0:
            eta = max(0.0, self.elapsed_seconds * (1.0 / progress - 1.0))
        return {
            "state": self.state,
            "backend": self.backend,
            "current_frame": self.current_frame,
            "total_frames": self.total_frames,
            "progress": progress,
            "message": self.message,
            "error": self.error,
            "elapsed_seconds": self.elapsed_seconds,
            "eta_seconds": eta,
            "tracking_fps": self.tracking_fps,
            "preview_frame_index": self.preview_frame_index,
            "video_url": f"/api/result/{self.output_dir.name}/video" if self.output_video else None,
            "csv_url": f"/api/result/{self.output_dir.name}/csv" if self.output_csv else None,
        }


class LightFCCPU:
    """CPU adaptation of the repository's official LightFC test tracker."""

    def __init__(self, checkpoint: Path, config_path: Path):
        if not checkpoint.is_file():
            raise FileNotFoundError(f"找不到模型文件：{checkpoint}")
        if not config_path.is_file():
            raise FileNotFoundError(f"找不到配置文件：{config_path}")

        self.cfg = load_yaml(str(config_path))
        self.network = LightFC(cfg=self.cfg, env_num=None, training=False)
        # This is a local, user-supplied LightFC training checkpoint.  Its
        # metadata contains LightFC Python objects, so weights_only=False is
        # necessary; only the `net` state dict is retained below.
        checkpoint_obj = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
        if not isinstance(checkpoint_obj, dict) or not isinstance(checkpoint_obj.get("net"), dict):
            raise RuntimeError("模型文件中没有 LightFC 的 `net` 权重。")
        self.network.load_state_dict(checkpoint_obj["net"], strict=True)

        for part in (self.network.backbone, self.network.head):
            for module in part.modules():
                if hasattr(module, "switch_to_deploy"):
                    module.switch_to_deploy()

        self.device = torch.device("cpu")
        self.network.to(self.device).eval()
        self.mean = torch.tensor(self.cfg.DATA.MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        self.std = torch.tensor(self.cfg.DATA.STD, dtype=torch.float32).view(1, 3, 1, 1)
        self.template_factor = float(self.cfg.TEST.TEMPLATE_FACTOR)
        self.template_size = int(self.cfg.TEST.TEMPLATE_SIZE)
        self.search_factor = float(self.cfg.TEST.SEARCH_FACTOR)
        self.search_size = int(self.cfg.TEST.SEARCH_SIZE)
        self.feat_size = self.search_size // int(self.cfg.MODEL.BACKBONE.STRIDE)
        self.output_window = hann2d(torch.tensor([self.feat_size, self.feat_size]), centered=True)
        self.state: Optional[list[float]] = None
        self.template_features = None

    def _preprocess(self, patch_rgb: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(np.ascontiguousarray(patch_rgb)).permute(2, 0, 1)
        tensor = tensor.unsqueeze(0).float().div_(255.0)
        return (tensor - self.mean) / self.std

    @torch.inference_mode()
    def initialize(self, frame_bgr: np.ndarray, bbox_xywh: list[float]) -> None:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        patch, _ = sample_target(frame_rgb, bbox_xywh, self.template_factor, self.template_size)
        self.template_features = self.network.forward_backbone(self._preprocess(patch))
        self.state = [float(value) for value in bbox_xywh]

    @torch.inference_mode()
    def track(self, frame_bgr: np.ndarray) -> tuple[list[float], float]:
        if self.state is None or self.template_features is None:
            raise RuntimeError("跟踪器尚未初始化。")
        height, width = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        patch, resize_factor = sample_target(
            frame_rgb, self.state, self.search_factor, self.search_size
        )
        output = self.network.forward_tracking(self.template_features, self._preprocess(patch))
        response = self.output_window * output["score_map"]
        box = self.network.head.cal_bbox(response, output["size_map"], output["offset_map"])
        cx, cy, box_width, box_height = (
            box.view(-1, 4).mean(dim=0) * self.search_size / resize_factor
        ).tolist()
        previous_cx = self.state[0] + 0.5 * self.state[2]
        previous_cy = self.state[1] + 0.5 * self.state[3]
        half_side = 0.5 * self.search_size / resize_factor
        mapped = [
            cx + previous_cx - half_side - 0.5 * box_width,
            cy + previous_cy - half_side - 0.5 * box_height,
            box_width,
            box_height,
        ]
        self.state = [float(value) for value in clip_box(mapped, height, width, margin=2)]
        confidence = float(output["score_map"].max().item())
        return self.state.copy(), confidence


class LightFCONNXCPU:
    """Split ONNX tracker that caches template features after initialization."""

    def __init__(self, backbone_path: Path, tracking_path: Path, config_path: Path):
        if not backbone_path.is_file():
            raise FileNotFoundError(f"找不到 ONNX 模板模型：{backbone_path}")
        if not tracking_path.is_file():
            raise FileNotFoundError(f"找不到 ONNX 跟踪模型：{tracking_path}")
        if not config_path.is_file():
            raise FileNotFoundError(f"找不到配置文件：{config_path}")
        self.cfg = load_yaml(str(config_path))
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.backbone_session = ort.InferenceSession(
            str(backbone_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.tracking_session = ort.InferenceSession(
            str(tracking_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.mean = np.asarray(self.cfg.DATA.MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
        self.std = np.asarray(self.cfg.DATA.STD, dtype=np.float32).reshape(1, 3, 1, 1)
        self.template_factor = float(self.cfg.TEST.TEMPLATE_FACTOR)
        self.template_size = int(self.cfg.TEST.TEMPLATE_SIZE)
        self.search_factor = float(self.cfg.TEST.SEARCH_FACTOR)
        self.search_size = int(self.cfg.TEST.SEARCH_SIZE)
        self.feat_size = self.search_size // int(self.cfg.MODEL.BACKBONE.STRIDE)
        self.output_window = hann2d(
            torch.tensor([self.feat_size, self.feat_size]), centered=True
        ).numpy()
        self.state: Optional[list[float]] = None
        self.template_features: Optional[np.ndarray] = None

    def _preprocess(self, patch_rgb: np.ndarray) -> np.ndarray:
        tensor = np.ascontiguousarray(patch_rgb.transpose(2, 0, 1)[None], dtype=np.float32)
        return (tensor / 255.0 - self.mean) / self.std

    def initialize(self, frame_bgr: np.ndarray, bbox_xywh: list[float]) -> None:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        patch, _ = sample_target(frame_rgb, bbox_xywh, self.template_factor, self.template_size)
        self.template_features = self.backbone_session.run(
            ["template_features"], {"template": self._preprocess(patch)}
        )[0]
        self.state = [float(value) for value in bbox_xywh]

    def track(self, frame_bgr: np.ndarray) -> tuple[list[float], float]:
        if self.state is None or self.template_features is None:
            raise RuntimeError("ONNX 跟踪器尚未初始化。")
        height, width = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        patch, resize_factor = sample_target(
            frame_rgb, self.state, self.search_factor, self.search_size
        )
        score_map, size_map, offset_map = self.tracking_session.run(
            ["score_map", "size_map", "offset_map"],
            {"template_features": self.template_features, "search": self._preprocess(patch)},
        )
        response = score_map * self.output_window
        index = int(np.argmax(response.reshape(-1)))
        index_y, index_x = divmod(index, self.feat_size)
        size = size_map[0, :, index_y, index_x]
        offset = offset_map[0, :, index_y, index_x]
        cx = (index_x + float(offset[0])) / self.feat_size
        cy = (index_y + float(offset[1])) / self.feat_size
        box_width, box_height = float(size[0]), float(size[1])
        cx, cy, box_width, box_height = (
            value * self.search_size / resize_factor
            for value in (cx, cy, box_width, box_height)
        )
        previous_cx = self.state[0] + 0.5 * self.state[2]
        previous_cy = self.state[1] + 0.5 * self.state[3]
        half_side = 0.5 * self.search_size / resize_factor
        mapped = [
            cx + previous_cx - half_side - 0.5 * box_width,
            cy + previous_cy - half_side - 0.5 * box_height,
            box_width,
            box_height,
        ]
        self.state = [float(value) for value in clip_box(mapped, height, width, margin=2)]
        return self.state.copy(), float(score_map.max())


def draw_box(frame: np.ndarray, bbox: list[float], confidence: float) -> None:
    x, y, width, height = bbox
    x1, y1 = int(round(x)), int(round(y))
    x2, y2 = int(round(x + width)), int(round(y + height))
    color = (49, 210, 139)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)
    cv2.putText(
        frame,
        f"LightFC {confidence:.3f}",
        (max(0, x1), max(25, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )


def update_preview(job: JobInfo, frame: np.ndarray, frame_index: int) -> None:
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if ok:
        job.preview_jpeg = encoded.tobytes()
        job.preview_frame_index = frame_index


def create_app(args: Optional[argparse.Namespace] = None) -> Flask:
    """Create the Flask app.

    ``args`` is optional so Flask's CLI can auto-discover and call this factory
    with ``FLASK_APP=web_demo.py``.  Direct script execution still supplies the
    values parsed by :func:`parse_args`.
    """
    if args is None:
        args = argparse.Namespace(
            checkpoint=str(DEFAULT_CHECKPOINT),
            onnx_backbone=str(DEFAULT_ONNX_BACKBONE),
            onnx_tracking=str(DEFAULT_ONNX_TRACKING),
            config=str(DEFAULT_CONFIG),
            work_dir=str(ROOT / "web_outputs"),
            max_upload_mb=4096,
        )
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = args.max_upload_mb * 1024 * 1024
    work_dir = Path(args.work_dir).resolve()
    uploads_dir, results_dir = work_dir / "uploads", work_dir / "results"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    videos: dict[str, VideoInfo] = {}
    jobs: dict[str, JobInfo] = {}
    registry_lock = threading.Lock()
    model_lock = threading.Lock()
    inference_lock = threading.Lock()
    onnx_backbone_path = Path(getattr(args, "onnx_backbone", DEFAULT_ONNX_BACKBONE))
    onnx_tracking_path = Path(getattr(args, "onnx_tracking", DEFAULT_ONNX_TRACKING))
    model_cache: dict[str, object] = {}

    def get_model(backend: str):
        with model_lock:
            if backend not in model_cache:
                if backend == "onnx":
                    model_cache[backend] = LightFCONNXCPU(
                        onnx_backbone_path, onnx_tracking_path, Path(args.config)
                    )
                else:
                    model_cache[backend] = LightFCCPU(Path(args.checkpoint), Path(args.config))
            return model_cache[backend]

    @app.get("/")
    def index():
        return send_file(INDEX_HTML)

    @app.get("/api/health")
    def health():
        return jsonify(
            model="LightFC",
            device="cpu",
            backends={
                "pytorch": {"available": Path(args.checkpoint).is_file(), "path": str(Path(args.checkpoint).resolve())},
                "onnx": {
                    "available": onnx_backbone_path.is_file() and onnx_tracking_path.is_file(),
                    "backbone_path": str(onnx_backbone_path.resolve()),
                    "tracking_path": str(onnx_tracking_path.resolve()),
                },
            },
        )

    @app.post("/api/upload")
    def upload_video():
        uploaded = request.files.get("video")
        if uploaded is None or not uploaded.filename:
            return jsonify(error="请选择一个视频文件。"), 400
        video_id = uuid.uuid4().hex
        suffix = Path(uploaded.filename).suffix.lower() or ".mp4"
        path = uploads_dir / f"{video_id}{suffix}"
        uploaded.save(path)
        cap = cv2.VideoCapture(str(path))
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            path.unlink(missing_ok=True)
            return jsonify(error="无法读取视频首帧，请尝试 MP4、AVI 或 MOV 格式。"), 400
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if not np.isfinite(fps) or fps <= 0:
            fps = 25.0
        first_path = uploads_dir / f"{video_id}.jpg"
        cv2.imwrite(str(first_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        info = VideoInfo(path, first_path, frame.shape[1], frame.shape[0], fps, frame_count)
        with registry_lock:
            videos[video_id] = info
        return jsonify(
            video_id=video_id,
            width=info.width,
            height=info.height,
            fps=info.fps,
            frame_count=info.frame_count,
            first_frame_url=f"/api/video/{video_id}/first-frame",
        )

    @app.get("/api/video/<video_id>/first-frame")
    def first_frame(video_id: str):
        info = videos.get(video_id)
        if info is None:
            return jsonify(error="视频不存在或服务已重启。"), 404
        return send_file(info.first_frame, mimetype="image/jpeg", max_age=0)

    def run_job(job_id: str, bbox: list[float]) -> None:
        job, info = jobs[job_id], videos[jobs[job_id].video_id]
        started = time.perf_counter()
        cap = None
        writer = None
        output_video = job.output_dir / "tracked.mp4"
        output_csv = job.output_dir / "bboxes.csv"
        try:
            job.state = "running"
            job.message = "正在加载 CPU 模型（首次运行会稍久）"
            with inference_lock:
                if job.cancel_event.is_set():
                    job.state, job.message = "cancelled", "任务已停止"
                    return
                tracker = get_model(job.backend)
                cap = cv2.VideoCapture(str(info.path))
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise RuntimeError("跟踪开始时无法读取视频。")
                writer = cv2.VideoWriter(
                    str(output_video),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    info.fps,
                    (info.width, info.height),
                )
                if not writer.isOpened():
                    raise RuntimeError("无法创建输出视频（OpenCV mp4v 编码器不可用）。")

                job.message = "正在提取目标模板"
                tracker.initialize(frame, bbox)
                draw_box(frame, bbox, 1.0)
                writer.write(frame)
                update_preview(job, frame, 0)
                job.current_frame = 1

                with output_csv.open("w", newline="", encoding="utf-8-sig") as csv_file:
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerow(("frame", "x", "y", "width", "height", "confidence"))
                    csv_writer.writerow((0, *bbox, 1.0))
                    frame_index = 1
                    recent_times: deque[float] = deque(maxlen=12)
                    while not job.cancel_event.is_set():
                        frame_started = time.perf_counter()
                        ok, frame = cap.read()
                        if not ok or frame is None:
                            break
                        tracked_bbox, confidence = tracker.track(frame)
                        draw_box(frame, tracked_bbox, confidence)
                        writer.write(frame)
                        update_preview(job, frame, frame_index)
                        csv_writer.writerow((frame_index, *tracked_bbox, confidence))
                        frame_index += 1
                        job.current_frame = frame_index
                        job.elapsed_seconds = time.perf_counter() - started
                        recent_times.append(time.perf_counter() - frame_started)
                        if sum(recent_times) > 0:
                            job.tracking_fps = len(recent_times) / sum(recent_times)
                        job.message = f"CPU 跟踪中：第 {frame_index} 帧"

            # Close media handles before publishing the completed state.  This
            # matters on Windows, where a just-finished file cannot be served
            # while OpenCV still owns its handle.
            cap.release()
            cap = None
            writer.release()
            writer = None
            job.output_video, job.output_csv = output_video, output_csv
            if job.cancel_event.is_set():
                job.state, job.message = "cancelled", "已停止；已处理部分仍可下载"
            else:
                job.state, job.message = "completed", "跟踪完成"
        except Exception as exc:
            job.state = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
            job.message = "跟踪失败"
            traceback.print_exc()
        finally:
            job.elapsed_seconds = time.perf_counter() - started
            if cap is not None:
                cap.release()
            if writer is not None:
                writer.release()

    @app.post("/api/track")
    def start_tracking():
        payload = request.get_json(silent=True) or {}
        backend = str(payload.get("backend", "pytorch")).lower()
        if backend not in {"pytorch", "onnx"}:
            return jsonify(error="推理引擎无效，请选择 PyTorch 或 ONNX。"), 400
        if backend == "onnx":
            missing = [
                str(path)
                for path in (onnx_backbone_path, onnx_tracking_path)
                if not path.is_file()
            ]
            if missing:
                return jsonify(error=f"找不到拆分 ONNX 模型：{', '.join(missing)}"), 400
        elif not Path(args.checkpoint).is_file():
            return jsonify(error=f"找不到 PYTORCH 模型文件：{args.checkpoint}"), 400
        video_id = str(payload.get("video_id", ""))
        info = videos.get(video_id)
        if info is None:
            return jsonify(error="视频不存在，请重新上传。"), 404
        try:
            box = payload["bbox"]
            x, y, width, height = (float(box[k]) for k in ("x", "y", "width", "height"))
        except (KeyError, TypeError, ValueError):
            return jsonify(error="框选坐标无效。"), 400
        if not all(np.isfinite(v) for v in (x, y, width, height)) or width < 2 or height < 2:
            return jsonify(error="目标框太小或坐标无效，请重新框选。"), 400
        x, y = max(0.0, min(x, info.width - 1.0)), max(0.0, min(y, info.height - 1.0))
        width = max(1.0, min(width, info.width - x))
        height = max(1.0, min(height, info.height - y))
        job_id = uuid.uuid4().hex
        output_dir = results_dir / job_id
        output_dir.mkdir(parents=True)
        job = JobInfo(video_id, output_dir, backend=backend, total_frames=info.frame_count)
        with registry_lock:
            jobs[job_id] = job
        threading.Thread(target=run_job, args=(job_id, [x, y, width, height]), daemon=True).start()
        return jsonify(job_id=job_id)

    @app.get("/api/status/<job_id>")
    def job_status(job_id: str):
        job = jobs.get(job_id)
        return jsonify(job.public()) if job else (jsonify(error="任务不存在或服务已重启。"), 404)

    @app.get("/api/preview/<job_id>")
    def job_preview(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            return jsonify(error="任务不存在。"), 404
        try:
            after = int(request.args.get("after", -1))
        except ValueError:
            after = -1
        if job.preview_jpeg is None or job.preview_frame_index <= after:
            return Response(status=204)
        return Response(
            job.preview_jpeg,
            mimetype="image/jpeg",
            headers={"Cache-Control": "no-store, max-age=0", "X-Frame-Index": str(job.preview_frame_index)},
        )

    @app.post("/api/cancel/<job_id>")
    def cancel_job(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            return jsonify(error="任务不存在。"), 404
        job.cancel_event.set()
        return jsonify(ok=True)

    @app.get("/api/result/<job_id>/<kind>")
    def result(job_id: str, kind: str):
        job = jobs.get(job_id)
        if job is None:
            return jsonify(error="任务不存在。"), 404
        path = job.output_video if kind == "video" else job.output_csv if kind == "csv" else None
        if path is None or not path.is_file():
            return jsonify(error="结果文件尚未生成。"), 404
        return send_file(
            path,
            mimetype="video/mp4" if kind == "video" else "text/csv",
            as_attachment=True,
            download_name=path.name,
        )

    @app.errorhandler(413)
    def upload_too_large(_error):
        return jsonify(error=f"视频超过上传限制 {args.max_upload_mb} MB。"), 413

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LightFC CPU 本地网页视频跟踪")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="LightFC checkpoint 路径")
    parser.add_argument("--onnx-backbone", default=str(DEFAULT_ONNX_BACKBONE), help="模板 Backbone ONNX 路径")
    parser.add_argument("--onnx-tracking", default=str(DEFAULT_ONNX_TRACKING), help="逐帧跟踪 ONNX 路径")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="LightFC 实验 YAML 路径")
    parser.add_argument("--work-dir", default=str(ROOT / "web_outputs"), help="上传与结果目录")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址，默认仅本机访问")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--threads", type=int, default=0, help="PyTorch CPU 线程数，0 表示自动")
    parser.add_argument("--max-upload-mb", type=int, default=4096)
    parser.add_argument("--open-browser", action="store_true", help="启动后自动打开浏览器")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    if cli_args.threads > 0:
        torch.set_num_threads(cli_args.threads)
    url = f"http://127.0.0.1:{cli_args.port}"
    print(f"LightFC 网页端：{url}")
    print("设备：CPU（不会调用 CUDA）")
    if cli_args.open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    create_app(cli_args).run(host=cli_args.host, port=cli_args.port, debug=False, threaded=True)
