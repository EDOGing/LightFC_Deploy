"""Real-time CPU-only LightFC tracking for RTSP cameras."""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
import threading
import time
from typing import Optional
import webbrowser

import cv2
from flask import Flask, Response, jsonify, request, send_file
import numpy as np
import torch

from web_demo import (
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    DEFAULT_ONNX_BACKBONE,
    DEFAULT_ONNX_TRACKING,
    LightFCCPU,
    LightFCONNXCPU,
    draw_box,
)


ROOT = Path(__file__).resolve().parent
INDEX_HTML = ROOT / "web" / "rtsp.html"
DEFAULT_RTSP_URL = "rtsp://192.168.6.116:554/live/av1"
DEFAULT_INT8_ONNX_BACKBONE = ROOT / "quantized" / "lightfc_w8a8_backbone_opset12.onnx"
DEFAULT_INT8_ONNX_TRACKING = ROOT / "quantized" / "lightfc_w8a8_tracking_opset12.onnx"


class RTSPTrackerService:
    """Own the camera reader, LightFC state, and browser MJPEG stream."""

    def __init__(
        self,
        checkpoint: Path,
        onnx_backbone: Path,
        onnx_tracking: Path,
        int8_onnx_backbone: Path,
        int8_onnx_tracking: Path,
        config: Path,
        jpeg_quality: int = 82,
    ):
        self.checkpoint = checkpoint
        self.onnx_backbone = onnx_backbone
        self.onnx_tracking = onnx_tracking
        self.int8_onnx_backbone = int8_onnx_backbone
        self.int8_onnx_tracking = int8_onnx_tracking
        self.config = config
        self.jpeg_quality = int(max(40, min(jpeg_quality, 95)))
        self.lock = threading.RLock()
        self.frame_ready = threading.Condition(self.lock)
        self.model_lock = threading.Lock()
        self.models: dict[str, object] = {}
        self.thread: Optional[threading.Thread] = None
        self.stop_event: Optional[threading.Event] = None
        self.capture: Optional[cv2.VideoCapture] = None
        self.generation = 0
        self.url = ""
        self.state = "disconnected"
        self.message = "尚未连接摄像机"
        self.error: Optional[str] = None
        self.width = 0
        self.height = 0
        self.source_fps = 0.0
        self.processing_fps = 0.0
        self.inference_fps = 0.0
        self.confidence: Optional[float] = None
        self.frame_number = 0
        self.jpeg: Optional[bytes] = None
        self.jpeg_sequence = 0
        self.tracking_requested = False
        self.tracking_active = False
        self.pending_bbox: Optional[list[float]] = None
        self.pending_backend = "pytorch"
        self.backend = "pytorch"
        self.last_bbox: Optional[list[float]] = None

    def _public_locked(self) -> dict:
        return {
            "state": self.state,
            "message": self.message,
            "error": self.error,
            "url": self.url,
            "width": self.width,
            "height": self.height,
            "source_fps": self.source_fps,
            "processing_fps": self.processing_fps,
            "inference_fps": self.inference_fps,
            "confidence": self.confidence,
            "frame_number": self.frame_number,
            "has_frame": self.jpeg is not None,
            "tracking": self.tracking_active,
            "tracking_requested": self.tracking_requested,
            "backend": self.backend,
            "bbox": self.last_bbox,
        }

    def status(self) -> dict:
        with self.lock:
            return self._public_locked()

    def connect(self, url: str) -> None:
        self.disconnect()
        with self.lock:
            self.generation += 1
            generation = self.generation
            stop_event = threading.Event()
            self.stop_event = stop_event
            self.url = url
            self.state = "connecting"
            self.message = "正在连接视频流…"
            self.error = None
            self.width = self.height = 0
            self.source_fps = self.processing_fps = self.inference_fps = 0.0
            self.confidence = None
            self.frame_number = 0
            self.jpeg = None
            self.tracking_requested = False
            self.tracking_active = False
            self.pending_bbox = None
            self.pending_backend = "pytorch"
            self.backend = "pytorch"
            self.last_bbox = None
            self.thread = threading.Thread(
                target=self._worker,
                args=(generation, url, stop_event),
                daemon=True,
                name="lightfc-rtsp-reader",
            )
            self.thread.start()

    def disconnect(self) -> None:
        with self.lock:
            old_thread = self.thread
            old_stop = self.stop_event
            self.generation += 1
            self.thread = None
            self.stop_event = None
            self.tracking_requested = False
            self.tracking_active = False
            self.pending_bbox = None
            self.pending_backend = "pytorch"
            self.last_bbox = None
            self.confidence = None
            self.processing_fps = self.inference_fps = 0.0
            self.state = "disconnected"
            self.message = "已断开摄像机"
            self.error = None
            self.frame_ready.notify_all()
        if old_stop is not None:
            old_stop.set()
        # Reads have a finite timeout. Avoid blocking an HTTP request if a
        # camera/driver takes longer than expected to return from read().
        if old_thread is not None and old_thread is not threading.current_thread():
            old_thread.join(timeout=0.25)

    def start_tracking(self, bbox: list[float], backend: str) -> None:
        with self.lock:
            if self.state not in {"live", "tracking", "initializing"} or self.jpeg is None:
                raise RuntimeError("摄像机尚未产生可用画面。")
            self.pending_bbox = bbox
            self.pending_backend = backend
            self.backend = backend
            self.tracking_requested = True
            self.tracking_active = False
            self.last_bbox = bbox.copy()
            self.confidence = None
            self.processing_fps = self.inference_fps = 0.0
            self.state = "initializing"
            self.message = "正在初始化目标模板…"
            self.error = None

    def cancel_tracking(self) -> None:
        with self.lock:
            self.tracking_requested = False
            self.tracking_active = False
            self.pending_bbox = None
            self.last_bbox = None
            self.confidence = None
            self.processing_fps = self.inference_fps = 0.0
            if self.state not in {"disconnected", "connecting", "reconnecting"}:
                self.state = "live"
                self.message = "跟踪已取消，可重新框选目标"

    def snapshot(self) -> Optional[bytes]:
        with self.lock:
            return self.jpeg

    def mjpeg(self):
        last_sequence = -1
        while True:
            with self.frame_ready:
                self.frame_ready.wait_for(
                    lambda: self.jpeg_sequence != last_sequence or self.state == "disconnected",
                    timeout=5.0,
                )
                jpeg = self.jpeg
                sequence = self.jpeg_sequence
                disconnected = self.state == "disconnected"
            if jpeg is not None and sequence != last_sequence:
                last_sequence = sequence
                yield b"--frame\r\nContent-Type: image/jpeg\r\nCache-Control: no-store\r\n\r\n" + jpeg + b"\r\n"
            elif disconnected:
                break

    @staticmethod
    def _open_capture(url: str) -> cv2.VideoCapture:
        params = [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
            5000,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC,
            3000,
        ]
        capture = cv2.VideoCapture()
        if not capture.open(url, cv2.CAP_FFMPEG, params):
            capture.release()
            return capture
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return capture

    def _set_connection_state(self, generation: int, state: str, message: str, error=None) -> bool:
        with self.lock:
            if generation != self.generation:
                return False
            self.state, self.message, self.error = state, message, error
            return True

    def _get_model(self, backend: str):
        with self.model_lock:
            if backend not in self.models:
                if backend == "onnx":
                    self.models[backend] = LightFCONNXCPU(
                        self.onnx_backbone, self.onnx_tracking, self.config
                    )
                elif backend == "onnx_int8":
                    self.models[backend] = LightFCONNXCPU(
                        self.int8_onnx_backbone, self.int8_onnx_tracking, self.config
                    )
                else:
                    self.models[backend] = LightFCCPU(self.checkpoint, self.config)
            return self.models[backend]

    def _worker(self, generation: int, url: str, stop_event: threading.Event) -> None:
        connected_once = False
        while not stop_event.is_set():
            state = "reconnecting" if connected_once else "connecting"
            message = "视频流中断，正在自动重连…" if connected_once else "正在连接视频流…"
            if not self._set_connection_state(generation, state, message):
                return
            capture = self._open_capture(url)
            with self.lock:
                if generation != self.generation or stop_event.is_set():
                    capture.release()
                    return
                self.capture = capture
            if not capture.isOpened():
                self._set_connection_state(
                    generation,
                    "reconnecting",
                    "连接失败，2 秒后重试…",
                    "无法打开 RTSP 地址，请检查地址、网络、账号和摄像机状态。",
                )
                if stop_event.wait(2.0):
                    return
                continue

            connected_once = True
            source_fps = float(capture.get(cv2.CAP_PROP_FPS))
            with self.lock:
                if generation != self.generation:
                    capture.release()
                    return
                self.source_fps = source_fps if np.isfinite(source_fps) and source_fps > 0 else 0.0
                self.state, self.message, self.error = "live", "实时画面已连接", None

            active = False
            active_backend = "pytorch"
            recent_inference_times: deque[float] = deque(maxlen=20)
            while not stop_event.is_set():
                ok, frame = capture.read()
                if not ok or frame is None:
                    break

                with self.lock:
                    if generation != self.generation:
                        capture.release()
                        return
                    requested = self.tracking_requested
                    new_bbox = self.pending_bbox
                    if new_bbox is not None:
                        self.pending_bbox = None
                        active_backend = self.pending_backend

                confidence = None
                tracked_bbox = None
                inference_elapsed = None
                try:
                    if not requested:
                        active = False
                        recent_inference_times.clear()
                    elif new_bbox is not None:
                        tracker = self._get_model(active_backend)
                        with self.model_lock:
                            tracker.initialize(frame, new_bbox)
                        active = True
                        recent_inference_times.clear()
                        tracked_bbox, confidence = new_bbox, 1.0
                    elif active:
                        tracker = self._get_model(active_backend)
                        with self.model_lock:
                            inference_started = time.perf_counter()
                            tracked_bbox, confidence = tracker.track(frame)
                            inference_elapsed = time.perf_counter() - inference_started
                except Exception as exc:
                    active = False
                    recent_inference_times.clear()
                    with self.lock:
                        if generation == self.generation:
                            self.tracking_requested = False
                            self.error = f"{type(exc).__name__}: {exc}"
                            self.state, self.message = "live", "跟踪失败，可重新框选"

                with self.lock:
                    still_requested = self.tracking_requested and generation == self.generation
                if active and still_requested and tracked_bbox is not None:
                    draw_box(frame, tracked_bbox, float(confidence))
                else:
                    active, tracked_bbox, confidence = False, None, None

                ok, encoded = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
                )
                if inference_elapsed is not None:
                    recent_inference_times.append(inference_elapsed)
                with self.frame_ready:
                    if generation != self.generation:
                        capture.release()
                        return
                    self.width, self.height = frame.shape[1], frame.shape[0]
                    self.frame_number += 1
                    self.tracking_active = active
                    self.last_bbox = tracked_bbox
                    self.confidence = confidence
                    if active:
                        self.state, self.message, self.error = "tracking", "LightFC CPU 实时跟踪中", None
                    elif self.state not in {"connecting", "reconnecting"}:
                        self.state = "live"
                        if not self.error:
                            self.message = "实时画面已连接，请拖框选择目标"
                    inference_seconds = sum(recent_inference_times)
                    if active and inference_seconds > 0:
                        self.inference_fps = len(recent_inference_times) / inference_seconds
                        # Keep the previous API field as a compatibility alias.
                        self.processing_fps = self.inference_fps
                    elif not active:
                        self.inference_fps = self.processing_fps = 0.0
                    if ok:
                        self.jpeg = encoded.tobytes()
                        self.jpeg_sequence += 1
                        self.frame_ready.notify_all()

            capture.release()
            with self.lock:
                if self.capture is capture:
                    self.capture = None
                if generation != self.generation or stop_event.is_set():
                    return
                self.tracking_requested = False
                self.tracking_active = False
                self.pending_bbox = None
                self.pending_backend = "pytorch"
                self.last_bbox = None
                self.confidence = None
                self.processing_fps = self.inference_fps = 0.0
                self.state, self.message = "reconnecting", "视频流中断，正在自动重连…"
            if stop_event.wait(1.0):
                return


def create_app(args: Optional[argparse.Namespace] = None) -> Flask:
    if args is None:
        args = argparse.Namespace(
            checkpoint=str(DEFAULT_CHECKPOINT),
            onnx_backbone=str(DEFAULT_ONNX_BACKBONE),
            onnx_tracking=str(DEFAULT_ONNX_TRACKING),
            int8_onnx_backbone=str(DEFAULT_INT8_ONNX_BACKBONE),
            int8_onnx_tracking=str(DEFAULT_INT8_ONNX_TRACKING),
            config=str(DEFAULT_CONFIG),
            default_url=DEFAULT_RTSP_URL,
            jpeg_quality=82,
        )
    app = Flask(__name__)
    onnx_backbone_path = Path(getattr(args, "onnx_backbone", DEFAULT_ONNX_BACKBONE))
    onnx_tracking_path = Path(getattr(args, "onnx_tracking", DEFAULT_ONNX_TRACKING))
    int8_onnx_backbone_path = Path(getattr(args, "int8_onnx_backbone", DEFAULT_INT8_ONNX_BACKBONE))
    int8_onnx_tracking_path = Path(getattr(args, "int8_onnx_tracking", DEFAULT_INT8_ONNX_TRACKING))
    service = RTSPTrackerService(
        Path(args.checkpoint),
        onnx_backbone_path,
        onnx_tracking_path,
        int8_onnx_backbone_path,
        int8_onnx_tracking_path,
        Path(args.config),
        args.jpeg_quality,
    )
    app.extensions["lightfc_rtsp_service"] = service

    @app.get("/")
    def index():
        return send_file(INDEX_HTML)

    @app.get("/api/config")
    def config():
        return jsonify(
            default_url=args.default_url,
            device="cpu",
            model="LightFC",
            backends={
                "pytorch": Path(args.checkpoint).is_file(),
                "onnx": onnx_backbone_path.is_file() and onnx_tracking_path.is_file(),
                "onnx_int8": int8_onnx_backbone_path.is_file() and int8_onnx_tracking_path.is_file(),
            },
        )

    @app.get("/api/status")
    def status():
        return jsonify(service.status())

    @app.post("/api/connect")
    def connect():
        payload = request.get_json(silent=True) or {}
        url = str(payload.get("url", "")).strip()
        if not url:
            return jsonify(error="RTSP 地址不能为空。"), 400
        if not (url.lower().startswith("rtsp://") or Path(url).is_file()):
            return jsonify(error="请输入以 rtsp:// 开头的地址。"), 400
        service.connect(url)
        return jsonify(ok=True)

    @app.post("/api/disconnect")
    def disconnect():
        service.disconnect()
        return jsonify(ok=True)

    @app.post("/api/track")
    def track():
        payload = request.get_json(silent=True) or {}
        backend = str(payload.get("backend", "pytorch")).lower()
        if backend not in {"pytorch", "onnx", "onnx_int8"}:
            return jsonify(error="推理引擎无效，请选择 PyTorch 或 ONNX。"), 400
        if backend in {"onnx", "onnx_int8"}:
            selected_paths = (
                (onnx_backbone_path, onnx_tracking_path)
                if backend == "onnx"
                else (int8_onnx_backbone_path, int8_onnx_tracking_path)
            )
            missing = [
                str(path)
                for path in selected_paths
                if not path.is_file()
            ]
            if missing:
                return jsonify(error=f"找不到拆分 ONNX 模型：{', '.join(missing)}"), 400
        elif not Path(args.checkpoint).is_file():
            return jsonify(error=f"找不到 PYTORCH 模型文件：{args.checkpoint}"), 400
        try:
            box = payload["bbox"]
            bbox = [float(box[key]) for key in ("x", "y", "width", "height")]
        except (KeyError, TypeError, ValueError):
            return jsonify(error="目标框坐标无效。"), 400
        status_now = service.status()
        if not all(np.isfinite(value) for value in bbox) or bbox[2] < 2 or bbox[3] < 2:
            return jsonify(error="目标框太小或坐标无效。"), 400
        if status_now["width"] and status_now["height"]:
            bbox[0] = max(0.0, min(bbox[0], status_now["width"] - 1.0))
            bbox[1] = max(0.0, min(bbox[1], status_now["height"] - 1.0))
            bbox[2] = max(1.0, min(bbox[2], status_now["width"] - bbox[0]))
            bbox[3] = max(1.0, min(bbox[3], status_now["height"] - bbox[1]))
        try:
            service.start_tracking(bbox, backend)
        except RuntimeError as exc:
            return jsonify(error=str(exc)), 409
        return jsonify(ok=True)

    @app.post("/api/cancel")
    def cancel():
        service.cancel_tracking()
        return jsonify(ok=True)

    @app.get("/api/stream")
    def stream():
        return Response(
            service.mjpeg(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
        )

    @app.get("/api/snapshot")
    def snapshot():
        jpeg = service.snapshot()
        if jpeg is None:
            return jsonify(error="当前没有可保存的画面。"), 404
        return Response(
            jpeg,
            mimetype="image/jpeg",
            headers={"Content-Disposition": 'attachment; filename="lightfc_snapshot.jpg"'},
        )

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LightFC CPU RTSP 实时目标跟踪")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--onnx-backbone", default=str(DEFAULT_ONNX_BACKBONE))
    parser.add_argument("--onnx-tracking", default=str(DEFAULT_ONNX_TRACKING))
    parser.add_argument("--int8-onnx-backbone", default=str(DEFAULT_INT8_ONNX_BACKBONE))
    parser.add_argument("--int8-onnx-tracking", default=str(DEFAULT_INT8_ONNX_TRACKING))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--default-url", default=DEFAULT_RTSP_URL)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--threads", type=int, default=0, help="PyTorch CPU 线程数，0 表示自动")
    parser.add_argument("--jpeg-quality", type=int, default=82)
    parser.add_argument("--open-browser", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    if cli_args.threads > 0:
        torch.set_num_threads(cli_args.threads)
    url = f"http://127.0.0.1:{cli_args.port}"
    print(f"LightFC RTSP 实时跟踪：{url}")
    print(f"默认 RTSP：{cli_args.default_url}")
    print("设备：CPU（不会调用 CUDA）")
    if cli_args.open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    create_app(cli_args).run(host=cli_args.host, port=cli_args.port, debug=False, threaded=True)
