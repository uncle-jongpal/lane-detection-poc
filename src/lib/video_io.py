"""영상 입력/출력 공통 유틸. infer_*.py 가 모두 사용."""

from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
import time
import cv2
import numpy as np


class VideoReader:
    """OpenCV VideoCapture 래퍼. for-loop 으로 frame yield."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise RuntimeError(f"영상 열기 실패: {self.path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __iter__(self):
        return self

    def __next__(self) -> np.ndarray:
        ok, frame = self.cap.read()
        if not ok:
            raise StopIteration
        return frame

    def release(self):
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()


class VideoWriter:
    """mp4 출력 — H.264 가 호환성 좋지만 OpenCV 내장은 mp4v 폴백."""

    def __init__(self, path: str | Path, fps: float, size: tuple[int, int]):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(self.path, fourcc, fps, size)
        if not self.writer.isOpened():
            raise RuntimeError(f"영상 쓰기 실패: {self.path}")

    def write(self, frame: np.ndarray):
        self.writer.write(frame)

    def release(self):
        self.writer.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()


@contextmanager
def fps_meter():
    """추론 루프 안에서 단일 프레임의 처리 시간 측정 — `with fps_meter() as m: ...; m.elapsed_ms`."""
    class M:
        elapsed_ms: float = 0.0
    m = M()
    t0 = time.perf_counter()
    try:
        yield m
    finally:
        m.elapsed_ms = (time.perf_counter() - t0) * 1000
