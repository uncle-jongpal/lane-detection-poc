"""차선 검출 결과를 영상 위에 그리는 시각화 유틸."""

from __future__ import annotations
import cv2
import numpy as np


# 모델별 색상 (BGR) — 비교 영상에서 같은 차선이라도 모델마다 다른 색.
COLORS = {
    "clrnet": (0, 255, 0),       # green
    "ultrafast": (0, 165, 255),  # orange
    "yolopv2": (255, 0, 255),    # magenta
}


def draw_lanes(
    frame: np.ndarray,
    lanes: list[list[tuple[int, int]]],
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 4,
) -> np.ndarray:
    """차선 polyline 을 frame 위에 그려서 새 frame 반환.

    Args:
        frame: BGR uint8 [H, W, 3]
        lanes: 각 lane 은 (x, y) 픽셀 좌표 리스트
        color: BGR
        thickness: 선 두께

    Returns:
        새 frame (원본 미변경).
    """
    out = frame.copy()
    for lane in lanes:
        if len(lane) < 2:
            continue
        pts = np.array(lane, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)
    return out


def draw_drivable(frame: np.ndarray, mask: np.ndarray, color=(0, 200, 0), alpha=0.35) -> np.ndarray:
    """주행가능영역 마스크를 반투명 오버레이 (YOLOPv2 등)."""
    out = frame.copy()
    if mask.dtype != bool:
        mask = mask.astype(bool)
    overlay = out.copy()
    overlay[mask] = color
    return cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)


def stamp_meta(frame: np.ndarray, label: str, fps: float | None = None, lane_count: int | None = None) -> np.ndarray:
    """좌상단에 모델명/FPS/검출된 차선 수 텍스트 박스."""
    out = frame.copy()
    lines = [label]
    if fps is not None:
        lines.append(f"{fps:.1f} FPS")
    if lane_count is not None:
        lines.append(f"lanes={lane_count}")

    pad = 8
    line_h = 26
    width = 220
    height = pad * 2 + line_h * len(lines)

    # 반투명 검정 배경
    overlay = out.copy()
    cv2.rectangle(overlay, (10, 10), (10 + width, 10 + height), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.55, out, 0.45, 0)

    for i, txt in enumerate(lines):
        y = 10 + pad + line_h * (i + 1) - 6
        cv2.putText(out, txt, (10 + pad, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return out
