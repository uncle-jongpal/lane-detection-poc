"""YOLOPv2 추론 — 차선 + 주행가능영역 + 객체 멀티태스크.

저장소: https://github.com/CAIC-AD/YOLOPv2 (GPLv3 — 상용 시 라이선스 주의)
CCRD 도메인 적합성: 차선 외에 주행가능영역까지 봄 → 도로 가장자리/갓길 인식.

사용:
    python src/infer_yolopv2.py \
        --input videos/input/sample.mp4 \
        --output videos/output/sample_yolopv2.mp4 \
        --metrics results/sample_yolopv2.json \
        --weights weights/yolopv2.pt
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external" / "YOLOPv2"))

from src.lib.video_io import VideoReader, VideoWriter, fps_meter
from src.lib.overlay import draw_lanes, draw_drivable, stamp_meta, COLORS


def load_model(weights: Path, device: str):
    """YOLOPv2 의 weights 는 jit-traced TorchScript (.pt) — torch.jit.load 한 번이면 됨."""
    try:
        import torch
    except ImportError as e:
        raise SystemExit(f"torch 미설치: {e}")
    model = torch.jit.load(str(weights), map_location=device)
    model.eval()
    return model


def letterbox(img: np.ndarray, new_shape=(384, 640), color=(114, 114, 114)):
    """YOLO 표준 letterbox — aspect 보존 + 패딩."""
    h, w = img.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw, dh = dw / 2, dh / 2
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)


def preprocess(frame: np.ndarray):
    import torch
    img, r, pad = letterbox(frame, (384, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    return torch.from_numpy(img).unsqueeze(0), r, pad


def postprocess_lane_mask_to_polylines(lane_mask: np.ndarray, frame_size: tuple[int, int]) -> list[list[tuple[int, int]]]:
    """이진 lane mask → polyline 후처리.

    YOLOPv2 의 lane head 는 segmentation mask 를 반환. 각 connected component 를
    column-by-column 으로 픽셀 평균해 중심선 polyline 생성.
    """
    H, W = frame_size
    if lane_mask.shape != (H, W):
        lane_mask = cv2.resize(lane_mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    n_components, labels = cv2.connectedComponents(lane_mask.astype(np.uint8))
    lanes = []
    for k in range(1, n_components):
        ys, xs = np.where(labels == k)
        if len(ys) < 50:
            continue
        # row 별 x 평균 → 중심선
        ymin, ymax = ys.min(), ys.max()
        pts = []
        for y in range(ymin, ymax + 1, 5):
            row_xs = xs[ys == y]
            if len(row_xs):
                pts.append((int(row_xs.mean()), int(y)))
        if len(pts) >= 2:
            lanes.append(pts)
    return lanes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--metrics", default=None)
    ap.add_argument("--weights", default="weights/yolopv2.pt")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise SystemExit(f"weights 없음: {weights_path}")

    import torch
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    if device != args.device:
        print(f"[warn] CUDA 미사용 — CPU 폴백")

    net = load_model(weights_path, device)
    print(f"[yolopv2] 모델 로드 완료 device={device}")

    metrics = []
    with VideoReader(args.input) as reader, \
         VideoWriter(args.output, fps=reader.fps, size=(reader.width, reader.height)) as writer:
        for i, frame in enumerate(reader):
            if args.max_frames and i >= args.max_frames:
                break
            with fps_meter() as m:
                inp, r, pad = preprocess(frame)
                inp = inp.to(device)
                with torch.no_grad():
                    pred, anchor_grid, seg, ll = net(inp)
                # seg = drivable area mask, ll = lane line mask (둘 다 [1,2,H,W])
                drivable_mask = seg.argmax(1)[0].cpu().numpy()    # 0 bg, 1 drivable
                lane_mask = ll.argmax(1)[0].cpu().numpy()         # 0 bg, 1 lane

                # letterbox 역변환 → 원본 좌표
                H, W = reader.height, reader.width
                drivable_full = cv2.resize(drivable_mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
                lanes = postprocess_lane_mask_to_polylines(lane_mask, (H, W))

            instant_fps = 1000.0 / max(m.elapsed_ms, 0.001)
            painted = draw_drivable(frame, drivable_full.astype(bool))
            painted = draw_lanes(painted, lanes, color=COLORS["yolopv2"], thickness=4)
            painted = stamp_meta(painted, "YOLOPv2", fps=instant_fps, lane_count=len(lanes))
            writer.write(painted)
            metrics.append({"frame": i, "infer_ms": m.elapsed_ms, "lanes": len(lanes), "drivable_px": int(drivable_full.sum())})
            if i % 30 == 0:
                print(f"  frame {i:5d}  {m.elapsed_ms:6.1f} ms  lanes={len(lanes)}")

    if args.metrics:
        Path(args.metrics).parent.mkdir(parents=True, exist_ok=True)
        Path(args.metrics).write_text(json.dumps({
            "model": "yolopv2",
            "input": args.input,
            "device": device,
            "frames": len(metrics),
            "avg_infer_ms": sum(x["infer_ms"] for x in metrics) / max(len(metrics), 1),
            "avg_lanes": sum(x["lanes"] for x in metrics) / max(len(metrics), 1),
            "per_frame": metrics,
        }, indent=2))
        print(f"[yolopv2] metrics → {args.metrics}")
    print(f"[yolopv2] 완료. {args.output}")


if __name__ == "__main__":
    main()
