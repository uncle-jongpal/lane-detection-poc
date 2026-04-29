"""Ultra-Fast-Lane-Detection 추론 — 영상 → 차선 오버레이 영상 + 프레임별 메트릭.

저장소: https://github.com/cfzd/Ultra-Fast-Lane-Detection (MIT)
임베디드 적합성 평가 — FPS 가 CLRNet 대비 3~5배 빠르지만 정확도는 낮음.

사용:
    python src/infer_ultrafast.py \
        --input videos/input/sample.mp4 \
        --output videos/output/sample_ultrafast.mp4 \
        --metrics results/sample_ultrafast.json \
        --weights weights/ufld_culane_18.pth \
        --backbone 18 \
        --dataset culane
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
sys.path.insert(0, str(ROOT / "external" / "Ultra-Fast-Lane-Detection"))

from src.lib.video_io import VideoReader, VideoWriter, fps_meter
from src.lib.overlay import draw_lanes, stamp_meta, COLORS


# Ultra-Fast 의 row anchor 좌표 (정규화). repo 의 data/constant.py 와 동일.
TUSIMPLE_ROW_ANCHOR = [64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128,
                      132, 136, 140, 144, 148, 152, 156, 160, 164, 168, 172, 176, 180, 184, 188,
                      192, 196, 200, 204, 208, 212, 216, 220, 224, 228, 232, 236, 240, 244, 248,
                      252, 256, 260, 264, 268, 272, 276, 280, 284]
CULANE_ROW_ANCHOR = [121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258,
                     267, 277, 287]


def load_model(weights: Path, backbone: int, num_lanes: int, num_gridding: int, num_cls_per_lane: int, device: str):
    try:
        import torch
        from model.model import parsingNet  # type: ignore (Ultra-Fast-Lane-Detection/model/model.py)
    except ImportError as e:
        raise SystemExit(
            "Ultra-Fast-Lane-Detection repo 가 external/ 에 없거나 의존성 미설치.\n"
            f"원인: {e}"
        )
    net = parsingNet(
        pretrained=False,
        backbone=str(backbone),
        cls_dim=(num_gridding + 1, num_cls_per_lane, num_lanes),
        use_aux=False,
    ).to(device)
    state = torch.load(str(weights), map_location=device)
    if "model" in state:
        state = state["model"]
    # repo 가 'module.' prefix 포함 — 제거
    state = {k.replace("module.", ""): v for k, v in state.items()}
    net.load_state_dict(state, strict=False)
    net.eval()
    return net


def preprocess(frame: np.ndarray, target_size=(800, 288)) -> "torch.Tensor":
    """BGR → 정규화된 [1,3,H,W] tensor. 학습 시 입력 사이즈 = (288, 800)."""
    import torch
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)  # (W, H)
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)
    return torch.from_numpy(img).unsqueeze(0)


def postprocess(out, frame_size: tuple[int, int], num_gridding: int, row_anchor: list[int], h_sample: int) -> list[list[tuple[int, int]]]:
    """Ultra-Fast 출력 [num_gridding+1, num_cls_per_lane, num_lanes] → lane 폴리라인."""
    import torch
    H, W = frame_size
    out = out.cpu().numpy() if hasattr(out, "cpu") else out
    out = out[0]  # batch 제거
    # softmax over location bins
    prob = np.exp(out[:-1, :, :])
    prob = prob / prob.sum(axis=0, keepdims=True)
    idx = np.arange(num_gridding) + 1
    loc = (prob * idx[:, None, None]).sum(axis=0)
    out_argmax = np.argmax(out, axis=0)
    loc[out_argmax == num_gridding] = 0  # background bin

    col_sample = np.linspace(0, 800 - 1, num_gridding)
    col_w = col_sample[1] - col_sample[0]

    lanes = []
    num_lanes = loc.shape[1]
    num_cls = loc.shape[0]
    for k in range(num_lanes):
        pts = []
        for i in range(num_cls):
            if loc[i, k] > 0:
                x = int(loc[i, k] * col_w * (W / 800.0)) - 1
                y = int(H * (row_anchor[num_cls - 1 - i] / float(h_sample)))
                pts.append((x, y))
        if len(pts) >= 2:
            lanes.append(pts)
    return lanes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--metrics", default=None)
    ap.add_argument("--weights", default="weights/ufld_culane_18.pth")
    ap.add_argument("--backbone", type=int, default=18, choices=[18, 34])
    ap.add_argument("--dataset", default="culane", choices=["culane", "tusimple"])
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise SystemExit(f"weights 없음: {weights_path}")

    if args.dataset == "culane":
        num_lanes, num_gridding, num_cls, row_anchor, h_sample = 4, 200, 18, CULANE_ROW_ANCHOR, 288
    else:
        num_lanes, num_gridding, num_cls, row_anchor, h_sample = 4, 100, 56, TUSIMPLE_ROW_ANCHOR, 288

    import torch
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    if device != args.device:
        print(f"[warn] CUDA 미사용 — CPU 폴백")

    net = load_model(weights_path, args.backbone, num_lanes, num_gridding, num_cls, device)
    print(f"[ufld] 모델 로드 완료 backbone={args.backbone} dataset={args.dataset} device={device}")

    metrics = []
    with VideoReader(args.input) as reader, \
         VideoWriter(args.output, fps=reader.fps, size=(reader.width, reader.height)) as writer:
        for i, frame in enumerate(reader):
            if args.max_frames and i >= args.max_frames:
                break
            with fps_meter() as m:
                inp = preprocess(frame).to(device)
                with torch.no_grad():
                    out = net(inp)
                lanes = postprocess(out, (reader.height, reader.width), num_gridding, row_anchor, h_sample)
            instant_fps = 1000.0 / max(m.elapsed_ms, 0.001)
            painted = draw_lanes(frame, lanes, color=COLORS["ultrafast"], thickness=4)
            painted = stamp_meta(painted, "Ultra-Fast", fps=instant_fps, lane_count=len(lanes))
            writer.write(painted)
            metrics.append({"frame": i, "infer_ms": m.elapsed_ms, "lanes": len(lanes)})
            if i % 30 == 0:
                print(f"  frame {i:5d}  {m.elapsed_ms:6.1f} ms  lanes={len(lanes)}")

    if args.metrics:
        Path(args.metrics).parent.mkdir(parents=True, exist_ok=True)
        Path(args.metrics).write_text(json.dumps({
            "model": "ultrafast",
            "input": args.input,
            "device": device,
            "frames": len(metrics),
            "avg_infer_ms": sum(x["infer_ms"] for x in metrics) / max(len(metrics), 1),
            "avg_lanes": sum(x["lanes"] for x in metrics) / max(len(metrics), 1),
            "per_frame": metrics,
        }, indent=2))
        print(f"[ufld] metrics → {args.metrics}")
    print(f"[ufld] 완료. {args.output}")


if __name__ == "__main__":
    main()
