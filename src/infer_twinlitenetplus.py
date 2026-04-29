"""TwinLiteNetPlus 추론 — 차선 + 주행가능영역 (lightweight, embedded용).

저장소: https://github.com/chequanghuy/TwinLiteNetPlus (MIT)

사용 예:
    python src/infer_twinlitenetplus.py \
        --input videos/input/01_clear.mp4 \
        --output videos/output/01_clear_twin_nano.mp4 \
        --metrics results/01_clear_twin_nano.json \
        --weights weights/twinlitenetplus/nano.pth \
        --config nano
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external" / "TwinLiteNetPlus"))

from src.lib.video_io import VideoReader, VideoWriter, fps_meter
from src.lib.overlay import draw_lanes, draw_drivable, stamp_meta, COLORS


def load_model(weights_path: Path, config: str, device: str, half: bool = True):
    import torch
    from model.model import TwinLiteNetPlus  # type: ignore

    args = SimpleNamespace(config=config, hyp=None)
    model = TwinLiteNetPlus(args)
    state = torch.load(str(weights_path), map_location=device, weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    model = model.to(device).eval()
    if half and device == "cuda":
        model = model.half()
    return model


def preprocess(frame: np.ndarray, target=(384, 640), half=True):
    """BGR → 정규화 [1,3,H,W] tensor. Letterbox padded."""
    import torch
    h, w = frame.shape[:2]
    r = min(target[0]/h, target[1]/w)
    new_unpad = (int(round(w*r)), int(round(h*r)))
    dw, dh = target[1] - new_unpad[0], target[0] - new_unpad[1]
    dw, dh = dw/2, dh/2
    img = cv2.resize(frame, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh-0.1)), int(round(dh+0.1))
    left, right = int(round(dw-0.1)), int(round(dw+0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114,114,114))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    tensor = torch.from_numpy(img).unsqueeze(0)
    if half:
        tensor = tensor.half()
    return tensor


def lane_mask_to_polylines(lane_mask: np.ndarray) -> list[list[tuple[int, int]]]:
    H, W = lane_mask.shape
    n_components, labels = cv2.connectedComponents(lane_mask.astype(np.uint8))
    lanes = []
    for k in range(1, n_components):
        ys, xs = np.where(labels == k)
        if len(ys) < 50:
            continue
        ymin, ymax = ys.min(), ys.max()
        pts = []
        for y in range(ymin, ymax + 1, 5):
            row = xs[ys == y]
            if len(row):
                pts.append((int(row.mean()), int(y)))
        if len(pts) >= 2:
            lanes.append(pts)
    return lanes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--metrics", default=None)
    ap.add_argument("--weights", default="weights/twinlitenetplus/nano.pth")
    ap.add_argument("--config", default="nano", choices=["nano","small","medium","large"])
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise SystemExit(f"weights 없음: {weights_path}")

    import torch
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    half = device == "cuda"
    if device != args.device:
        print(f"[warn] CUDA 미사용 — CPU 폴백")

    net = load_model(weights_path, args.config, device, half=half)
    print(f"[twin+] 모델 로드 완료 config={args.config} device={device} half={half}")

    metrics = []
    with VideoReader(args.input) as reader, \
         VideoWriter(args.output, fps=reader.fps, size=(reader.width, reader.height)) as writer:
        for i, frame in enumerate(reader):
            if args.max_frames and i >= args.max_frames:
                break
            with fps_meter() as m:
                tensor = preprocess(frame, half=half).to(device)
                with torch.no_grad():
                    out = net(tensor)
                # out: (drivable, lane) each [1, 2, H, W]
                if isinstance(out, (list, tuple)):
                    drivable_logit, lane_logit = out[0], out[1]
                else:
                    drivable_logit = out
                    lane_logit = out
                drivable_mask = drivable_logit.argmax(1)[0].cpu().numpy().astype(np.uint8)
                lane_mask = lane_logit.argmax(1)[0].cpu().numpy().astype(np.uint8)

                H, W = reader.height, reader.width
                drivable_full = cv2.resize(drivable_mask, (W, H), interpolation=cv2.INTER_NEAREST)
                lane_full = cv2.resize(lane_mask, (W, H), interpolation=cv2.INTER_NEAREST)
                lanes = lane_mask_to_polylines(lane_full)

            instant_fps = 1000.0 / max(m.elapsed_ms, 0.001)
            painted = draw_drivable(frame, drivable_full.astype(bool))
            painted = draw_lanes(painted, lanes, color=COLORS.get("twin", (0, 255, 255)), thickness=4)
            painted = stamp_meta(painted, f"TwinLiteNet+ {args.config}", fps=instant_fps, lane_count=len(lanes))
            writer.write(painted)
            metrics.append({"frame": i, "infer_ms": m.elapsed_ms, "lanes": len(lanes), "drivable_px": int(drivable_full.sum())})
            if i % 30 == 0:
                print(f"  frame {i:5d}  {m.elapsed_ms:6.1f} ms  lanes={len(lanes)}")

    if args.metrics:
        Path(args.metrics).parent.mkdir(parents=True, exist_ok=True)
        Path(args.metrics).write_text(json.dumps({
            "model": f"twinlitenetplus-{args.config}",
            "input": args.input,
            "device": device,
            "half": half,
            "frames": len(metrics),
            "avg_infer_ms": sum(x["infer_ms"] for x in metrics) / max(len(metrics), 1),
            "avg_lanes": sum(x["lanes"] for x in metrics) / max(len(metrics), 1),
            "per_frame": metrics,
        }, indent=2))
        print(f"[twin+] metrics → {args.metrics}")
    print(f"[twin+] 완료. {args.output}")


if __name__ == "__main__":
    main()
