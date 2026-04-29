"""HybridNets 추론 — 차선 + 주행가능영역 + 객체 멀티태스크.

저장소: https://github.com/datvuthanh/HybridNets (MIT)

사용 예:
    python src/infer_hybridnets.py \
        --input videos/input/01_clear.mp4 \
        --output videos/output/01_clear_hybridnets.mp4 \
        --metrics results/01_clear_hybridnets.json \
        --weights weights/hybridnets.pth
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
sys.path.insert(0, str(ROOT / "external" / "HybridNets"))

from src.lib.video_io import VideoReader, VideoWriter, fps_meter
from src.lib.overlay import draw_lanes, draw_drivable, stamp_meta, COLORS


def load_model(weights_path: Path, device: str):
    import torch
    from backbone import HybridNetsBackbone  # type: ignore
    from utils.utils import Params  # type: ignore
    from utils.constants import BINARY_MODE, MULTICLASS_MODE, MULTILABEL_MODE  # type: ignore

    params = Params(str(ROOT / "external" / "HybridNets" / "projects" / "bdd100k.yml"))
    weight = torch.load(str(weights_path), map_location=device, weights_only=False)
    state = weight.get("model", weight)
    weight_seg = state["segmentation_head.0.weight"]
    if weight_seg.size(0) == 1:
        seg_mode = BINARY_MODE
    elif params.seg_multilabel:
        seg_mode = MULTILABEL_MODE
    else:
        seg_mode = MULTICLASS_MODE

    model = HybridNetsBackbone(
        num_classes=len(params.obj_list),
        compound_coef=3,
        ratios=eval(params.anchors_ratios),
        scales=eval(params.anchors_scales),
        seg_classes=len(params.seg_list),
        backbone_name=None,
        seg_mode=seg_mode,
    )
    model.load_state_dict(state)
    model = model.to(device).eval()
    return model, params, seg_mode


def _letterbox(img, new_shape=(384, 640), color=(114, 114, 114)):
    h, w = img.shape[:2]
    r = min(new_shape[0]/h, new_shape[1]/w)
    new_unpad = (int(round(w*r)), int(round(h*r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw, dh = dw/2, dh/2
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh-0.1)), int(round(dh+0.1))
    left, right = int(round(dw-0.1)), int(round(dw+0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)


def preprocess(frame: np.ndarray, target=(384, 640)):
    """Resize to (H=384, W=640) with letterbox + ImageNet normalization."""
    import torch
    from torchvision import transforms
    img, ratio, (dw, dh) = _letterbox(frame, target)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = norm(img).unsqueeze(0)
    return tensor, ratio, (dw, dh)


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
    ap.add_argument("--weights", default="weights/hybridnets.pth")
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

    net, params, seg_mode = load_model(weights_path, device)
    print(f"[hybridnets] 모델 로드 완료 device={device} seg_mode={seg_mode}")

    metrics = []
    with VideoReader(args.input) as reader, \
         VideoWriter(args.output, fps=reader.fps, size=(reader.width, reader.height)) as writer:
        for i, frame in enumerate(reader):
            if args.max_frames and i >= args.max_frames:
                break
            with fps_meter() as m:
                tensor, ratio, (dw, dh) = preprocess(frame)
                tensor = tensor.to(device)
                with torch.no_grad():
                    features, regression, classification, anchors, seg = net(tensor)
                # seg [1, num_seg_classes, 384, 640] -> argmax for lane/drivable
                seg = seg[0]
                if seg.shape[0] >= 2:
                    seg_argmax = seg.argmax(0).cpu().numpy()
                    # bdd100k seg_list: ['road', 'lane'] typically. drivable=1, lane=2
                    drivable_mask = (seg_argmax == 1).astype(np.uint8)
                    lane_mask = (seg_argmax == 2).astype(np.uint8) if seg.shape[0] >= 3 else (seg_argmax == 1).astype(np.uint8)
                else:
                    seg_b = (seg[0] > 0.5).cpu().numpy().astype(np.uint8)
                    drivable_mask = seg_b
                    lane_mask = seg_b

                # letterbox 역으로 원본 크기로
                H, W = reader.height, reader.width
                drivable_full = cv2.resize(drivable_mask, (W, H), interpolation=cv2.INTER_NEAREST)
                lane_full = cv2.resize(lane_mask, (W, H), interpolation=cv2.INTER_NEAREST)
                lanes = lane_mask_to_polylines(lane_full)

            instant_fps = 1000.0 / max(m.elapsed_ms, 0.001)
            painted = draw_drivable(frame, drivable_full.astype(bool))
            painted = draw_lanes(painted, lanes, color=COLORS.get("hybridnets", (255, 0, 255)), thickness=4)
            painted = stamp_meta(painted, "HybridNets", fps=instant_fps, lane_count=len(lanes))
            writer.write(painted)
            metrics.append({"frame": i, "infer_ms": m.elapsed_ms, "lanes": len(lanes), "drivable_px": int(drivable_full.sum())})
            if i % 30 == 0:
                print(f"  frame {i:5d}  {m.elapsed_ms:6.1f} ms  lanes={len(lanes)}")

    if args.metrics:
        Path(args.metrics).parent.mkdir(parents=True, exist_ok=True)
        Path(args.metrics).write_text(json.dumps({
            "model": "hybridnets",
            "input": args.input,
            "device": device,
            "frames": len(metrics),
            "avg_infer_ms": sum(x["infer_ms"] for x in metrics) / max(len(metrics), 1),
            "avg_lanes": sum(x["lanes"] for x in metrics) / max(len(metrics), 1),
            "per_frame": metrics,
        }, indent=2))
        print(f"[hybridnets] metrics → {args.metrics}")
    print(f"[hybridnets] 완료. {args.output}")


if __name__ == "__main__":
    main()
