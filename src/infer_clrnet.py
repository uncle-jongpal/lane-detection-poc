"""CLRNet 추론 — 영상 → 차선 오버레이 영상 + 프레임별 메트릭 JSON.

저장소: https://github.com/Turoad/CLRNet (Apache-2.0)
모델 weights: download_weights.py 가 자동으로 받음 (CULane/TuSimple pretrained).

사용:
    python src/infer_clrnet.py \
        --input videos/input/sample.mp4 \
        --output videos/output/sample_clrnet.mp4 \
        --metrics results/sample_clrnet.json \
        --weights weights/clrnet_culane_resnet18.pth

GPU 없으면 --device cpu (속도 1/20 떨어짐).
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external" / "CLRNet"))

from src.lib.video_io import VideoReader, VideoWriter, fps_meter
from src.lib.overlay import draw_lanes, stamp_meta, COLORS


def load_model(weights: Path, config: Path | None, device: str):
    """CLRNet 모델 로드.

    CLRNet 공식 repo 의 inference 흐름:
        from clrnet.models.registry import build_net
        from clrnet.utils.config import Config
        cfg = Config.fromfile(config)
        net = build_net(cfg)
        net.load_state_dict(torch.load(weights)['net'])
        net.eval()

    동적 import — 외부 repo 가 없으면 친절한 에러.
    """
    try:
        import torch
        from clrnet.models.registry import build_net  # type: ignore
        from clrnet.utils.config import Config  # type: ignore
    except ImportError as e:
        raise SystemExit(
            "CLRNet repo 가 external/CLRNet 에 없거나 의존성 미설치.\n"
            "  cd external && git clone https://github.com/Turoad/CLRNet.git\n"
            f"원인: {e}"
        )

    if config is None:
        # repo 의 configs/clrnet/clr_resnet18_culane.py 가 기본
        config = ROOT / "external" / "CLRNet" / "configs" / "clrnet" / "clr_resnet18_culane.py"
    cfg = Config.fromfile(str(config))
    cfg.gpus = 1 if device == "cuda" else 0
    net = build_net(cfg)
    state = torch.load(str(weights), map_location=device)
    if "net" in state:
        state = state["net"]
    net.load_state_dict(state, strict=False)
    net.to(device).eval()
    return net, cfg


def preprocess(frame: np.ndarray, cfg) -> "torch.Tensor":
    """BGR uint8 → CLRNet 입력 텐서 [1, 3, H, W] float32 정규화."""
    import torch
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = cfg.img_h, cfg.img_w
    img = cv2.resize(img, (w, h))
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)
    return torch.from_numpy(img).unsqueeze(0)


def postprocess(output, cfg, frame_size: tuple[int, int]) -> list[list[tuple[int, int]]]:
    """모델 출력 → 원본 frame 좌표계의 lane polyline 리스트.

    CLRNet 출력은 보통 lanes 객체 — 각 lane 의 점들을 normalized 또는 픽셀로 반환.
    repo 의 datasets/process/transforms.py 의 inverse 를 적용해 원본 좌표로.
    """
    H, W = frame_size
    try:
        # heads 의 get_lanes 가 lane 객체 list 반환
        lanes = output["lanes"][0] if isinstance(output, dict) else output
    except Exception:
        return []
    result = []
    for lane in lanes:
        # lane.points 는 [(x, y), ...] in [0, 1] normalized
        try:
            points = lane.to_array(cfg.sample_y) if hasattr(lane, "to_array") else lane
        except Exception:
            continue
        pts = []
        for p in points:
            x, y = float(p[0]), float(p[1])
            if 0 <= x <= 1:
                pts.append((int(x * W), int(y * H)))
            else:
                pts.append((int(x), int(y)))
        if len(pts) >= 2:
            result.append(pts)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="입력 영상 (.mp4)")
    ap.add_argument("--output", required=True, help="출력 영상 (.mp4)")
    ap.add_argument("--metrics", default=None, help="프레임별 메트릭 JSON (옵션)")
    ap.add_argument("--weights", default="weights/clrnet_culane_resnet18.pth")
    ap.add_argument("--config", default=None, help="(선택) CLRNet config .py 경로")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--max-frames", type=int, default=0, help="0=전체, 그 외 처음 N 프레임만")
    args = ap.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise SystemExit(f"weights 없음: {weights_path} (먼저 python scripts/download_weights.py 실행)")

    import torch
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    if device != args.device:
        print(f"[warn] CUDA 미사용 — CPU 폴백")

    net, cfg = load_model(weights_path, Path(args.config) if args.config else None, device)
    print(f"[clrnet] 모델 로드 완료 device={device}")

    metrics = []
    with VideoReader(args.input) as reader, \
         VideoWriter(args.output, fps=reader.fps, size=(reader.width, reader.height)) as writer:
        print(f"[clrnet] 입력 {reader.width}x{reader.height} @ {reader.fps:.1f}fps, {reader.frame_count} 프레임")
        for i, frame in enumerate(reader):
            if args.max_frames and i >= args.max_frames:
                break
            with fps_meter() as m:
                inp = preprocess(frame, cfg).to(device)
                with torch.no_grad():
                    out = net(inp)
                lanes = postprocess(out, cfg, (reader.height, reader.width))
            instant_fps = 1000.0 / max(m.elapsed_ms, 0.001)
            painted = draw_lanes(frame, lanes, color=COLORS["clrnet"], thickness=4)
            painted = stamp_meta(painted, "CLRNet", fps=instant_fps, lane_count=len(lanes))
            writer.write(painted)
            metrics.append({"frame": i, "infer_ms": m.elapsed_ms, "lanes": len(lanes)})
            if i % 30 == 0:
                print(f"  frame {i:5d}  {m.elapsed_ms:6.1f} ms  lanes={len(lanes)}")

    if args.metrics:
        Path(args.metrics).parent.mkdir(parents=True, exist_ok=True)
        Path(args.metrics).write_text(json.dumps({
            "model": "clrnet",
            "input": args.input,
            "device": device,
            "frames": len(metrics),
            "avg_infer_ms": sum(x["infer_ms"] for x in metrics) / max(len(metrics), 1),
            "avg_lanes": sum(x["lanes"] for x in metrics) / max(len(metrics), 1),
            "per_frame": metrics,
        }, indent=2))
        print(f"[clrnet] metrics → {args.metrics}")
    print(f"[clrnet] 완료. {args.output}")


if __name__ == "__main__":
    main()
