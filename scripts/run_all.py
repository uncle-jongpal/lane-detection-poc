"""한 영상에 모든 모델 추론을 직렬로 돌림. 결과는 videos/output + results/ 에 저장.

사용:
    python scripts/run_all.py --input videos/input/sample.mp4
    python scripts/run_all.py --inputs videos/input/*.mp4
"""

from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


MODELS = [
    {
        "name": "clrnet",
        "script": "src/infer_clrnet.py",
        "weights": "weights/clrnet_culane_resnet18.pth",
    },
    {
        "name": "ultrafast",
        "script": "src/infer_ultrafast.py",
        "weights": "weights/ufld_culane_18.pth",
    },
    {
        "name": "yolopv2",
        "script": "src/infer_yolopv2.py",
        "weights": "weights/yolopv2.pt",
    },
]


def run_one(input_path: Path, model: dict, max_frames: int, device: str):
    stem = input_path.stem
    out_video = ROOT / "videos" / "output" / f"{stem}_{model['name']}.mp4"
    out_metrics = ROOT / "results" / f"{stem}_{model['name']}.json"
    cmd = [
        sys.executable,
        str(ROOT / model["script"]),
        "--input", str(input_path),
        "--output", str(out_video),
        "--metrics", str(out_metrics),
        "--weights", str(ROOT / model["weights"]),
        "--device", device,
    ]
    if max_frames:
        cmd.extend(["--max-frames", str(max_frames)])
    print(f"\n=== {model['name']} on {input_path.name} ===")
    print(" ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"  [! fail] {model['name']}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="입력 영상 1개")
    ap.add_argument("--inputs", nargs="+", help="입력 영상 여러 개 (--input 보다 우선)")
    ap.add_argument("--max-frames", type=int, default=0, help="모델당 처음 N 프레임만")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--only", nargs="+", default=None, help="특정 모델만 (clrnet|ultrafast|yolopv2)")
    args = ap.parse_args()

    inputs = []
    if args.inputs:
        inputs = [Path(p) for p in args.inputs]
    elif args.input:
        inputs = [Path(args.input)]
    else:
        inputs = sorted((ROOT / "videos" / "input").glob("*.mp4"))
        if not inputs:
            raise SystemExit("입력 영상 없음. videos/input/ 에 .mp4 추가하거나 --input 지정.")

    models = MODELS if not args.only else [m for m in MODELS if m["name"] in args.only]

    for input_path in inputs:
        for model in models:
            run_one(input_path, model, args.max_frames, args.device)

    print(f"\n전체 완료. 다음:\n  python scripts/compare.py")


if __name__ == "__main__":
    main()
