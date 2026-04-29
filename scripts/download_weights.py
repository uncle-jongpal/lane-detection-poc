"""모델 weights 다운로드. 한 번 실행하면 weights/ 에 모두 저장됨.

CLRNet 과 YOLOPv2 는 Google Drive 호스팅이라 gdown 사용.
Ultra-Fast 도 google drive — 동일.

URL 정확성은 각 repo 의 README 와 동기화. 변경되면 여기 업데이트.

사용:
    python scripts/download_weights.py
    python scripts/download_weights.py --only clrnet      # 특정 모델만
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = ROOT / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

# (이름, 출력 파일명, gdown ID 또는 URL, 메모)
DOWNLOADS = {
    "clrnet": [
        # CLRNet/Turoad README 의 CULane resnet18 (small) checkpoint.
        ("clrnet_culane_resnet18.pth", "https://github.com/Turoad/CLRNet/releases/download/models/clrnet_culane_resnet18.pth",
         "CLRNet CULane R18 — 정확도 vs 속도 절충"),
    ],
    "ultrafast": [
        # cfzd/Ultra-Fast-Lane-Detection - Google Drive
        ("ufld_culane_18.pth", "1zXBRTw50WOzhU0Gej5ZdXCSiK7Fl3JK4",
         "Ultra-Fast CULane ResNet18"),
    ],
    "yolopv2": [
        # CAIC-AD/YOLOPv2 - github release
        ("yolopv2.pt", "https://github.com/CAIC-AD/YOLOPv2/releases/download/V0.0.1/yolopv2.pt",
         "YOLOPv2 jit-traced TorchScript"),
    ],
}


def download(name: str, url_or_id: str, out: Path):
    if out.exists():
        print(f"  [skip] {out.name} (이미 있음)")
        return
    if url_or_id.startswith("http://") or url_or_id.startswith("https://"):
        print(f"  curl → {out.name}")
        subprocess.run(["curl", "-fL", "-o", str(out), url_or_id], check=True)
    else:
        print(f"  gdown {url_or_id} → {out.name}")
        try:
            import gdown
            gdown.download(id=url_or_id, output=str(out), quiet=False)
        except ImportError:
            raise SystemExit("gdown 미설치. `pip install gdown`")
    print(f"  ✓ {out} ({out.stat().st_size/1e6:.1f}MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(DOWNLOADS.keys()), default=None,
                    help="특정 모델만 다운로드")
    args = ap.parse_args()

    targets = [args.only] if args.only else list(DOWNLOADS.keys())
    for model in targets:
        print(f"\n[{model}]")
        for filename, url_or_id, note in DOWNLOADS[model]:
            print(f"  {note}")
            try:
                download(model, url_or_id, WEIGHTS_DIR / filename)
            except subprocess.CalledProcessError as e:
                print(f"  ! 실패: {e}")
                sys.exit(1)
            except Exception as e:
                print(f"  ! 실패: {e}")
                sys.exit(1)

    print(f"\n완료. weights/ 디렉토리:")
    for f in sorted(WEIGHTS_DIR.iterdir()):
        if f.is_file() and not f.name.startswith("."):
            print(f"  {f.name}: {f.stat().st_size/1e6:.1f}MB")


if __name__ == "__main__":
    main()
