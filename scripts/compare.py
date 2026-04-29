"""results/*.json 메트릭들을 합쳐 모델 × 영상 비교 표 생성.

산출:
- results/comparison.csv     : 모델/영상별 평균 FPS, 평균 검출 차선, 분포
- results/comparison.md      : 마크다운 표 (REPORT.md 에 paste)
- results/lanes_timeline.png : 프레임 진행에 따른 검출 차선 수 (모델별 line)
- results/inference_ms.png   : 프레임 진행에 따른 추론 시간

사용:
    python scripts/compare.py
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load_all() -> pd.DataFrame:
    rows = []
    for f in sorted(RESULTS.glob("*.json")):
        try:
            d = json.loads(f.read_text())
        except Exception as e:
            print(f"[skip] {f.name}: {e}")
            continue
        rows.append({
            "video": Path(d.get("input", f.stem)).stem,
            "model": d.get("model", f.stem.split("_")[-1]),
            "device": d.get("device", "?"),
            "frames": d.get("frames", 0),
            "avg_infer_ms": d.get("avg_infer_ms", 0.0),
            "avg_lanes": d.get("avg_lanes", 0.0),
            "avg_fps": (1000.0 / d["avg_infer_ms"]) if d.get("avg_infer_ms") else 0.0,
            "_per_frame": d.get("per_frame", []),
        })
    return pd.DataFrame(rows)


def write_csv(df: pd.DataFrame):
    out = RESULTS / "comparison.csv"
    df.drop(columns=["_per_frame"], errors="ignore").to_csv(out, index=False)
    print(f"  CSV → {out}")


def write_md(df: pd.DataFrame):
    out = RESULTS / "comparison.md"
    cols = ["video", "model", "device", "frames", "avg_infer_ms", "avg_fps", "avg_lanes"]
    table = df[cols].copy()
    table["avg_infer_ms"] = table["avg_infer_ms"].round(1)
    table["avg_fps"] = table["avg_fps"].round(1)
    table["avg_lanes"] = table["avg_lanes"].round(2)
    md = table.to_markdown(index=False)
    out.write_text("# 모델 비교 표\n\n" + md + "\n")
    print(f"  MD  → {out}")


def write_charts(df: pd.DataFrame):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [skip] matplotlib 미설치 — 차트 스킵")
        return

    # 프레임별 검출 차선 수 (모델별 line, 영상별 subplot)
    videos = sorted(df["video"].unique())
    fig, axes = plt.subplots(len(videos), 1, figsize=(10, 3.5 * len(videos)), squeeze=False)
    for r, v in enumerate(videos):
        ax = axes[r][0]
        for _, row in df[df["video"] == v].iterrows():
            per = pd.DataFrame(row["_per_frame"])
            if per.empty:
                continue
            ax.plot(per["frame"], per["lanes"], label=row["model"], alpha=0.8)
        ax.set_title(f"{v}: 프레임당 검출된 차선 수")
        ax.set_xlabel("frame")
        ax.set_ylabel("lanes")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    out = RESULTS / "lanes_timeline.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  PNG → {out}")

    # 추론 시간
    fig, axes = plt.subplots(len(videos), 1, figsize=(10, 3.5 * len(videos)), squeeze=False)
    for r, v in enumerate(videos):
        ax = axes[r][0]
        for _, row in df[df["video"] == v].iterrows():
            per = pd.DataFrame(row["_per_frame"])
            if per.empty:
                continue
            ax.plot(per["frame"], per["infer_ms"], label=row["model"], alpha=0.8)
        ax.set_title(f"{v}: 프레임당 추론 시간 (ms)")
        ax.set_xlabel("frame")
        ax.set_ylabel("ms")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    out = RESULTS / "inference_ms.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  PNG → {out}")


def main():
    df = load_all()
    if df.empty:
        print("results/*.json 없음 — 먼저 추론 실행")
        sys.exit(1)
    print(f"메트릭 {len(df)}개 로드:")
    print(df[["video", "model", "frames", "avg_infer_ms", "avg_fps", "avg_lanes"]].to_string(index=False))
    print()
    write_csv(df)
    write_md(df)
    write_charts(df)


if __name__ == "__main__":
    main()
