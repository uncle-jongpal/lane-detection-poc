"""TwinLiteNet+ → ONNX export (Dev PC 에서 실행, Jetson 에서 TensorRT 변환).

ONNX 는 GPU 아키텍처와 무관해서 portable. 같은 .onnx 파일을 Jetson Nano 에서
trtexec 로 변환하면 sm_53 용 engine 이 생성됨.

사용:
    python scripts/export_onnx.py \
        --weights weights/twinlitenetplus/nano.pth \
        --config nano \
        --output weights/twinlitenetplus/nano.onnx \
        --opset 13 \
        --input-shape 1 3 384 640

Notes:
- TensorRT 8.2 (Jetson Nano JetPack 4.6.1) 가 ONNX opset 13 까지 안정 지원.
- dynamic_axes 는 batch 만 가변 (TRT INT8 calibration 위해), H/W 는 고정.
- FP16 변환은 ONNX 단계에선 안 함 — Jetson 에서 trtexec --fp16 로.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external" / "TwinLiteNetPlus"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="PyTorch state_dict (.pth)")
    ap.add_argument("--config", default="nano", choices=["nano", "small", "medium", "large"])
    ap.add_argument("--output", required=True, help="출력 ONNX 경로 (.onnx)")
    ap.add_argument("--opset", type=int, default=13)
    ap.add_argument("--input-shape", nargs=4, type=int, default=[1, 3, 384, 640],
                    help="N C H W (default: 1 3 384 640)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--simplify", action="store_true", help="onnxsim 으로 단순화 (있을 때만)")
    args = ap.parse_args()

    import torch
    from model.model import TwinLiteNetPlus  # type: ignore

    weights_path = Path(args.weights)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu"
    print(f"[export] device={device} config={args.config}")

    model_args = SimpleNamespace(config=args.config, hyp=None)
    model = TwinLiteNetPlus(model_args).to(device).eval()
    state = torch.load(str(weights_path), map_location=device, weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[export] WARN missing keys: {len(missing)}")
    if unexpected:
        print(f"[export] WARN unexpected keys: {len(unexpected)}")

    dummy = torch.randn(*args.input_shape, device=device)
    print(f"[export] input shape={list(dummy.shape)} → exporting opset={args.opset}")

    # output names — TwinLiteNet+ 는 (drivable_area, lane) 두 개 출력
    torch.onnx.export(
        model, dummy, str(output_path),
        input_names=["image"],
        output_names=["drivable", "lane"],
        opset_version=args.opset,
        dynamic_axes={
            "image": {0: "batch"},
            "drivable": {0: "batch"},
            "lane": {0: "batch"},
        },
        do_constant_folding=True,
        export_params=True,
    )
    size_mb = output_path.stat().st_size / (1024*1024)
    print(f"[export] wrote {output_path} ({size_mb:.2f} MB)")

    if args.simplify:
        try:
            import onnx
            from onnxsim import simplify
            m = onnx.load(str(output_path))
            m_simp, ok = simplify(m)
            if ok:
                onnx.save(m_simp, str(output_path))
                size_mb = output_path.stat().st_size / (1024*1024)
                print(f"[export] simplified → {size_mb:.2f} MB")
            else:
                print("[export] simplify check failed, kept original")
        except ImportError:
            print("[export] onnxsim not installed, skipping (pip install onnxsim)")

    # ONNX validity check
    try:
        import onnx
        m = onnx.load(str(output_path))
        onnx.checker.check_model(m)
        print(f"[export] ✓ ONNX 검증 통과 — opset {m.opset_import[0].version}")
    except ImportError:
        print("[export] onnx 미설치 (pip install onnx) — 검증 스킵")
    except Exception as e:
        print(f"[export] ⚠ ONNX 검증 경고: {e}")


if __name__ == "__main__":
    main()
