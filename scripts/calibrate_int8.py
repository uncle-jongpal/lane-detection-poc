"""INT8 calibration data 준비 + cache 생성 (Jetson 에서 실행).

INT8 양자화는 floating-point activation 의 동적 범위를 학습 데이터 분포로
추정해야 함. trtexec 가 자동으로 추정하긴 하지만, 실제 도메인 영상으로
calibrate 해야 정확도 손실이 작음.

흐름:
1. 입력 영상에서 N 프레임 균등 샘플 → preprocess → .npy 로 저장
2. trt builder 의 IInt8EntropyCalibrator2 구현 → cache 파일 생성
3. cache 를 build_trt_engine.sh 의 --calib 인자로 전달

사용:
    python3 scripts/calibrate_int8.py \\
        --onnx weights/twinlitenetplus/nano.onnx \\
        --videos videos/input/01_clear.mp4 videos/input/02_rural.mp4 \\
        --num-frames 200 \\
        --cache weights/twinlitenetplus/calib_cache.bin

Notes:
- TensorRT 8.2 의 IInt8EntropyCalibrator2 는 cache 를 binary 로 저장.
- 200 프레임이면 10 분 정도 빌드 시간. 50 만으로 시작해도 됨.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import cv2
import numpy as np


def preprocess(frame, target=(384, 640)):
    h, w = frame.shape[:2]
    r = min(target[0]/h, target[1]/w)
    new_unpad = (int(round(w*r)), int(round(h*r)))
    dw, dh = target[1] - new_unpad[0], target[0] - new_unpad[1]
    dw, dh = dw/2, dh/2
    img = cv2.resize(frame, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh-0.1)), int(round(dh+0.1))
    left, right = int(round(dw-0.1)), int(round(dw+0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114,114,114))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)[None]
    return np.ascontiguousarray(img)


def sample_frames(videos, num_frames):
    """여러 영상에서 균등 샘플."""
    samples = []
    per_video = max(1, num_frames // len(videos))
    for v in videos:
        cap = cv2.VideoCapture(str(v))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n < per_video:
            indices = list(range(n))
        else:
            indices = np.linspace(0, n-1, per_video, dtype=int).tolist()
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                samples.append(preprocess(frame))
        cap.release()
        print(f"[calib] {v}: sampled {len(indices)}/{n}")
    return samples[:num_frames]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--videos", nargs="+", required=True, help="calibration 용 영상 (여러 개)")
    ap.add_argument("--num-frames", type=int, default=200)
    ap.add_argument("--cache", required=True, help="저장할 cache 파일 경로")
    ap.add_argument("--engine-out", help="(옵션) calibration 후 INT8 engine 도 같이 빌드해 저장")
    ap.add_argument("--workspace-mb", type=int, default=256)
    args = ap.parse_args()

    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401

    # 1. 샘플 수집
    print(f"[calib] sampling {args.num_frames} frames from {len(args.videos)} videos")
    samples = sample_frames(args.videos, args.num_frames)
    if not samples:
        raise RuntimeError("No samples collected — 영상 경로 / num-frames 확인")
    arr = np.concatenate(samples, axis=0).astype(np.float32)  # (N, 3, 384, 640)
    print(f"[calib] sample tensor: {arr.shape} {arr.dtype} ({arr.nbytes/1024/1024:.1f} MB)")

    # 2. Calibrator 구현
    class FrameCalibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self, frames, cache_path, batch_size=1):
            super().__init__()
            self.frames = frames
            self.cache_path = cache_path
            self.batch_size = batch_size
            self.idx = 0
            self.dev_input = cuda.mem_alloc(frames[0].nbytes)

        def get_batch_size(self):
            return self.batch_size

        def get_batch(self, names):
            if self.idx >= len(self.frames):
                return None
            cuda.memcpy_htod(self.dev_input, self.frames[self.idx])
            self.idx += 1
            if self.idx % 50 == 0:
                print(f"[calib] batch {self.idx}/{len(self.frames)}")
            return [int(self.dev_input)]

        def read_calibration_cache(self):
            p = Path(self.cache_path)
            if p.exists():
                print(f"[calib] reading existing cache: {p}")
                return p.read_bytes()
            return None

        def write_calibration_cache(self, cache):
            Path(self.cache_path).write_bytes(cache)
            print(f"[calib] wrote cache → {self.cache_path}")

    # 각 프레임을 별도 ndarray 로 split (calibrator 가 batch 단위 호출)
    frames = [arr[i:i+1] for i in range(arr.shape[0])]
    calib = FrameCalibrator(frames, args.cache)

    # 3. INT8 engine build (cache 만 만들 거면 engine_out 안 줘도 됨)
    if args.engine_out:
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        config = builder.create_builder_config()
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)   # INT8 + FP16 fallback
        config.int8_calibrator = calib
        config.max_workspace_size = args.workspace_mb * 1024 * 1024

        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        with open(args.onnx, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(parser.get_error(i))
                raise RuntimeError("ONNX 파싱 실패")

        print(f"[calib] building INT8 engine (cache: {args.cache}) → {args.engine_out}")
        engine = builder.build_engine(network, config)
        if engine is None:
            raise RuntimeError("INT8 engine 빌드 실패")
        Path(args.engine_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.engine_out, "wb") as f:
            f.write(engine.serialize())
        size_mb = Path(args.engine_out).stat().st_size / 1024 / 1024
        print(f"[calib] ✓ engine: {args.engine_out} ({size_mb:.1f} MB)")
    else:
        # cache 만 만들기 — dummy build 한 번 돌려서 calibrator 호출 트리거
        print("[calib] cache only mode (engine_out 미지정) — dummy build 로 calibration 진행")
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        config = builder.create_builder_config()
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = calib
        config.max_workspace_size = args.workspace_mb * 1024 * 1024
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        with open(args.onnx, "rb") as f:
            parser.parse(f.read())
        builder.build_engine(network, config)   # 결과는 버림 — cache 만 필요

    print("[calib] DONE")


if __name__ == "__main__":
    main()
