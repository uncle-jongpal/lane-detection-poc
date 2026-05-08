"""TensorRT engine inference (Jetson Nano 용).

PyTorch 와 별도로, 빌드된 TRT engine 으로 추론하고 FPS / latency 측정.
src/infer_twinlitenetplus.py 의 전후처리 (preprocess / postprocess) 와 동일하게
맞춰서 같은 차선 검출 결과를 비교 가능하게 함.

사용:
    python3 src/infer_trt.py \\
        --engine weights/twinlitenetplus/nano_fp16.engine \\
        --input videos/input/01_clear.mp4 \\
        --output videos/output/01_clear_trt_fp16.mp4 \\
        --metrics results/01_clear_trt_fp16.json

Notes:
- TensorRT 8.2 Python API 를 사용 (Jetson L4T R32.7 기본 제공).
- 입력 shape (1, 3, 384, 640) 고정 — export_onnx.py 와 일치 필요.
- 출력 두 개: drivable area + lane mask (TwinLiteNet+ multi-task).
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np


def load_engine(engine_path):
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine


class TRTInference:
    def __init__(self, engine_path: Path, input_shape=(1, 3, 384, 640)):
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401  (CUDA context 초기화)

        self.cuda = cuda
        self.engine = load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.input_shape = input_shape
        self.context.set_binding_shape(0, input_shape)

        # Allocate I/O buffers
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()
        for i in range(self.engine.num_bindings):
            shape = self.context.get_binding_shape(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            size = int(np.prod(shape))
            host_mem = cuda.pagelocked_empty(size, dtype)
            dev_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(dev_mem))
            entry = {"host": host_mem, "device": dev_mem, "shape": shape, "dtype": dtype, "name": self.engine.get_binding_name(i)}
            if self.engine.binding_is_input(i):
                self.inputs.append(entry)
            else:
                self.outputs.append(entry)

    def infer(self, x: np.ndarray):
        """x: (1,3,H,W) float32. returns dict {output_name: np.array}."""
        np.copyto(self.inputs[0]["host"], x.ravel())
        self.cuda.memcpy_htod_async(self.inputs[0]["device"], self.inputs[0]["host"], self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        results = {}
        for out in self.outputs:
            self.cuda.memcpy_dtoh_async(out["host"], out["device"], self.stream)
        self.stream.synchronize()
        for out in self.outputs:
            results[out["name"]] = out["host"].reshape(out["shape"])
        return results


def preprocess(frame: np.ndarray, target=(384, 640)) -> np.ndarray:
    """src/infer_twinlitenetplus.py 의 preprocess 와 동일 letterbox 전처리."""
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
    img = img.transpose(2, 0, 1)[None]
    return np.ascontiguousarray(img)


def postprocess_lane_mask(lane_logits: np.ndarray) -> np.ndarray:
    """TwinLiteNet+ 의 lane head 출력 (1, 2, H, W) → binary mask (H, W)."""
    if lane_logits.ndim == 4 and lane_logits.shape[1] == 2:
        lane = np.argmax(lane_logits[0], axis=0).astype(np.uint8)
    elif lane_logits.ndim == 4 and lane_logits.shape[1] == 1:
        lane = (lane_logits[0, 0] > 0.5).astype(np.uint8)
    else:
        lane = (lane_logits.squeeze() > 0.5).astype(np.uint8)
    return lane


def lane_mask_to_polylines(lane_mask: np.ndarray) -> list:
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
                pts.append((int(np.mean(row)), int(y)))
        if len(pts) >= 3:
            lanes.append(pts)
    return lanes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, help="TensorRT .engine 파일")
    ap.add_argument("--input", required=True, help="입력 mp4")
    ap.add_argument("--output", help="(옵션) 차선 그린 출력 mp4")
    ap.add_argument("--metrics", help="(옵션) 결과 JSON")
    ap.add_argument("--limit", type=int, default=0, help="처리할 max frames (0=전부)")
    ap.add_argument("--warmup", type=int, default=10, help="측정 전 warmup 프레임 수")
    args = ap.parse_args()

    eng = TRTInference(Path(args.engine))
    print(f"[trt] engine 로드 완료. inputs={[i['name'] for i in eng.inputs]} outputs={[o['name'] for o in eng.outputs]}")

    cap = cv2.VideoCapture(args.input)
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps_in, (W, H))

    times_ms = []
    avg_lanes_list = []
    n = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        x = preprocess(frame)
        t0 = time.perf_counter()
        out = eng.infer(x)
        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000

        # warmup 이후만 측정
        if n >= args.warmup:
            times_ms.append(elapsed_ms)

        # output mp4 만드는 경우 lane overlay
        if writer is not None:
            # output 이름 매칭 (TwinLiteNet+ 기준 'lane' 출력)
            lane_logits = out.get("lane") or list(out.values())[-1]
            lane_mask = postprocess_lane_mask(lane_logits)
            # mask 를 원본 해상도로 리사이즈
            lane_mask_full = cv2.resize(lane_mask, (W, H), interpolation=cv2.INTER_NEAREST)
            polys = lane_mask_to_polylines(lane_mask_full)
            avg_lanes_list.append(len(polys))
            painted = frame.copy()
            for pl in polys:
                pts = np.array(pl, dtype=np.int32)
                cv2.polylines(painted, [pts], False, (0, 255, 0), 4)
            cv2.putText(painted, f"TRT {Path(args.engine).stem}  {1000/elapsed_ms:5.1f} fps  lanes={len(polys)}",
                        (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(painted)

        n += 1
        if args.limit and n >= args.limit:
            break
        if n % 60 == 0:
            cur_fps = 1000 / np.mean(times_ms[-60:]) if times_ms else 0
            print(f"[trt] {n}/{n_total} cur_fps={cur_fps:.1f}")

    cap.release()
    if writer is not None:
        writer.release()

    avg_ms = float(np.mean(times_ms)) if times_ms else 0
    fps = 1000 / avg_ms if avg_ms else 0
    avg_lanes = float(np.mean(avg_lanes_list)) if avg_lanes_list else 0
    print(f"[trt] DONE — frames={n} avg={avg_ms:.2f}ms fps={fps:.2f} avg_lanes={avg_lanes:.2f}")

    if args.metrics:
        Path(args.metrics).parent.mkdir(parents=True, exist_ok=True)
        with open(args.metrics, "w") as f:
            json.dump({
                "model": f"trt-{Path(args.engine).stem}",
                "input": str(args.input),
                "device": "jetson-nano",
                "engine": str(args.engine),
                "frames": n,
                "warmup_frames": args.warmup,
                "avg_infer_ms": round(avg_ms, 2),
                "fps": round(fps, 2),
                "avg_lanes": round(avg_lanes, 3),
                "per_frame_ms": [round(t, 3) for t in times_ms[:600]],
            }, f, indent=2)
        print(f"[trt] metrics → {args.metrics}")


if __name__ == "__main__":
    main()
