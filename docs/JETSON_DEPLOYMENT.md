# Jetson Nano 배포 가이드 — TwinLiteNet+ Nano

> **목표**: 데스크톱 PyTorch 베이스라인 (RTX 3060 Ti) 으로 측정 끝낸 TwinLiteNet+ Nano 를 Jetson Nano 임베디드에 양자화 + TensorRT 엔진 형태로 배포하고, 동일 영상에서 FPS 를 비교 측정.

## 0. 왜 이 가이드가 필요한가

- **TensorRT 엔진은 GPU 아키텍처 의존**: RTX 3060 Ti (sm_86) 에서 빌드한 .engine 은 Jetson Nano (sm_53) 에서 안 돌아감.
- 따라서 **portable 한 ONNX 만 옮기고, engine 은 Jetson 에서 직접 빌드** 해야 함.
- 이 가이드는 두 머신 분업 파이프라인을 단계별로 정리.

## 1. 전체 흐름

```
Dev PC (RTX 3060 Ti)            Jetson Nano (smhaccp@100.66.149.44)
──────────────────              ─────────────────────────────────────
1. PyTorch baseline 측정        4. git clone (이미 끝)
2. PyTorch → ONNX 변환    ─────→ 5. ONNX → TRT engine (FP16, INT8)
3. ONNX commit + push            6. TRT 추론 + FPS 측정
                                 7. 결과 JSON 커밋 → Dev PC pull
```

## 2. 단계별 실행

### 2.1 Dev PC (RTX 3060 Ti / Win PC) — ONNX export

```bash
# 1) repo 동기화
cd /c/work/dev/07_lane-detection-poc
git pull

# 2) Python 환경 (이미 docker-compose 환경 있음)
docker compose run --rm lane-detection bash
# 또는 호스트 직접:
# pip install onnx onnxsim

# 3) ONNX export
python scripts/export_onnx.py \
    --weights weights/twinlitenetplus/nano.pth \
    --config nano \
    --output weights/twinlitenetplus/nano.onnx \
    --opset 13 \
    --input-shape 1 3 384 640 \
    --simplify

# 결과: ~5MB nano.onnx
ls -la weights/twinlitenetplus/nano.onnx

# 4) git 에 commit (ONNX 는 weights/ gitignore 에서 예외 처리 필요 또는 별도 디렉토리)
#    .gitignore 에 추가:  !weights/twinlitenetplus/*.onnx
git add weights/twinlitenetplus/nano.onnx
git commit -m "feat: nano.onnx export for Jetson TRT pipeline"
git push
```

> 💡 ONNX 파일 (~5MB) 은 git 에 직접 커밋 가능. 100MB 초과면 git LFS 또는 외부 호스팅 (S3, Hugging Face) 권장.

### 2.2 Jetson Nano — engine 빌드 + 추론

```bash
# 1) repo 동기화 (ONNX 받기)
ssh smhaccp@100.66.149.44
cd ~/lane-detection-poc
git pull

# 2) 의존성 (이미 깔린 패키지 확인)
python3 -c "import tensorrt; print(tensorrt.__version__)"   # 8.2.1.x 기대
python3 -c "import pycuda.driver; print('pycuda OK')"
# 안 깔렸으면:
sudo apt install python3-pycuda
pip3 install --user opencv-python==4.5.5.64

# 3) FP16 engine 빌드 (가장 안전, ~10분)
bash scripts/build_trt_engine.sh \
    weights/twinlitenetplus/nano.onnx \
    fp16

# 결과: weights/twinlitenetplus/nano_fp16.engine (~3-5MB)

# 4) FP16 추론 + FPS 측정
python3 src/infer_trt.py \
    --engine weights/twinlitenetplus/nano_fp16.engine \
    --input videos/input/01_clear.mp4 \
    --output videos/output/01_clear_trt_fp16.mp4 \
    --metrics results/01_clear_trt_fp16.json

# 5) INT8 calibration 후 빌드 (정확도 약간 떨어지지만 1.5-2x 빠름)
python3 scripts/calibrate_int8.py \
    --onnx weights/twinlitenetplus/nano.onnx \
    --videos videos/input/01_clear.mp4 videos/input/02_rural.mp4 \
    --num-frames 200 \
    --cache weights/twinlitenetplus/calib_cache.bin \
    --engine-out weights/twinlitenetplus/nano_int8.engine

# 6) INT8 추론 + FPS 측정
python3 src/infer_trt.py \
    --engine weights/twinlitenetplus/nano_int8.engine \
    --input videos/input/01_clear.mp4 \
    --output videos/output/01_clear_trt_int8.mp4 \
    --metrics results/01_clear_trt_int8.json

# 7) 결과 커밋
git add results/*trt*.json
git commit -m "feat: Jetson Nano TRT FP16/INT8 benchmark results"
git push
```

### 2.3 Dev PC — 결과 통합

```bash
git pull
# results/01_clear_trt_*.json 받음
python scripts/compare.py    # 비교 표 갱신
# slides_assets/ 의 차트도 갱신 (수동 또는 자동)
```

## 3. 예상 / 측정값 (TwinLiteNet+ Nano, 01_clear, 600 frames)

| Stage | Hardware | Precision | Engine size | FPS (예상) | 측정 |
|---|---|---|---|---|---|
| Baseline | RTX 3060 Ti | PyTorch FP16 | — (.pth 0.2MB) | 40 | ✅ 끝 |
| Export | Dev PC | ONNX | ~5MB | — | (예정) |
| Jetson PyTorch | Jetson Nano | PyTorch FP16 | — | 5-10 | (skip — install 까다로움) |
| **Jetson TRT FP16** | **Jetson Nano** | **TensorRT FP16** | **~5MB** | **15-25** | (예정) |
| **Jetson TRT INT8** | **Jetson Nano** | **TensorRT INT8** | **~3MB** | **25-40** | (예정) |

## 4. 트러블슈팅

### `trtexec: command not found`
```bash
export PATH=$PATH:/usr/src/tensorrt/bin
```

### `pycuda.driver.LogicError: cuMemAlloc failed: out of memory`
Jetson Nano 메모리 4GB 제한. Chrome 등 닫고:
```bash
sudo systemctl stop nvargus-daemon  # 카메라 데몬 중지 (인퍼런스만 할 때)
```
또는 `--workspace=128` 으로 줄임.

### ONNX opset 호환성
TensorRT 8.2 는 opset 13 까지 안정 지원. 더 높은 opset 쓰면 일부 op 가 fallback 으로 떨어져 FP16/INT8 효과 반감.

### INT8 calibration 후 품질 저하
`--num-frames` 늘림 (200 → 500), 영상 다양성 확보 (clear / rural / rain 균등 샘플).
또는 FP16 만 쓰고 INT8 포기 — 정확도 vs 속도 trade-off.

### Jetson Nano 발열 / throttling
긴 빌드 / 추론 시 nvpmodel 모드 확인:
```bash
sudo nvpmodel -m 0   # MAXN 모드 (10W)
sudo jetson_clocks   # CPU/GPU 클럭 최대 고정
```

## 5. 참고

- TwinLiteNet+: https://github.com/chequanghuy/TwinLiteNetPlus (MIT)
- TensorRT 8.2 docs: https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-822/
- Jetson Zoo (PyTorch wheel): https://elinux.org/Jetson_Zoo
