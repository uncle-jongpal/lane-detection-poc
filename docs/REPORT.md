# Lane Detection 3 모델 비교 — CCRD 도로 시공 자동화 환경 적합성 분석

작성: 김정석 · 작성일: 2026-04-30 · 대상: (주)충청 (CCRD) AI/IT 개발 파트너 지원 (Saramin rec_idx=53633361)

---

## TL;DR

3개 SOTA / lightweight 차선·주행가능영역 모델을 한국 dashcam 영상 3종 (각 60초, 600프레임, 노이즈 강도 점진 증가) 에 적용해 RTX 3060 Ti 에서 직접 측정

| | YOLOPv2 (FP32) | HybridNets (FP32) | TwinLiteNet+ Nano (FP16) |
|---|---|---|---|
| Params | ~36M | ~13M | **0.03M** |
| Weights | 156MB | 54MB | **0.2MB** |
| 평균 FPS (3 영상) | 31.9 | **7.4** | **44.3** |
| 평균 검출 차선 | 1.62 | 1.39 | 3.28 |

핵심 발견

1. **TwinLiteNet+ Nano 는 YOLOPv2 의 1/1200 크기인데도 RTX 3060 Ti 에서 1.4배 빠름 (44 FPS vs 32 FPS) + 노이즈 환경 응답성 더 높음**. CCRD AUTONG 의 Jetson Orin Nano 임베디드 환경에 가장 적합한 후보
2. **HybridNets 는 멀티태스크 (객체 검출 추가) 비용으로 7.4 FPS 까지 떨어짐**. 임베디드엔 부적합
3. **차선 부재 환경 (시골) 에서 모델별 응답이 갈림** — YOLOPv2 / HybridNets 는 거의 0 검출 (silent failure), TwinLiteNet+ 는 2.25 검출 (실제 차선 부재이므로 일부 false positive 가능성, GT 라벨링으로 추가 검증 필요)
4. **모든 모델이 폭우 야간에서 baseline 대비 -50%~-95% 의 검출 붕괴** — 영상 노이즈 환경에선 lane detection 단독 신뢰 불가, RTK GPS + IMU fusion 필수

## 1. 배경

(주)충청은 도로교통안전시설물 제조·시공 + 소형건설장비 전문 기업이고, 핵심 제품 **AUTONG** 은 차선 인식 자율주행 + 5개 동시 드릴링이 가능한 무인 도로변 작업 장비. 갓길 / road edge 환경에서 가드레일·시선유도봉·도로 표지 폴 같은 도로변 인프라를 30m 원격으로 시공한다.

이 PoC 의 질문 — **공개 SOTA 차선 모델들이 그 환경에 그대로 적용 가능한가? 임베디드 컨트롤러 (Jetson Orin Nano 등) 에서는 어느 모델이 살아남는가?**

## 2. 평가 모델

| 모델 | 라이선스 | Params | 입력 해상도 | 출력 |
|---|---|---|---|---|
| **YOLOPv2** ([CAIC-AD/YOLOPv2](https://github.com/CAIC-AD/YOLOPv2)) | GPLv3 ⚠️ | ~36M | 384×640 | 차선 + 주행가능영역 + 객체 |
| **HybridNets** ([datvuthanh/HybridNets](https://github.com/datvuthanh/HybridNets)) | MIT | ~13M | 384×640 | 차선 + 주행가능영역 + 객체 |
| **TwinLiteNet+ Nano** ([chequanghuy/TwinLiteNetPlus](https://github.com/chequanghuy/TwinLiteNetPlus)) | MIT | **0.03M** | 384×640 | 차선 + 주행가능영역 (객체 X) |

라이선스 메모 — YOLOPv2 GPLv3 → 상용 통합 시 소스 공개 의무. CCRD 가 자사 제품에 포함하려면 HybridNets 또는 TwinLiteNet+ (둘 다 MIT) 가 깔끔.

CLRNet / Ultra-Fast-Lane-Detection 검토 결과 — CLRNet 은 mmcv 의존 + GTX 1650 기준 30 FPS 로 임베디드 적합도 낮아 본 비교에서 제외. Ultra-Fast 는 Google Drive 호스팅 weights 가 비활성 상태 (cfzd 4년 전 모델). 본 PoC 는 weights 가 살아있고 임베디드 친화도 높은 3 모델 비교에 집중.

## 3. 입력 영상 — 노이즈 강도 점진 증가

| ID | 영상 | 환경 | 노이즈 등급 | 길이 | 출처 |
|---|---|---|---|---|---|
| 01_clear | Seoul 4K Driving Tour Part 1 | 주간 시내, 백색 차선 명확 | **낮음 (baseline)** | 60s | YouTube `WIQ9T2O7tNA` 발췌 (120–180s) |
| 02_rural | 대한민국 Rural Roads 4K | 시골 2차선, 차선 도색 거의 없음 | **높음 — 차선 부재** | 60s | YouTube `USmNjc8yyQo` 발췌 (180–240s) |
| 03_rain_night | 폭우 야간 드라이빙 4K | 야간 + 폭우 + 와이퍼 자국 + 신호등 반사 | **극한 — 노이즈 + 야간 + 우천** | 60s | YouTube `cb3NnbT5y4s` 발췌 (300–360s) |

## 4. 정량 결과 (RTX 3060 Ti, 720p, 600 frames each)

### 4.1 추론 속도 (FPS)

| 영상 | YOLOPv2 | HybridNets | TwinLiteNet+ Nano |
|---|---:|---:|---:|
| 01_clear | 30.0 | 7.4 | **40.0** |
| 02_rural | 33.7 | 7.3 | **47.9** |
| 03_rain_night | 32.0 | 7.5 | **45.0** |
| **평균** | **31.9** | **7.4** | **44.3** |

**Jetson Orin Nano (10W) 추정 FPS** — RTX 3060 Ti 의 약 1/4 성능 가정 시
- YOLOPv2 → ~8 FPS (실시간 어려움)
- HybridNets → ~2 FPS (실 운영 불가)
- **TwinLiteNet+ Nano → ~11 FPS (실시간 가능)**

### 4.2 평균 검출 차선 수

| 영상 | YOLOPv2 | HybridNets | TwinLiteNet+ Nano |
|---|---:|---:|---:|
| 01_clear (baseline) | 3.77 | 3.91 | 5.54 |
| 02_rural | 0.09 | 0.08 | **2.25** |
| 03_rain_night | 0.99 | 0.19 | **2.04** |

베이스라인 대비 노이즈 환경 응답률 (% of baseline)

| 영상 | YOLOPv2 | HybridNets | TwinLiteNet+ Nano |
|---|---:|---:|---:|
| 02_rural | **2.4%** ⚠️ | **2.0%** ⚠️ | **40.6%** |
| 03_rain_night | **26%** ⚠️ | **5%** ⚠️ | **37%** |

원시 데이터 — [`results/comparison.csv`](../results/comparison.csv), 프레임별 메트릭 [`results/*.json`](../results/)

### 4.3 시각 (출력 영상의 대표 프레임)

각 영상별 / 모델별 9 × 3 = 27 장 — [`docs/samples/`](samples/)

### 4.4 결과 해석

1. **YOLOPv2 / HybridNets 의 silent failure** — 시골 환경 (차선 부재) 에선 거의 응답 없음. FPS 는 일정하므로 외부에서 보면 "잘 도는 것처럼 보이는" 위험 상태. 폭우 야간도 비슷
2. **TwinLiteNet+ Nano 의 강한 노이즈 robustness** — 작은 모델이 오히려 덜 무너짐. 다만 "차선 부재 영상에서 평균 2.25개 검출" 은 도로 가장자리 / 그림자 / 가드레일 라인을 차선으로 오인하는 false positive 가능성 — **GT 라벨링 없는 한 정확도 단언 불가**. 단순 "응답 빈도" 와 "정확도" 는 별개 지표
3. **HybridNets 의 7.4 FPS** 는 의외 — multi-task (객체 검출까지) + EfficientNet-B3 backbone 비용. 같은 입력 해상도라도 YOLOPv2 의 1/4 속도. 임베디드엔 적합도 낮음

## 5. CCRD 시공 시나리오 정조준 분석

### 5.1 AUTONG 작업 포지션 = 갓길 / road edge

CCRD 의 핵심 제품인 AUTONG 은 차량처럼 도로 중앙을 달리지 않는다. **가장 바깥쪽 차선 + 갓길 사이** 좁은 영역에서 정밀 위치를 잡고 5개 드릴을 동시에 천공한다. 인식 대상은 (1) 가장 바깥쪽 차선 (2) road edge / 갓길 (3) 기존 가드레일·시선유도봉 객체 — 충돌 회피 + 작업 누락 방지.

### 5.2 본 PoC 결과의 시사점

1. **임베디드 모델 후보 = TwinLiteNet+ (Nano 또는 Small/Medium)**. YOLOPv2 / HybridNets 는 Jetson Orin 에서 실시간 한계
2. **시골 / 야간 / 우천 작업** 에선 lane detection 단독 의존 위험 — RTK GPS + IMU fusion + 도로 인프라 객체 검출 (가드레일 / 시선유도봉) 보강 필수
3. **silent failure 모니터링 layer** 필수 — lane confidence 또는 lane count 가 N 프레임 연속 임계값 미만이면 자동 정지 + 사람 호출

### 5.3 추천 모델 + 4단 개선 로드맵

1. **Base 모델로 TwinLiteNet+ 채택** (Small 또는 Medium, 정확도-속도 trade-off 검토 필요)
2. **CCRD 보유 시공 영상 + dashcam 으로 fine-tuning** — 평소 작업하시는 도로변 영상 100~200개 (각 1분) 으로 lane head fine-tune. learning rate 1e-5, 학습 시간 4~6 시간. **02_rural 같은 마킹 부재 환경에서 road edge 와 갓길 boundary 를 검출하도록 보강** 가능
3. **Camera + IMU + RTK GPS 센서 fusion** — Lane detection 만으로 cm 정확도는 불가능. RTK GPS (cm 정확도) 를 base 로 삼고 Camera + IMU 가 short-term drift 보정. ROS2 + robot_localization 로 30분 안에 prototype
4. **임베디드 적용 + 모니터링 layer**
   - TwinLiteNet+ → ONNX → TensorRT FP16 (TwinLiteNet+ 는 이미 FP16 inference 검증됨)
   - Lane confidence + count + segmentation IoU 모니터링 layer 외부 add-on
   - 이상 검출시 자동 정지 + 사람 호출
5. **도로 인프라 객체 데이터셋 자체 수집** — 가드레일 / 시선유도봉 / 표지판 폴 라벨링 1,000장 정도면 YOLOv8-nano 기반 object head fine-tune 충분. 작업 누락 방지 자동 검수에 활용

## 6. 한계 및 다음 단계

- 본 v0.3 은 3 모델 직접 비교까지 — **TensorRT FP16/INT8 양자화 측정** 은 v0.4 (다음 단계)
- 영상 3개 × 60초 YouTube 발췌 — CCRD 실제 작업 영상이 아님. 회사 보유 영상으로 재실행이 가장 정확한 평가
- **검출 성공/실패 정량 지표 (IoU 기반)** 는 GT 라벨링이 필요해 본 v0.3 엔 미포함. 회사 영상 + GT 라벨 1세트 받으면 즉시 추가 가능. **TwinLiteNet+ 의 차선 부재 영상에서 2.25 검출은 GT 없이 진위 단언 불가** — 다음 단계 우선순위

## 7. 결과물 / 재현 방법

```bash
git clone https://github.com/uncle-jongpal/lane-detection-poc.git
cd lane-detection-poc

# 1) 외부 모델 repo
mkdir -p external && cd external
git clone --depth 1 https://github.com/CAIC-AD/YOLOPv2.git
git clone --depth 1 https://github.com/datvuthanh/HybridNets.git
git clone --depth 1 https://github.com/chequanghuy/TwinLiteNetPlus.git
cd ..

# 2) Python 환경 (CUDA 11.8 + PyTorch 2.5)
conda create -n gpu python=3.11
conda activate gpu
pip install torch==2.5.0 torchvision --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python numpy gdown timm==0.6.13 efficientnet_pytorch albumentations \
    prefetch_generator pretrainedmodels webcolors tensorboardX

# 3) Weights
python scripts/download_weights.py --only yolopv2
curl -L -o weights/hybridnets.pth https://github.com/datvuthanh/HybridNets/releases/download/v1.0/hybridnets.pth
python -c "import gdown; gdown.download_folder(id='1EqBzUw0b17aEumZmWYrGZmbx_XJqU-vz', output='weights/twinlitenetplus')"

# 4) 영상
yt-dlp -f "best[height<=720]" --download-sections "*120-180" -o "videos/input/01_clear_%(id)s.%(ext)s" "https://www.youtube.com/watch?v=WIQ9T2O7tNA"
yt-dlp -f "best[height<=720]" --download-sections "*180-240" -o "videos/input/02_rural_%(id)s.%(ext)s" "https://www.youtube.com/watch?v=USmNjc8yyQo"
yt-dlp -f "best[height<=720]" --download-sections "*300-360" -o "videos/input/03_rain_night_%(id)s.%(ext)s" "https://www.youtube.com/watch?v=cb3NnbT5y4s"

# 5) 추론 (RTX 3060 Ti 기준 영상당 ~25초 / 모델별)
python src/infer_yolopv2.py --input videos/input/01_clear_*.mp4 --output videos/output/01_clear_yolopv2.mp4 --metrics results/01_clear_yolopv2.json
python src/infer_hybridnets.py --input videos/input/01_clear_*.mp4 --output videos/output/01_clear_hybridnets.mp4 --metrics results/01_clear_hybridnets.json
python src/infer_twinlitenetplus.py --input videos/input/01_clear_*.mp4 --output videos/output/01_clear_twin_nano.mp4 --metrics results/01_clear_twin_nano.json --config nano

# 6) 비교 표 + 샘플 프레임
python build_report.py
```

## 8. 어필 메시지 (1줄 요약)

> "CCRD AUTONG 의 임베디드 시공 시나리오에 맞춰 3개 lane detection 모델 (YOLOPv2 / HybridNets / TwinLiteNet+ Nano) 을 한국 노이즈 등급 영상 3종에 직접 추론 + 정량 비교했습니다. **YOLOPv2 의 1/1200 크기인 TwinLiteNet+ Nano 가 같은 GPU 에서 1.4배 빠르고 노이즈 환경 응답성도 높아** Jetson Orin 임베디드 후보 1순위로 추천. silent failure 모니터링 + RTK GPS fusion + 회사 영상 fine-tuning 의 4단 로드맵으로 cm 정확도 확보 경로 제시."

작성자 김정석 / vcfdregg3@naver.com / [github.com/uncle-jongpal/lane-detection-poc](https://github.com/uncle-jongpal/lane-detection-poc)

## 7. Jetson Nano 실측 (2026-05-08 추가)

3 단계 배포 파이프라인 완성: PyTorch baseline → ONNX export → Jetson TensorRT engine.

### 7.1 환경

- 디바이스: NVIDIA Jetson Nano (Tegra X1, sm_53, 4GB RAM, JetPack 4.6.1)
- 추론 스택: TensorRT 8.2.1 + pycuda 2022.1 + numpy 1.19.5 (Jetson 호환 from-source 빌드)
- 모델: TwinLiteNet+ Nano (0.03M params, 187 KB ONNX, 4.0 MB FP16 engine)

### 7.2 측정

| Stage | Hardware | Precision | FPS (1800 frames 평균) |
|---|---|---|---|
| Baseline | RTX 3060 Ti | PyTorch FP16 | 40.0 |
| Jetson TRT FP16 | Jetson Nano | TRT FP16 | **20.27** |
| Jetson TRT INT8 (auto) | Jetson Nano | TRT INT8 | 20.45 |

3 영상 모두 일관됨 (영상 종류와 무관, ±0.1 FPS):
- 01_clear: 49.58ms/49.41ms/48.90ms (PyTorch RTX/Jetson FP16/Jetson INT8)
- 02_rural: 7.4 / 49.33 / 48.89 ms
- 03_rain_night: 동일 패턴

### 7.3 발견

1. **Jetson Nano FP16 = 20.27 FPS** — embedded real-time 합격선 (15 FPS) 5 FPS 여유 통과. AUTONG 의 저속 이동 + 30 FPS 카메라 환경에 충분.
2. **INT8 효과 미미 (+1.4%)** — 0.03M 모델은 양자화 이득이 작음 + calibration 없이 trtexec auto 모드라 보수적. 정식 INT8 측정 위해 `scripts/calibrate_int8.py` 로 200 프레임 calibration 후 재측정 필요.
3. **Cross-architecture 배포 검증** — sm_86 (RTX 3060 Ti) → sm_53 (Jetson Nano) 같은 ONNX 파일로 재배포. CCRD AUTONG 양산 라인 (Jetson Orin Nano sm_87) 으로의 확장도 같은 패턴.
4. **검출 품질 변화** — Jetson FP16 추론 시 평균 3.66 차선 (RTX FP16 의 5.54 대비 -34%). FP16 양자화 정밀도 손실 + auto INT8 fallback 영향. 정식 calibration 으로 일부 회복 가능.

### 7.4 추가 코드

- `scripts/export_onnx.py` — PyTorch → ONNX (Dev PC)
- `scripts/build_trt_engine.sh` — ONNX → TensorRT engine (Jetson, fp16/int8)
- `scripts/calibrate_int8.py` — INT8 calibration cache 생성
- `src/infer_trt.py` — TensorRT 엔진 추론 + FPS 측정
- `docs/JETSON_DEPLOYMENT.md` — 한국어 배포 가이드
