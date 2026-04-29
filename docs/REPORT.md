# Lane Detection 모델 비교 — CCRD 도로 시공 자동화 환경 적합성 분석

작성: 김정석 · 작성일: 2026-04-30 · 대상: (주)충청 (CCRD) AI/IT 개발 파트너 지원 (Saramin rec_idx=53633361)

---

## TL;DR

- 도시 도로 학습 SOTA 모델 **YOLOPv2** 를 한국 일반 dashcam · 시골길 · 도로공사 영상 3종 (각 60초) 에 적용. RTX 3060 Ti 기준 **평균 30 FPS** 를 유지하며 실시간 처리 가능을 확인.
- 그러나 **시골길 (흐려진 차선) 환경에서 평균 검출 차선 수가 정상 환경의 2.3배로 부풀려져** OCR 노이즈성 false positive 가 상당함을 정량화. CCRD AUTONG 의 **갓길 시공 시나리오** 에서는 이 노이즈가 제어 명령에 직접 영향을 줄 수 있음.
- CCRD 가 보유한 **실제 도로변 시공 영상으로 fine-tuning + Camera/IMU/RTK GPS 센서 fusion** 이 cm 정확도 확보의 가장 빠른 경로라고 봄. 임베디드 (Jetson Orin) 적용 시 양자화 + ONNX Runtime 권장.

## 1. 배경

(주)충청은 2019 설립된 도로교통안전시설물 제조·시공 + 소형건설장비 전문 기업이다. 핵심 제품 **AUTONG** 은 차선 인식 자율주행 + 5개 동시 드릴링이 가능한 무인 도로변 작업 장비 — 가드레일·시선유도봉·도로 표지 폴 같은 도로변 인프라를 사람이 도로 위에 들어가지 않고도 30m 원격으로 시공한다.

이 PoC 의 질문 **공개된 SOTA lane detection 모델을 그 환경에 그대로 적용 가능한가? 어떤 보강이 필요한가?**

## 2. 평가 모델

| 모델 | 라이선스 | 비교 위치 | 비고 |
|---|---|---|---|
| **YOLOPv2** ([CAIC-AD/YOLOPv2](https://github.com/CAIC-AD/YOLOPv2)) | GPLv3 ⚠️ | **본 PoC 직접 추론** | 차선 + 주행가능영역 + 객체 멀티태스크. CCRD 의 도로변 객체(가드레일·표지) 검출까지 한 번에 평가 가능 |
| CLRNet ([Turoad/CLRNet](https://github.com/Turoad/CLRNet)) | Apache-2.0 | public benchmark 인용 | CULane F1 79.7 SOTA. mmcv 의존성으로 Windows native 셋업 보류 — Docker 환경에서 추가 예정 (v0.2) |
| Ultra-Fast-Lane-Detection ([cfzd/Ultra-Fast](https://github.com/cfzd/Ultra-Fast-Lane-Detection)) | MIT | public benchmark 인용 | CULane ~150 FPS, 임베디드 적합. weights 미러 이슈로 v0.2 추가 |

**라이선스 메모** YOLOPv2 GPLv3 — 회사 상용 제품에 정적/동적 통합 시 소스 공개 의무. PoC 비교용으로는 OK이며, 실 채택 시 CLRNet (Apache-2.0) 또는 Ultra-Fast (MIT) 로 교체 가능.

## 3. 입력 영상

| ID | 영상 | 환경 | 해상도 | 길이 | 출처 |
|---|---|---|---|---|---|
| 01_normal | Jeju 해안 4K dashcam | 정상 차선, 직선 + 곡선 | 720p | 60s | YouTube `9fcaTnhL0v0` 발췌 (60–120s) |
| 02_rural | 한국 시골길 dashcam | 흐려진 차선, 일부 그림자 | 720p | 60s | YouTube `OFskJ_OREIo` 발췌 (30–90s) |
| 03_construction | 도로공사 dashcam | 공사 표시, 임시 차선 | 720p | 60s | YouTube `pWSR394_GP0` 발췌 (0–60s) |

## 4. 정량 결과 (YOLOPv2, RTX 3060 Ti, 720p, 600 frames each)

| 영상 | 평균 추론 시간 (ms) | 평균 FPS | 평균 검출 차선 수 |
|---|---:|---:|---:|
| 01_normal | 33.19 | **30.1** | 3.21 |
| 02_rural | 37.23 | 26.9 | **7.48** |
| 03_construction | 33.84 | 29.6 | 3.55 |

원시 데이터: [`results/comparison.csv`](../results/comparison.csv), 프레임별 메트릭 [`results/*.json`](../results/)

### 4.1 FPS 분석

- 평균 30 FPS 유지 — RTX 3060 Ti 데스크탑에서 **실시간 (30 fps) 입력에 대한 1:1 처리 가능**. 시골길에서 약간 (~3 FPS) 느린 것은 뒷단 lane mask 후처리 (connected components + polyline fitting) 가 차선 후보 많을 때 더 무거워지기 때문.
- **임베디드 적용 가능성** — Jetson Orin Nano (10W) 기준 RTX 3060 Ti 의 약 1/4 성능 → 약 7~8 FPS 예상. 실시간엔 부족하므로 **TensorRT FP16 양자화 + 입력 해상도 640×384 → 416×256 다운스케일** 시 20+ FPS 달성 가능. AUTONG 의 30m 원격 작업 거리에선 충분.

### 4.2 검출 안정성 — false positive 분석 (가장 중요한 발견)

- 정상 환경 (01_normal) 평균 3.2 lanes / 프레임. 한국 일반 4차선 도로면 차선 2~4개 잡혀야 자연스러움. **3.2 → 약간 노이즈는 있으나 합리적**.
- 시골길 (02_rural) 평균 **7.5 lanes / 프레임 — 정상 대비 2.3배**. 라벨 영역만 봐서는 차선이 그렇게 많을 수 없음. **흐려진 차선/노면 균열/그림자 패턴을 차선으로 오인** 한 것.
- 공사장 (03_construction) 3.6 lanes / 프레임 — 정상에 가까움. 본 클립은 차선 마킹이 비교적 명확한 구간이었기에 큰 차이 없음. **단 임시 콘이나 다른 콘 패턴이 한국 공사장에 추가되면 더 나빠질 가능성**.

### 4.3 시각 (출력 영상의 대표 프레임)

- [`docs/samples/01_normal_yolopv2_t1.jpg`](samples/01_normal_yolopv2_t1.jpg), `_t2.jpg`, `_t3.jpg` (10% / 50% / 85% 시점)
- [`docs/samples/02_rural_yolopv2_*.jpg`](samples/) — 흐려진 차선에서 다중 polyline 가중 가시화
- [`docs/samples/03_construction_yolopv2_*.jpg`](samples/)
- 전체 출력 영상 9개는 `videos/output/` 에 저장 (.gitignore 처리, 재현은 `scripts/run_all.py`)

## 5. CCRD 시공 시나리오 정조준 분석

### 5.1 AUTONG 작업 포지션 = 갓길 / road edge

CCRD 의 핵심 제품인 AUTONG 은 차량처럼 도로 중앙을 달리지 않는다. **가장 바깥쪽 차선 + 갓길 사이** 좁은 영역에서 정밀 위치를 잡고 5개 드릴을 동시에 천공한다. 즉 인식 대상이

1. **가장 바깥쪽 차선** (white solid line) — 자기 차량의 위치 기준점
2. **road edge / 갓길** — 작업 한계
3. **기존 도로 인프라 객체** — 이미 박힌 가드레일 폴, 시선유도봉 (충돌 회피 + 작업 누락 방지)

### 5.2 본 PoC 결과의 시사점

- **YOLOPv2 의 차선 검출** 만으로는 (1) 만 다룸. (2) 갓길은 drivable area mask 의 boundary 로 추출 가능하나 추가 후처리 필요
- **YOLOPv2 의 객체 검출** (현재 미사용 head) 활용 시 (3) 도 일부 가능하나 도로 인프라 객체는 COCO/BDD 학습 데이터에 없음 → fine-tuning 필수
- **시골길 false positive 2.3배** 결과는 한국 비도시 환경에서 학습된 적 없는 모델의 도메인 격차를 정량화함

### 5.3 개선 경로 (가장 빠른 cm 정확도 확보 시나리오)

1. **CCRD 보유 시공 영상 + dashcam 으로 fine-tuning** — 평소 작업하시는 도로변 영상 100개 (각 1분) 으로 lane head 만 fine-tune, learning rate 1e-5 예상 학습 시간 4~6 시간. 시골길 false positive 2.3배 → 1.2배 이하로 수렴 예상
2. **Camera + IMU + RTK GPS 센서 fusion** — Lane detection 만으로 cm 정확도는 불가능. RTK GPS (cm 정확도) 를 base 로 삼고 Camera + IMU 가 short-term drift 를 보정. ROS2 + robot_localization 로 30분 안에 prototype 가능
3. **임베디드 적용** — Jetson Orin Nano (또는 Orin NX) + TensorRT FP16 + 입력 해상도 다운. AUTONG 컨트롤러 무게/전력 제약 충족
4. **도로 인프라 객체 데이터셋 자체 수집** — 가드레일 / 시선유도봉 / 표지판 폴 라벨링 1,000장 정도면 YOLOv8 기반 object head fine-tune 충분. 작업 누락 방지 자동 검수에 활용

## 6. 한계 및 다음 단계

- 본 v0.1 은 **YOLOPv2 단일 모델 + 3 영상** 제한. CLRNet · Ultra-Fast 추가는 v0.2 (Docker 환경 또는 weights 직접 다운 후) 진행
- 영상은 **YouTube 발췌 60초씩** — CCRD 실제 작업 영상이 아님. 회사 보유 영상으로 재실행이 가장 정확한 평가
- **검출 성공/실패 정량 지표 (IoU 기반)** 는 GT 라벨링이 필요해 본 v0.1 엔 미포함. 회사 영상 + GT 라벨 1세트 받으면 즉시 추가 가능

## 7. 결과물 / 재현 방법

```bash
git clone https://github.com/uncle-jongpal/lane-detection-poc.git
cd lane-detection-poc

# 1) 모델 weights
python scripts/download_weights.py --only yolopv2

# 2) 영상 (별도 다운, 본 PoC 과 동일한 3개)
yt-dlp -f "best[height<=720]" --download-sections "*60-120" -o "videos/input/01_normal_%(id)s.%(ext)s" "https://www.youtube.com/watch?v=9fcaTnhL0v0"
# 02_rural 와 03_construction 는 README 참조

# 3) 추론 (RTX 3060 Ti 기준 영상당 ~25초)
python src/infer_yolopv2.py --input videos/input/01_normal_*.mp4 --output videos/output/01_normal_yolopv2.mp4 --metrics results/01_normal_yolopv2.json

# 4) 비교 표 + 샘플 프레임 생성
python build_report.py
```

## 8. 어필 메시지 (1줄 요약)

> "CCRD 가 다루는 도로변 시공 시나리오에서 SOTA 차선 모델의 한계 (시골길 false positive 2.3배, 갓길 인식 불완전, 인프라 객체 미학습) 를 정량화하고, 회사 보유 영상 fine-tuning + Camera/IMU/RTK GPS fusion + Jetson Orin 양자화 의 3단 개선 로드맵을 제시합니다. 본 PoC repo + 출력 영상 + 정량 데이터로 즉시 재현 가능합니다."

작성자 김정석 / vcfdregg3@naver.com / [github.com/uncle-jongpal/lane-detection-poc](https://github.com/uncle-jongpal/lane-detection-poc)
