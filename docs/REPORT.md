# Lane Detection 모델 비교 — CCRD 도로 시공 자동화 환경 적합성 분석

작성: 김정석 · 작성일: 2026-04-30 · 대상: (주)충청 (CCRD) AI/IT 개발 파트너 지원 (Saramin rec_idx=53633361)

---

## TL;DR

도시 도로 학습 SOTA 모델 **YOLOPv2** 를 환경 노이즈 강도가 다른 한국 dashcam 영상 3종 (각 60초, 600프레임) 에 적용

- **명확한 도시 차선** — 평균 30 FPS · 3.77 차선 검출. 정상 동작
- **차선 도색 없는 시골 2차선 도로** — 평균 33.7 FPS · **0.09 차선 검출** — 99% 프레임에서 차선 미검출. **사실상 작동 불능**
- **폭우 야간** — 평균 32 FPS · **0.99 차선 검출** — 정상 대비 -74%. 신호등·반사 불빛만 있는 환경에서도 1개 정도만 간신히 인식

CCRD AUTONG 의 갓길 시공 시나리오는 위 두 노이즈 환경의 합집합 (시골/지방도 + 야간/우천 작업 가능성) 에 정확히 해당. **공개 SOTA 모델만으로는 fallback safety margin 부족**, 회사 보유 시공 영상으로 fine-tuning + Camera/IMU/RTK GPS 센서 fusion + Jetson Orin 양자화 의 3단 보강이 cm 정확도 확보의 가장 빠른 경로라고 봄.

## 1. 배경

(주)충청은 2019 설립된 도로교통안전시설물 제조·시공 + 소형건설장비 전문 기업이다. 핵심 제품 **AUTONG** 은 차선 인식 자율주행 + 5개 동시 드릴링이 가능한 무인 도로변 작업 장비 — 가드레일·시선유도봉·도로 표지 폴 같은 도로변 인프라를 30m 원격으로 시공한다.

이 PoC 의 질문 — **공개 SOTA lane detection 모델을 그 환경에 그대로 적용 가능한가? 노이즈 환경에서 어떻게 무너지는가?**

## 2. 평가 모델

| 모델 | 라이선스 | 비교 위치 | 비고 |
|---|---|---|---|
| **YOLOPv2** ([CAIC-AD/YOLOPv2](https://github.com/CAIC-AD/YOLOPv2)) | GPLv3 ⚠️ | **본 PoC 직접 추론** | 차선 + 주행가능영역 + 객체 멀티태스크. CCRD 의 도로변 객체(가드레일·표지) 검출까지 한 번에 평가 가능 |
| CLRNet ([Turoad/CLRNet](https://github.com/Turoad/CLRNet)) | Apache-2.0 | public benchmark 인용 | CULane F1 79.7 SOTA. mmcv 의존성으로 Windows native 셋업 보류 — Docker 환경에서 추가 예정 (v0.3) |
| Ultra-Fast-Lane-Detection ([cfzd/Ultra-Fast](https://github.com/cfzd/Ultra-Fast-Lane-Detection)) | MIT | public benchmark 인용 | CULane ~150 FPS, 임베디드 적합. weights 미러 이슈로 v0.3 추가 |

라이선스 메모 — YOLOPv2 GPLv3 → 회사 상용 제품에 정적/동적 통합 시 소스 공개 의무. PoC 비교용으론 OK이며, 실 채택 시 CLRNet (Apache-2.0) 또는 Ultra-Fast (MIT) 로 교체 가능.

## 3. 입력 영상 — 노이즈 강도 점진 증가

| ID | 영상 | 환경 | 노이즈 등급 | 길이 | 출처 |
|---|---|---|---|---|---|
| 01_clear | Seoul 4K Driving Tour Part 1 | 주간 시내, 백색 차선 명확 | **낮음 (baseline)** | 60s | YouTube `WIQ9T2O7tNA` 발췌 (120–180s) |
| 02_rural | 대한민국 Rural Roads 4K | 시골 2차선, 차선 도색 거의 없음 | **높음 — 차선 부재** | 60s | YouTube `USmNjc8yyQo` 발췌 (180–240s) |
| 03_rain_night | 폭우 야간 드라이빙 4K | 야간 + 폭우 + 와이퍼 자국 + 신호등 반사 | **극한 — 노이즈 + 야간 + 우천** | 60s | YouTube `cb3NnbT5y4s` 발췌 (300–360s) |

## 4. 정량 결과 (YOLOPv2, RTX 3060 Ti, 720p, 600 frames each)

| 영상 | 평균 추론 시간 (ms) | 평균 FPS | 평균 검출 차선 수 | baseline 대비 |
|---|---:|---:|---:|---:|
| 01_clear | 33.32 | **30.0** | 3.77 | 100% |
| 02_rural | 29.70 | 33.7 | **0.09** | **2.4%** ⚠️ |
| 03_rain_night | 31.23 | 32.0 | **0.99** | **26%** ⚠️ |

원시 데이터 — [`results/comparison.csv`](../results/comparison.csv), 프레임별 메트릭 [`results/*.json`](../results/)

### 4.1 FPS 일관성

추론 시간은 30~33ms 로 노이즈 환경과 무관하게 일정. 즉 **detect 가 0개여도 inference 자체는 멀쩡히 도는 silent failure** — 시스템 외부에선 "잘 도는 것처럼 보이는" 위험한 상태. 이게 자율주행 시공 로봇에 그대로 들어가면 차선 못 본 채 작업 명령을 보내는 사고로 이어짐.

→ 실 운영 시 **lane confidence threshold 외부 모니터링 + fallback 모드** 필수.

### 4.2 환경별 검출 안정성

- **01_clear (baseline)** 평균 3.77 lanes / 프레임. 한국 시내 4차선 도로면 3~4 차선 자연. 안정적
- **02_rural** 평균 **0.09 lanes / 프레임 — 99% 프레임 검출 0**. 차선 마킹 자체가 없는 환경에서 모델은 거의 응답 못 함. 곡선 도로면이나 가드레일 라인을 차선으로 오인하는 sporadic 검출 0.09개
- **03_rain_night** 평균 **0.99 lanes / 프레임 — 정상 대비 -74%**. 와이퍼 자국 + 빗방울 + 신호등 반사 + 백색 라인 잘 안 보이는 야간이라 model 이 잘해야 1개 lane 정도 잡음. 자율주행 차선 추적엔 turn signal 도 못 줄 수준

### 4.3 시각 (출력 영상의 대표 프레임)

- [`docs/samples/01_clear_yolopv2_*.jpg`](samples/) — 주간 시내, 백색 차선 polyline + 주행가능영역 mask 정상
- [`docs/samples/02_rural_yolopv2_*.jpg`](samples/) — 시골 2차선, polyline 거의 없음 (모델 전 detection 실패)
- [`docs/samples/03_rain_night_yolopv2_*.jpg`](samples/) — 폭우 야간, 빗방울 반사 위에 sparse 한 polyline 1개 정도

## 5. CCRD 시공 시나리오 정조준 분석

### 5.1 AUTONG 작업 포지션 = 갓길 / road edge

CCRD 의 핵심 제품인 AUTONG 은 차량처럼 도로 중앙을 달리지 않는다. **가장 바깥쪽 차선 + 갓길 사이** 좁은 영역에서 정밀 위치를 잡고 5개 드릴을 동시에 천공한다. 인식 대상은 (1) 가장 바깥쪽 차선 (2) road edge / 갓길 (3) 기존 가드레일·시선유도봉 객체 — 충돌 회피 + 작업 누락 방지.

### 5.2 본 PoC 결과의 시사점 — silent failure 위험

위 02_rural / 03_rain_night 두 시나리오는 한국 도로변 시공 현장에 **빈번히 등장**한다

1. 시공 대상 도로의 상당수는 시골/지방도 — 차선 도색이 흐려지거나 아예 없는 구간. **02_rural 의 0.09 lanes/프레임은 그 환경에서 모델이 사실상 사용 불가**임을 의미
2. 야간 작업 또는 우천 작업 — 교통 차단 시간을 줄이려면 야간/이른 새벽 시공이 빈번. **03_rain_night 의 0.99 lanes/프레임은 모델 의존이 위험**함을 의미
3. FPS 가 노이즈 환경에서도 변동 없이 일정하므로 **외부에서 보면 "잘 돌고 있는 것처럼 보이는" silent failure** 위험 — 시스템 모니터링 layer 가 반드시 lane confidence + count 를 함께 추적해야 함

### 5.3 개선 경로 (가장 빠른 cm 정확도 확보 시나리오)

1. **CCRD 보유 시공 영상 + dashcam 으로 fine-tuning** — 평소 작업하시는 도로변 영상 100~200개 (각 1분) 으로 lane head fine-tune. learning rate 1e-5, 학습 시간 4~6 시간. **02_rural 같은 마킹 부재 환경에서 road edge 와 갓길 boundary 를 검출하도록 보강** 가능
2. **Camera + IMU + RTK GPS 센서 fusion** — Lane detection 만으로 cm 정확도는 불가능. RTK GPS (cm 정확도) 를 base 로 삼고 Camera + IMU 가 short-term drift 보정. ROS2 + robot_localization 로 30분 안에 prototype. **03_rain_night 같은 영상 노이즈 환경에서 RTK 가 backbone 역할 — 핵심**
3. **임베디드 적용** — Jetson Orin Nano (10W) + TensorRT FP16 + 입력 해상도 다운스케일. 30m 원격 작업 거리에 충분
4. **도로 인프라 객체 데이터셋 자체 수집** — 가드레일 / 시선유도봉 / 표지판 폴 라벨링 1,000장 정도면 YOLOv8 기반 object head fine-tune 충분. 작업 누락 방지 자동 검수에 활용
5. **Lane confidence + count 모니터링 layer** — 모델 silent failure 검출. lane count = 0 또는 confidence < threshold 가 연속 N 프레임 발생하면 자동 정지 + 사람 호출

## 6. 한계 및 다음 단계

- 본 v0.2 는 **YOLOPv2 단일 모델 + 3 영상** 제한. CLRNet · Ultra-Fast 추가는 v0.3 (Docker 환경 + Ultra-Fast weights GitHub 미러 우회) 진행
- 영상은 **YouTube 발췌 60초씩** — CCRD 실제 작업 영상이 아님. 회사 보유 영상으로 재실행이 가장 정확한 평가
- **검출 성공/실패 정량 지표 (IoU 기반)** 는 GT 라벨링이 필요해 본 v0.2 엔 미포함. 회사 영상 + GT 라벨 1세트 받으면 즉시 추가 가능

## 7. 결과물 / 재현 방법

```bash
git clone https://github.com/uncle-jongpal/lane-detection-poc.git
cd lane-detection-poc

# 1) 모델 weights
python scripts/download_weights.py --only yolopv2

# 2) 영상 (별도 다운, 본 PoC 과 동일한 3개)
yt-dlp -f "best[height<=720]" --download-sections "*120-180" -o "videos/input/01_clear_%(id)s.%(ext)s" "https://www.youtube.com/watch?v=WIQ9T2O7tNA"
yt-dlp -f "best[height<=720]" --download-sections "*180-240" -o "videos/input/02_rural_%(id)s.%(ext)s" "https://www.youtube.com/watch?v=USmNjc8yyQo"
yt-dlp -f "best[height<=720]" --download-sections "*300-360" -o "videos/input/03_rain_night_%(id)s.%(ext)s" "https://www.youtube.com/watch?v=cb3NnbT5y4s"

# 3) 추론 (RTX 3060 Ti 기준 영상당 ~20초)
python src/infer_yolopv2.py --input videos/input/01_clear_*.mp4 --output videos/output/01_clear_yolopv2.mp4 --metrics results/01_clear_yolopv2.json

# 4) 비교 표 + 샘플 프레임 생성
python build_report.py
```

## 8. 어필 메시지 (1줄 요약)

> "CCRD 가 다루는 도로변 시공 시나리오에서 SOTA 차선 모델의 silent failure 한계 (시골 마킹 부재시 검출 -98%, 야간 우천시 -74%) 를 정량화하고, 회사 보유 영상 fine-tuning + Camera/IMU/RTK GPS fusion + Jetson Orin 양자화 + lane confidence 모니터링 의 4단 개선 로드맵을 제시합니다. 본 PoC repo + 출력 영상 + 정량 데이터로 즉시 재현 가능합니다."

작성자 김정석 / vcfdregg3@naver.com / [github.com/uncle-jongpal/lane-detection-poc](https://github.com/uncle-jongpal/lane-detection-poc)
