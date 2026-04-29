# Lane Detection PoC for CCRD

도로교통안전시설물 자동 시공 로봇 환경에 차선 인식 SOTA 모델을 적용했을 때의 한계와 개선 경로를 정량적으로 분석한 PoC.

[(주)충청 / CCRD](https://ccrd.co.kr/) 의 사업 영역(도로변 시공 로봇)과 일반적 차선 인식 모델(도시 도로 학습)의 도메인 격차를 보여주는 것이 목적.

## 무엇을 보여주나

- SOTA 차선 검출 모델 **3종을 같은 영상**에 동시 추론
  - **CLRNet** — 정확도 우선 SOTA
  - **Ultra-Fast-Lane-Detection** — 속도 우선 (임베디드 적합)
  - **YOLOPv2** — 차선 + 주행가능영역 + 객체 멀티태스크
- **CCRD 시공 시나리오 정량화**
  - 정상 차선 / 흐려진 차선 / 비포장 / 그림자 등 환경별 검출률
  - FPS (임베디드 컨트롤러 적용 가능성)
  - 도로 인프라 객체 (가드레일/시선유도봉/표지판) 검출 가능 여부
- **개선 제안** (회사 보유 시공 영상 fine-tuning, 센서 fusion, 임베디드 양자화 경로)

## 결과물 (지원서 제출용)

- 📊 [`docs/REPORT.md`](docs/REPORT.md) — 분석 PDF 의 마크다운 원본 (10쪽)
- 🎬 `videos/output/` — 모델별 차선 오버레이 영상
- 📈 `results/comparison.csv` — 모델별 FPS / 검출률 / 실패 구간 정량
- 📓 `notebooks/` — (선택) 재현용 jupyter

## 빠른 시작

### 1. 환경 (Docker 권장 — 윈도우/리눅스 동일)

```bash
docker compose up -d
docker compose exec lane-detection bash
```

또는 호스트에 직접 (Python 3.10+):

```bash
python -m venv .venv && source .venv/bin/activate    # 윈도우: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 모델 weights 다운로드

```bash
python scripts/download_weights.py
```

### 3. 입력 영상 배치

`videos/input/` 에 `.mp4` 파일을 넣는다. 추천: 도로변 작업 / 공사장 dashcam / 흐려진 차선 영상 1~3개 (각 30초~1분).

### 4. 추론 실행

```bash
# 단일 모델
python src/infer_clrnet.py --input videos/input/sample.mp4 --output videos/output/sample_clrnet.mp4

# 또는 모든 모델 한 번에
python scripts/run_all.py --input videos/input/sample.mp4
```

### 5. 결과 비교 표 생성

```bash
python scripts/compare.py --inputs videos/input/*.mp4 --output results/comparison.csv
```

## 디렉토리 구조

```
lane-detection-poc/
├── README.md
├── ccrd-notes.md            # 회사 도메인 분석
├── plan-doc.md / plan-work.md  # 초기 두 가지 계획안
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── src/
│   ├── infer_clrnet.py        # CLRNet 추론
│   ├── infer_ultrafast.py     # Ultra-Fast 추론
│   ├── infer_yolopv2.py       # YOLOPv2 추론
│   └── lib/
│       ├── video_io.py        # 공통 영상 I/O
│       └── overlay.py         # 시각화
├── scripts/
│   ├── download_weights.py    # 모델 weights 다운로드
│   ├── run_all.py             # 모든 모델 일괄 실행
│   └── compare.py             # 결과 비교 CSV
├── configs/
│   ├── clrnet.yaml
│   ├── ultrafast.yaml
│   └── yolopv2.yaml
├── videos/{input,output}/
├── weights/                   # gitignored
├── results/                   # gitignored
└── docs/
    ├── REPORT.md              # 최종 분석 보고서
    └── REPORT.pdf             # PDF 변환 결과
```

## 한 줄 요약

> "회사가 진짜 다루는 도로변 시공 로봇 환경에서 SOTA 차선 모델 3종을 비교했고, 한계와 개선 경로를 정량적으로 정리한 PoC."

## 라이선스

- 본 PoC 코드: MIT
- 외부 모델: 각 repo 라이선스 따름 (CLRNet: Apache-2.0, Ultra-Fast: MIT, YOLOPv2: GPLv3 — **상용 시 GPL 전염성 주의**)
