# Lane Detection 모델 비교 — CCRD 도로 시공 자동화 환경 적합성 분석

작성: ___ (이름) | 작성일: 2026-__-__ | 대상: (주)충청 (CCRD) AI/IT 개발 파트너 지원

---

## TL;DR

> 도시 도로 학습 SOTA 차선 검출 모델 3종을 **CCRD 가 실제 다루는 도로변 시공 환경 영상**에 적용하여 한계와 개선 경로를 정량화했습니다. 정상 차선에서는 ___% 의 검출률을 보였으나 **흐려진 차선·비포장 구간·작업 차량 주변에서 ___% 까지 떨어졌고**, 회사가 보유한 시공 영상으로 fine-tuning + 센서 fusion 이 cm 정확도 확보의 빠른 경로라고 봅니다.

## 1. 배경

(주)충청은 도로교통안전시설물 제조·시공과 자동천공기·소형건설시공로봇을 개발하는 회사다. 일반적 자율주행 자동차의 "달리는 차선" 인식이 아니라, **도로변에서 작업하는 로봇이 차선·갓길·도로 인프라 객체를 인식해야 하는 환경** 이다.

이 PoC 의 질문: **공개된 SOTA 차선 검출 모델이 그 환경에 그대로 적용 가능한가? 아니면 어떤 보강이 필요한가?**

## 2. 비교 대상 모델

| 모델 | 라이선스 | 강점 | 약점 |
|---|---|---|---|
| **CLRNet** ([Turoad/CLRNet](https://github.com/Turoad/CLRNet)) | Apache-2.0 | 정확도 SOTA (CULane F1 79.7) | 무거움 (~30 FPS GTX 1650) |
| **Ultra-Fast-Lane-Detection** ([cfzd/Ultra-Fast](https://github.com/cfzd/Ultra-Fast-Lane-Detection)) | MIT | 임베디드 (~150 FPS) | 정확도 떨어짐, 곡선 약함 |
| **YOLOPv2** ([CAIC-AD/YOLOPv2](https://github.com/CAIC-AD/YOLOPv2)) | GPLv3 ⚠️ | 차선+주행가능영역+객체 동시 | GPL 전염성, 상용 시 라이선스 검토 |

**라이선스 메모**: YOLOPv2 는 GPLv3 — 회사 상용 제품에 통합하면 소스 공개 의무. PoC 비교용으로는 OK, 실제 채택 시 이 점 명시.

## 3. 입력 영상

| 영상 | 출처 | 길이 | 환경 | URL/파일 |
|---|---|---|---|---|
| `sample_01.mp4` | (예: YouTube 도로 시공 영상) | 30초 | 정상 차선, 포장 | (URL) |
| `sample_02.mp4` | (예: 흐려진 차선 dashcam) | 45초 | 흐려진 차선, 비포장 일부 | (URL) |
| `sample_03.mp4` | (예: 공사장 진입 차로) | 60초 | 공사장, 임시 차선 콘 | (URL) |

> 여기 영상 후보는 ccrd.co.kr 정독 후 회사가 다루는 환경에 가깝게 선택. 가드레일 시공이 메인이면 갓길·road edge 보이는 영상 필수.

## 4. 정량 결과

### 4.1 평균 추론 속도 (GTX 1650 Ti Mobile, 4GB VRAM)

| 영상 | CLRNet | Ultra-Fast | YOLOPv2 |
|---|---|---|---|
| sample_01 | ___ FPS | ___ FPS | ___ FPS |
| sample_02 | ___ FPS | ___ FPS | ___ FPS |
| sample_03 | ___ FPS | ___ FPS | ___ FPS |

→ Ultra-Fast 가 임베디드 (Jetson Orin Nano 등) 적용 시 후보. CLRNet 은 정확도 우선 시나리오.

### 4.2 평균 검출 차선 수 (정상 4차선 도로 기준)

| 영상 | CLRNet | Ultra-Fast | YOLOPv2 |
|---|---|---|---|
| sample_01 | ___ | ___ | ___ |
| sample_02 | ___ | ___ | ___ |
| sample_03 | ___ | ___ | ___ |

### 4.3 환경별 검출 성공률 (눈으로 검수 + 정량)

| 환경 | CLRNet | Ultra-Fast | YOLOPv2 |
|---|---|---|---|
| 정상 차선 (마킹 선명, 직선) | ___% | ___% | ___% |
| 흐려진 차선 | ___% | ___% | ___% |
| 곡선/굽이 | ___% | ___% | ___% |
| 그림자/역광 | ___% | ___% | ___% |
| 공사장 임시 차선 (콘, 페인트 미흡) | ___% | ___% | ___% |
| 비포장 갓길 | ___% | ___% | ___% |

> 측정 방법: 영상에서 5초 간격 키프레임 N개를 GT 로 라벨링(차선 polygon), 모델 출력과 IoU > 0.5 면 검출 성공으로 카운트.

### 4.4 정성 분석 (실패 케이스 모음)

→ `videos/output/` 의 출력 영상에서 캡처한 대표 실패 프레임 4~6 장.

- **CLRNet**: ___ (예: 차선이 가려지면 보간 실패)
- **Ultra-Fast**: ___ (예: 곡선 차선에서 점들이 끊김)
- **YOLOPv2**: ___ (예: 주행가능영역은 잘 잡지만 차선 자체는 굵게 뭉침)

## 5. CCRD 시공 시나리오 정조준 분석

### 5.1 도로변 시공 로봇이 필요한 인식 능력

회사가 자동천공기·소형건설시공로봇을 운영한다면 **차량 주행 컨텍스트가 아닌 시공 컨텍스트** 의 인식이 필요하다:

1. **차선 자체** — 작업 영역 경계
2. **갓길·road edge** — 로봇이 안전하게 위치할 영역
3. **시선유도봉·가드레일·표지판** — 시공·교체 대상 객체
4. **노면 상태** — 천공 가능한 표면 구간 식별

### 5.2 모델별 적합성 매트릭스

| 능력 | CLRNet | Ultra-Fast | YOLOPv2 | 비고 |
|---|---|---|---|---|
| 정상 차선 인식 | 🟢 | 🟡 | 🟢 | YOLOPv2 는 mask 라 굵음 |
| 흐려진 차선 인식 | 🟡 | 🔴 | 🟡 | 모두 한계 — fine-tuning 필요 |
| Road edge 인식 | 🔴 | 🔴 | 🟢 | YOLOPv2 의 drivable area 활용 가능 |
| 도로 인프라 객체 (가드레일/표지판) | 🔴 | 🔴 | 🟡 | YOLOPv2 의 객체 head 일부, **별도 detector 필요** |
| FPS (임베디드) | 🔴 | 🟢 | 🟡 | Ultra-Fast 양자화 시 더 좋아짐 |
| 라이선스 (상용) | 🟢 Apache | 🟢 MIT | 🔴 GPL | YOLOPv2 는 그대로 채택 곤란 |

### 5.3 핵심 결론

**공개 모델 1개를 그대로 채택해서는 시공 컨텍스트 풀 커버 불가.** 다만:

- **Ultra-Fast (MIT) + YOLOPv2 의 drivable head 차용 + 자체 학습 객체 detector** 조합이 가장 합리적
- 또는 **CLRNet 백본 + 시공 도메인 fine-tuning + 별도 객체 detector** 가 정확도 우선 경로

## 6. 개선 제안 (다음 단계)

### 6.1 단기 (1~2주)

1. **회사 보유 시공 영상으로 fine-tuning**
   - CULane format 또는 LaneATT 호환 라벨 200~500장이면 의미있는 개선
   - 라벨링 도구: CVAT (오픈소스), Roboflow
   - 라벨링 가이드 1~2 페이지 동봉 가능
2. **YOLOv8 으로 도로 인프라 객체 detection 추가**
   - 가드레일/시선유도봉/표지판 클래스만 50~100장으로 충분 (Ultralytics 추천 baseline)

### 6.2 중기 (1~2개월)

3. **센서 fusion**: 카메라 + IMU + RTK GPS → 시공 cm 정확도
4. **임베디드 최적화**: ONNX Runtime + INT8 양자화 → Jetson/RaspberryPi + Coral 후보 비교
5. **온도 센서/조도 센서** 결합 → 흐려진 차선 자동 감지 보조 신호

### 6.3 장기

6. **Self-supervised representation** — 시공 영상 대량 수집 → MAE/DINOv2 백본 사전학습
7. **시공 후 결과 검증 센서**: 천공 위치/깊이/각도 자동 측정 + 비전 검증

## 7. 데모 / 재현

- GitHub: ___ (URL)
- 실행 한 줄 (Docker): `docker compose up -d && docker compose exec lane-detection bash -c "python scripts/run_all.py --inputs videos/input/*.mp4 && python scripts/compare.py"`
- 전체 산출물: 영상 3개 × 모델 3개 = **출력 영상 9편** + `results/comparison.csv` + `results/*.png`

## 8. 한계와 솔직한 메모

- **TuSimple/CULane 학습 모델** 이라 도시 도로 가정. 도로변 시공 환경은 학습 분포 밖 — 그래서 fine-tuning 이 거의 필수.
- **Day 1 PoC 이라 GT 라벨링 양 적음** — 환경별 검출 성공률 N=___ 키프레임 기준, 통계적 신뢰구간 넓음.
- **GTX 1650 Ti Mobile** 는 컨슈머급 — 프로덕션 GPU (A2000, T4 등) 에서 FPS 1.5~2배 향상 예상.
- **YOLOPv2 GPL** — 비교용 OK, 채택 시 회사 정책 검토 필요.

## 9. 면접/지원서용 요약 (1줄, 2줄, 5줄)

**1줄**: SOTA 차선 검출 모델 3종을 CCRD 도로변 시공 환경 가정 영상에 적용해 한계와 개선 경로를 정량화했다.

**2줄**: 차선 검출만으로는 도로 인프라 객체·갓길·흐려진 차선 모두 커버 불가. Ultra-Fast(MIT) + YOLOv8 객체 + 회사 보유 영상 fine-tuning 조합을 1~2주 내 적용 가능한 빠른 경로로 제안한다.

**5줄**: (위 본문 §5.3 + §6.1 합쳐 작성)

---

*PoC 코드와 결과는 모두 `lane-detection-poc/` 디렉토리에 재현 가능하게 정리되어 있습니다. 인터뷰 시 라이브 데모 / 추가 영상 즉시 추론 가능.*
