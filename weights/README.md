# weights/

모델 가중치 저장. **gitignore 됨** — 직접 다운로드 필요.

## 다운로드

```bash
python scripts/download_weights.py
```

## 파일 목록 (다운로드 후)

| 파일 | 크기 | 모델 | 출처 |
|---|---|---|---|
| `clrnet_culane_resnet18.pth` | ~150MB | CLRNet R18 CULane | [Turoad/CLRNet](https://github.com/Turoad/CLRNet) |
| `ufld_culane_18.pth` | ~50MB | Ultra-Fast R18 CULane | [cfzd/Ultra-Fast](https://github.com/cfzd/Ultra-Fast-Lane-Detection) |
| `yolopv2.pt` | ~75MB | YOLOPv2 jit | [CAIC-AD/YOLOPv2](https://github.com/CAIC-AD/YOLOPv2) |

## 다른 backbone / 데이터셋 weights

각 모델 repo 의 README 에서 다른 사전학습 weights URL 확인 후
`scripts/download_weights.py` 의 `DOWNLOADS` dict 추가.

예: TuSimple 학습 모델은 흰 차선 위주 → 도시 차선 적합. CULane 학습 모델은 야간/카메라 흔들림 강건 → CCRD 도메인 더 가까움.
