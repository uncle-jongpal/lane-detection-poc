# 윈도우 실행 가이드

리눅스에서 셋업한 PoC 를 윈도우 GPU 머신에서 끝까지 돌리는 절차.

## 사전 요구

- Windows 10/11
- NVIDIA GPU (GTX 1660 이상 권장, 1650 Ti 이상이면 가능)
- Python 3.10+ (또는 Docker Desktop + WSL2)
- Git
- 디스크 여유 30GB (모델 + 영상)

## A. Docker 경로 (권장 — 재현성↑)

### A.1 Docker Desktop 설치 + WSL2 + GPU

1. [Docker Desktop](https://www.docker.com/products/docker-desktop/) 설치 시 "WSL2 backend" 활성화.
2. NVIDIA Container Toolkit 자동 통합 — Docker Desktop 4.x 이상이면 별도 설치 불필요. (안 되면 [공식 가이드](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#install-guide-windows))
3. PowerShell 에서 GPU 보이는지 확인:
   ```powershell
   docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
   ```

### A.2 Repo clone + 빌드 + 실행

```powershell
git clone <REPO_URL> lane-detection-poc
cd lane-detection-poc

# 빌드 (이미지 ~7GB, 5~10분)
docker compose build

# 한 번 들어가서 weights 다운로드 + 영상 배치
docker compose run --rm lane-detection bash
# (컨테이너 안에서)
python scripts/download_weights.py
exit

# 영상 추가: 호스트의 lane-detection-poc/videos/input/ 에 .mp4 넣기
# (Docker 가 볼륨 마운트로 컨테이너에서 자동으로 보임)

# 추론 + 비교
docker compose run --rm lane-detection bash -c "python scripts/run_all.py --inputs videos/input/*.mp4 && python scripts/compare.py"
```

### A.3 결과 확인

호스트의 `videos/output/`, `results/` 폴더에 모두 저장됨. 윈도우 탐색기에서 바로 열림.

## B. 호스트 직접 실행 (Docker 안 쓸 때)

### B.1 Python 환경

```powershell
# Python 3.10 설치 후
python -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip

# PyTorch (CUDA 12.1)
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# 나머지
pip install -r requirements.txt

# GPU 확인
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### B.2 외부 모델 repo clone

```powershell
mkdir external
cd external
git clone --depth 1 https://github.com/Turoad/CLRNet.git
git clone --depth 1 https://github.com/cfzd/Ultra-Fast-Lane-Detection.git
git clone --depth 1 https://github.com/CAIC-AD/YOLOPv2.git
cd ..
```

### B.3 weights + 영상 + 추론

```powershell
python scripts/download_weights.py

# 영상 넣기 (탐색기로 .mp4 를 videos/input/ 에)

python scripts/run_all.py --inputs videos\input\*.mp4
python scripts/compare.py
```

## 영상 후보 추천 (CCRD 도메인)

YouTube 검색 키워드:
- "도로 가드레일 시공 영상"
- "차선 도색 작업"
- "construction zone dashcam"
- "road shoulder maintenance"

영상 받기:
```powershell
# yt-dlp 설치
pip install yt-dlp

# 다운로드 (1080p 권장)
yt-dlp -f "bestvideo[height<=1080]+bestaudio/best[height<=1080]" -o "videos\input\sample_01.mp4" "<유튜브 URL>"
```

영상 길이는 30초~1분 권장 (추론 시간 + 분석 부담).

## 자주 막히는 부분

| 증상 | 원인 / 해결 |
|---|---|
| `CUDA out of memory` | `--max-frames 100` 으로 제한, 또는 CLRNet → Ultra-Fast 만 사용 |
| Docker 에서 GPU 안 보임 | Docker Desktop "Use GPU" 옵션 + WSL2 활성화 확인 |
| `mmcv` 빌드 실패 | `pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html` |
| weights 다운로드 403 | `scripts/download_weights.py` 의 URL 수동 업데이트 (모델 repo README 와 동기화) |
| 영상 출력이 깨짐 | OpenCV 의 mp4v 가 일부 플레이어에서 재생 안 됨 — VLC 또는 ffmpeg 으로 H.264 재인코딩 |

## 결과 정리 → 지원서 첨부

1. `docs/REPORT.md` 의 빈 칸 (___) 을 결과로 채우기
2. `docs/REPORT.md` → PDF 변환 (VS Code "Markdown PDF" 확장 또는 `pandoc`)
3. GitHub repo public 으로 push, README 에 GIF 1~2개
4. 사람인 지원서: GitHub URL + PDF 첨부

## 시간 가이드

| 단계 | 예상 |
|---|---|
| 환경 셋업 (Docker A 경로) | 30분~1시간 (이미지 빌드 + 모델 다운로드) |
| 영상 1개 모델 3개 추론 (60초 영상 기준) | 5~10분 (GTX 1650) |
| 영상 3개 전체 + compare | 30분~1시간 |
| 결과 키프레임 라벨링 (검출률 정량용) | 1~2시간 |
| REPORT.md 작성 | 1시간 |
| **합계** | **반나절~1일** |
