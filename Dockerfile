FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt /workspace/
RUN pip install --no-cache-dir -r requirements.txt

# 외부 모델 repo 는 빌드 시 미리 clone 해 두면 컨테이너 부팅 빠름.
# 단, weights 는 용량 커서 별도 다운로드.
RUN mkdir -p /workspace/external && cd /workspace/external && \
    git clone --depth 1 https://github.com/Turoad/CLRNet.git && \
    git clone --depth 1 https://github.com/cfzd/Ultra-Fast-Lane-Detection.git && \
    git clone --depth 1 https://github.com/CAIC-AD/YOLOPv2.git

CMD ["bash"]
