#!/usr/bin/env bash
# Jetson Nano (TensorRT 8.2 + CUDA 10.2) 에서 ONNX → TRT engine 빌드.
# 같은 ONNX 로 FP16 / INT8 두 종 빌드 후 비교 측정.
#
# 사용:
#   bash scripts/build_trt_engine.sh weights/twinlitenetplus/nano.onnx fp16
#   bash scripts/build_trt_engine.sh weights/twinlitenetplus/nano.onnx int8
#
# Notes:
# - INT8 모드는 calibration data 필요 (--calib=...). 미리 scripts/calibrate_int8.py 로 준비.
# - Jetson Nano 는 DLA 미지원 (Orin 부터 지원) — GPU 추론만.
# - workspace 메모리 256 MB 제한 (Nano 4GB 시스템 메모리 한정).
set -euo pipefail

ONNX=${1:-}
PRECISION=${2:-fp16}    # fp16 | int8 | fp32
CALIB=${3:-calib_cache.bin}

if [[ -z "$ONNX" || ! -f "$ONNX" ]]; then
    echo "ERR: ONNX not found: $ONNX"
    echo "사용: $0 <onnx> [fp16|int8|fp32] [calib_cache]"
    exit 1
fi

ONNX_BASE=$(basename "$ONNX" .onnx)
ENGINE="weights/twinlitenetplus/${ONNX_BASE}_${PRECISION}.engine"
mkdir -p "$(dirname "$ENGINE")"

echo "[build] ONNX=$ONNX → engine=$ENGINE  precision=$PRECISION"

# trtexec 위치 — Jetson 기본 PATH 에 있음. 없으면 절대경로.
TRTEXEC=$(which trtexec 2>/dev/null || echo /usr/src/tensorrt/bin/trtexec)
if [[ ! -x "$TRTEXEC" ]]; then
    echo "ERR: trtexec not found. tensorrt-bin 패키지 설치 필요."
    exit 1
fi

# 공통 플래그
ARGS=(
    --onnx="$ONNX"
    --saveEngine="$ENGINE"
    --workspace=256        # MB. Jetson Nano 4GB 시스템 메모리 한정
    --verbose
)

case "$PRECISION" in
    fp32) ;;  # default precision
    fp16) ARGS+=( --fp16 ) ;;
    int8)
        ARGS+=( --int8 --fp16 )    # int8 + fp16 fallback
        if [[ -f "$CALIB" ]]; then
            ARGS+=( --calib="$CALIB" )
            echo "[build] using calibration cache: $CALIB"
        else
            echo "[build] WARN: $CALIB 없음 — calibrate_int8.py 먼저 실행 권장. trtexec 가 dynamic range 자동 추정으로 빌드 시도 (정확도 떨어질 수 있음)."
        fi
        ;;
    *) echo "ERR: precision must be fp16/int8/fp32"; exit 1 ;;
esac

echo "[build] trtexec ${ARGS[@]}"
echo "[build] (소요 시간: 5~15 분 — Jetson Nano 에선 ONNX → engine 빌드가 느림)"

START=$(date +%s)
"$TRTEXEC" "${ARGS[@]}" 2>&1 | tee "/tmp/trt_build_${PRECISION}.log"
END=$(date +%s)
ELAPSED=$((END-START))

if [[ -f "$ENGINE" ]]; then
    SIZE=$(du -h "$ENGINE" | awk '{print $1}')
    echo "[build] ✓ 빌드 완료 — $ENGINE ($SIZE, ${ELAPSED}s)"
    echo "[build] 로그: /tmp/trt_build_${PRECISION}.log"
else
    echo "[build] ✗ 빌드 실패 — 로그 확인: /tmp/trt_build_${PRECISION}.log"
    exit 2
fi
