@echo off
REM Pure-ONNX real-time action recognition demo
REM Requires: numpy, opencv-python, onnxruntime-gpu, nvidia-cudnn-cu12
REM NO mmcv, mmpose, mmdet needed!

cd /d %~dp0\..
call conda activate pyskl

echo ============================================
echo   ONNX Real-Time Action Recognition Demo
echo ============================================
echo.

REM Default: GPU det/pose, CPU recognition
python demo/demo_onnx.py ^
    --device cuda ^
    --recog-device cpu ^
    --window-frames 30 ^
    --clip-len 100 ^
    --det-every 1 ^
    --short-side 480 ^
    --det-score-thr 0.5 ^
    %*

pause
