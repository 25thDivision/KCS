#!/bin/bash
PIPELINE_ID=GT_color_d5

echo "=== [1단계] d=5 학습 ==="
# d=5 병렬 학습
python3 run_suite.py --train --id $PIPELINE_ID -c color_code -d 5 -g 0 -m GraphTransformer -n realistic/dp0.001_mf0.01_rf0.01_gd0.008 &
python3 run_suite.py --train --id $PIPELINE_ID -c color_code -d 5 -g 1 -m GraphTransformer -n realistic/dp0.005_mf0.02_rf0.02_gd0.015 realistic/dp0.01_mf0.05_rf0.05_gd0.01 &

echo "=== [대기] 1단계 작업들이 모두 끝날 때까지 얌전히 대기 중... ==="
wait

echo "=== [2단계] d=5 하드웨어 실험 ==="
# 방금 만들어진 d=5는 학습 시작, 학습이 끝난 d=3은 바로 하드웨어 실험 시작
python3 run_suite.py --experiment ionq --id $PIPELINE_ID -d 5 -m GraphTransformer &

echo "=== [대기] 2단계 작업들이 모두 끝날 때까지 얌전히 대기 중... ==="
wait

echo "=== 🎉 KCS 프로젝트 통합 파이프라인 모든 과정 완료! ==="