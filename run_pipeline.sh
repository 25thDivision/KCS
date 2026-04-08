#!/bin/bash

echo "=== [1단계] d=7 데이터 생성 & d=3 CNN 학습 & d=3 IBM(QPU), IonQ 하드웨어 실험 (CNN 빠짐) (교차 병렬) ==="
# 
python3 run_suite.py --generate -c color_code surface_code -d 7 --cpu full &
python3 run_suite.py --train -c surface_code color_code -d 3 -m CNN &
python3 run_suite.py --experiment ionq ibm -d 3 -m GCNII APPNP GraphTransformer GCN GAT GNN GraphMamba &

echo "=== [대기] 1단계 작업들이 모두 끝날 때까지 대기 중... ==="
wait

echo "=== [2단계] d=7 데이터 학습 & d=3 CNN IBM(QPU), IonQ 하드웨어 실험 (교차 병렬) ==="
# 
python3 run_suite.py --train -c color_code surface_code -d 7 --gpu0 CNN GCNII APPNP GraphTransformer --gpu1 GCN GAT GNN GraphMamba &
python3 run_suite.py --experiment ionq ibm -d 3 -m CNN &

echo "=== [대기] 2단계 작업들이 모두 끝날 때까지 대기 중... ==="
wait

echo "=== [3단계] d=5 CNN 학습 (동기) ==="
# 
python3 run_suite.py --train -c color_code surface_code -d 5 --gpu0 CNN &

echo "=== 🎉 KCS 프로젝트 통합 파이프라인 모든 과정 완료! ==="