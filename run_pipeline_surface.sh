#!/bin/bash

echo "=== [1단계] d=3 데이터 생성 (동기) | Code: surface ==="
# 맨 뒤에 &가 없으므로 여기서 데이터가 다 만들어질 때까지 대기
python3 run_suite.py --generate -c surface_code -d 3 --cpu full

echo "=== [2단계] d=3 학습 & d=5,7 데이터 생성 (병렬) ==="
# 데이터 생성이 끝난 d=3은 바로 학습 시작, 동시에 남은 d=5,7 데이터 생성
python3 run_suite.py --train -c surface_code -d 3 --gpu0 CNN GCNII APPNP GraphTransformer --gpu1 GCN GAT GNN GraphMamba &
python3 run_suite.py --generate -c surface_code -d 5 7 --cpu full &

echo "=== [대기] 2단계 작업들이 모두 끝날 때까지 얌전히 대기 중... ==="
wait

echo "=== [3단계] d=5,7 학습 & d=3 하드웨어 실험 (교차 병렬) ==="
# 방금 만들어진 d=5,7은 학습 시작, 학습이 끝난 d=3은 바로 하드웨어 실험 시작
python3 run_suite.py --train -c surface_code -d 5 7 --gpu0 CNN GCNII APPNP GraphTransformer --gpu1 GCN GAT GNN GraphMamba &
python3 run_suite.py --experiment ibm -c surface_code -d 3 -m CNN GCNII APPNP GraphTransformer GCN GAT GNN GraphMamba &

echo "=== [대기] 3단계 작업들이 모두 끝날 때까지 얌전히 대기 중... ==="
wait

echo "=== [4단계] d=5,7 최종 하드웨어 실험 (동기) ==="
# 마지막으로 남은 d=5,7 모델들에 대한 실험 진행
python3 run_suite.py --experiment ibm -c surface_code -d 5 7 -m CNN GCNII APPNP GraphTransformer GCN GAT GNN GraphMamba

echo "=== 🎉 KCS 프로젝트 통합 파이프라인 모든 과정 완료! ==="