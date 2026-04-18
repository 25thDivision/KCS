#!/bin/bash
PIPELINE_ID=New_Surface_ibm_miami

echo "=== [1단계] d=3 데이터 생성 (동기) | Code: surface ==="
# 맨 뒤에 &가 없으므로 여기서 데이터가 다 만들어질 때까지 대기
python3 run_suite.py --generate -c surface -d 3 --cpu full --id $PIPELINE_ID

echo "=== [2단계] d=3 학습 & d=5 데이터 생성 (병렬) ==="
# 데이터 생성이 끝난 d=3은 바로 학습 시작, 동시에 남은 d=5 데이터 생성
python3 run_suite.py --train -c surface -d 3 --gpu0 CNN GCNII APPNP GraphTransformer --gpu1 GCN GAT GNN GraphMamba --id $PIPELINE_ID &
python3 run_suite.py --generate -c surface -d 5 --cpu full --id $PIPELINE_ID &

echo "=== [대기] 2단계 작업들이 모두 끝날 때까지 얌전히 대기 중... ==="
wait

echo "=== [3단계] d=5 학습 & d=3 하드웨어 실험 (교차 병렬) ==="
# 방금 만들어진 d=5는 학습 시작, 학습이 끝난 d=3은 바로 하드웨어 실험 시작
python3 run_suite.py --train -c surface -d 5 --gpu0 CNN GCNII APPNP GraphTransformer --gpu1 GCN GAT GNN GraphMamba --id $PIPELINE_ID &
python3 run_suite.py --experiment ibm -c surface -b ibm_miami -d 3 -m CNN GCNII APPNP GraphTransformer GCN GAT GNN GraphMamba --id $PIPELINE_ID &
echo "=== [대기] 3단계 작업들이 모두 끝날 때까지 얌전히 대기 중... ==="
wait

echo "=== [4단계] d=5 최종 하드웨어 실험 (동기) ==="
# 마지막으로 남은 d=5 모델들에 대한 실험 진행
python3 run_suite.py --experiment ibm -c surface -b ibm_miami -d 5 -m CNN GCNII APPNP GraphTransformer GCN GAT GNN GraphMamba --id $PIPELINE_ID

echo "=== 🎉 KCS 프로젝트 통합 파이프라인 모든 과정 완료! ==="