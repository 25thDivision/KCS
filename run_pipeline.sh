#!/bin/bash

echo "=== KCS 프로젝트 통합 파이프라인 시작 ==="

echo "=== [1단계] ==="
# 
python3 run_suite.py --train -c color_code -d 5 --gpu0 CNN GCNII APPNP GraphTransformer --gpu1 GCN GAT GNN GraphMamba > logs/color_code_5_train.log 2>&1 &
python3 run_suite.py --generate -c heavyhex_surface_code -n realistic/dp0.005_mf0.02_rf0.02_gd0.015 realistic/dp0.01_mf0.05_rf0.05_gd0.01 -d 3 --cpu full > logs/heavyhex_3_generate1.log 2>&1 &

echo "=== [대기] 1단계 작업들이 모두 끝날 때까지 대기 중... ==="
wait

echo "=== [2단계] ==="
# 
python3 run_suite.py --experiment ionq -c color_code -d 5 -m CNN GCNII APPNP GraphTransformer GCN GAT GNN GraphMamba > logs/color_code_5_experiment.log 2>&1 &
python3 stim_simulation/simulation/generate_dataset_image.py -c heavyhex_surface_code -n realistic/dp0.001_mf0.01_rf0.01_gd0.008 -d 3 --cpu full > logs/heavyhex_3_generate2.log 2>&1 &

echo "=== [대기] 2단계 작업들이 모두 끝날 때까지 대기 중... ==="
wait

echo "=== [3단계] ==="
# 

python3 run_suite.py --train -c heavyhex_surface_code -d 3 --gpu0 CNN GCNII APPNP GraphTransformer --gpu1 GCN GAT GNN GraphMamba > logs/heavyhex_3_train.log 2>&1 &

echo "=== [대기] 3단계 작업들이 모두 끝날 때까지 대기 중... ==="
wait

echo "=== [4단계] ==="
# 
python3 run_suite.py --experiment ibm -d 3 -m CNN GCNII APPNP GraphTransformer GCN GAT GNN GraphMamba > logs/heavyhex_3_experiment.log 2>&1 &

echo "=== [대기] 4단계 작업들이 모두 끝날 때까지 대기 중... ==="
wait

echo "=== 🎉 KCS 프로젝트 통합 파이프라인 모든 과정 완료! ==="