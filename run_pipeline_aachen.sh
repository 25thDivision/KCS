#!/bin/bash
PIPELINE_ID=Final_Results

# 각 프로세서를 독립적으로 5회 순차 실행, 프로세서 간은 병렬
run_sequential() {
    local platform=$1
    local backend=$2
    local instance=$3
    local runs=("1st" "2nd" "3rd" "4th" "5th")
    local models="CNN GCNII APPNP GraphTransformer GCN GAT GNN GraphMamba"

    for run in "${runs[@]}"; do
        echo "============================================"
        echo "[${platform}/${backend:-default}/${instance:-default}] ${run} 시작"
        echo "============================================"
        if [ "$platform" == "ibm" ]; then
            python3 run_suite.py --id $PIPELINE_ID/$run --experiment ibm -b $backend -i $instance -m $models
        else
            python3 run_suite.py --id $PIPELINE_ID/$run --experiment ionq -m $models
        fi
        echo "============================================"
        echo "[${platform}/${backend:-default}/${instance:-default}] ${run} 완료"
        echo "============================================"
    done
}

# 3개 프로세서 병렬, 각각 내부적으로 5회 순차
# run_sequential ibm ibm_boston &
# run_sequential ibm ibm_pittsburgh &
run_sequential ibm ibm_aachen Yonsei_internal-eu &
# run_sequential ionq &

wait
echo "=== 🎉 데이터 수집 완료 ==="