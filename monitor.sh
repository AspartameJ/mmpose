#!/bin/bash
test_path_dir=`pwd`
RANK_ID=$1
CODE_NAME="hrnet_om_infer_end.py"

for i in $(seq 1 500000)
do
  job_num=`ps -efww|grep -v grep|grep -c "$CODE_NAME"`
  if [ ${job_num} -eq 0 ]
  then
    break
  fi
  time=`date |awk '{print $4}'`
  mpstat 1 1 >> ${CURRENT_DIR}/log/cpu_monitor.log &
  echo "$time " `npu-smi info | grep -E "/" | awk -v line=$((2*RANK_ID+1)) 'NR==line{printf "NPU=%s,   Power(W)=%s,\tTemp(C):%s   HBM-Usage(MB):%s/%s,  ",$3,$7,$8,$9,$11};NR==line+1{printf "AICore(%%):%s,   Memory-Usage(MB):%s/%s  ",$7,$8,$10}'` >>${test_path_dir}/log/ascend_monitor.log &
  # echo "$time " `npu-smi info | grep -E "/" | awk -v line=$((2*RANK_ID+1)) 'NR==line{printf "NPU=%s,   Power(W)=%s,\tTemp(C):%s,  ",$3,$7,$8};NR==line+1{printf "AICore(%%):%s,   Memory-Usage(MB):%s/%s, HBM-Usage(MB):%s/%s",$6,$7,$9,$10,$12}'` >>${test_path_dir}/log/ascend_monitor.log &
  echo  "$time "`free |grep Mem` >> ${test_path_dir}/log/mem_monitor.log &
  sleep 1
done
