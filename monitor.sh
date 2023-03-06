#!/bin/bash
test_path_dir=`pwd`
RANK_ID=$1

if [ ! -d ${test_path_dir}/output/${RANK_ID} ];then
  mkdir -p ${test_path_dir}/output/${RANK_ID}
fi

for i in $(seq 1 500000)
do
  #job_num=`ps -efww|grep benchmark |grep -v grep|wc -l`
  job_num=`ps -efww|grep -v grep|grep -c "hrnet_om_infer_end.py"`
  if [ ${job_num} -eq 0 ]
  then
    echo "infer done-------------------"
    break
  fi
  #echo "Record the status of  CPU NPU MEM"
  time=`date |awk '{print $4}'`
  mpstat >>${test_path_dir}/output/${RANK_ID}/cpu_monitor.log &
  echo "$time "`npu-smi info | grep -E "310|0000:" | awk -v line=$((2*RANK_ID+1)) 'NR==line{printf "NPU=%s,   Power(W)=%s,\tTemp(C):%s   ",$3,$7,$8};NR==line+1{printf "AICore(%%):%s,   Memory-Usage(MB):%s/%s,  HBM-Usage(MB):%s/%s\n",$6,$7,$9,$10,$12}'` >>${test_path_dir}/output/${RANK_ID}/ascend_monitor.log &
  echo  "$time "`free |grep Mem` >> ${test_path_dir}/output/${RANK_ID}/mem_monitor.log &
  sleep 1
done
