#!/bin/bash
CURRENT_DIR=`pwd`

start_time=$(date +%s)

if [ -d ${CURRENT_DIR}/log ]; then
    cp -rf ${CURRENT_DIR}/log ${CURRENT_DIR}/log_${start_time}
    rm -rf ${CURRENT_DIR}/log
    mkdir -p ${CURRENT_DIR}/log
else
    mkdir -p ${CURRENT_DIR}/log
fi

python3 hrnet_om_infer_end.py &

# monitor
nohup bash monitor.sh 0 &