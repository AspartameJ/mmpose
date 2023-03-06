#!/bin/bash

python3 hrnet_om_infer_end.py &

# monitor
nohup bash monitor.sh 0 &