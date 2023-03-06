FROM ascendhub.huawei.com/public-ascendhub/pytorch-modelzoo:22.0.RC3-1.8.1

VOLUME ["/usr/local/Ascend/driver", "/usr/local/Ascend/add-ons", "/var/log/npu"]

COPY . /root/hrnet

USER root

WORKDIR /root/hrnet

RUN apt-get update
RUN apt-get install -y dstat sysstat
RUN ./Ascend-cann-toolkit_6.0.RC1_linux-x86_64.run --full
RUN echo '. /usr/local/Ascend/ascend-toolkit/set_env.sh' >> ~/.bashrc
RUN /bin/bash -c "source ~/.bashrc"
RUN pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/ascend/torch1.8.0/index.html
RUN pip install torchvision==0.9.1 onnx onnxruntime onnx-simplifier xtcocotools json_tricks munkres
RUN pip install -v 'git+https://gitee.com/ascend/tools.git#egg=aclruntime&subdirectory=ais-bench_workload/tool/ais_bench/backend'
RUN pip install -v 'git+https://gitee.com/ascend/tools.git#egg=ais_bench&subdirectory=ais-bench_workload/tool/ais_bench'
RUN rm -rf Ascend-cann-toolkit_6.0.RC1_linux-x86_64.run
