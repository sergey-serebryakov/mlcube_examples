# Works for Hoppper Volta
FROM nvcr.io/nvidia/pytorch:22.12-py3

RUN apt-get update && \
    apt-get install --no-install-recommends -y git && \
    rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/NVIDIA/nccl-tests.git && \
    cd nccl-tests && \
    make

COPY nccl.py /opt/air/nccl/nccl.py

ENTRYPOINT ["python", "/opt/air/nccl/nccl.py"]