# Hopper
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04
# Volta Ampere
# FROM nvidia/cuda:12.5.1-cudnn-devel-ubuntu22.04
ARG GPU_TARGET=Hopper


RUN apt update && \
    apt install -y --no-install-recommends \
                git python3 cmake libblas-dev liblapack-dev

WORKDIR /opt/air/magma
RUN git clone https://github.com/icl-utk-edu/magma.git && \
    cd ./magma && \
    git checkout v2.9.0 && \
    echo "BACKEND = cuda\nFORT = true\nGPU_TARGET = ${GPU_TARGET}" > ./make.inc && \
    make generate && \
    mkdir build && \
    cd build && \
    cmake -DMAGMA_ENABLE_CUDA=ON -DCMAKE_INSTALL_PREFIX=/usr/local/magma -DLAPACK_LIBRARIES="-llapack -lblas" \
          -DGPU_TARGET=${GPU_TARGET} -DBLA_VENDOR=Generic -DUSE_FORTRAN=off .. && \
    make -j$(nproc)

COPY magma.py /opt/air/magma/magma.py

WORKDIR /opt/air/magma
ENTRYPOINT ["python3", "magma.py"]