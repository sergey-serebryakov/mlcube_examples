# AI Readiness (AIR)

- [nvidia-smi](./nvidia-smi/) Run NVIDIA-SMI in a docker container. Successful execution of this ML cube does not guarantee applications can actually use GPUs. Run GPU Check next.
- [gpu-check](./gpu-check/) Check user application can actually allocate a GPU memory. If not successful, double check NVIDIA fabric manager is installed.
- [nccl](./nccl/) Run single-node multi-GPU NCCL tests.
- [magma](./magma/) Run linear algebra tests (single / double gemm - general matrix multiply).