FROM nvcr.io/nvidia/pytorch:22.12-py3

RUN apt-get update

RUN pip install pytorch-lightning==1.9.0

COPY gpu_check.py /opt/air/gpu_check/gpu_check.py

ENTRYPOINT ["python", "/opt/air/gpu_check/gpu_check.py"]
