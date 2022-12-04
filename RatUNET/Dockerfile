FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime
WORKDIR /src

RUN apt update; exit 0
# add apt dependencies here
RUN apt-get update && apt-get install -y git curl wget ffmpeg libsm6 libxext6 \
    zsh && \
    mkdir /data
COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

