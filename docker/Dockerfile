FROM --platform=linux nvidia/cuda:11.3.1-base-ubuntu20.04

RUN apt update && apt -y upgrade && DEBIAN_FRONTEND=noninteractive apt install -y \
        python3=3.8.2-0ubuntu2 \
        python3-pip \
        python3-venv=3.8.2-0ubuntu2 \
        build-essential \
        libssl-dev \
        libffi-dev \
        python3-dev=3.8.2-0ubuntu2 \
        python3-opencv \
        ffmpeg \
        git \
        npm

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv /opt/vfe-env
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY . /vocalexplore/
WORKDIR /vocalexplore

# Install dependencies and run set up
RUN pip install --no-cache-dir -r requirements.txt && rm requirements.txt \
 && pip3 install --no-cache-dir torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 -f https://download.pytorch.org/whl/cu113/torch_stable.html pytorch-lightning==1.6.0 lightning-bolts==0.5.0 torch_tb_profiler \
 && pip install --no-cache-dir --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda110 \
 && pip install --no-cache-dir ftfy regex tqdm \
 && pip install --no-cache-dir git+https://github.com/openai/CLIP.git \
 && python3 setup.py develop \
 && npm install --global http-server

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,video,utility
