FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    FORCE_CUDA="1" \
    CUDA_DEVICE_ORDER="PCI_BUS_ID"

# fetch the key refer to https://forums.developer.nvidia.com/t/18-04-cuda-docker-image-is-broken/212892/9
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub 32
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

WORKDIR /app

RUN mkdir src input output

# Install Poetry
RUN pip install --upgrade pip && pip install poetry

# Copy poetry files
COPY ./pyproject.toml ./poetry.lock /app/

# Install additional dependencies
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 ffmpeg libsm6 libxext6 git ninja-build libxrender-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry dependencies
RUN poetry config virtualenvs.create false
RUN poetry install

# Install MMEngine and MMCV
RUN pip install openmim && \
    mim install "mmengine>=0.7.1" "mmcv>=2.0.0rc4,<2.2"

# Clone and install mmdetection
RUN conda clean --all \
    && git clone https://github.com/open-mmlab/mmdetection.git ./mmdetection \
    && cd ./mmdetection \
    && pip install -r requirements/albu.txt \
    && pip install --no-cache-dir -e .

WORKDIR /app/src

# Copy source code and scripts
COPY ./src /app/src

# Set a more flexible entrypoint
CMD ["bash", "Process_SkillEval.sh", "GRS"]
