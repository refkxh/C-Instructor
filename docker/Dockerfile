# Matterport3DSimulator
# Requires nvidia gpu with driver 396.37 or higher

FROM nvidia/cudagl:11.4.2-devel

# Install cudnn
# ENV CUDNN_VERSION 8.2.4.15
# LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"

# RUN apt-get update && apt-get install -y --no-install-recommends \
#     libcudnn8=$CUDNN_VERSION-1+cuda11.4 \
# libcudnn8-dev=$CUDNN_VERSION-1+cuda11.4 \
# && \
#     apt-mark hold libcudnn8 && \
#     rm -rf /var/lib/apt/lists/*

# openssh-server for sshd
# sudo for switch user
RUN apt-get update && apt-get install -y --no-install-recommends openssh-server sudo

# Allow sshd PasswordAuthentication
RUN sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/g' /etc/ssh/sshd_config

# Install a few libraries to support both EGL and OSMESA options
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y wget doxygen curl libjsoncpp-dev libepoxy-dev libglm-dev libosmesa6 libosmesa6-dev libglew-dev libopencv-dev python3-setuptools python3-dev python3-pip git htop tmux libaio-dev zip
RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple torch torchvision torchaudio opencv-python numpy pandas networkx fairscale sentencepiece gradio nvitop h5py progressbar2 lmdb jsonlines easydict tensorboard ipykernel
RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple deepspeed
RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple git+https://github.com/csuhan/timm_0_3_2.git
RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple git+https://github.com/openai/CLIP.git

#install latest cmake
# ADD https://cmake.org/files/v3.27/cmake-3.27.1-linux-x86_64.sh /cmake-3.27.1-linux-x86_64.sh
# RUN mkdir /opt/cmake
# RUN sh /cmake-3.27.1-linux-x86_64.sh --prefix=/opt/cmake --skip-license
# RUN ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake
RUN cmake --version

ENV PYTHONPATH=/root/mount/Matterport3DSimulator/build
