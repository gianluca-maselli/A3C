FROM ubuntu:20.04
RUN apt update && apt install htop git wget vim curl -y

WORKDIR /root/
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh
RUN chmod +x Miniconda3-py38_4.10.3-Linux-x86_64.sh
RUN /root/Miniconda3-py38_4.10.3-Linux-x86_64.sh -b && eval "$(/root/miniconda3/bin/conda shell.bash hook)" && /root/miniconda3/bin/conda clean -afy
RUN /root/miniconda3/bin/conda init
RUN echo 'conda activate main' >> ~/.bashrc
RUN /root/miniconda3/bin/conda create --name main python==3.9
# Env vars for the nvidia-container-runtime.
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute


RUN /root/miniconda3/bin/conda run -n main pip install torch torchvision torchaudio
RUN /root/miniconda3/bin/conda run -n main pip install opencv-python
RUN /root/miniconda3/bin/conda run -n main conda install numpy
RUN /root/miniconda3/bin/conda run -n main pip install matplotlib
RUN /root/miniconda3/bin/conda run -n main pip install IPython
RUN /root/miniconda3/bin/conda run -n main pip install gym[atari]
RUN /root/miniconda3/bin/conda run -n main pip install "gym[atari, accept-rom-license]"
RUN /root/miniconda3/bin/conda run -n main pip install pip install imageio
RUN /root/miniconda3/bin/conda run -n main pip install cmake


ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

WORKDIR /workspace
