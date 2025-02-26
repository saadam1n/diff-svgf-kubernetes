FROM python:3.11-slim
WORKDIR /dir

RUN apt update
RUN apt install wget cmake g++ libgles2-mesa-dev libglew-dev libglfw3-dev libglm-dev zlib1g-dev git -y

RUN git clone https://github.com/saadam1n/infinigen.git
RUN ls
RUN cd infinigen
RUN INFINIGEN_MINIMAL_INSTALL=True pip install -e .

