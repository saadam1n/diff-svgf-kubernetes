FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /docker


RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 git -y

RUN pip install opencv-python

COPY . .

CMD ["/bin/sh", "train.sh"]
 