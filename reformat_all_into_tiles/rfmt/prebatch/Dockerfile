FROM python:3.9-slim

RUN pip3 install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install openexr_numpy

WORKDIR /rfmt/

COPY . .
