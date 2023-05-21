# syntax=docker/dockerfile:1
FROM python:3.9
ENV PYTHONPATH "/usr/lib/python3/dist-packages:/usr/local/lib/python3.9/site-packages"
WORKDIR /code
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get autoremove -y \
    && apt-get install -y \
        gcc \
        build-essential \
        zlib1g-dev \
        wget \
        unzip \
        cmake \
        python3-dev \
        gfortran \
        libblas-dev \
        liblapack-dev \
        libatlas-base-dev \
    && apt-get clean

COPY requirements.txt requirements.txt
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN python3 -m pip install --upgrade pip setuptools wheel
RUN python3 -m pip install -r requirements.txt --no-cache-dir
EXPOSE 5000
COPY . .
