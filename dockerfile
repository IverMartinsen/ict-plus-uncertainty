# Use a specific version of the base image
FROM --platform=amd64 tensorflow/tensorflow:2.8.4-gpu

# Set working directory inside image
WORKDIR /src

COPY requirements.txt /src

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
