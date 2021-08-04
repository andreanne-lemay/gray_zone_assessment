# Start with the base CUDA image
FROM projectmonai/monai:latest

WORKDIR /

ENV TZ=America/Montreal
ARG DEBIAN_FRONTEND=noninteractive

# Install useful programs and build tools
RUN apt update && apt install -y \
       build-essential \
       cifs-utils \
       cmake \
       git \
       vim \
       swig \
       tmux \
       wget && \
       apt clean


# Install Python packages from a requirements file
COPY requirements.txt /
RUN pip install --upgrade pip==20.1.1 && pip install --no-cache-dir -r requirements.txt
