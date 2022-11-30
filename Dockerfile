# Start from NVIDIA CUDA container (Ubuntu 20.04)
FROM nvcr.io/nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

COPY requirements.txt /tmp/requirements.txt
COPY fsct /opt/fsct

# Set as non-interactve
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y  \
	    apt-utils git curl vim unzip wget \
	    build-essential 
	
# Install pip and CUDA compatabiilty layer
RUN apt-get install -y python3-pip cuda-compat-11-3

# Create symlink for libcusolver, needed for tensorflow
RUN	cd /usr/local/cuda-11.3/lib64/ && \
		ln -s libcusolver.so.11 libcusolver.so.10 && \
		cd -

# Install requirements using pip
RUN pip install -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Make sure scripts are excecutable
RUN chmod a+x /opt/fsct/run.py /opt/fsct/points2trees.py

# Set environmental variables
ENV  LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.3/lib:/usr/local/cuda-11.3/lib64:/usr/local/cuda/compat
ENV PYTHONPATH=/opt/:$PYTHONPATH
ENV PATH=/opt/fsct:$PATH

