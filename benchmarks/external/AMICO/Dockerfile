FROM nvcr.io/nvidia/pytorch:23.10-py3

# Install system dependencies for SPAMS and AMICO
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# dipy, dmri-dicelib, and nibabel are required by AMICO
RUN pip install --no-cache-dir \
    dipy \
    dmri-dicelib \
    nibabel

# Install SPAMS from source with patch for ARM64 (removing -m64 flag)
RUN pip download --no-binary :all: --dest /tmp/spams_source spams && \
    tar -xvf /tmp/spams_source/spams-*.tar.gz -C /tmp && \
    cd /tmp/spams-* && \
    sed -i 's/-m64//g' setup.py && \
    pip install .

# Copy AMICO source code
COPY . /opt/AMICO

# Install AMICO
WORKDIR /opt/AMICO
RUN pip install .

# Set up staging directory for data exchange
RUN mkdir -p /data
WORKDIR /data

# Default command (will be overridden by the oracle wrapper)
CMD ["python", "--version"]
