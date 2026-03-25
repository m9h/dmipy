Bootstrap: docker
From: nvcr.io/nvidia/pytorch:25.12-py3

%post
    # Set working directory
    mkdir -p /app
    cd /app

    # Install system dependencies if needed (none explicitly in Dockerfile but apt-get here if needed)
    
    # Install dependencies
    pip install "sbi==0.22.0" dipy nibabel scipy

    # Copy files
    # Note: %files copies from host to container. 
    # We copy current directory content to /app in container
    
%files
    . /app

%post
    cd /app
    
    # Install dmipy-jax and its dependencies
    # Using the same command as Dockerfile
    pip install -U "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    pip install .

    # Create data directory
    mkdir -p /data

%runscript
    python -c "import jax; print(jax.devices())"

%labels
    Author dmipy-jax-team
    Version v0.1.0
