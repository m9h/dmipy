FROM nvcr.io/nvidia/pytorch:25.12-py3

WORKDIR /app

# Install dependencies
RUN pip install "sbi==0.22.0" dipy nibabel scipy

# Copy the current directory explicitly to avoid missing files if using a bind mount isn't enough for build
WORKDIR /app
COPY . /app

# Install dmipy-jax and its dependencies
# Using uv for faster installation if available, or just pip
# We need to make sure we get the cuda version of jax
RUN pip install -U "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip install .

# Create data directory
RUN mkdir -p /data

# Default command
CMD ["python", "-c", "import jax; print(jax.devices())"]
