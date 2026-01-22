FROM nvcr.io/nvidia/pytorch:25.12-py3

WORKDIR /app

# Install dependencies
RUN pip install "sbi==0.22.0" dipy nibabel scipy

# Clone the SBI_dMRI repository
RUN git clone https://github.com/SPMIC-UoN/SBI_dMRI.git

# Set Python path to include the repository
ENV PYTHONPATH="${PYTHONPATH}:/app/SBI_dMRI"

# Copy the simulation scripts
COPY generate_oracle_data.py /app/generate_oracle_data.py
COPY generate_complex_oracle.py /app/generate_complex_oracle.py
COPY generate_connectome_oracle.py /app/generate_connectome_oracle.py

# Create data directory
RUN mkdir -p /data

# Default command can be overridden
CMD ["python", "generate_complex_oracle.py"]
