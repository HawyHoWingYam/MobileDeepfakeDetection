# Multi-stage build for AWARE-NET
# Base image with CUDA support
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    CUDA_VISIBLE_DEVICES=0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    unzip \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgtk-3-0 \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /workspace/aware-net

# Install conda environment
COPY environment.yml .
RUN conda env create -f environment.yml

# Activate environment
SHELL ["conda", "run", "-n", "aware-net", "/bin/bash", "-c"]

# Development stage
FROM base AS development

# Copy source code
COPY . .

# Install additional development tools
RUN conda run -n aware-net pip install \
    jupyterlab==4.0.5 \
    jupyterlab-git==0.42.0 \
    ipywidgets==8.1.0

# Set up Jupyter Lab
RUN conda run -n aware-net jupyter lab --generate-config

# Expose ports for Jupyter and TensorBoard
EXPOSE 8888 6006

# Default command for development
CMD ["conda", "run", "-n", "aware-net", "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# Production stage
FROM base AS production

# Copy only necessary files
COPY src/ ./src/
COPY configs/ ./configs/
COPY requirements.txt .

# Set up production environment
RUN conda run -n aware-net pip install --no-deps -e .

# Create non-root user
RUN useradd -m -u 1000 aware && \
    chown -R aware:aware /workspace/aware-net
USER aware

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD conda run -n aware-net python -c "import torch; print(torch.cuda.is_available())"

# Default command for production
CMD ["conda", "run", "-n", "aware-net", "python", "-m", "src.stage_00.train_baseline"]

# Inference stage (lightweight)
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime AS inference

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install minimal dependencies for inference
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    timm==0.9.7 \
    opencv-python==4.8.0.76 \
    numpy==1.24.3 \
    pillow==10.0.0

# Copy trained models and inference code
COPY --from=production /workspace/aware-net/models/ ./models/
COPY --from=production /workspace/aware-net/src/inference/ ./src/inference/

# Set up inference user
RUN useradd -m -u 1000 inference && \
    chown -R inference:inference /app
USER inference

# Expose inference API port
EXPOSE 8000

CMD ["python", "-m", "src.inference.api"]