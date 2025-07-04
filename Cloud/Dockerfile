# Prometheus_Agent/Cloud/Dockerfile

# --- Stage 1: Build Stage ---
# Use a full Python image to build dependencies that might need C compilation.
# Using a specific, modern version like 3.11 ensures reproducibility.
FROM python:3.11-slim-bookworm AS builder

# Set a non-root user for 06a_coding_security best practices
RUN useradd --create-home --shell /bin/bash prometheus_user
WORKDIR /home/prometheus_user/app

# Install system-level build dependencies.
# build-essential is needed for compiling some Python packages.
# The other libraries (libgl1, etc.) are kept in case any dependency
# (like a plotting library) needs them during installation.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements file to leverage Docker's layer caching.
COPY --chown=prometheus_user:prometheus_user requirements.txt .

# Create a virtual environment inside the builder stage. This isolates dependencies.
RUN python -m venv /home/prometheus_user/venv
ENV PATH="/home/prometheus_user/venv/bin:$PATH"

# Install all Python dependencies into the virtual environment.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# --- Stage 2: Final Runtime Stage ---
# Start from a fresh, slim base image for a lean final product.
FROM python:3.11-slim-bookworm

# Create and set the non-root user for the runtime environment.
RUN useradd --create-home --shell /bin/bash prometheus_user
USER prometheus_user
WORKDIR /home/prometheus_user/app

# Install only the necessary runtime shared libraries.
# Note: build-essential is NOT included here, making the final image smaller and more secure.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the pre-built virtual environment from the builder stage.
COPY --from=builder /home/prometheus_user/venv /home/prometheus_user/venv

# Copy the entire project source code into the final image.
COPY --chown=prometheus_user:prometheus_user . .

# Set the PATH environment variable to use the virtual environment's Python.
ENV PATH="/home/prometheus_user/venv/bin:$PATH"

# --- CORE ARCHITECTURAL CHANGE ---
# The old Dockerfile was set to run a GUI. The new goal is to run the agent's
# core logic as a sandboxed API server. We expose the port for the API.
EXPOSE 8000

# The command to run when the container starts.
# This launches the FastAPI server, which is the new entry point for the agent's core.
# We use "--host 0.0.0.0" to make it accessible from outside the container.
CMD ["uvicorn", "main_api:app", "--host", "0.0.0.0", "--port", "8000"]
