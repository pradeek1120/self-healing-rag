# Dockerfile
# Builds from official openenv-base image (required by hackathon)
# Build:  docker build -t self-healing-rag .
# Run:    docker run -p 7860:7860 self-healing-rag

ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE}

WORKDIR /app

# Copy requirements and install
COPY server/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy entire project
COPY . .

# Expose port (7860 for Hugging Face Spaces)
EXPOSE 7860

# Start the FastAPI server with uvicorn
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
