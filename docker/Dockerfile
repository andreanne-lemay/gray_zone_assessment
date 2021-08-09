# Start with the base CUDA image
FROM projectmonai/monai:latest

WORKDIR /

# Install Python packages from a requirements file
COPY requirements.txt /
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt
