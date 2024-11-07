# Use Python 3.10 as base image
FROM python:3.10-slim

# Set working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip first
RUN pip install --upgrade pip

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files into the container
COPY code/ ./code/
COPY data/ ./data/
COPY models/ ./models/
COPY src/ ./src/

# Set the working directory to where the training script is located
WORKDIR /app/code

# Command to run the training script
CMD ["python", "model_training.py", "--train-path", "../data/application_train.csv", "--model-output", "../models/model.pkl"]