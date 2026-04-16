# Base image
FROM python:3.9-slim

# Working directory set karein
WORKDIR /app

# Requirements file copy aur install karein
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pura code copy karein
COPY . .

# Default command (Wese pipeline runs ke waqt KFP ise override kar deta hai)
CMD ["python", "pipeline.py"]
