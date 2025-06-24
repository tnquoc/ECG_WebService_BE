# Use an official Python runtime as a parent image
FROM python:3.10-slim

ENV TZ=Asia/Ho_Chi_Minh

# Install gcc and other build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*  # Clean up to reduce image size

# Set the working directory in the container
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir uvicorn

# Set the PYTHONPATH environment variable
ENV PYTHONPATH=/app

# Copy the current directory contents into the container at /app
COPY . /app

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run app.py when the container launches
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]