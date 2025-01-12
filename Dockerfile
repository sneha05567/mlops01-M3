# Use the official Python base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install the required Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y \
    libglib2.0-dev \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Expose the port Flask will run on
EXPOSE 5000

# Command to run the application
CMD ["python", "app/app.py"]
