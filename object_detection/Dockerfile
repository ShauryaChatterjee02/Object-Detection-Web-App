# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed dependencies
RUN pip install --no-cache-dir streamlit pillow matplotlib torch torchvision

# Expose the port streamlit runs on
EXPOSE 8501

# Run streamlit command
CMD ["streamlit", "run", "App.py"]

