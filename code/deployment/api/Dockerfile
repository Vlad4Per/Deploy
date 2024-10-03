# Use an official Python runtime as a base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /api

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the FastAPI port (8000)
EXPOSE 8000

# Command to run FastAPI on container start
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
