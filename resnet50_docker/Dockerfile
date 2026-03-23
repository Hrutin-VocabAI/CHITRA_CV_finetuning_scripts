# Use the official TensorFlow GPU image
# This automatically handles the nightmare of CUDA, cuDNN, and TensorFlow version matching
FROM tensorflow/tensorflow:latest-gpu

# Set the working directory inside the container
WORKDIR /app

# The build context is set to the parent folder so we can copy the script
# Copy the pure Python script into the container
COPY resnet_50_d2d_train.py /app/train.py

# Set the entrypoint to Python. 
# We don't define the arguments here because we will pass them dynamically through docker-compose.
ENTRYPOINT ["python", "/app/train.py"]
