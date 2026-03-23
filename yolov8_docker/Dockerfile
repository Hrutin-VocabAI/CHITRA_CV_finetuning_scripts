# Use the official Ultralytics Docker image.
# Why? It comes pre-configured with CUDA, PyTorch, OpenCV, and all dependencies required by YOLOv8.
# This prevents 99% of "works on my machine" issues and saves you from dependency hell in Python/C++.
<<<<<<< HEAD
# Specifically using the '-python' tag to get a pure Python environment without extra bloated CLI tools.
=======
# Specifically using the fully-featured latest tag which correctly bundles CUDA 11/12 GPU bindings
>>>>>>> 5627fae ( local updates differ from server)
FROM ultralytics/ultralytics:latest

# Set the working directory inside the container
# This is where all the action will happen inside the Docker environment.
WORKDIR /app

# Copy the generic training script from your local machine to the container's working directory
COPY train.py /app/train.py

# Optional: if you have custom packages (e.g. for API tracking), copy requirements and install them.
# COPY requirements.txt /app/requirements.txt
# RUN pip install -r requirements.txt

# The ENTRYPOINT defines the default command the container will execute when it starts.
# When someone runs `docker run ... your_image`, it automatically executes `python /app/train.py`.
# Since train.py reads environment variables, the user doesn't need to touch this file again.
ENTRYPOINT ["python", "/app/train.py"]
