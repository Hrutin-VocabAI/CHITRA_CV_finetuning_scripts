# 🐳 Production-Ready YOLOv8 Training with Docker

This folder contains a fully containerized, industry-standard setup for fine-tuning YOLOv8 models. By using Docker, you eliminate the "works on my machine" problem, avoid corrupting your local Python environment, and make it trivial to run training jobs on remote cloud GPU servers.

## 🧠 Why is this "Production-Ready & Future-Orientated"?

When moving from Jupyter Notebooks (`.ipynb`) to production systems, there are a few major "Edge Cases" and best practices you must follow:

### 1. 🌐 Environment Variables for Reusability
You wanted **one image for all training scripts**. Hardcoding values inside a Python script means you must edit the code every time you train.
**The Fix:** The `train.py` script reads from standard OS Environment Variables (`os.getenv`). Now, the Docker Image is totally generic. You never have to touch `train.py` or rebuild the Docker `image` again simply to change the `epochs` or `batch` size. You just change the `.env` file!

### 2. 📂 Volume Mounting for Transient Data
**The Edge Case:** Docker containers are "ephemeral" (temporary). If you train a model *inside* the container and the container finishes or crashes, the `.pt` file (your trained weights) is instantly **deleted** forever.
**The Fix:** We use **Volumes** in `docker-compose.yml`. We link a local `./runs` folder on your Mac/Linux host to `/app/runs` inside the container. When YOLO saves the `.pt` file inside the container, Docker reaches through the "wormhole" and drops it securely onto your local hard drive!

### 3. 🤔 Separating Host Paths vs Container Paths
**The Edge Case:** In your Jupyter notebook, your `data.yaml` path is hardcoded as: `/media/vocab/DATA5/.../data.yaml`. The Docker container doesn't have a `media` folder or a `vocab` user. It is isolated.
**The Fix:** `docker-compose.yml` mounts your huge dataset folder into `/app/data`. We tell YOLO inside the container to *only* look at `/app/data/data.yaml`. All it knows is `/app/data`, regardless of where the data actually lives on your host computer.

### 4. 🧨 Docker and GPU Access
**The Edge Case:** By default, Docker containers **cannot see your GPU**, even if you have one. They run purely on CPU.
**The Fix:** We use the `deploy -> resources -> nvidia` tag in `docker-compose.yml`. This tells the Docker Engine to securely pass your host machine's physical GPU inside the container. Also, we base the `Dockerfile` entirely off `ultralytics/ultralytics:latest-python` which already handles the nightmare of CUDA + PyTorch dependencies for you.

---

## 🚀 How To Run It (The Workflow)

### Step 1: Prepare the Environment File
Copy the example file to create your actual `.env` file (Docker Compose reads `.env` automatically).
```bash
cp .env.example .env
```

### Step 2: Edit `.env`
Open `.env` in an editor and change:
1. `HOST_DATA_PATH`: Point this to the absolute local directory where your dataset and `data.yaml` live.
2. `PROJECT_NAME` / `RUN_NAME`: Give your training run a recognizable name.
3. Hyperparameters like `EPOCHS` and `BATCH_SIZE`.

### Step 3: Build & Run using Docker Compose
The magic command. It will:
- Build the `train.py` script into a fresh Docker image (only takes seconds because the base image is heavily cached).
- Read the `.env` file.
- Mount your GPU.
- Start the training.

Run the following inside the `yolov8_docker` directory:
```bash
docker compose up --build
```
*(Tip: If you want it to run entirely in the background, use `docker compose up --build -d`. You can check the logs later with `docker compose logs -f`.)*

### Step 4: Collect Your Weights
When training finishes, check the new `./runs/<PROJECT_NAME>/<RUN_NAME>/weights` folder right next to your docker-compose file. Your shiny new `best.pt` file will be waiting for you!

## 🎉 Summary
Now, if you want to train on Barcodes instead of White Cement, simply duplicate the `.env` file, change `HOST_DATA_PATH` to point to the barcode dataset, change `RUN_NAME` to `barcodes_run`, and type `docker compose up`! You never have to touch Python code or Jupyter Notebooks again.
