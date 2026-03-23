# 🐳 Production-Ready ResNet50 Training with Docker

This folder dockerizes your `resnet_50_d2d_train.py` script perfectly using the official `tensorflow/tensorflow:latest-gpu` image.

## 🚀 The Docker + Argparse Trick (Important!)

Unlike your `train.py` script for YOLO, the `resnet_50_d2d_train.py` script uses Python's `argparse` to expect command-line flags like `--img_size 224`.

You do **NOT** need to edit `resnet_50_d2d_train.py` to use `os.getenv`!

Instead, we use a classic Docker "Magic Bridge" trick in `docker-compose.yml`. We configure the `command:` property to read your `.env` variables and literally type out the arguments for the Python script automatically. It acts exactly as if you stood at the keyboard and typed `python train.py --img_size 224 ...` by yourself!

---

## 🛠️ How To Run It

### Step 1: Create your Environment File
```bash
cp .env.example .env
```

### Step 2: Configure `.env`
Open `.env` in any editor. The most critical one is:
- `HOST_SPLITS_DIR`: Set this to the absolute path of the folder containing your dataset. **This folder must have the `train`, `val`, and `test` directories inside it!**
- Tweak the other parameters like `BATCH_SIZE`, `EPOCHS_HEAD`, `MODEL_FILENAME` as desired.

### Step 3: Train!
Run the magical Compose command. Because the context points back one folder (`..`), Docker will automatically find your `resnet_50_d2d_train.py` script and pull it into the image.

Run the following inside the `resnet50_docker` directory:
```bash
docker compose up --build
```
*(Use `docker compose up --build -d` to run it in the background!)*

### Step 4: Collect the Trained Kernel
Once training and testing finishes, your fine-tuned model `.keras` file will magically deploy into `resnet50_docker/output_models/` on your host machine.

---
## 🎉 Moving to Docker Hub (A Recap!)
Just like the YOLO script:
1. `docker build -t your_username/resnet50-trainer:latest -f Dockerfile ..`
2. `docker push your_username/resnet50-trainer:latest`
3. Then move `docker-compose.yml` and `.env` anywhere in the world and run it!
