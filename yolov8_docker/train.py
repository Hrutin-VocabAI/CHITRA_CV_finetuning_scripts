import os
import torch
from ultralytics import YOLO

def str2bool(v):
    return String(v).lower() in ("yes", "true", "t", "1")

def main():
    print("=========================================")
    print("       YOLOv8 Docker Training Script     ")
    print("=========================================")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: No GPU detected! Training will be on CPU and VERY slow.")
    print("=========================================\n")

    # Read hyperparameters from environment variables with sensible defaults
    model_name = os.getenv("MODEL_NAME", "yolov8n.pt")
    
    # Path inside the container (this is where the host volume will be mounted)
    data_path = os.getenv("DATA_PATH", "/app/data/data.yaml")
    
    epochs = int(os.getenv("EPOCHS", "60"))
    imgsz = int(os.getenv("IMGSZ", "640"))
    
    # Use GPU 0 by default if available, otherwise fallback to CPU
    device = os.getenv("DEVICE", "0" if torch.cuda.is_available() else "cpu")
    batch_size = int(os.getenv("BATCH_SIZE", "16"))
    plots_env = os.getenv("PLOTS", "True")
    plots = str2bool(plots_env) if isinstance(plots_env, str) else True
    
    project_name = os.getenv("PROJECT_NAME", "training_project")
    run_name = os.getenv("RUN_NAME", "run_1")

    print("\n--- Training Configuration ---")
    print(f"Model:        {model_name}")
    print(f"Data Path:    {data_path}")
    print(f"Epochs:       {epochs}")
    print(f"Image Size:   {imgsz}")
    print(f"Device:       {device}")
    print(f"Batch Size:   {batch_size}")
    print(f"Plots:        {plots}")
    print(f"Project Name: {project_name}")
    print(f"Run Name:     {run_name}")
    print("------------------------------\n")

    # Validate that data.yaml exists inside the container
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Data file NOT FOUND at: {data_path}\n"
            f"Please ensure you mounted the dataset correctly using Docker volumes "
            f"and that the data.yaml file is present inside the mounted directory!"
        )

    print(f"Loading Model: {model_name}...")
    model = YOLO(model_name)

    print("Starting Training...")
    # 'project' specifies where ultralytics saves the outputs. 
    # We save to /app/runs so that we can map a local volume to grab the trained weights!
    results = model.train(
        data=data_path,
        epochs=epochs,
        imgsz=imgsz,
        device=device,
        batch=batch_size,
        plots=plots,
        project=f"/app/runs/{project_name}",
        name=run_name
    )

    print("\nTraining Complete! 🎉")
    if hasattr(results, 'results_dict'):
        print(results.results_dict)

if __name__ == "__main__":
    main()
