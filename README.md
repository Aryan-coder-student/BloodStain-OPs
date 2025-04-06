---

# Blood Detection MLOps Pipeline

This repository implements an end-to-end MLOps pipeline for training a YOLOv8 model to detect blood in images, versioning data and models with DVC, orchestrating the workflow with Apache Airflow, and deploying the trained model as a FastAPI service in a Docker container. The pipeline integrates with Roboflow for dataset retrieval and DagsHub for remote storage and tracking.

## Project Overview
- **Objective**: Automate the process of downloading a blood detection dataset, training a YOLOv8 model, versioning artifacts, and deploying the model as a REST API.
- **Tools Used**:
  - **Roboflow**: Dataset source.
  - **Ultralytics YOLOv8**: Model training.
  - **DVC**: Data and model versioning.
  - **Airflow**: Workflow orchestration.
  - **FastAPI**: Model deployment.
  - **Docker**: Containerization.
  - **DagsHub**: Remote storage and tracking.

## Folder Structure
```
mlops_blood_detection/
├── dags/                     # Airflow DAGs for workflow orchestration
│   └── train_dag.py          # DAG to run the pipeline
├── src/                      # Source code for pipeline stages and API
│   ├── download_data.py     # Script to download dataset from Roboflow
│   ├── train_model.py       # Script to train the YOLOv8 model
│   └── app.py               # FastAPI app to serve the trained model
├── Dockerfile               # Docker configuration for FastAPI deployment
├── config.yaml              # Configuration file for pipeline parameters
├── requirements.txt         # Python dependencies
├── dvc.yaml                 # DVC pipeline definition for versioning
├── data/                    # Dataset storage (tracked by DVC)
├── models/                  # Model storage (tracked by DVC)
└── README.md                # Project documentation (this file)
```

## File Descriptions

### `dags/train_dag.py`
- **Purpose**: An Apache Airflow DAG that orchestrates the pipeline by running the data download, model training, and DVC push stages.
- **Details**:
  - Defines a DAG named `blood_detection_train_dag` with a daily schedule.
  - Uses `BashOperator` to execute Python scripts and DVC commands.
  - Task sequence: `download_data` → `train_model` → `dvc_push`.
- **Key Code**:
  ```python
  download_task = BashOperator(
      task_id="download_data",
      bash_command="python /opt/airflow/src/download_data.py",
  )
  train_task = BashOperator(
      task_id="train_model",
      bash_command="python /opt/airflow/src/train_model.py",
  )
  dvc_push_task = BashOperator(
      task_id="dvc_push",
      bash_command="dvc push",
  )
  download_task >> train_task >> dvc_push_task
  ```

### `src/download_data.py`
- **Purpose**: Downloads the blood detection dataset from Roboflow using the configuration in `config.yaml`.
- **Details**: Uses the Roboflow API to fetch version 2 of the `blood-detection-v3` project in YOLOv8 format.

### `src/train_model.py`
- **Purpose**: Trains a YOLOv8 model on the downloaded dataset and saves the trained model.
- **Details**: Loads parameters from `config.yaml`, trains the model for 10 epochs, and saves it to `models/yolov8n_blood_detection.pt`.

### `src/app.py`
- **Purpose**: A FastAPI application that serves the trained YOLOv8 model for blood detection and blurring.
- **Details**: Accepts image uploads via POST requests, processes them with the model, blurs detected regions, and returns a response.

### `Dockerfile`
- **Purpose**: Defines the Docker image for deploying the FastAPI app.
- **Details**: Uses `python:3.9-slim`, installs dependencies, copies source code and models, and runs the app with Uvicorn on port 8000.

### `config.yaml`
- **Purpose**: Central configuration file for the pipeline.
- **Details**:
  ```yaml
  data:
    roboflow_api_key: ""  # Your Roboflow API key
    workspace: "bloodviolenceweapons"
    project: "blood-detection-v3"
    version: 2
    format: "yolov8"
  training:
    model: "yolov8n.pt"
    epochs: 10
    imgsz: 640
    batch: 16
    name: "yolov8n_blood_detection"
  paths:
    data_dir: "data"
    model_dir: "models"
    output_model: "models/yolov8n_blood_detection.pt"
  ```

### `requirements.txt`
- **Purpose**: Lists Python dependencies required for the pipeline and FastAPI app.
- **Content**: Includes `roboflow`, `ultralytics`, `fastapi`, `uvicorn`, `dvc`, `apache-airflow`, etc.

### `dvc.yaml`
- **Purpose**: Defines the DVC pipeline stages for data downloading and model training.
- **Details**:
  ```yaml
  stages:
    download_data:
      cmd: python src/download_data.py
      outs:
        - data/
    train_model:
      cmd: python src/train_model.py
      deps:
        - data/
        - src/train_model.py
        - config.yaml
      outs:
        - models/yolov8n_blood_detection.pt
  ```
  - **Stages**:
    - `download_data`: Downloads the dataset and tracks the `data/` directory.
    - `train_model`: Trains the model, depends on the dataset and scripts, and tracks the output model.

### `data/` and `models/`
- **Purpose**: Directories for storing the dataset and trained model, respectively.
- **Details**: Managed by DVC and pushed to DagsHub for versioning.

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/<your-username>/mlops_blood_detection.git
   cd mlops_blood_detection
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure DVC**:
   - Initialize DVC:
     ```bash
     dvc init
     ```
   - Set up DagsHub remote (replace with your credentials):
     ```bash
     dvc remote add -d origin https://dagshub.com/<your-username>/mlops_blood_detection.dvc
     dvc remote modify origin --local auth basic
     dvc remote modify origin --local user <your-dagshub-username>
     dvc remote modify origin --local password <your-dagshub-token>
     ```

4. **Run the DVC Pipeline**:
   ```bash
   dvc repro
   dvc push
   ```

5. **Set Up Airflow**:
   - Initialize Airflow database:
     ```bash
     airflow db init
     ```
   - Copy `dags/train_dag.py` to your Airflow `dags/` folder (e.g., `/opt/airflow/dags/`).
   - Start Airflow:
     ```bash
     airflow webserver -p 8080 &
     airflow scheduler &
     ```

6. **Build and Run Docker**:
   ```bash
   docker build -t blood-detection-api .
   docker run -p 8000:8000 blood-detection-api
   ```

7. **Test the API**:
   ```bash
   curl -X POST -F "file=@test_image.png" http://localhost:8000/predict/
   ```

## Usage
- **Pipeline**: Trigger the Airflow DAG via the UI or CLI (`airflow dags trigger -e blood_detection_train_dag`).
- **API**: Send images to `http://localhost:8000/predict/` to get blood detection and blurring results.

## Notes
- Replace placeholders in `config.yaml` (e.g., `roboflow_api_key`) with your actual credentials.
- Ensure Airflow and Docker are properly configured on your system.
- Use DagsHub to track experiments and artifacts.

---
