import yaml
from ultralytics import YOLO

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

train_config = config["training"]
data_config = config["data"]
paths_config = config["paths"]

model = YOLO(train_config["model"])
model.train(
    data=f"{paths_config['data_dir']}/{data_config['project']}-{data_config['version']}/data.yaml",
    epochs=train_config["epochs"],
    imgsz=train_config["imgsz"],
    batch=train_config["batch"],
    name=train_config["name"],
)

# Save the trained model
model.save(paths_config["output_model"])
print(f"Model saved to: {paths_config['output_model']}")