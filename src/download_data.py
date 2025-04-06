import yaml
from roboflow import Roboflow

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

data_config = config["data"]
rf = Roboflow(api_key=data_config["roboflow_api_key"])
project = rf.workspace(data_config["workspace"]).project(data_config["project"])
version = project.version(data_config["version"])
dataset = version.download(data_config["format"])

print(f"Dataset downloaded to: {dataset.location}")