from huggingface_hub import HfApi
import os

repo_id = "hyluo/atmt_a3_MQA"
model_dir = "/data/atmt_2025_assignment3/cz-en"
token = os.getenv("HF_TOKEN")

api.upload_folder(
    folder_path=model_dir,
    repo_id=repo_id,
    repo_type="model",
    token=token
)
