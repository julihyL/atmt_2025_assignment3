from huggingface_hub import HfApi
import os

repo_id = "hyluo/atmt_a3_MQA"  
model_dir = "/data/atmt_2025_assignment3/cz-en"

# Read Hugging Face token from environment variable
token = os.getenv("HF_TOKEN")
if not token:
    raise ValueError("❌ Environment variable HF_TOKEN is not set. "
                     "Run 'export HF_TOKEN=\"your_token_here\"' before executing this script.")

# Initialize the Hugging Face API
api = HfApi()

# Upload the entire local folder to Hugging Face Hub repository
print(f"🚀 Uploading folder '{model_dir}' to https://huggingface.co/{repo_id}")
api.upload_folder(
    folder_path=model_dir,   # Local model directory
    repo_id=repo_id,         # Destination repository on Hugging Face
    repo_type="model",       # Type of repository (model / dataset / space)
    token=token              # Authentication token
)

print("✅ Upload complete! You can check your model here:")
print(f"https://huggingface.co/{repo_id}")

