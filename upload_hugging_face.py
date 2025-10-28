from huggingface_hub import HfApi
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import os

repo_id = "hyluo/atmt_a3_MQA"
model_dir = "/data/atmt_2025_assignment3/cz-en"

token = os.getenv("HF_TOKEN")
if not token:
    raise ValueError("❌ Missing HF_TOKEN environment variable")

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

tokenizer.push_to_hub(repo_id, token=token)
model.push_to_hub(repo_id, token=token)
