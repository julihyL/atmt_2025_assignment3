from huggingface_hub import login
from huggingface_hub import HfApi

# 需要在huggingface里面注册后获取的API
login("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")

api = HfApi()
model_name = "hyLuo/atmt_MQA_model"  # 定义微调后的模型名称
api.create_repo(model_name, exist_ok=True)  # 创建一个新的 Hugging Face 公开仓库

api.upload_folder(
    folder_path="./cz-en",  # 你的微调模型输出路径
    repo_id=model_name,  # 你的 Hugging Face 仓库
    repo_type="model"
)
print('模型上传成功')