
from safetensors.torch import save_file
import torch

# 加载 .pt 文件
pt_file_path = "/mnt/nfs-mnj-home-43/i24_ziliu/pretrained_models/VAEs/ClearVAE_V2.3_fp16.pt" # 替换为你的 .pt 文件路径
model_state_dict = torch.load(pt_file_path, map_location="cpu")['state_dict']  # 将模型加载到CPU中
# 保存为 .safetensors 文件
safetensors_file_path = "/mnt/nfs-mnj-home-43/i24_ziliu/pretrained_models/VAEs/ClearVAE_V2.3_fp16.safetensors"  # 替换为你想保存的 .safetensors 文件路径
save_file(model_state_dict, safetensors_file_path)

print(f"Successfully converted {pt_file_path} to {safetensors_file_path}")