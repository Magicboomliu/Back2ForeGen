import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import os
from tqdm import tqdm

# 加载预训练的 BLIP-2 模型和处理器
processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl")

# 检查设备（GPU 或 CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.half()
model.to(device)

quit()


# 使用 BLIP-2 生成图片描述
def generate_image_description(image):
    # 将图片处理为模型可识别的格式
    inputs = processor(images=image, return_tensors="pt").to(device)
    # 生成图片描述
    generated_ids = model.generate(**inputs, max_length=50)
    # 解码生成的文本描述
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return generated_text


if __name__=="__main__":
    
    root_path = "/mnt/nfs-mnj-home-43/i24_ziliu/dataset/CrossDomainData/damon/images"
    filenames = sorted(os.listdir(root_path))
    description_list = []
    
    for image in os.listdir(root_path):
        fname = os.path.join(root_path,image)
        image = Image.open(fname)
        # 获取原始尺寸
        original_size = image.size
        # 计算缩放比例
        max_side = max(original_size)
        if max_side > 512:
            scale_ratio = 512 / max_side
            new_size = (int(original_size[0] * scale_ratio), int(original_size[1] * scale_ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)

        # 输出图片描述
        description = generate_image_description(image)
        description_list.append(description)

    
    with open("description.txt",'w') as f:
        for idx, line in enumerate(description_list):
            if idx!=len(description_list)-1:
                f.writelines(line+"\n")
            else:
                f.writelines(line)