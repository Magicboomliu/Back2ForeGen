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

def remove_repeated_words(sentence):
    stop_words = {"is", "are", "the", "in", "and", "of", "to", "a", "an", "on", "with", "by", "at", "for","so","this","that","two","three"}
    words = sentence.split()
    word_count = {}
    
    # Count the occurrences of each word
    for word in words:
        word_lower = word.lower()  # To ensure case-insensitivity
        if word_lower in word_count:
            word_count[word_lower] += 1
        else:
            word_count[word_lower] = 1
    
    # Reconstruct the sentence without words that appear more than once, except stop words
    result = [word for word in words if word_count[word.lower()] == 1 or word.lower() in stop_words]
    
    return ' '.join(result)

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
    
    fname = "/home/zliu/PFN_Internship/DDIM_Inversion/diffusers_ddim_inversion/anime.jpg"
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
    description=  remove_repeated_words(description)
    
    print(description)
        