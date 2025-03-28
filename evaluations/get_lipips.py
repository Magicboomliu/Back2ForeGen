import os
import torch
import lpips
from PIL import Image
import torchvision.transforms as transforms
import os
from tqdm import tqdm
import json



def calculate_mean_lpips(est_folder, GT_folder, model='alex', image_size=(256, 256)):
    """
    计算两个文件夹中相同文件名图像的平均LPIPS。

    参数:
    est_folder (str): 生成图像文件夹路径。
    GT_folder (str): 真实图像文件夹路径。
    model (str): LPIPS 模型，默认使用 'alex'。
    image_size (tuple): 图像预处理的尺寸，默认 (256, 256)。

    返回:
    float: 平均LPIPS得分。
    """
    # 初始化LPIPS模型
    lpips_model = lpips.LPIPS(net=model)

    # 加载和预处理图像
    def load_image(image_path):
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])
        image = Image.open(image_path).convert('RGB')
        return transform(image).unsqueeze(0)

    # 获取两个文件夹中共有的文件名
    est_images = set([f for f in os.listdir(est_folder) if f.endswith(('png', 'jpg', 'jpeg'))])
    GT_images = set([f for f in os.listdir(GT_folder) if f.endswith(('png', 'jpg', 'jpeg'))])
    common_images = est_images.intersection(GT_images)

    # 计算LPIPS得分
    lpips_scores = []
    for img_name in common_images:
        est_img_path = os.path.join(est_folder, img_name)
        GT_img_path = os.path.join(GT_folder, img_name)

        est_img = load_image(est_img_path)
        GT_img = load_image(GT_img_path)

        # 计算LPIPS
        lpips_score = lpips_model(est_img, GT_img)
        lpips_scores.append(lpips_score.item())

    # 计算平均LPIPS
    mean_lpips = sum(lpips_scores) / len(lpips_scores) if lpips_scores else 0
    return mean_lpips


import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Infenrece")

    parser.add_argument(
        "--root_folder",
        type=str,
        default="/home/zliu/PFN/PFN24/PFN24/i24_ziliu/ForB/outputs/evaluation_results/inpainting_pfn_with_initial",
        required=True,
        help="/data1/liu/PFN/mnt/nfs-mnj-home-43/i24_ziliu/dataset/Synthesis_Images/")

    parser.add_argument(
        "--saved_name",
        type=str,
        default=None,
        help="../outputs/evaluation_results/inpainting_pfn_with_initial")

    
    args = parser.parse_args()
    
    return args

if __name__=="__main__":
    
    args = parse_args()
    
    root_folder = args.root_folder
    root_folder_example = "{}/ours_adaIN".format(root_folder)
    
    fname_list = os.listdir(root_folder_example)
    
    saved_dict = {}
    for fname in tqdm(fname_list):
        # est_folder 和 GT_folder 的路径
        
        saved_dict[fname] = dict()
        
        # ours adaIN
        ours_adaIN_est_folder = '{}/ours_adaIN/{}/'.format(root_folder,fname)
        GT_folder = '{}/original_images/{}/'.format(root_folder,fname)
        fid_value_ours_adaIN = calculate_mean_lpips(est_folder=ours_adaIN_est_folder,GT_folder=GT_folder)
        saved_dict[fname]['ours_adaIN'] = fid_value_ours_adaIN
        
        # ours attn
        ours_attn_est_folder = '{}/ours_attn/{}/'.format(root_folder,fname)
        fid_value_ours_attn = calculate_mean_lpips(est_folder=ours_attn_est_folder,GT_folder=GT_folder)
        saved_dict[fname]['ours_attn'] = fid_value_ours_attn
        
        # our attn + adaIN
        ours_adaIN_ours_attn_est_folder = '{}/ours_adaIN_ours_attn/{}/'.format(root_folder,fname)
        fid_value_ours_adaIN_ours_attn = calculate_mean_lpips(est_folder=ours_adaIN_ours_attn_est_folder,GT_folder=GT_folder)
        saved_dict[fname]['ours_attn_adaIN'] = fid_value_ours_adaIN_ours_attn
        
        
        
        # normal inpainting
        normal_inpainting_est_folder = '{}/normal_inpainting/{}/'.format(root_folder,fname)
        fid_value_normal_inpainting = calculate_mean_lpips(est_folder=normal_inpainting_est_folder,GT_folder=GT_folder)
        saved_dict[fname]['normal_inpainting'] = fid_value_normal_inpainting

        # off adaIN
        official_adaIN_est_folder = '{}/official_adaIN/{}/'.format(root_folder,fname)
        fid_value_official_adaIN = calculate_mean_lpips(est_folder=official_adaIN_est_folder,GT_folder=GT_folder)
        saved_dict[fname]['offical_adaIN'] = fid_value_official_adaIN
        
        # off attn
        official_attn_est_folder = '{}/official_attn/{}/'.format(root_folder,fname)
        fid_value_official_attn = calculate_mean_lpips(est_folder=official_attn_est_folder,GT_folder=GT_folder)
        saved_dict[fname]['offical_attn'] = fid_value_official_attn
        
        # off AdaIN + Attn
        official_attn_adaIN_est_folder = '{}/official_attn_adaIN/{}/'.format(root_folder,fname)
        fid_value_official_attn_adaIN = calculate_mean_lpips(est_folder=official_attn_adaIN_est_folder,GT_folder=GT_folder)
        saved_dict[fname]['offical_attn_adain'] = fid_value_official_attn_adaIN
        
    
    
    with open(args.saved_name, 'w', encoding='utf-8') as f:
        json.dump(saved_dict, f, ensure_ascii=False, indent=4)

