from huggingface_hub import hf_hub_download
import torch
import torch.nn as nn
import torch.nn.functional as F


if __name__=="__main__":

    # download from a single file
    hf_hub_download(repo_id="skytnt/anime-seg", filename="isnetis.ckpt",
    local_dir="/mnt/nfs-mnj-home-43/i24_ziliu/Foreground2Background/AnimeSegmentation/anime-segmentation/pretrained_models")

    print('Downloaded Successfully')
    pass