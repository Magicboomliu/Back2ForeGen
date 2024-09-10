import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import sys
import os
import logging
sys.path.append("../..")
from ForB.dataloader.anime_dataset import Anime_Dataset
from ForB.dataloader.utils import transforms

def collate_fn_concat(batch):
    sample = dict()
    sample["image"] = torch.cat([item['image'].unsqueeze(0) for item in batch], dim=0)  # (batch_size, C, H, W)
    sample["bg_mask"] = torch.cat([item['bg_mask'].unsqueeze(0) for item in batch], dim=0)
    sample['prompt'] = [item['prompt'] for item in batch]  # (batch_size,)
    
    example_item = batch[0]
    if 'fg_mask' in example_item.keys():
        sample['fg_mask'] = torch.cat([item['fg_mask'].unsqueeze(0) for item in batch], dim=0)
    if 'fg_image' in example_item.keys():
        sample['fg_image'] = torch.cat([item['fg_image'].unsqueeze(0) for item in batch], dim=0)
    
    return sample

def prepare_dataset(datapath,
                    trainlist,
                    vallist,
                    logger=None,
                    batch_size=1,
                    test_size=1,
                    datathread=4,
                    target_resolution=(512,512),
                    use_foreground=False):

    train_transform_list = [transforms.ToTensor()]
    train_transform = transforms.Compose(train_transform_list)

    test_transform_list = [transforms.ToTensor()]
    test_transform = transforms.Compose(test_transform_list)
    

    anime_train_dataset = Anime_Dataset(datapath=datapath,
                            trainlist=trainlist,
                            vallist=vallist,
                            transform=train_transform,
                            save_filename=False,
                            mode='train',
                            target_resolution=(512,512),
                            get_foreground=use_foreground)

    anime_test_dataset = Anime_Dataset(datapath=datapath,
                            trainlist=trainlist,
                            vallist=vallist,
                            transform=test_transform,
                            save_filename=False,
                            mode='test',
                            target_resolution=(512,512),
                            get_foreground=use_foreground)

    datathread=datathread
    if os.environ.get('datathread') is not None:
        datathread = int(os.environ.get('datathread'))
    if logger is not None:
        logger.info("Use %d processes to load data..." % datathread)
    
    train_loader = DataLoader(anime_train_dataset, batch_size=batch_size, \
                                shuffle=True, num_workers=datathread,
                                pin_memory=True,
                                collate_fn=collate_fn_concat)

    test_loader = DataLoader(anime_test_dataset, batch_size=test_size, \
                                shuffle=False, num_workers=datathread,
                                pin_memory=True,
                                collate_fn=collate_fn_concat)
    
    num_batches_per_epoch = len(train_loader)

    return (train_loader,test_loader),num_batches_per_epoch


def image_normalization(image_tensor):
    image_normalized = image_tensor * 2.0 - 1.0
    return image_normalized

def image_denormalization(image_tensor):
    image_denormalized = (image_tensor + 1.0)/2.0
    image_denormalized = torch.clamp(image_denormalized,min=0,
                                    max=1.0)
    return image_denormalized

if __name__=="__main__":
    
    import matplotlib.pyplot as plt

    datapath = "/mnt/nfs-mnj-home-43/i24_ziliu/dataset/SD_Gen_Images"
    trainlist = "descriptions_train.txt"
    vallist = "descriptions_test.txt"
    logger = None
    batch_size = 1
    test_size = 1
    datathread = 4

    # dataloder 
    (train_loader,test_loader),num_batches_per_epoch = prepare_dataset(datapath=datapath,
                    trainlist=trainlist,
                    vallist=vallist,
                    logger=logger,
                    batch_size=1,
                    test_size=1,
                    datathread=4,
                    target_resolution=(512,512),
                    use_foreground=True
                    )
    
    for idx, sample in enumerate(test_loader):
        
        
        plt.subplot(2,2,1)
        plt.imshow(sample['image'].squeeze(0).permute(1,2,0).cpu().numpy())
        plt.subplot(2,2,2)
        plt.imshow(sample['bg_mask'].squeeze(0).squeeze(0).cpu().numpy(),cmap='gray')
        plt.subplot(2,2,3)
        plt.imshow(sample['fg_image'].squeeze(0).permute(1,2,0).cpu().numpy())
        plt.subplot(2,2,4)
        plt.imshow(sample['fg_mask'].squeeze(0).squeeze(0).cpu().numpy(),cmap='gray')
        plt.savefig("example.png")
        print(sample['prompt'])

        quit()
        # print(sample['image'].shape)
        # print(sample['bg_mask'].shape)
        # print(sample['prompt'])
        # print(sample['fg_image'].shape)
        # print(sample['fg_mask'].shape)
        # quit()








