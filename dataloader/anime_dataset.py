from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from torch.utils.data import Dataset

import os
import sys
sys.path.append("../..")
from ForB.dataloader.utils.file_io import read_img,read_mask,resize_image,get_resize_foreground_and_mask
from ForB.dataloader.utils.utils import read_text_lines,get_id_and_prompt
from ForB.dataloader.utils import transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm



class Anime_Dataset(Dataset):
    def __init__(self,
                 datapath,
                 trainlist,
                 vallist,
                 mode='train',
                 transform=None,
                 target_resolution=(512,512),
                 save_filename=False,
                 get_foreground=False):
        super(Anime_Dataset,self).__init__()
        
        self.datapath = datapath
        self.trainlist = trainlist
        self.vallist = vallist
        self.mode = mode
        self.transform = transform
        self.save_filename = save_filename
        self.target_resolution = target_resolution
        self.get_foreground = get_foreground
        
        dataset_dict = {
            'train': self.trainlist,
            'val': self.vallist,
            "test": self.vallist
        }
        self.samples  =[]
        
        # get the lines
        lines = read_text_lines(os.path.join(self.datapath,dataset_dict[mode]))
        
        for line in lines:
            parts = line.split(' ', 2)
            image_path = parts[0]
            bg_mask_path = parts[1]
            description = parts[2]

            # get the images and the background_mask
            image_path = os.path.join(self.datapath,image_path)
            background_path = os.path.join(self.datapath,bg_mask_path)
            prompt = description
            assert os.path.exists(image_path) and os.path.exists(background_path)

            sample = dict()
            if self.save_filename:
                sample['filename'] = prompt_id

            sample['prompt'] = prompt
            sample['image_path'] = image_path
            sample['background_path'] = background_path
            self.samples.append(sample)
    
    
    def __getitem__(self, index):

        sample = {}
        sample_path = self.samples[index]

        if self.save_filename:
            sample['filename'] = sample_path['filenme']
        
        sample['image'] = read_img(sample_path["image_path"])
        sample['bg_mask'] = read_mask(sample_path['background_path'])
        sample['prompt'] = sample_path['prompt']



        if self.target_resolution is not None:
            sample["image"],sample['origin_size'] = resize_image(sample['image'],
                                                    size=self.target_resolution,is_mask=False)
            sample['bg_mask'],sample['origin_size'] = resize_image(sample['bg_mask'],
                                                    size=self.target_resolution,is_mask=True)
        
        sample['bg_mask'] = sample['bg_mask'][:,:,:1]

        if self.get_foreground:
            # get the foreground_mask
            sample['fg_mask'] = np.ones_like(sample['bg_mask'].astype(np.float32)) - sample['bg_mask'].astype(np.float32)
            sample['fg_image'] = sample["image"] * sample['fg_mask']
            # sample['fg_image'], sample['fg_mask'] = get_resize_foreground_and_mask(image=sample['fg_image'],
            #                                             mask=sample['fg_mask'])

        if self.transform is not None:
            sample = self.transform(sample)
            
        return sample
    
    def __len__(self):
        return len(self.samples)




if __name__=="__main__":
    import skimage.io
    import matplotlib.pyplot as plt

    datapath = "/mnt/nfs-mnj-home-43/i24_ziliu/dataset"
    trainlist = "/mnt/nfs-mnj-home-43/i24_ziliu/i24_ziliu/ForB/filenames/training_data_all_selected.txt"
    vallist = "/mnt/nfs-mnj-home-43/i24_ziliu/i24_ziliu/ForB/filenames/training_data_all_selected.txt"

    train_transform_list = [transforms.ToTensor()]
    train_transform = transforms.Compose(train_transform_list)

    test_transform_list = [transforms.ToTensor()]
    test_transform = transforms.Compose(test_transform_list)

    anime_train_dataset = Anime_DatasetV2(datapath=datapath,
                            trainlist=trainlist,
                            vallist=vallist,
                            transform=train_transform,
                            save_filename=False,
                            mode='train',
                            target_resolution=(512,512))
    anime_test_dataset = Anime_DatasetV2(datapath=datapath,
                            trainlist=trainlist,
                            vallist=vallist,
                            transform=test_transform,
                            save_filename=False,
                            mode='test',
                            target_resolution=(512,512))


    for sample in tqdm(anime_train_dataset):

        image = sample['image']
        bg_mask = sample['bg_mask']
        prompt = sample['prompt']

        print(prompt)

        









