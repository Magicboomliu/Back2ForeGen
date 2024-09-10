import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torch.nn.init import kaiming_normal
import sys
sys.path.append("../..")
from ForB.networks.utils_net import conv,conv3x3,conv_Relu,convbn,ResBlock,BasicBlock,get_positional_encoding

class Mid_Block_Network(nn.Module):
    def __init__(self, in_channels, time_embed_dim=512, text_prompt_dim=768):
        super(Mid_Block_Network, self).__init__()
        self.in_channels = in_channels

        # Convolutional Layers for mid_block_feat
        self.conv1 = nn.Conv2d(in_channels=self.in_channels+1, out_channels=self.in_channels//2, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=self.in_channels//2, out_channels=self.in_channels, kernel_size=3, padding=1)

        # Fully Connected Layers for mean
        self.fc1 = nn.Linear(self.in_channels*2, self.in_channels)
        self.fc2 = nn.Linear(self.in_channels, self.in_channels)


        # Time Embedding layers
        self.time_embed_fc = nn.Linear(time_embed_dim, self.in_channels)
        # Text Prompt layers
        self.text_prompt_fc = nn.Linear(text_prompt_dim, self.in_channels)

        
        # Output layers
        self.output1 = nn.Linear(self.in_channels, self.in_channels)
        self.output2 = nn.Linear(self.in_channels, self.in_channels)

    def forward(self, mid_block_mean_list, mid_block_var_list, mid_block_feat_list, time_embed, text_prompt, foreground_mask):

        mid_block_mean = mid_block_mean_list[0]  # [1,1280,1,1]
        mid_block_var = mid_block_var_list[0]    # [1,1280,1,1]
        mid_block_feat = mid_block_feat_list[0]  # [1,1280,8,8]

        # Process foreground mask
        feat_h,feat_w = mid_block_feat.shape[-2:]
        mask = F.interpolate(foreground_mask,size=[feat_h,feat_w],mode='nearest')

        mid_block_mean_var_concated = torch.cat((mid_block_mean,mid_block_var),dim=1) #[1,1280*2,1,1]
        # Process mid_block_feat
        mask = mask.type_as(mid_block_feat)
        x1 = self.conv1(torch.cat((mid_block_feat,mask),dim=1))
        x1 = self.relu(x1)
        x1 = self.conv2(x1)
        x1 = self.relu(x1)
        x1 = torch.mean(x1, dim=(2, 3), keepdim=True)

        # Process mid_block_mean
        x2 = mid_block_mean_var_concated.view(mid_block_mean_var_concated.size(0), -1)
        x2 = self.fc1(x2)
        x2 = self.relu(x2)
        x2 = self.fc2(x2)
        x2 = x2.view(x2.size(0), x2.size(1), 1, 1) #[1,1280,1,1]

        time_embed = self.time_embed_fc(time_embed)  # [1, in_channels]
        time_embed = time_embed.view(time_embed.size(0), time_embed.size(1), 1, 1)  # Reshape to [1, in_channels, 1, 1]
        # Process text prompt
        text_prompt = torch.mean(text_prompt, dim=1)  # Average across the sequence length dimension
        text_prompt = self.text_prompt_fc(text_prompt)  # [1, in_channels]
        text_prompt = text_prompt.view(text_prompt.size(0), text_prompt.size(1), 1, 1)  # Reshape to [1, in_channels, 1, 1]

        # Combine all features
        x = x1 + x2 + time_embed + text_prompt
        # Generate outputs
        out1 = self.output1(x.view(x.size(0), -1)).view(x.size(0), -1, 1, 1)
        out2 = self.output2(x.view(x.size(0), -1)).view(x.size(0), -1, 1, 1)

        converted_mean_list = [out1]
        converted_var_list = [out2]

        return converted_mean_list, converted_var_list


class CrossAttnDownBlock2D_Network(nn.Module):
    def __init__(self, in_channels_list=[320, 640, 1280], time_embed_dim=512, text_prompt_dim=768):
        super(CrossAttnDownBlock2D_Network, self).__init__()
        self.in_channels_list = in_channels_list

        self.layers = nn.ModuleList()
        self.mean_layers = nn.ModuleList()
        self.variance_layers = nn.ModuleList()
        self.output1_layers = nn.ModuleList()
        self.output2_layers = nn.ModuleList()
        self.projection_layers_0 = nn.ModuleList()
        self.projection_layers_1 = nn.ModuleList()

        # Time Embedding layers
        self.time_embed_fc = nn.Linear(time_embed_dim, in_channels_list[0])
        # Text Prompt layers
        self.text_prompt_fc = nn.Linear(text_prompt_dim, in_channels_list[0])

        for in_channels in self.in_channels_list:
            # Conv layers for x1
            self.layers.append(nn.Sequential(
                nn.Conv2d(in_channels+1, in_channels // 2, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels // 2, in_channels, kernel_size=3, padding=1),
                nn.ReLU()
            ))

            # Fully connected layers for x2 (mean)
            self.mean_layers.append(nn.Sequential(
                nn.Linear(in_channels*2, in_channels),
                nn.ReLU(),
                nn.Linear(in_channels, in_channels)
            ))


            self.projection_layers_0.append(
                nn.Conv2d(in_channels_list[0],in_channels,kernel_size=1, padding=0)
            )
            self.projection_layers_1.append(
                nn.Conv2d(in_channels_list[0],in_channels,kernel_size=1, padding=0)
            )


            # Output layers
            self.output1_layers.append(nn.Linear(in_channels, in_channels))
            self.output2_layers.append(nn.Linear(in_channels, in_channels))

    def forward(self, 
                CrossAttnDownBlock2D_block_mean_list_1,
                CrossAttnDownBlock2D_block_mean_list_2,
                CrossAttnDownBlock2D_block_mean_list_3,
                CrossAttnDownBlock2D_block_var_list_1,
                CrossAttnDownBlock2D_block_var_list_2,
                CrossAttnDownBlock2D_block_var_list_3,
                CrossAttnDownBlock2D_block_feat_list_1,
                CrossAttnDownBlock2D_block_feat_list_2,
                CrossAttnDownBlock2D_block_feat_list_3,
                time_embed, text_prompt, foreground_mask):
        
        # Process time embedding and text prompt
        time_embed_processed = self.time_embed_fc(time_embed)
        time_embed_processed = time_embed_processed.view(time_embed_processed.size(0), time_embed_processed.size(1), 1, 1)
        
        text_prompt_processed = torch.mean(text_prompt, dim=1)
        text_prompt_processed = self.text_prompt_fc(text_prompt_processed)
        text_prompt_processed = text_prompt_processed.view(text_prompt_processed.size(0), text_prompt_processed.size(1), 1, 1)




        CrossAttnDownBlock2D_block_mean_list_1 = [item[0] for item in CrossAttnDownBlock2D_block_mean_list_1]
        CrossAttnDownBlock2D_block_mean_1 = torch.cat(CrossAttnDownBlock2D_block_mean_list_1, dim=0)

        CrossAttnDownBlock2D_block_mean_list_2 = [item[0] for item in CrossAttnDownBlock2D_block_mean_list_2]
        CrossAttnDownBlock2D_block_mean_2 = torch.cat(CrossAttnDownBlock2D_block_mean_list_2, dim=0)

        CrossAttnDownBlock2D_block_mean_list_3 = [item[0] for item in CrossAttnDownBlock2D_block_mean_list_3]
        CrossAttnDownBlock2D_block_mean_3 = torch.cat(CrossAttnDownBlock2D_block_mean_list_3, dim=0)

        CrossAttnDownBlock2D_block_var_list_1 = [item[0] for item in CrossAttnDownBlock2D_block_var_list_1]
        CrossAttnDownBlock2D_block_var_1 = torch.cat(CrossAttnDownBlock2D_block_var_list_1, dim=0)

        CrossAttnDownBlock2D_block_var_list_2 = [item[0] for item in CrossAttnDownBlock2D_block_var_list_2]
        CrossAttnDownBlock2D_block_var_2 = torch.cat(CrossAttnDownBlock2D_block_var_list_2, dim=0)

        CrossAttnDownBlock2D_block_var_list_3 = [item[0] for item in CrossAttnDownBlock2D_block_var_list_3]
        CrossAttnDownBlock2D_block_var_3 = torch.cat(CrossAttnDownBlock2D_block_var_list_3, dim=0)

        CrossAttnDownBlock2D_block_feat_list_1 = [item[0] for item in CrossAttnDownBlock2D_block_feat_list_1]
        CrossAttnDownBlock2D_block_feat_1 = torch.cat(CrossAttnDownBlock2D_block_feat_list_1, dim=0)

        CrossAttnDownBlock2D_block_feat_list_2 = [item[0] for item in CrossAttnDownBlock2D_block_feat_list_2]
        CrossAttnDownBlock2D_block_feat_2 = torch.cat(CrossAttnDownBlock2D_block_feat_list_2, dim=0)

        CrossAttnDownBlock2D_block_feat_list_3 = [item[0] for item in CrossAttnDownBlock2D_block_feat_list_3]
        CrossAttnDownBlock2D_block_feat_3 = torch.cat(CrossAttnDownBlock2D_block_feat_list_3, dim=0)



        inputs = [CrossAttnDownBlock2D_block_feat_1, CrossAttnDownBlock2D_block_feat_2, CrossAttnDownBlock2D_block_feat_3]
        means = [CrossAttnDownBlock2D_block_mean_1, CrossAttnDownBlock2D_block_mean_2, CrossAttnDownBlock2D_block_mean_3]
        variances = [CrossAttnDownBlock2D_block_var_1, CrossAttnDownBlock2D_block_var_2, CrossAttnDownBlock2D_block_var_3]
        fg_lists = [F.interpolate(foreground_mask,size=[res,res],mode='nearest') for res in [64,32,16]]


        new_means = []
        new_variances = []

        for i in range(len(inputs)):
            x1 = inputs[i]
            x2 = means[i]
            x3 = variances[i]
            fg_mask = fg_lists[i]



            # Process x1 (feature map)
            if fg_mask.shape[0]!=x1.shape[0]:
                fg_mask = fg_mask.repeat(x1.shape[0],1,1,1)
                fg_mask = fg_mask[:x1.shape[0],:,:,:]
            

            x1 = torch.cat((x1,fg_mask),dim=1)
            
            x1 = self.layers[i](x1)
            x1 = torch.mean(x1, dim=(2, 3), keepdim=True)  # Global average pooling

            # Process x2 (mean)
            x2 = torch.cat((x2,x3),dim=1) 
            x2 = x2.view(x2.size(0), -1)  # Flatten
            x2 = self.mean_layers[i](x2)
            x2 = x2.view(x2.size(0), x2.size(1), 1, 1)  # Reshape to [B, C, 1, 1]


            current_time_embed_processed = self.projection_layers_0[i](time_embed_processed)
            current_text_prompt_processed = self.projection_layers_1[i](text_prompt_processed)


            combined = x1 + x2
            current_dim = x1.shape[1]
            
            if current_text_prompt_processed.shape[0]!=1:
                if current_text_prompt_processed.shape[0]<combined.shape[0]:
                    current_text_prompt_processed = current_text_prompt_processed.repeat(2,1,1,1)
            


            combined1 = combined + current_time_embed_processed + current_text_prompt_processed
            combined2 = combined1


            # Generate outputs
            out1 = self.output1_layers[i](combined1.view(combined1.size(0), -1)).view(combined1.size(0), -1, 1, 1)
            out2 = self.output2_layers[i](combined2.view(combined2.size(0), -1)).view(combined2.size(0), -1, 1, 1)

            new_means.append(out1)
            new_variances.append(out2)


        return_mean = [list(torch.chunk(new_mean,dim=0,chunks=2)) for new_mean in new_means]
        return_var= [list(torch.chunk(new_var,dim=0,chunks=2)) for new_var in new_variances]

        returned_CrossAttnDownBlock2D_block_mean_list_1 = return_mean[0]
        returned_CrossAttnDownBlock2D_block_mean_list_2 = return_mean[1]
        returned_CrossAttnDownBlock2D_block_mean_list_3 = return_mean[2]

        returned_CrossAttnDownBlock2D_block_var_list_1 = return_var[0]
        returned_CrossAttnDownBlock2D_block_var_list_2 = return_var[1]
        returned_CrossAttnDownBlock2D_block_var_list_3 = return_var[2]


        return returned_CrossAttnDownBlock2D_block_mean_list_1,returned_CrossAttnDownBlock2D_block_mean_list_2,returned_CrossAttnDownBlock2D_block_mean_list_3, \
            returned_CrossAttnDownBlock2D_block_var_list_1,returned_CrossAttnDownBlock2D_block_var_list_2,returned_CrossAttnDownBlock2D_block_var_list_3


class DownBlock_Branch_Network(nn.Module):
    def __init__(self,in_channels, time_embed_dim=512, text_prompt_dim=768):
        super(DownBlock_Branch_Network, self).__init__()
        self.in_channels = in_channels
        # Convolutional Layers for mid_block_feat
        self.conv1 = nn.Conv2d(in_channels=self.in_channels+1, out_channels=self.in_channels // 2, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=self.in_channels // 2, out_channels=self.in_channels, kernel_size=3, padding=1)

        # Fully Connected Layers for mean
        self.fc1 = nn.Linear(self.in_channels*2, self.in_channels)
        self.fc2 = nn.Linear(self.in_channels, self.in_channels)


        # Time Embedding layers
        self.time_embed_fc = nn.Linear(time_embed_dim, self.in_channels)
        # Text Prompt layers
        self.text_prompt_fc = nn.Linear(text_prompt_dim, self.in_channels)

        
        # Output layers
        self.output1 = nn.Linear(self.in_channels, self.in_channels)
        self.output2 = nn.Linear(self.in_channels, self.in_channels)


    def forward(self,down_block_mean_list, down_block_var_list, down_block_feat_list, time_embed, text_prompt, foreground_mask):

        down_block_mean_list = [item[0] for item in down_block_mean_list]
        down_block_mean = torch.cat(down_block_mean_list,dim=0)

        down_block_var_list = [item[0] for item in down_block_var_list]
        down_block_var = torch.cat(down_block_var_list,dim=0)

        down_block_feat_list = [item[0] for item in down_block_feat_list]
        down_block_feat = torch.cat(down_block_feat_list,dim=0)

        x1 = down_block_feat
        x2 = down_block_mean
        x3 = down_block_var

        # Process foreground mask
        feat_h,feat_w = x1.shape[-2:]
        mask = F.interpolate(foreground_mask,size=[feat_h,feat_w],mode='nearest')
        if mask.shape[0]!=x1.shape[0]:
            mask = mask.repeat(x1.shape[0],1,1,1)
            mask = mask[:x1.shape[0],:,:,:]

        x2 = torch.cat((x2,x3),dim=1) #[1,1280*2,1,1]

        
        x1 = torch.cat((x1,mask),dim=1)

        # Apply Convolution -> LayerNorm -> ReLU to the first input
        x1 = self.conv1(x1)
        # x1 = self.ln1(x1)
        x1 = self.relu(x1)
        x1 = self.conv2(x1)
        # x1 = self.ln2(x1)
        x1 = self.relu(x1)
        x1 = torch.mean(x1, dim=(2, 3), keepdim=True)
        
        # Flatten, Fully Connected -> LayerNorm -> ReLU for x2 and x3
        x2 = x2.view(x2.size(0), -1)
        x2 = self.fc1(x2)
        # x2 = self.ln_fc1(x2)
        x2 = self.relu(x2)
        x2 = self.fc2(x2)
        # x2 = self.ln_fc2(x2)
        x2 = x2.view(x2.size(0), x2.size(1), 1, 1)

        
        # Process time embedding
        time_embed = self.time_embed_fc(time_embed)  # [1, in_channels]
        time_embed = time_embed.view(time_embed.size(0), time_embed.size(1), 1, 1)  # Reshape to [1, in_channels, 1, 1]
        # Process text prompt
        text_prompt = torch.mean(text_prompt, dim=1)  # Average across the sequence length dimension
        text_prompt = self.text_prompt_fc(text_prompt)  # [1, in_channels]
        text_prompt = text_prompt.view(text_prompt.size(0), text_prompt.size(1), 1, 1)  # Reshape to [1, in_channels, 1, 1]


        # Combine all features
        if text_prompt.shape[0]!=1:
            if text_prompt.shape[0]<x1.shape[0]:
                text_prompt = text_prompt.repeat(2,1,1,1)

        x = x1 + x2 + time_embed + text_prompt 

        out1 = self.output1(x.view(x.size(0), -1)).view(x.size(0), -1, 1, 1)
        out2 = self.output2(x.view(x.size(0), -1)).view(x.size(0), -1, 1, 1)

        return_down_block_mean_list = list(torch.chunk(out1,dim=0,chunks=2))
        return_down_block_var_list = list(torch.chunk(out2,dim=0,chunks=2))

        return return_down_block_mean_list,return_down_block_var_list


class UpBlock2D_Branch_Network(nn.Module):
    def __init__(self,in_channels, time_embed_dim=512, text_prompt_dim=768):
        super(UpBlock2D_Branch_Network, self).__init__()
        self.in_channels = in_channels
        # Convolutional Layers for mid_block_feat
        self.conv1 = nn.Conv2d(in_channels=self.in_channels+1, out_channels=self.in_channels // 2, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=self.in_channels // 2, out_channels=self.in_channels, kernel_size=3, padding=1)

        # Fully Connected Layers for mean
        self.fc1 = nn.Linear(self.in_channels*2, self.in_channels)
        self.fc2 = nn.Linear(self.in_channels, self.in_channels)


        # Time Embedding layers
        self.time_embed_fc = nn.Linear(time_embed_dim, self.in_channels)
        # Text Prompt layers
        self.text_prompt_fc = nn.Linear(text_prompt_dim, self.in_channels)

        
        # Output layers
        self.output1 = nn.Linear(self.in_channels, self.in_channels)
        self.output2 = nn.Linear(self.in_channels, self.in_channels)


    def forward(self,down_block_mean_list, down_block_var_list, down_block_feat_list, time_embed, text_prompt, foreground_mask):

        down_block_mean_list = [item[0] for item in down_block_mean_list]
        down_block_mean = torch.cat(down_block_mean_list,dim=0)

        down_block_var_list = [item[0] for item in down_block_var_list]
        down_block_var = torch.cat(down_block_var_list,dim=0)

        down_block_feat_list = [item[0] for item in down_block_feat_list]
        down_block_feat = torch.cat(down_block_feat_list,dim=0)

        x1 = down_block_feat
        x2 = down_block_mean
        x3 = down_block_var

        # Process foreground mask
        feat_h,feat_w = x1.shape[-2:]
        mask = F.interpolate(foreground_mask,size=[feat_h,feat_w],mode='nearest')
        if mask.shape[0]!=x1.shape[0]:
            mask = mask.repeat(x1.shape[0],1,1,1)
            mask = mask[:x1.shape[0],:,:,:]

        x2 = torch.cat((x2,x3),dim=1) #[1,1280*2,1,1]
        x1 = torch.cat((x1,mask),dim=1)

        # Apply Convolution -> LayerNorm -> ReLU to the first input
        x1 = self.conv1(x1)
        # x1 = self.ln1(x1)
        x1 = self.relu(x1)
        x1 = self.conv2(x1)
        # x1 = self.ln2(x1)
        x1 = self.relu(x1)
        x1 = torch.mean(x1, dim=(2, 3), keepdim=True)
        
        # Flatten, Fully Connected -> LayerNorm -> ReLU for x2 and x3
        x2 = x2.view(x2.size(0), -1)
        x2 = self.fc1(x2)
        # x2 = self.ln_fc1(x2)
        x2 = self.relu(x2)
        x2 = self.fc2(x2)
        # x2 = self.ln_fc2(x2)
        x2 = x2.view(x2.size(0), x2.size(1), 1, 1)

        
        # Process time embedding
        time_embed = self.time_embed_fc(time_embed)  # [1, in_channels]
        time_embed = time_embed.view(time_embed.size(0), time_embed.size(1), 1, 1)  # Reshape to [1, in_channels, 1, 1]
        # Process text prompt
        text_prompt = torch.mean(text_prompt, dim=1)  # Average across the sequence length dimension
        text_prompt = self.text_prompt_fc(text_prompt)  # [1, in_channels]
        text_prompt = text_prompt.view(text_prompt.size(0), text_prompt.size(1), 1, 1)  # Reshape to [1, in_channels, 1, 1]


        # Combine all features
        if text_prompt.shape[0]!=1:
            if text_prompt.shape[0]<x1.shape[0]:
                text_prompt = text_prompt.repeat(3,1,1,1)

        # Combine all features
        x = x1 + x2 + time_embed + text_prompt 

        out1 = self.output1(x.view(x.size(0), -1)).view(x.size(0), -1, 1, 1)
        out2 = self.output2(x.view(x.size(0), -1)).view(x.size(0), -1, 1, 1)

        return_down_block_mean_list = list(torch.chunk(out1,dim=0,chunks=3))
        return_down_block_var_list = list(torch.chunk(out2,dim=0,chunks=3))


        return return_down_block_mean_list,return_down_block_var_list


class CrossAttnUpBlock2D_Network(nn.Module):
    def __init__(self, in_channels_list=[1280, 640, 320], time_embed_dim=512, text_prompt_dim=768):
        super(CrossAttnUpBlock2D_Network, self).__init__()
        self.in_channels_list = in_channels_list

        self.layers = nn.ModuleList()
        self.mean_layers = nn.ModuleList()
        self.variance_layers = nn.ModuleList()
        self.output1_layers = nn.ModuleList()
        self.output2_layers = nn.ModuleList()
        self.projection_layers_0 = nn.ModuleList()
        self.projection_layers_1 = nn.ModuleList()

        # Time Embedding layers
        self.time_embed_fc = nn.Linear(time_embed_dim, in_channels_list[-1])
        # Text Prompt layers
        self.text_prompt_fc = nn.Linear(text_prompt_dim, in_channels_list[-1])


        for in_channels in self.in_channels_list:
            # Conv layers for x1
            self.layers.append(nn.Sequential(
                nn.Conv2d(in_channels+1, in_channels // 2, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels // 2, in_channels, kernel_size=3, padding=1),
                nn.ReLU()
            ))

            # Fully connected layers for x2 (mean)
            self.mean_layers.append(nn.Sequential(
                nn.Linear(in_channels*2, in_channels),
                nn.ReLU(),
                nn.Linear(in_channels, in_channels)
            ))

            self.projection_layers_0.append(
                nn.Conv2d(in_channels_list[-1],in_channels,kernel_size=1, padding=0)
            )
            self.projection_layers_1.append(
                nn.Conv2d(in_channels_list[-1],in_channels,kernel_size=1, padding=0)
            )

            # Output layers
            self.output1_layers.append(nn.Linear(in_channels, in_channels))
            self.output2_layers.append(nn.Linear(in_channels, in_channels))

    def forward(self, 
                CrossAttnUpBlock2D_block_mean_list_1,
                CrossAttnUpBlock2D_block_mean_list_2,
                CrossAttnUpBlock2D_block_mean_list_3,
                CrossAttnUpBlock2D_block_var_list_1,
                CrossAttnUpBlock2D_block_var_list_2,
                CrossAttnUpBlock2D_block_var_list_3,
                CrossAttnUpBlock2D_block_feat_list_1,
                CrossAttnUpBlock2D_block_feat_list_2,
                CrossAttnUpBlock2D_block_feat_list_3,
                time_embed, text_prompt, foreground_mask):
        
        # Process time embedding and text prompt
        time_embed_processed = self.time_embed_fc(time_embed)
        time_embed_processed = time_embed_processed.view(time_embed_processed.size(0), time_embed_processed.size(1), 1, 1)
        
        text_prompt_processed = torch.mean(text_prompt, dim=1)
        text_prompt_processed = self.text_prompt_fc(text_prompt_processed)
        text_prompt_processed = text_prompt_processed.view(text_prompt_processed.size(0), text_prompt_processed.size(1), 1, 1)





        CrossAttnUpBlock2D_block_mean_list_1 = [item[0] for item in CrossAttnUpBlock2D_block_mean_list_1]
        CrossAttnUpBlock2D_block_mean_1 = torch.cat(CrossAttnUpBlock2D_block_mean_list_1, dim=0)

        CrossAttnUpBlock2D_block_mean_list_2 = [item[0] for item in CrossAttnUpBlock2D_block_mean_list_2]
        CrossAttnUpBlock2D_block_mean_2 = torch.cat(CrossAttnUpBlock2D_block_mean_list_2, dim=0)

        CrossAttnUpBlock2D_block_mean_list_3 = [item[0] for item in CrossAttnUpBlock2D_block_mean_list_3]
        CrossAttnUpBlock2D_block_mean_3 = torch.cat(CrossAttnUpBlock2D_block_mean_list_3, dim=0)

        CrossAttnUpBlock2D_block_var_list_1 = [item[0] for item in CrossAttnUpBlock2D_block_var_list_1]
        CrossAttnUpBlock2D_block_var_1 = torch.cat(CrossAttnUpBlock2D_block_var_list_1, dim=0)

        CrossAttnUpBlock2D_block_var_list_2 = [item[0] for item in CrossAttnUpBlock2D_block_var_list_2]
        CrossAttnUpBlock2D_block_var_2 = torch.cat(CrossAttnUpBlock2D_block_var_list_2, dim=0)

        CrossAttnUpBlock2D_block_var_list_3 = [item[0] for item in CrossAttnUpBlock2D_block_var_list_3]
        CrossAttnUpBlock2D_block_var_3 = torch.cat(CrossAttnUpBlock2D_block_var_list_3, dim=0)

        CrossAttnUpBlock2D_block_feat_list_1 = [item[0] for item in CrossAttnUpBlock2D_block_feat_list_1]
        CrossAttnUpBlock2D_block_feat_1 = torch.cat(CrossAttnUpBlock2D_block_feat_list_1, dim=0)

        CrossAttnUpBlock2D_block_feat_list_2 = [item[0] for item in CrossAttnUpBlock2D_block_feat_list_2]
        CrossAttnUpBlock2D_block_feat_2 = torch.cat(CrossAttnUpBlock2D_block_feat_list_2, dim=0)

        CrossAttnUpBlock2D_block_feat_list_3 = [item[0] for item in CrossAttnUpBlock2D_block_feat_list_3]
        CrossAttnUpBlock2D_block_feat_3 = torch.cat(CrossAttnUpBlock2D_block_feat_list_3, dim=0)

        inputs = [CrossAttnUpBlock2D_block_feat_1, CrossAttnUpBlock2D_block_feat_2, CrossAttnUpBlock2D_block_feat_3]
        means = [CrossAttnUpBlock2D_block_mean_1, CrossAttnUpBlock2D_block_mean_2, CrossAttnUpBlock2D_block_mean_3]
        variances = [CrossAttnUpBlock2D_block_var_1, CrossAttnUpBlock2D_block_var_2, CrossAttnUpBlock2D_block_var_3]
        fg_lists = [F.interpolate(foreground_mask,size=[res,res],mode='nearest') for res in [16,32,64]]

        new_means = []
        new_variances = []



        for i in range(len(inputs)):
            x1 = inputs[i]
            x2 = means[i]
            x3 = variances[i]

            fg_mask = fg_lists[i]

            # Process x1 (feature map)
            if fg_mask.shape[0]!=x1.shape[0]:
                fg_mask = fg_mask.repeat(x1.shape[0],1,1,1)
                fg_mask = fg_mask[:x1.shape[0],:,:,:]

            x1 = torch.cat((x1,fg_mask),dim=1)
            # Process x1 (feature map)
            x1 = self.layers[i](x1)
            x1 = torch.mean(x1, dim=(2, 3), keepdim=True)  # Global average pooling

            

            # Process x2 (mean)
            x2 = torch.cat((x2,x3),dim=1) 
            x2 = x2.view(x2.size(0), -1)  # Flatten
            x2 = self.mean_layers[i](x2)
            x2 = x2.view(x2.size(0), x2.size(1), 1, 1)  # Reshape to [B, C, 1, 1]


            # Combine x1, x2, and x3 with the combined embedding and foreground mask
            current_time_embed_processed = self.projection_layers_0[i](time_embed_processed)
            current_text_prompt_processed = self.projection_layers_1[i](text_prompt_processed)



            combined = x1 + x2 
            current_dim = x1.shape[1]

            if current_text_prompt_processed.shape[0]!=1:
                if current_text_prompt_processed.shape[0]<combined.shape[0]:
                    current_text_prompt_processed = current_text_prompt_processed.repeat(3,1,1,1)

            combined1 = combined + current_time_embed_processed + current_text_prompt_processed 
            combined2 = combined1


            # Generate outputs
            out1 = self.output1_layers[i](combined1.view(combined1.size(0), -1)).view(combined1.size(0), -1, 1, 1)
            out2 = self.output2_layers[i](combined2.view(combined2.size(0), -1)).view(combined2.size(0), -1, 1, 1)

            new_means.append(out1)
            new_variances.append(out2)


        return_mean = [list(torch.chunk(new_mean,dim=0,chunks=3)) for new_mean in new_means]
        return_var= [list(torch.chunk(new_var,dim=0,chunks=3)) for new_var in new_variances]

        returned_CrossAttnUpBlock2D_block_mean_list_1 = return_mean[0]
        returned_CrossAttnUpBlock2D_block_mean_list_2 = return_mean[1]
        returned_CrossAttnUpBlock2D_block_mean_list_3 = return_mean[2]

        returned_CrossAttnUpBlock2D_block_var_list_1 = return_var[0]
        returned_CrossAttnUpBlock2D_block_var_list_2 = return_var[1]
        returned_CrossAttnUpBlock2D_block_var_list_3 = return_var[2]


        return returned_CrossAttnUpBlock2D_block_mean_list_1,returned_CrossAttnUpBlock2D_block_mean_list_2,returned_CrossAttnUpBlock2D_block_mean_list_3, \
            returned_CrossAttnUpBlock2D_block_var_list_1,returned_CrossAttnUpBlock2D_block_var_list_2,returned_CrossAttnUpBlock2D_block_var_list_3



class F2All_Converter(nn.Module):
    def __init__(self,mid_channels=1280):
        super(F2All_Converter, self).__init__()
        self.mid_channels = mid_channels

        self.Mid_Block_Branch = Mid_Block_Network(in_channels=self.mid_channels)
        self.CrossAttnDownBlock2D_Branch = CrossAttnDownBlock2D_Network(in_channels_list=[320,640,1280])
        self.DownBlock_Branch = DownBlock_Branch_Network(in_channels=self.mid_channels)
        self.UpBlock2D_Branch = UpBlock2D_Branch_Network(in_channels=self.mid_channels)
        self.CrossAttnUpBlock2D_Branch = CrossAttnUpBlock2D_Network(in_channels_list=[1280,640,320])

    
    def forward(self,mean_bank,var_bank,feat_bank,time_embed,text_embed,foreground_mask):
        
        result_mean_dict = dict()
        result_var_dict = dict()

        # mid_block 0 
        mid_block_mean_list = mean_bank[('mid_block', 0)]
        mid_block_var_list = var_bank[('mid_block', 0)]
        mid_block_feat_list = feat_bank[('mid_block', 0)]

        # time_embed, text_prompt, foreground_mask
        convered_mean_list, converted_var_list = self.Mid_Block_Branch(mid_block_mean_list=mid_block_mean_list,
                                                                       mid_block_var_list=mid_block_var_list,
                                                                       mid_block_feat_list=mid_block_feat_list,
                                                                       time_embed=time_embed,
                                                                       text_prompt = text_embed,
                                                                       foreground_mask = foreground_mask
                                                                       )
        result_mean_dict[('mid_block', 0)] = convered_mean_list
        result_var_dict[('mid_block', 0)] = converted_var_list

        '''-----------------------------------------------------------------------'''

        # ('CrossAttnDownBlock2D', 1)
        CrossAttnDownBlock2D_block_mean_list_1 = mean_bank[('CrossAttnDownBlock2D', 1)]
        CrossAttnDownBlock2D_block_var_list_1 = var_bank[('CrossAttnDownBlock2D', 1)]
        CrossAttnDownBlock2D_block_feat_list_1 = feat_bank[('CrossAttnDownBlock2D', 1)]
        # ('CrossAttnDownBlock2D', 2)
        CrossAttnDownBlock2D_block_mean_list_2 = mean_bank[('CrossAttnDownBlock2D', 2)]
        CrossAttnDownBlock2D_block_var_list_2 = var_bank[('CrossAttnDownBlock2D', 2)]
        CrossAttnDownBlock2D_block_feat_list_2 = feat_bank[('CrossAttnDownBlock2D', 2)]
        # ('CrossAttnDownBlock2D', 3)
        CrossAttnDownBlock2D_block_mean_list_3 = mean_bank[('CrossAttnDownBlock2D', 3)]
        CrossAttnDownBlock2D_block_var_list_3 = var_bank[('CrossAttnDownBlock2D', 3)]
        CrossAttnDownBlock2D_block_feat_list_3 = feat_bank[('CrossAttnDownBlock2D', 3)]    


        #time_embed, text_prompt, foreground_mask
        returned_CrossAttnDownBlock2D_block_mean_list_1,returned_CrossAttnDownBlock2D_block_mean_list_2,returned_CrossAttnDownBlock2D_block_mean_list_3, \
            returned_CrossAttnDownBlock2D_block_var_list_1,returned_CrossAttnDownBlock2D_block_var_list_2,returned_CrossAttnDownBlock2D_block_var_list_3 = self.CrossAttnDownBlock2D_Branch(CrossAttnDownBlock2D_block_mean_list_1,
                    CrossAttnDownBlock2D_block_mean_list_2,
                    CrossAttnDownBlock2D_block_mean_list_3,
                    CrossAttnDownBlock2D_block_var_list_1,
                    CrossAttnDownBlock2D_block_var_list_2,
                    CrossAttnDownBlock2D_block_var_list_3,
                    CrossAttnDownBlock2D_block_feat_list_1,
                    CrossAttnDownBlock2D_block_feat_list_2,
                    CrossAttnDownBlock2D_block_feat_list_3,
                    time_embed=time_embed,
                    text_prompt = text_embed,
                    foreground_mask = foreground_mask)

        result_mean_dict[('CrossAttnDownBlock2D', 1)] = returned_CrossAttnDownBlock2D_block_mean_list_1
        result_var_dict[('CrossAttnDownBlock2D', 1)] = returned_CrossAttnDownBlock2D_block_var_list_1
        result_mean_dict[('CrossAttnDownBlock2D', 2)] = returned_CrossAttnDownBlock2D_block_mean_list_2
        result_var_dict[('CrossAttnDownBlock2D', 2)] = returned_CrossAttnDownBlock2D_block_var_list_2
        result_mean_dict[('CrossAttnDownBlock2D', 3)] = returned_CrossAttnDownBlock2D_block_mean_list_3
        result_var_dict[('CrossAttnDownBlock2D', 3)] = returned_CrossAttnDownBlock2D_block_var_list_3
        
        '''--------------------------------------------------------------------------------------------'''

        # ('DownBlock2D', 4)
        DownBlock2D_mean_list = mean_bank[('DownBlock2D', 4)]
        DownBlock2D_var_list = var_bank[('DownBlock2D', 4)]
        DownBlock2D_feat_list = feat_bank[('DownBlock2D', 4)]

        return_down_block_mean_list,return_down_block_var_list = self.DownBlock_Branch(DownBlock2D_mean_list,DownBlock2D_var_list,DownBlock2D_feat_list,\
                            time_embed=time_embed,
                    text_prompt = text_embed,
                    foreground_mask = foreground_mask)
        result_mean_dict[('DownBlock2D', 4)] = return_down_block_mean_list
        result_var_dict[('DownBlock2D', 4)] = return_down_block_var_list




        '''------------------------------------------------------------'''

        # ('UpBlock2D', 5)
        UpBlock2D_mean_list = mean_bank[('UpBlock2D', 5)]
        UpBlock2D_var_list = var_bank[('UpBlock2D', 5)]
        UpBlock2D_feat_list = feat_bank[('UpBlock2D', 5)]

        return_up_block_mean_list,return_up_block_var_list = self.UpBlock2D_Branch(UpBlock2D_mean_list,UpBlock2D_var_list,UpBlock2D_feat_list,
                    time_embed=time_embed,
                    text_prompt = text_embed,
                    foreground_mask = foreground_mask)
        result_mean_dict[('UpBlock2D', 5)] = return_up_block_mean_list
        result_var_dict[('UpBlock2D', 5)] = return_up_block_var_list

        '''----------------------------------------------------------------'''

        # ('CrossAttnUpBlock2D', 6)
        CrossAttnUpBlock2D_block_mean_list_6 = mean_bank[('CrossAttnUpBlock2D', 6)]
        CrossAttnUpBlock2D_block_var_list_6 = var_bank[('CrossAttnUpBlock2D', 6)]
        CrossAttnUpBlock2D_block_feat_list_6 = feat_bank[('CrossAttnUpBlock2D', 6)]

        # ('CrossAttnUpBlock2D', 7)
        CrossAttnUpBlock2D_block_mean_list_7 = mean_bank[('CrossAttnUpBlock2D', 7)]
        CrossAttnUpBlock2D_block_var_list_7 = var_bank[('CrossAttnUpBlock2D', 7)]
        CrossAttnUpBlock2D_block_feat_list_7 = feat_bank[('CrossAttnUpBlock2D', 7)]

        # ('CrossAttnUpBlock2D', 8)])
        CrossAttnUpBlock2D_block_mean_list_8 = mean_bank[('CrossAttnUpBlock2D', 8)]
        CrossAttnUpBlock2D_block_var_list_8 = var_bank[('CrossAttnUpBlock2D', 8)]
        CrossAttnUpBlock2D_block_feat_list_8 = feat_bank[('CrossAttnUpBlock2D', 8)]


        returned_CrossAttnUpBlock2D_block_mean_list_6,returned_CrossAttnUpBlock2D_block_mean_list_7,returned_CrossAttnUpBlock2D_block_mean_list_8, \
            returned_CrossAttnUpBlock2D_block_var_list_6,returned_CrossAttnUpBlock2D_block_var_list_7,returned_CrossAttnUpBlock2D_block_var_list_8 = self.CrossAttnUpBlock2D_Branch(CrossAttnUpBlock2D_block_mean_list_6,
                    CrossAttnUpBlock2D_block_mean_list_7,
                    CrossAttnUpBlock2D_block_mean_list_8,
                    CrossAttnUpBlock2D_block_var_list_6,
                    CrossAttnUpBlock2D_block_var_list_7,
                    CrossAttnUpBlock2D_block_var_list_8,
                    CrossAttnUpBlock2D_block_feat_list_6,
                    CrossAttnUpBlock2D_block_feat_list_7,
                    CrossAttnUpBlock2D_block_feat_list_8,
                    time_embed=time_embed,
                    text_prompt = text_embed,
                    foreground_mask = foreground_mask)

        result_mean_dict[('CrossAttnUpBlock2D', 6)] = returned_CrossAttnUpBlock2D_block_mean_list_6
        result_var_dict[('CrossAttnUpBlock2D', 6)] = returned_CrossAttnUpBlock2D_block_var_list_6

        result_mean_dict[('CrossAttnUpBlock2D', 7)] = returned_CrossAttnUpBlock2D_block_mean_list_7
        result_var_dict[('CrossAttnUpBlock2D', 7)] = returned_CrossAttnUpBlock2D_block_var_list_7

        result_mean_dict[('CrossAttnUpBlock2D', 8)] = returned_CrossAttnUpBlock2D_block_mean_list_8
        result_var_dict[('CrossAttnUpBlock2D', 8)] = returned_CrossAttnUpBlock2D_block_var_list_8


        return result_mean_dict,result_var_dict



if __name__=="__main__":
    input_examples = []

    # dict_keys(
    # [('mid_block', 0), 
    # ('CrossAttnDownBlock2D', 1), 
    # ('CrossAttnDownBlock2D', 2), 
    # ('CrossAttnDownBlock2D', 3), 
    # ('DownBlock2D', 4), 
    # ('UpBlock2D', 5),
    # ('CrossAttnUpBlock2D', 6), 
    # ('CrossAttnUpBlock2D', 7), 
    # ('CrossAttnUpBlock2D', 8)])

    all_keys= [('mid_block', 0), 
    ('CrossAttnDownBlock2D', 1), ('CrossAttnDownBlock2D', 2), 
    ('CrossAttnDownBlock2D', 3), ('DownBlock2D', 4), 
    ('UpBlock2D', 5), ('CrossAttnUpBlock2D', 6), 
    ('CrossAttnUpBlock2D', 7), ('CrossAttnUpBlock2D', 8)]

    feat_bank_dict = dict()
    mean_bank_dict = dict()
    var_bank_dict = dict()
    for key in all_keys:

        if key ==('mid_block', 0):
            mean_bank_dict[key] = [torch.randn(1,1280,1,1)]
            var_bank_dict[key] = [torch.randn(1,1280,1,1)]
            feat_bank_dict[key] = [torch.randn(1,1280,8,8)]
        if key ==('CrossAttnDownBlock2D', 1):
            mean_bank_dict[key] = [[torch.randn(1,320,1,1)],[torch.randn(1,320,1,1)]]
            var_bank_dict[key] = [[torch.randn(1,320,1,1)],[torch.randn(1,320,1,1)]]
            feat_bank_dict[key] = [[torch.randn(1,320,64,64)],[torch.randn(1,320,64,64)]]
        if key ==('CrossAttnDownBlock2D', 2):
            mean_bank_dict[key] = [[torch.randn(1,640,1,1)],[torch.randn(1,640,1,1)]]
            var_bank_dict[key] = [[torch.randn(1,640,1,1)],[torch.randn(1,640,1,1)]]
            feat_bank_dict[key] = [[torch.randn(1,640,32,32)],[torch.randn(1,640,32,32)]]
        if key ==('CrossAttnDownBlock2D', 3):
            mean_bank_dict[key] = [[torch.randn(1,1280,1,1)],[torch.randn(1,1280,1,1)]]
            var_bank_dict[key] = [[torch.randn(1,1280,1,1)],[torch.randn(1,1280,1,1)]]
            feat_bank_dict[key] = [[torch.randn(1,1280,16,16)],[torch.randn(1,1280,16,16)]]
        if key ==('DownBlock2D', 4):
            mean_bank_dict[key] = [[torch.randn(1,1280,1,1)],[torch.randn(1,1280,1,1)]]
            var_bank_dict[key] = [[torch.randn(1,1280,1,1)],[torch.randn(1,1280,1,1)]]
            feat_bank_dict[key] = [[torch.randn(1,1280,8,8)],[torch.randn(1,1280,8,8)]]
        if key ==('UpBlock2D', 5):
            mean_bank_dict[key] = [[torch.randn(1,1280,1,1)],[torch.randn(1,1280,1,1)],[torch.randn(1,1280,1,1)]]
            var_bank_dict[key] = [[torch.randn(1,1280,1,1)],[torch.randn(1,1280,1,1)],[torch.randn(1,1280,1,1)]]
            feat_bank_dict[key] = [[torch.randn(1,1280,8,8)],[torch.randn(1,1280,8,8)],[torch.randn(1,1280,8,8)]]
        if key ==('CrossAttnUpBlock2D', 6):
            mean_bank_dict[key] = [[torch.randn(1,1280,1,1)],[torch.randn(1,1280,1,1)],[torch.randn(1,1280,1,1)]]
            var_bank_dict[key] = [[torch.randn(1,1280,1,1)],[torch.randn(1,1280,1,1)],[torch.randn(1,1280,1,1)]]
            feat_bank_dict[key] = [[torch.randn(1,1280,16,16)],[torch.randn(1,1280,16,16)],[torch.randn(1,1280,16,16)]]
        if key ==('CrossAttnUpBlock2D', 7):
            mean_bank_dict[key] = [[torch.randn(1,640,1,1)],[torch.randn(1,640,1,1)],[torch.randn(1,640,1,1)]]
            var_bank_dict[key] = [[torch.randn(1,640,1,1)],[torch.randn(1,640,1,1)],[torch.randn(1,640,1,1)]]
            feat_bank_dict[key] = [[torch.randn(1,640,32,32)],[torch.randn(1,640,32,32)],[torch.randn(1,640,32,32)]]
        if key ==('CrossAttnUpBlock2D', 8):
            mean_bank_dict[key] = [[torch.randn(1,320,1,1)],[torch.randn(1,320,1,1)],[torch.randn(1,320,1,1)]]
            var_bank_dict[key] = [[torch.randn(1,320,1,1)],[torch.randn(1,320,1,1)],[torch.randn(1,320,1,1)]]
            feat_bank_dict[key] = [[torch.randn(1,320,64,64)],[torch.randn(1,320,64,64)],[torch.randn(1,320,64,64)]]



    f2a_converter = F2All_Converter()

    time_embed = torch.randn(1,512)
    text_embed = torch.randn(1,77,768)
    foreground_mask = torch.randn(1,1,512,512)

    f2a_converter(mean_bank = mean_bank_dict,
                    var_bank=var_bank_dict,
                    feat_bank=feat_bank_dict,
                    time_embed=time_embed,
                    text_embed=text_embed,
                    foreground_mask=foreground_mask)
    

