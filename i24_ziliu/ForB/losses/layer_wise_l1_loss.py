import torch
import torch.nn as nn
import torch.nn.functional as F


def LayerWise_L1_Loss(est_mean_bank,est_variance_bank,gt_mean_bank,gt_variance_bank):

    # dict_keys([('mid_block', 0), ('CrossAttnDownBlock2D', 1), ('CrossAttnDownBlock2D', 2), ('CrossAttnDownBlock2D', 3), ('DownBlock2D', 4), ('UpBlock2D', 5), ('CrossAttnUpBlock2D', 6), ('CrossAttnUpBlock2D', 7), ('CrossAttnUpBlock2D', 8)])
    mean_loss = 0
    var_loss = 0
    count = 0
    for key in est_mean_bank.keys():
        if key == ('mid_block', 0):
            est_mean_list = est_mean_bank[key]
            est_var_list = est_variance_bank[key]
            gt_mean_list = gt_mean_bank[key]
            gt_var_list = gt_variance_bank[key]

            for idx in range(len(est_mean_list)):
                est_mean = est_mean_list[idx]
                gt_mean = gt_mean_list[idx]
                mean_loss+=F.l1_loss(est_mean,gt_mean,size_average=True,reduction=True)

                est_var = est_var_list[idx]
                gt_var= gt_var_list[idx]
                var_loss+=F.l1_loss(est_var,gt_var,size_average=True,reduction=True)
                count+=1
            
        elif key == ('CrossAttnDownBlock2D', 1):
            est_mean_list = est_mean_bank[key]
            est_var_list = est_variance_bank[key]
            gt_mean_list = gt_mean_bank[key]
            gt_var_list = gt_variance_bank[key]

            for idx in range(len(est_mean_list)):
                
                est_mean = est_mean_list[idx]
                gt_mean = gt_mean_list[idx]
                mean_loss+=F.l1_loss(est_mean,gt_mean[0],size_average=True,reduction=True)

                est_var = est_var_list[idx]
                gt_var= gt_var_list[idx]
                var_loss+=F.l1_loss(est_var,gt_var[0],size_average=True,reduction=True)
                count+=1

        
        elif key == ('CrossAttnDownBlock2D', 2):
            est_mean_list = est_mean_bank[key]
            est_var_list = est_variance_bank[key]
            gt_mean_list = gt_mean_bank[key]
            gt_var_list = gt_variance_bank[key]

            for idx in range(len(est_mean_list)):
                
                est_mean = est_mean_list[idx]
                gt_mean = gt_mean_list[idx]
                mean_loss+=F.l1_loss(est_mean,gt_mean[0],size_average=True,reduction=True)

                est_var = est_var_list[idx]
                gt_var= gt_var_list[idx]
                var_loss+=F.l1_loss(est_var,gt_var[0],size_average=True,reduction=True)
                count+=1
        
        elif key == ('CrossAttnDownBlock2D', 3):
            est_mean_list = est_mean_bank[key]
            est_var_list = est_variance_bank[key]
            gt_mean_list = gt_mean_bank[key]
            gt_var_list = gt_variance_bank[key]

            for idx in range(len(est_mean_list)):
                
                est_mean = est_mean_list[idx]
                gt_mean = gt_mean_list[idx]
                mean_loss+=F.l1_loss(est_mean,gt_mean[0],size_average=True,reduction=True)

                est_var = est_var_list[idx]
                gt_var= gt_var_list[idx]
                var_loss+=F.l1_loss(est_var,gt_var[0],size_average=True,reduction=True)
                count+=1
        elif key == ('DownBlock2D', 4):
            est_mean_list = est_mean_bank[key]
            est_var_list = est_variance_bank[key]
            gt_mean_list = gt_mean_bank[key]
            gt_var_list = gt_variance_bank[key]

            for idx in range(len(est_mean_list)):
                
                est_mean = est_mean_list[idx]
                gt_mean = gt_mean_list[idx]
                mean_loss+=F.l1_loss(est_mean,gt_mean[0],size_average=True,reduction=True)

                est_var = est_var_list[idx]
                gt_var= gt_var_list[idx]
                var_loss+=F.l1_loss(est_var,gt_var[0],size_average=True,reduction=True)
                count+=1
        elif key ==('UpBlock2D', 5):
            est_mean_list = est_mean_bank[key]
            est_var_list = est_variance_bank[key]
            gt_mean_list = gt_mean_bank[key]
            gt_var_list = gt_variance_bank[key]

            for idx in range(len(est_mean_list)):
                
                est_mean = est_mean_list[idx]
                gt_mean = gt_mean_list[idx]
                mean_loss+=F.l1_loss(est_mean,gt_mean[0],size_average=True,reduction=True)

                est_var = est_var_list[idx]
                gt_var= gt_var_list[idx]
                var_loss+=F.l1_loss(est_var,gt_var[0],size_average=True,reduction=True)
                count+=1
        elif key == ('CrossAttnUpBlock2D', 6):
            est_mean_list = est_mean_bank[key]
            est_var_list = est_variance_bank[key]
            gt_mean_list = gt_mean_bank[key]
            gt_var_list = gt_variance_bank[key]

            for idx in range(len(est_mean_list)):
                
                est_mean = est_mean_list[idx]
                gt_mean = gt_mean_list[idx]
                mean_loss+=F.l1_loss(est_mean,gt_mean[0],size_average=True,reduction=True)

                est_var = est_var_list[idx]
                gt_var= gt_var_list[idx]
                var_loss+=F.l1_loss(est_var,gt_var[0],size_average=True,reduction=True)
                count+=1
        elif key == ('CrossAttnUpBlock2D', 7):
            est_mean_list = est_mean_bank[key]
            est_var_list = est_variance_bank[key]
            gt_mean_list = gt_mean_bank[key]
            gt_var_list = gt_variance_bank[key]

            for idx in range(len(est_mean_list)):
                
                est_mean = est_mean_list[idx]
                gt_mean = gt_mean_list[idx]
                mean_loss+=F.l1_loss(est_mean,gt_mean[0],size_average=True,reduction=True)

                est_var = est_var_list[idx]
                gt_var= gt_var_list[idx]
                var_loss+=F.l1_loss(est_var,gt_var[0],size_average=True,reduction=True)
                count+=1
        elif key == ('CrossAttnUpBlock2D', 8):
            est_mean_list = est_mean_bank[key]
            est_var_list = est_variance_bank[key]
            gt_mean_list = gt_mean_bank[key]
            gt_var_list = gt_variance_bank[key]

            for idx in range(len(est_mean_list)):
                
                est_mean = est_mean_list[idx]
                gt_mean = gt_mean_list[idx]
                mean_loss+=F.l1_loss(est_mean,gt_mean[0],size_average=True,reduction=True)

                est_var = est_var_list[idx]
                gt_var= gt_var_list[idx]
                var_loss+=F.l1_loss(est_var,gt_var[0],size_average=True,reduction=True)
                count+=1
        else:
            raise NotImplementedError


    mean_loss = mean_loss/max(count,1)
    var_loss = var_loss/max(count,1)

    return mean_loss,var_loss

