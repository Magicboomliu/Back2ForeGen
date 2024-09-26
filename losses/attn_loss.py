import torch
import torch.nn as nn
import torch.nn.functional as F


def Attn_loss(list1,list2):
    
    assert len(list1) == len(list2)
    
    loss = 0
    for idx, sample in enumerate(list1):
        ele_lst1 =  list1[idx]
        ele_lst2 = list2[idx]
        
        loss += F.smooth_l1_loss(ele_lst1,ele_lst2,size_average=True,reduction='mean')
    
    return loss/len(list1)