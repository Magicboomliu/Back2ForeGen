import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("../..")
from networks.adaIN_converter import F2All_Converter
from networks.attention_converter import AggregationNet



class ConverterNetwork(nn.Module):
    def __init__(self) -> None:
        super(ConverterNetwork,self).__init__()
        
        self.adain_net = F2All_Converter()
        self.attn_net = AggregationNet()
    
        
    def forward(self,mean_bank,var_bank,feat_bank,time_embed,text_embed,foreground_mask,inputs,use_seperate=None):
        
        
        if use_seperate=='adaIN':
            fg2all_mean_bank, fg2all_variance_bank = self.adain_net(mean_bank,var_bank,feat_bank,time_embed,text_embed,foreground_mask)
            converted_fg_attn_banks_list = None
        elif use_seperate=='attn':
            converted_fg_attn_banks_list = self.attn_net(inputs,text_embed,time_embed)
            fg2all_mean_bank = None 
            fg2all_variance_bank = None
            
            
        else:
            fg2all_mean_bank, fg2all_variance_bank = self.adain_net(mean_bank,var_bank,feat_bank,time_embed,text_embed,foreground_mask)
            
            converted_fg_attn_banks_list = self.attn_net(inputs,text_embed,time_embed)
        
        return fg2all_mean_bank, fg2all_variance_bank,converted_fg_attn_banks_list
