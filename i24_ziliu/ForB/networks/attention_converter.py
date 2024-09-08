import torch
import torch.nn as nn

class AggregationNet(nn.Module):
    def __init__(self):
        super(AggregationNet, self).__init__()
        
        # 定义每个输入的线性变换或卷积变换来调整维度
        self.first_encode = nn.Sequential(
                                nn.Linear(1280*5, 1280*5),
                                nn.Linear(1280*5,1280*5))
    
        self.second_encode = nn.Linear(1280, 1280)

        # 文本 embedding 处理
        self.text_proj = nn.Linear(768, 256)
        # 时间嵌入处理
        self.time_proj = nn.Linear(512, 256)
        self.fusion_layer = nn.Linear(78*256,256)
        


        # 对 attention 输出进行投影，使其与输入形状匹配
        self.attn_proj_4096_320 = nn.Linear(256, 320)
        self.attn_proj_1024_640 = nn.Linear(256, 640)
        self.attn_proj_256_1280 = nn.Linear(256, 1280)
        self.attn_proj_64_1280 = nn.Linear(256, 1280)

        # 输出变换，调整到期望的形状
        self.third_encode = nn.Sequential(
                                nn.Linear(640*5, 640*5),
                                nn.Linear(640*5,640*5))

        self.four_encode = nn.Sequential(
                                nn.Linear(320*5, 320*5),
                                nn.Linear(320*5,320*5))
        


        self.output_branch1 = nn.Linear(1280*5,1280*5)
        self.output_branch2  = nn.Linear(1280,1280)
        self.output_branch3 = nn.Linear(640*5,640*5)
        self.output_branch4 = nn.Linear(320*5,320*5)
        

        
        

    def forward(self, inputs, text_embed, time_embed):
        # 提取每个输入的张量
        x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16 = inputs
        
        # # 将文本嵌入和时间嵌入进行线性投影
        text_feat = self.text_proj(text_embed)
        time_feat = self.time_proj(time_embed).unsqueeze(1)
        
        conditional_prompt = torch.cat((text_feat,time_feat),dim=1).view(time_feat.shape[0],-1)
        prompt_feat = self.fusion_layer(conditional_prompt).unsqueeze(1)
        
        # # 对 attention 输出进行线性投影，使其形状与输入相匹配
        attn_4096_320 = self.attn_proj_4096_320(prompt_feat)  # 投影到 320
        attn_1024_640 = self.attn_proj_1024_640(prompt_feat)  # 投影到 640
        attn_256_1280 = self.attn_proj_256_1280(prompt_feat)  # 投影到 1280
        attn_64_1280 = self.attn_proj_64_1280(prompt_feat)    # 投影到 1280

        # 对每个输入特征进行线性变换保持维度
        first_branch = torch.cat((x1,x2,x3,x4,x5),dim=-1)
        
        first_branch_encode = self.first_encode(first_branch)
        first_branch_encode = first_branch_encode + attn_256_1280.repeat(1,1,5)
        first_branch_decode = self.output_branch1(first_branch_encode)        
        returned_x1, returned_x2,returned_x3, returned_x4,returned_x5 = torch.chunk(first_branch_decode,chunks=5,dim=-1)
        
        
        second_branch_encode = self.second_encode(x6)
        second_branch_encode = second_branch_encode + attn_64_1280
        returned_x6 = self.output_branch2(second_branch_encode)



        third_branch = torch.cat((x7,x8,x9,x10,x11),dim=-1)
        third_branch_encode = self.third_encode(third_branch)
        third_branch_encode = third_branch_encode + attn_1024_640.repeat(1,1,5)
        third_branch_decode = self.output_branch3(third_branch_encode)        
        returned_x7,returned_x8, returned_x9,returned_x10,returned_x11 = torch.chunk(third_branch_decode,chunks=5,dim=-1)
        

        fourth_branch = torch.cat((x12,x13,x14,x15,x16),dim=-1)
        fourth_branch_encode = self.four_encode(fourth_branch)
        fourth_branch_encode = fourth_branch_encode + attn_4096_320.repeat(1,1,5)
        fourth_branch_decode = self.output_branch4(fourth_branch_encode)
        returned_x12,returned_x13,returned_x14, returned_x15,returned_x16 = torch.chunk(fourth_branch_decode ,chunks=5,dim=-1)
        
        

        
        return [returned_x1,returned_x2,returned_x3,returned_x4,returned_x5,returned_x6,returned_x7,returned_x8,
                returned_x9,returned_x10,returned_x11,returned_x12,returned_x13,returned_x14,returned_x15,returned_x16]
        






if __name__=="__main__":
    x1 = torch.randn((2,4096,320)).cuda()
    x2 = torch.randn((2,4096,320)).cuda()
    
    x3 = torch.randn((2,1024,640)).cuda()
    x4 = torch.randn((2,1024,640)).cuda()
    
    x5 = torch.randn((2,256,1280)).cuda()
    x6 = torch.randn((2,256,1280)).cuda()
    
    x7 = torch.randn((2,64,1280)).cuda()

    x8 = torch.randn((2,256,1280)).cuda()
    x9 = torch.randn((2,256,1280)).cuda()
    x10 = torch.randn((2,256,1280)).cuda()
    
    x11 = torch.randn((2,1024,640)).cuda()
    x12 = torch.randn((2,1024,640)).cuda()
    x13 = torch.randn((2,1024,640)).cuda()

    x14 = torch.randn((2,4096,320)).cuda()
    x15 = torch.randn((2,4096,320)).cuda()
    x16 = torch.randn((2,4096,320)).cuda()
    
    
    text_embed = torch.randn((2,77,768)).cuda()
    
    time_embed = torch.randn((2,512)).cuda()


    inputs = [
        torch.randn((2, 4096, 320)).cuda(),
        torch.randn((2, 4096, 320)).cuda(),
        torch.randn((2, 1024, 640)).cuda(),
        torch.randn((2, 1024, 640)).cuda(),
        torch.randn((2, 256, 1280)).cuda(),
        torch.randn((2, 256, 1280)).cuda(),
        torch.randn((2, 64, 1280)).cuda(),
        torch.randn((2, 256, 1280)).cuda(),
        torch.randn((2, 256, 1280)).cuda(),
        torch.randn((2, 256, 1280)).cuda(),
        torch.randn((2, 1024, 640)).cuda(),
        torch.randn((2, 1024, 640)).cuda(),
        torch.randn((2, 1024, 640)).cuda(),
        torch.randn((2, 4096, 320)).cuda(),
        torch.randn((2, 4096, 320)).cuda(),
        torch.randn((2, 4096, 320)).cuda()
    ]
    # 使用网络
    net = AggregationNet().cuda()
    outputs = net(inputs, text_embed, time_embed)

    # 检查输出形状是否一致
    for output in outputs:
        print(output.shape)