# Harmonious Background Generation Driven by Foreground Images

[Project Slide](https://docs.google.com/presentation/d/1CAWBnfH-Yx8kBsKU2P_xrVJmqkB3Ze7m/edit?usp=sharing&ouid=112605403951022205460&rtpof=true&sd=true)  

![image](figures/result.png)
### Dependencies
- Pytorch=2.0.1
- Diffusers (lastest)
- accelerate

Run the following code to create the conda env.
```
conda create --name pfn_creative python=3.9
pip install -r requirements.txt
```

### Dataset and Pretrained Model Download

```

```

### Training the Converters

Before training, initialize the `accelerate` environment
```
accelerate config default
```

- Train Start with Pure Black Background

```
cd scripts
sh train_sd_f2b.sh
```
Set the training options  in the scirpts into `"TRAIN_SD15_Inpainting_F2B_Mix_with_Attn_And_AdaIN_Simple"` . 

- Train Start with Initial Inpainting Background
```
cd scripts
sh train_sd_f2b.sh
```
Set the training options  in the scirpts into `"TRAIN_SD15_Inpainting_F2B_Mix_with_Attn_And_AdaIN_Start_With_Inpaint"` . 


### Inference the Background using Converters

```
cd scripts
sh  inference.sh
```

set the infernece options to change the models.


### Evaluate the Results with LPIPS and FID
```
cd scripts

sh evaluation.sh
```