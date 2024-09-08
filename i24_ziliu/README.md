# Foreground to Background Synthesis

### Dependencies
- Pytorch=2.0.1
- Diffusers (lastest)
- accelerate

Run the following code to create the conda env.
```
conda create --name pfn_creative python=3.9
pip install -r requirements.txt
```

## Data Preparation
- [Anime Background and Foreground Image Dataset](https://huggingface.co/datasets/skytnt/anime-segmentation/tree/main). (~18 G) 
---
|   Dir  |                Descriptions                | Format | Images |
|:------:|:------------------------------------------:|:------:|:------:|
|   bg   |              background images             |   jpg  |  8057  |
|   fg   |  foreground images, transparent background |   png  |  11802 |
| images | real images with background and foreground |   jpg  |  1111  |
|  masks |               labels for imgs              |   jpg  |  1111  |
---

Download the dataset:
```
cd Foreground2Background/datasets
python download_dataset.py
```

## Reference Third-Party Code
- [Animate Segmentation](https://github.com/SkyTNT/anime-segmentation)
- [Stable Diffusion Image Inpanting](https://huggingface.co/docs/diffusers/using-diffusers/inpaint)


## Progress Doc / Notion

[Google Doc](https://docs.google.com/document/d/1OQZwOXmSKQt9Nbv7AwpwHO1k9VH066dCZndnlA5-7jc/edit#heading=h.z5pa86dit7n8)
