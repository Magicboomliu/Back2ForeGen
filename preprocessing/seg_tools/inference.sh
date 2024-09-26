Get_BackGround_Mask(){
net="isnet_is"
ckpt="/mnt/nfs-mnj-home-43/i24_ziliu/Foreground2Background/AnimeSegmentation/anime-segmentation/pretrained_models/isnetis.ckpt"
data="/mnt/nfs-mnj-home-43/i24_ziliu/dataset/Flat2D/images"
out="/mnt/nfs-mnj-home-43/i24_ziliu/dataset/Flat2D/bg_mask_v1"
img_size=1024

python inference.py --net $net \
                    --ckpt $ckpt \
                    --data $data \
                    --out $out \
                    --img-size $img_size


}


Get_BackGround_Mask
