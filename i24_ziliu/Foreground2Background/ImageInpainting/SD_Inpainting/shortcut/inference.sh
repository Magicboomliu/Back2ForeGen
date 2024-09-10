RUN_SD_Based_Impanting(){
cd ..
cd scripts

pipeline_type="SD_IM"
weights_type='from_hub'
pretrained_weight="diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
image_folder="/mnt/nfs-mnj-home-43/i24_ziliu/i24_ziliu/Foreground2Background/example_data/initial_image"
mask_folder="/mnt/nfs-mnj-home-43/i24_ziliu/i24_ziliu/Foreground2Background/example_data/generated_mask"
saved_folder="/mnt/nfs-mnj-home-43/i24_ziliu/i24_ziliu/Foreground2Background/example_data/results"
seed=114514



python run_sd_inpainting.py --pipeline_type $pipeline_type \
                            --weights_type $weights_type \
                            --pretrained_weight $pretrained_weight \
                            --image_folder $image_folder \
                            --mask_folder $mask_folder \
                            --saved_folder $saved_folder \
                            --seed $seed


}







RUN_SD_Based_Impanting