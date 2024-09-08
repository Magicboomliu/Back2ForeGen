TRAIN_SD15_F2B(){
cd ..
cd trainer/
pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5"
pretrained_single_file="/mnt/nfs-mnj-home-43/i24_ziliu/pretrained_models/sd15_anime.safetensors"
pretrained_VAE_single_file="/mnt/nfs-mnj-home-43/i24_ziliu/pretrained_models/pretrained_VAE/vae-ft-mse-840000-ema-pruned_fp16.safetensors"
dataset_name="SD_Gen_Images"
dataset_path="/mnt/nfs-mnj-home-43/i24_ziliu/dataset/SD_Gen_Images"
trainlist="descriptions_train.txt"
vallist="descriptions_test.txt"
output_dir="../outputs/$dataset_name/sd15_anime_specific_vae"
seed=100
train_batch_size=1
num_train_epochs=200
gradient_accumulation_steps=32
learning_rate=1e-4
lr_scheduler="cosine"
lr_warmup_steps=0
dataloader_num_workers=4
logging_dir="../logs/$dataset_name"
checkpointing_steps=1000

CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision="fp16"  sd_gen_image_trainer.py \
                            --pretrained_model_name_or_path $pretrained_model_name_or_path \
                            --pretrained_single_file $pretrained_single_file \
                            --dataset_name $dataset_name \
                            --dataset_path $dataset_path \
                            --trainlist $trainlist \
                            --vallist $vallist \
                            --output_dir $output_dir \
                            --seed $seed \
                            --train_batch_size $train_batch_size \
                            --num_train_epochs $num_train_epochs \
                            --gradient_accumulation_steps $gradient_accumulation_steps \
                            --learning_rate $learning_rate \
                            --lr_scheduler $lr_scheduler \
                            --lr_warmup_steps $lr_warmup_steps \
                            --dataloader_num_workers $dataloader_num_workers \
                            --logging_dir $logging_dir \
                            --checkpointing_steps $checkpointing_steps \
                            --pretrained_VAE_single_file $pretrained_VAE_single_file \
                            --use_adIN
}

TRAIN_SD15_F2B_Mix(){
cd ..
cd trainer/
pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5"
pretrained_single_file="/mnt/nfs-mnj-home-43/i24_ziliu/pretrained_models/sd15_anime.safetensors"
pretrained_VAE_single_file="/mnt/nfs-mnj-home-43/i24_ziliu/pretrained_models/pretrained_VAE/vae-ft-mse-840000-ema-pruned_fp16.safetensors"
dataset_name="Mixed_Dataset"
dataset_path="/mnt/nfs-mnj-home-43/i24_ziliu/dataset/"
trainlist="/mnt/nfs-mnj-home-43/i24_ziliu/i24_ziliu/ForB/filenames/training_data_all_selected.txt"
vallist="/mnt/nfs-mnj-home-43/i24_ziliu/i24_ziliu/ForB/filenames/training_data_all_selected.txt"
output_dir="../outputs/$dataset_name/sd15_learnable_VAE"
seed=100
train_batch_size=1
num_train_epochs=200
gradient_accumulation_steps=32
learning_rate=1e-4
lr_scheduler="cosine"
lr_warmup_steps=0
dataloader_num_workers=4
logging_dir="../logs/$dataset_name"
checkpointing_steps=1000

CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision="fp16"  sd_gen_image_trainer_mix.py \
                            --pretrained_model_name_or_path $pretrained_model_name_or_path \
                            --pretrained_single_file $pretrained_single_file \
                            --dataset_name $dataset_name \
                            --dataset_path $dataset_path \
                            --trainlist $trainlist \
                            --vallist $vallist \
                            --output_dir $output_dir \
                            --seed $seed \
                            --train_batch_size $train_batch_size \
                            --num_train_epochs $num_train_epochs \
                            --gradient_accumulation_steps $gradient_accumulation_steps \
                            --learning_rate $learning_rate \
                            --lr_scheduler $lr_scheduler \
                            --lr_warmup_steps $lr_warmup_steps \
                            --dataloader_num_workers $dataloader_num_workers \
                            --logging_dir $logging_dir \
                            --checkpointing_steps $checkpointing_steps \
                            --pretrained_VAE_single_file $pretrained_VAE_single_file \
                            --use_adIN
}


TRAIN_SD15_F2B_Mix

# TRAIN_SD15_F2B