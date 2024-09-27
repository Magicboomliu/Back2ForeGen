TRAIN_SD15_Inpainting_F2B_Mix_with_Attn_And_AdaIN_Simple(){
cd ..
cd trainer/
pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5"
pretrained_single_file="/home/zliu/PFN/pretrained_models/base_models/anything45Inpainting_v10-inpainting.safetensors"
pretrained_VAE_single_file="none"
dataset_name="Mixed_Dataset"
dataset_path="/data1/liu/PFN/mnt/nfs-mnj-home-43/i24_ziliu/dataset/Synthesis_Images/"
trainlist="/home/zliu/PFN/Example/PFN24/filenames/training_data_all_selected.txt"
vallist="/home/zliu/PFN/Example/PFN24/filenames/training_data_all_selected.txt"
output_dir="../outputs/$dataset_name/sd15_inpainting_mix_converters_simple"
seed=70
train_batch_size=1
num_train_epochs=100
gradient_accumulation_steps=16
learning_rate=1e-4
lr_scheduler="cosine"
lr_warmup_steps=0
dataloader_num_workers=4
logging_dir="../logs/$dataset_name"
checkpointing_steps=1000
report_to="tensorboard"

CUDA_VISIBLE_DEVICES=0,1 accelerate launch --mixed_precision="fp16"  pfn_inpaininting_simple_mix_learner.py \
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
                            --use_adIN \
                            --use_attn \
                            --report_to  $report_to 
}



TRAIN_SD15_Inpainting_F2B_Mix_with_Attn_And_AdaIN_Start_With_Inpaint(){
cd ..
cd trainer/
pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5"
pretrained_single_file="/home/zliu/PFN/pretrained_models/base_models/anything45Inpainting_v10-inpainting.safetensors"
pretrained_VAE_single_file="none"
dataset_name="Mixed_Dataset"
dataset_path="/data1/liu/PFN/mnt/nfs-mnj-home-43/i24_ziliu/dataset/Synthesis_Images/"
trainlist="/home/zliu/PFN/Example/PFN24/filenames/training_data_all_selected.txt"
vallist="/home/zliu/PFN/Example/PFN24/filenames/training_data_all_selected.txt"
output_dir="../outputs/$dataset_name/sd15_inpainting_mix_converters_start_with_inpaint"
seed=70
train_batch_size=1
num_train_epochs=100
gradient_accumulation_steps=16
learning_rate=1e-4
lr_scheduler="cosine"
lr_warmup_steps=0
dataloader_num_workers=4
logging_dir="../logs/$dataset_name"
checkpointing_steps=1000
report_to="tensorboard"

CUDA_VISIBLE_DEVICES=0,1 accelerate launch --mixed_precision="fp16"  pfn_inpaininting_simple_mix_learner.py \
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
                            --use_adIN \
                            --use_attn \
                            --report_to  $report_to \
                            --start_with_inpaint
}



TRAIN_SD15_Inpainting_F2B_Mix_With_AdaIN_Converter_Only(){

cd ..
cd trainer/
pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5"
pretrained_single_file="/home/zliu/PFN/pretrained_models/base_models/anything45Inpainting_v10-inpainting.safetensors"
pretrained_VAE_single_file="none"
dataset_name="Mixed_Dataset"
dataset_path="/data1/liu/PFN/mnt/nfs-mnj-home-43/i24_ziliu/dataset/Synthesis_Images/"
trainlist="/home/zliu/PFN/Example/PFN24/filenames/training_data_all_selected.txt"
vallist="/home/zliu/PFN/Example/PFN24/filenames/training_data_all_selected.txt"
seed=70
train_batch_size=1
num_train_epochs=100
gradient_accumulation_steps=16
learning_rate=1e-4
lr_scheduler="cosine"
lr_warmup_steps=0
dataloader_num_workers=4
logging_dir="../logs/$dataset_name"
checkpointing_steps=1000
report_to="tensorboard"
adaIN_converter_type='simple'
output_dir="../outputs/$dataset_name/sd15_inpainting_mix_adaIN_conveters_only_simple"

CUDA_VISIBLE_DEVICES=0,1 accelerate launch --mixed_precision="fp16"  pfn_inpainting_simple_adaIN_converter_only.py \
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
                            --use_adIN \
                            --report_to  $report_to \
                            --start_with_inpaint \
                            --adaIN_converter_type $adaIN_converter_type








}

TRAIN_SD15_Inpainting_F2B_Mix_With_AdaIN_Converter_Only

# TRAIN_SD15_Inpainting_F2B_Mix_with_Attn_And_AdaIN_Simple
# TRAIN_SD15_Inpainting_F2B_Mix_with_Attn_And_AdaIN_Start_With_Inpaint