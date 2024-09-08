Inference_SD15_F2B(){
cd ..
cd pipelines
pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5"
pretrained_single_file="none"
dataset_name="SD_Gen_Images"
dataset_path="/mnt/nfs-mnj-home-43/i24_ziliu/dataset/SD_Gen_Images"
trainlist="descriptions_train.txt"
vallist="descriptions_test.txt"
output_dir="../outputs/inference_results/$dataset_name"
seed=100
dataloader_num_workers=4
pretrained_converter_path="/mnt/nfs-mnj-home-43/i24_ziliu/i24_ziliu/ForB/outputs/SD_Gen_Images/ckpt_9001.pt"

CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision="fp16"  stablediffusion_adaIN_pipeline.py \
                            --pretrained_model_name_or_path $pretrained_model_name_or_path \
                            --pretrained_single_file $pretrained_single_file \
                            --dataset_name $dataset_name \
                            --dataset_path $dataset_path \
                            --trainlist $trainlist \
                            --vallist $vallist \
                            --output_dir $output_dir \
                            --seed $seed \
                            --dataloader_num_workers $dataloader_num_workers \
                            --pretrained_converter_path $pretrained_converter_path \
                            --use_adIN


}


Inference_SD15_F2B