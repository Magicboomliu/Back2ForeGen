

Inference_start_with_inpaint(){
cd ..
cd  infernece
root_path="/data1/liu/PFN/mnt/nfs-mnj-home-43/i24_ziliu/dataset/Synthesis_Images/"
saved_path="../outputs/evaluation_results/inpainting_pfn_with_initial"
base_model_path="/home/zliu/PFN/pretrained_models/base_models/anything45Inpainting_v10-inpainting.safetensors"
validation_list_path="/home/zliu/PFN/Example/PFN24/filenames/validation_select_data.txt"
without_noise_model="/home/zliu/PFN/pretrained_models/Converter/Mix_inpainting_with_initial/ckpt_13001.pt"
seed=1204
inference_type='ours_adaIN' # ["off_adaIN",'off_attn',"off_attn_adaIN","off_inpainting","ours_attn","ours_adaIN","ous_attn_adaIN"]


python sequential_pfn_inpainting_reference_only_with_initial_inpainting.py \
            --root_path $root_path \
            --saved_path $saved_path \
            --base_model_path $base_model_path \
            --validation_list_path $validation_list_path \
            --without_noise_model $without_noise_model \
            --seed $seed \
            --inference_type $inference_type
}



Inference_start_Simple(){
cd ..
cd  infernece
root_path="/data1/liu/PFN/mnt/nfs-mnj-home-43/i24_ziliu/dataset/Synthesis_Images/"
saved_path="../outputs/evaluation_results/inpainting_pfn_simple"
base_model_path="/home/zliu/PFN/pretrained_models/base_models/anything45Inpainting_v10-inpainting.safetensors"
validation_list_path="/home/zliu/PFN/Example/PFN24/filenames/validation_select_data.txt"
without_noise_model="/home/zliu/PFN/pretrained_models/Converter/Mix_Inpainting/ckpt_14001.pt"
seed=1204
inference_type='ours_adaIN' # ["off_adaIN",'off_attn',"off_attn_adaIN","off_inpainting","ours_attn","ours_adaIN","ous_attn_adaIN"]


python sequential_pfn_inpainting_reference_only_inference.py \
            --root_path $root_path \
            --saved_path $saved_path \
            --base_model_path $base_model_path \
            --validation_list_path $validation_list_path \
            --without_noise_model $without_noise_model \
            --seed $seed \
            --inference_type $inference_type
}

Inference_start_Simple
# Inference_start_with_inpaint


