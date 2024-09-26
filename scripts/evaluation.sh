GET_LPIPS(){
cd ..
cd evaluations
root_folder="/home/zliu/PFN/PFN24/PFN24/i24_ziliu/ForB/outputs/evaluation_results/inpainting_pfn_with_initial"
saved_name="output_lpips.json"

python get_lipips.py --root_folder $root_folder --saved_name $saved_name
}


GET_FID(){
cd ..
cd evaluations
root_folder="/home/zliu/PFN/PFN24/PFN24/i24_ziliu/ForB/outputs/evaluation_results/inpainting_pfn_with_initial"
saved_name="output_fid.json"

python get_fid.py --root_folder $root_folder --saved_name $saved_name
}


GET_LPIPS

# GET_FID