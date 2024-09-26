import os
import sys
sys.path.append("..")


def read_text_lines(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines = [l.rstrip() for l in lines]
    return lines

if __name__=="__main__":

    all_saved = []

    merged_txt_folder_path = "/mnt/nfs-mnj-home-43/i24_ziliu/i24_ziliu/ForB/playground/Dataset_Configuration/data_txt_folder"
    awaited_processed_txt = [os.path.join(merged_txt_folder_path,path) for path in os.listdir(merged_txt_folder_path)]

    for txt in awaited_processed_txt:
        all_saved.extend(read_text_lines(txt))
    

    with open("all_training_data.txt",'w') as f:
        for idx ,line in enumerate(all_saved):
            if idx!=len(all_saved)-1:
                f.writelines(line+"\n")
            else:
                f.writelines(line)


    pass