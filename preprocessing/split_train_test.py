import os
import sys
sys.path.append("../..")
from dataloader.utils.utils import read_text_lines
from sklearn.model_selection import train_test_split

def write_list_into_files(content_list,fname):
    with open(fname,'w') as f:
        for idx, line in enumerate(content_list):
            if idx!=len(content_list)-1:
                f.writelines(line+"\n")
            else:
                f.writelines(line)


if __name__=="__main__":

    total_descriptions_file_name = "/mnt/nfs-mnj-home-43/i24_ziliu/dataset/SD_Gen_Images/descriptions.txt"
    # 1000 lines
    descriptions_contents = read_text_lines(total_descriptions_file_name)

    new_total_lines = []
    for idx, description in enumerate(descriptions_contents):
        descriptions_with_id = str(idx) + " "+ description
        new_total_lines.append(descriptions_with_id)
    
    # 使用 train_test_split 函数将数据集随机分为训练集和测试集
    train_data, test_data = train_test_split(new_total_lines, test_size=0.1, random_state=42)
    
    # saved train data
    train_data_name = total_descriptions_file_name.replace("descriptions","descriptions_train")
    write_list_into_files(train_data,train_data_name)

    # saved test data
    test_data_name = total_descriptions_file_name.replace("descriptions","descriptions_test")
    write_list_into_files(test_data,test_data_name)