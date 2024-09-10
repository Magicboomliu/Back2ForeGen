import os
import sys
import json
import torch
from glob import glob
import logging
import numpy as np
import re


def get_id_and_prompt(prompt):
    match = re.match(r"(\d+)\s+(.*)", prompt)
    if match:
        number_part = match.group(1)  # 数字部分
        text_part = match.group(2)    # 文本部分
    else:
        first_space_index = input_string.find(' ')
        number_part = prompt[:first_space_index]
        text_part = prompt[first_space_index+1,:]
    

    return int(number_part),text_part
    

def read_text_lines(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines = [l.rstrip() for l in lines]
    return lines

def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    # fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    fmt = "[%(asctime)s] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger