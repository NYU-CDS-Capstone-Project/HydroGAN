import os
import re


def get_output_folder(run_in_jupyter = False):
    
    if run_in_jupyter:
        folder_base = "jupyter-output"
    else:
        folder_base = "drive-output"
    
    list_dir = os.listdir()
    
    list_dir = [x for x in list_dir if folder_base in x]
    list_dir = [re.findall('\d+',x) for x in list_dir]
    
    last_folder_no = [int(item) for sublist in list_dir for item in sublist]
    last_folder_no.append(0)
    last_folder_no = max(last_folder_no)
#     print(last_folder_no)
    
    new_folder = folder_base + "-" + str(last_folder_no+1)
    return new_folder