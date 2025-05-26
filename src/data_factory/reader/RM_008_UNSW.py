from utils import load_data
import os
import numpy as np
from utils import fix_byte_order

def read(file_path,*args):
    data = load_data(file_path, file_type='mat')
    # Define keys in specific order
    keys = ["accH", "accV", "enc1", "enc2", "loadCell", "tacho"]
    # Extract arrays from the dictionary
    arrays = [data[key] for key in keys if key in data]
    # Concatenate along the last dimension (C)
    data = np.concatenate(arrays, axis=1)
    
    data = fix_byte_order(data)
    

    return data
if __name__ == "__main__":
    file_path = "/home/user/data/a_bearing/a_008_UNSW_raw/Test 2/6Hz/vib_000005664_06.mat"
    data = read(file_path)
    print(data)
    
    file_path = "/home/user/data/a_bearing/a_008_UNSW_raw/Test 3/Multiple speeds/vib_000334687_15.mat"
    data = read(file_path)
    print(data.shape)
    
