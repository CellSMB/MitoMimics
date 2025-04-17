import argparse
import napari
import os
from pathlib import Path
import numpy as np
import skimage

def main():
    parser = argparse.ArgumentParser(description='Process a folder location.')
    parser.add_argument('--loc', type=str, help='Path to the folder containing the fake data.')
    
    
    args = parser.parse_args()
    real_folder = Path(args.loc)
    
    
    real_numpy_dict = dict()
    
    for file in os.listdir(real_folder):
        # check if file is numpy
        if file.endswith('.npy'):
            # load the numpy file
            temp_file_name = file.split('.')[0]
            real_numpy_dict[temp_file_name] = np.load(os.path.join(real_folder, file))
            
    

    
    viewer = napari.view_image(np.zeros((64,1024,1024)))

    for key, value in real_numpy_dict.items():
        d = viewer.add_image(value, name = key)
        d.contrast_limits = (0,1)
    

    



    napari.run()

if __name__ == '__main__':
    main()