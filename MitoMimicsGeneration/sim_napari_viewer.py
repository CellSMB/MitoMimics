import argparse
import napari
import os
from pathlib import Path
import numpy as np
import skimage

def main():
    parser = argparse.ArgumentParser(description='Process a folder location.')
    parser.add_argument('--loc', type=str, help='Path to the folder containing the fake data.')
    parser.add_argument('--rloc', type=str, help='Path to the npy of a real raw data.')

    args = parser.parse_args()

    # Access the folder location
    folder_location = Path(args.loc)
    real_numpy_location = args.rloc
    

    fmask_path = os.path.join(folder_location, 'double_res_fmask.npy')
    skeleton_path = os.path.join(folder_location, 'double_res_skeleton.npy')
    double_res_path = os.path.join(folder_location, 'double_res_raw.npy')
    
    skeleton_weights_path = os.path.join(folder_location, 'skeleton_weights.npy')
    fmask_weights_path = os.path.join(folder_location, 'fmask_weights.npy')


    fmask_arr = np.load(fmask_path)
    skeleton_arr = np.load(skeleton_path)
    double_res_arr = np.load(double_res_path)
    
    #skeleton_weights_arr = np.load(skeleton_weights_path)
    #fmask_weights_arr = np.load(fmask_weights_path)


    
    
    if real_numpy_location is not None:
    
        real_numpy = np.load(Path(real_numpy_location))
    
    
    
    raw_shape = double_res_arr.shape
    
    viewer = napari.Viewer()
    viewer.add_image(np.zeros(raw_shape))

    
    if real_numpy_location is not None:
        r1 = viewer.add_image(real_numpy)
        r1.contrast_limits = (0,1)

    raw_double_resL = viewer.add_image(double_res_arr, name = 'raw_double_res')
    raw_double_resL.contrast_limits = (0,256)
    
    skeletonL = viewer.add_labels(skeleton_arr, name = 'Skeleton')
    fmaskL = viewer.add_labels(fmask_arr, name = 'FMASK')
    
    #skeleton_weights = viewer.add_image(skeleton_weights_arr, name = 'skeleton_weights')
    #fmask_weights = viewer.add_image(fmask_weights_arr, name = 'fmask_weights')
    
        # switch 1 timestep earlier
    @viewer.bind_key('t')
    def update_func(_):
        vz_plane = viewer.dims.order[0]
        current_vz = viewer.dims.current_step[vz_plane]
        viewer.dims.set_current_step(vz_plane, current_vz-1)


    # switch 1 timestep later
    @viewer.bind_key('y')
    def update_func2(_):
        vz_plane = viewer.dims.order[0]
        current_vz = viewer.dims.current_step[vz_plane]
        viewer.dims.set_current_step(vz_plane, current_vz+1)
        


    napari.run()

if __name__ == '__main__':
    main()
