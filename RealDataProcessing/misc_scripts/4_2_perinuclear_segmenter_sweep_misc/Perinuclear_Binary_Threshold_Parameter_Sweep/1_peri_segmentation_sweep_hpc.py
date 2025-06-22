import os
import time
import gc
import numpy as np
from functools import partial
import concurrent.futures
from tqdm import tqdm
from scipy.ndimage import gaussian_filter, uniform_filter1d
import skimage.filters
from skimage.measure import regionprops
from skimage.morphology import remove_small_objects
import argparse
import multiprocessing

multiprocessing.set_start_method('fork', force=True)

def process_single_frame_threshold(frame_idx, raw_arr, params):
    
    sigma_value, thresh_adj_mult, min_obj_size, window_size, averaging_threshold = params
    
    # Process Single Frames
    current_frame = raw_arr[frame_idx]
    
    # Apply Gaussian smoothing
    smoothed = gaussian_filter(current_frame, sigma=sigma_value)
    
    # Apply Otsu thresholding
    thresh = skimage.filters.threshold_otsu(smoothed)
    adjusted_thresh = thresh * thresh_adj_mult
    binary_mask = smoothed > adjusted_thresh
    
    # Remove small objects
    binary_cleaned = remove_small_objects(binary_mask, min_size=min_obj_size)
    
    # Explicitly clear large intermediate variables
    del smoothed, binary_mask
    gc.collect()

    return frame_idx, binary_cleaned


def process_perinuclear_mask(raw_arr, output_path, sigma_value=10, thresh_adj_mult=1.3, min_obj_size=5000, window_size=10, averaging_threshold=0.5, max_workers=None):
    # Process in parallel

    start_time=time.time()

    if max_workers==None:
        max_workers=8
        
    print(max_workers)
    num_frames, height, width = raw_arr.shape
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    binary_masks_filename = output_path.replace('.npz', '_binary_masks.npz')

    params = (sigma_value, thresh_adj_mult, min_obj_size, window_size, averaging_threshold)



    
    # Store the binary mask
    temp_binary_masks = np.zeros((num_frames, height, width), dtype=bool)

    process_binary_mask_partial = partial(process_single_frame_threshold, 
                                          raw_arr=raw_arr, 
                                          params=params)

    print(f"\n Starting Perinuclear Mask Threshold Parallel Processing of {num_frames} Frames")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_binary_mask_partial, frame_idx) for frame_idx in range(num_frames)]
        
        with tqdm(total=len(futures), desc="Processing Frame") as pbar:
            for future in concurrent.futures.as_completed(futures):
                frame_idx, binary_cleaned = future.result()
                temp_binary_masks[frame_idx] = binary_cleaned
                
                pbar.update(1)
                del binary_cleaned
                gc.collect()



    
    print(f"\n Averaging Binary Masks Across Windows of Size {window_size}")
    binary_masks_full = uniform_filter1d(temp_binary_masks, size=window_size, axis=0, mode='nearest')
    del temp_binary_masks

    np.savez_compressed(binary_masks_filename, data=binary_masks_full)

    total_time = time.time() - start_time

    del binary_masks_full
    gc.collect()

    print(f"\n Total Processing Time: {total_time}")


def process_single_parameter(sample_num, thresh_adj_mult, data_dir, output_root_dir, 
                           sigma_value=10, min_obj_size=5000,
                           window_size=10, averaging_threshold=0.5, max_workers=None):
    """Process a single sample with a single threshold multiplier value"""
    
    print(f"\n ==== Processing Sample {sample_num}, Threshold Multiplier {thresh_adj_mult:.1f} ====")
    
    # Paths for this sample
    sample_dir = os.path.join(data_dir, str(sample_num))
    raw_path = os.path.join(sample_dir, "raw.npz")
    
    # Check if files exist
    if not os.path.exists(raw_path):
        print(f"\n Error: Required files not found for sample {sample_num}.")
        return
    
    # Load data
    print(f"\n Loading data for sample {sample_num}...")
    raw_data = np.load(raw_path, mmap_mode='r')['arr_0']

    # Create output directory for this sample
    sample_output_dir = os.path.join(output_root_dir, str(sample_num))
    os.makedirs(sample_output_dir, exist_ok=True)
    
    # Create output directory for this parameter set
    param_dir_name = f"Sample{sample_num}_thresh_adj_mult_{thresh_adj_mult:.1f}"
    param_output_dir = os.path.join(sample_output_dir, param_dir_name)
    os.makedirs(param_output_dir, exist_ok=True)
    
    # Output path for results
    output_path = os.path.join(param_output_dir, f"results.npz")
    
    # Process with these parameters
    process_perinuclear_mask(
        raw_arr=raw_data,
        output_path=output_path,
        sigma_value=sigma_value,
        thresh_adj_mult=thresh_adj_mult,
        min_obj_size=min_obj_size,
        window_size=window_size,
        averaging_threshold=averaging_threshold,
        max_workers=max_workers
    )

    print(f"\n Completed processing for sample {sample_num}, thresh_adj_mult={thresh_adj_mult:.1f}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a single sample with a single parameter value')
    parser.add_argument('--data_dir', type=str, required=True, help='Root directory containing sample folders')
    parser.add_argument('--output_dir', type=str, required=True, help='Root directory for output results')
    parser.add_argument('--sample', type=int, required=True, help='Sample number to process')
    parser.add_argument('--thresh_adj_mult', type=float, required=True, help='Threshold adjustment multiplier value')
    parser.add_argument('--sigma', type=float, default=10, help='Sigma value for Gaussian filter')
    parser.add_argument('--min_obj_size', type=int, default=5000, help='Minimum object size to keep')
    parser.add_argument('--window_size', type=int, default=10, help='Window size for temporal averaging')
    parser.add_argument('--avg_threshold', type=float, default=0.5, help='Threshold for averaged binary masks')
    parser.add_argument('--max_workers', type=int, default=4, help='Maximum number of worker processes')
    
    args = parser.parse_args()
    
    # Process the single parameter
    process_single_parameter(
        sample_num=args.sample,
        thresh_adj_mult=args.thresh_adj_mult,
        data_dir=args.data_dir,
        output_root_dir=args.output_dir,
        sigma_value=args.sigma,
        min_obj_size=args.min_obj_size,
        window_size=args.window_size,
        averaging_threshold=args.avg_threshold,
        max_workers=args.max_workers
    )
