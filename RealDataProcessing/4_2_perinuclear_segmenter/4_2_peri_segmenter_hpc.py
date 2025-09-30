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
    """
    Threshold binarisation of perinuclear region for each frame
    """
    sigma_value, thresh_adj_mult, min_obj_size, overlap_thresh, transition_thresh, window_size = params
    
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



def process_single_frame_instance_separation(frame_idx, raw_arr, instance_arr, averaged_binary_masks, params):
    """
    Separate instance objects into true perinuclear, transition perinuclear, or telenuclear microdomains for each frame
    """
    sigma_value, thresh_adj_mult, min_obj_size, overlap_thresh, transition_thresh, window_size = params
    
    # Get objects for current frame
    frame_objects = instance_arr[frame_idx]
    
    binary_averaged_per_frame = averaged_binary_masks[frame_idx]
        
    perinuclear_labels = np.zeros_like(frame_objects, dtype=np.uint32)
    transition_labels = np.zeros_like(frame_objects, dtype=np.uint32)
    telenuclear_labels = np.zeros_like(frame_objects, dtype=np.uint32)

    # Label objects - only process non-zero areas
    if np.any(frame_objects):
        # Process objects to categorize them
        for region in regionprops(frame_objects):
            # Get coordinates of this object
            coords = region.coords
            
            # Calculate overlap ratio
            mask_values = binary_averaged_per_frame[coords[:, 0], coords[:, 1]]
            overlap_area = np.sum(mask_values)
            overlap_ratio = overlap_area / region.area
            
            # Check if the overlap ratio meets the thresholds where 
            # transition threshold defines regions with greater than overlap mask ratios as being true perinuclear 
            # and overlap threshold defines the percentage when less than or equal of the instance mask overlaps with the perinuclear binary mask
            if transition_thresh > overlap_ratio >= overlap_thresh:
                # Mark as transition perinuclear
                for y, x in coords:
                    transition_labels[y, x] = region.label
            elif overlap_ratio >= transition_thresh:
                # Mark as true perinuclear
                for y, x in coords:
                    perinuclear_labels[y, x] = region.label
            else:
                # Mark as telenuclear
                for y, x in coords:
                    telenuclear_labels[y, x] = region.label
                    
    del frame_objects, binary_averaged_per_frame
    gc.collect()

    return frame_idx, perinuclear_labels, telenuclear_labels, transition_labels



def process_perinuclear_mask(raw_arr, instance_arr, output_path, sigma_value=10, thresh_adj_mult=1.3, min_obj_size=5000, overlap_thresh=0.05, transition_thresh=0.65, window_size=10, max_workers=None):
    """
    Process perinuclear mask segmentation and instance separation in parallel on CPU
    """

    start_time=time.time()

    if max_workers==None:
        max_workers=8
        
    print(max_workers)
    num_frames, height, width = raw_arr.shape
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    perinuclear_filename = output_path.replace('.npz', 'true_perinuclear_instance_masks.npz')
    transition_filename = output_path.replace('.npz','perinuclear_transition_instance_masks.npz')
    telenuclear_filename = output_path.replace('.npz', 'telenuclear_instance_masks.npz')
    binary_masks_filename = output_path.replace('.npz', 'perinuclear_binary_masks.npz')
    combined_perinuclear_transition_filename = output_path.replace('.npz','full_combined_perinuclear_instance_masks.npz')

    perinuclear_full = np.zeros((num_frames, height, width), dtype=np.uint32)
    transition_full = np.zeros((num_frames, height, width), dtype=np.uint32)
    telenuclear_full = np.zeros((num_frames, height, width), dtype=np.uint32)
    combined_perinuclear_transition_full = np.zeros((num_frames, height, width), dtype=np.uint32)

    params = (sigma_value, thresh_adj_mult, min_obj_size, overlap_thresh, transition_thresh, window_size)



    
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

    
    process_instances_partial = partial(process_single_frame_instance_separation, 
                                        raw_arr=raw_arr, 
                                        instance_arr=instance_arr, 
                                        averaged_binary_masks=binary_masks_full,
                                        params=params)
    
    print(f"\n Starting Instance Separation Parallel Processing of {num_frames} Frames")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_instances_partial, frame_idx) for frame_idx in range(num_frames)]
        
        with tqdm(total=len(futures), desc="Processing Frame") as pbar:
            for future in concurrent.futures.as_completed(futures):
                frame_idx, perinuclear_labels, telenuclear_labels, transition_labels = future.result()
                perinuclear_full[frame_idx] = perinuclear_labels
                transition_full[frame_idx] = transition_labels
                telenuclear_full[frame_idx] = telenuclear_labels
                combined_perinuclear_transition_full[frame_idx] = perinuclear_labels + transition_labels
                
                pbar.update(1)
                del perinuclear_labels, telenuclear_labels, transition_labels
                gc.collect()


    print("\n Saving Files")
    np.savez_compressed(perinuclear_filename, data=perinuclear_full)
    np.savez_compressed(transition_filename, data=transition_full)
    np.savez_compressed(telenuclear_filename, data=telenuclear_full)
    np.savez_compressed(binary_masks_filename, data=binary_masks_full)
    np.savez_compressed(combined_perinuclear_transition_filename, data=combined_perinuclear_transition_full)

    total_time = time.time() - start_time

    del perinuclear_full, telenuclear_full, binary_masks_full, transition_full, combined_perinuclear_transition_full
    gc.collect()

    print(f"\n Total Processing Time: {total_time}")


def process_single_parameter(sample_num, thresh_adj_mult, overlap_thresh, transition_thresh, data_dir, output_root_dir, sigma_value=10, min_obj_size=5000, window_size=10, max_workers=None):
    
    """
    Process a single sample with a single threshold multiplier value
    """
    
    print(f"\n ==== Processing Sample {sample_num}, Threshold Multiplier {thresh_adj_mult:.1f}, Telenuclear Overlap Ratio {overlap_thresh:.2f}, Perinuclear Transition Overlap Ratio {transition_thresh:.2f} ====")
    
    # Paths for this sample
    sample_dir = os.path.join(data_dir, str(sample_num))
    raw_path = os.path.join(sample_dir, "raw.npz")
    instance_path = os.path.join(sample_dir, "instance_masks.npz")
    
    # Check if files exist
    if not (os.path.exists(raw_path) and os.path.exists(instance_path)):
        print(f"\n Error: Required files not found for sample {sample_num}.")
        return
    
    # Load data
    print(f"\n Loading data for sample {sample_num}...")
    raw_data = np.load(raw_path, mmap_mode='r')['arr_0']
    instance_data = np.load(instance_path, mmap_mode='r')['arr_0']

    # Create output directory for this sample
    sample_output_dir = os.path.join(output_root_dir, str(sample_num))
    os.makedirs(sample_output_dir, exist_ok=True)
    
    # Create output directory for this parameter set
    param_dir_name = f"perinuclear_segmentations"
    param_output_dir = os.path.join(sample_output_dir, param_dir_name)
    os.makedirs(param_output_dir, exist_ok=True)
    
    # Output path for results
    output_path = os.path.join(param_output_dir, f".npz")
    
    # Process with these parameters
    process_perinuclear_mask(
        raw_arr=raw_data,
        instance_arr=instance_data,
        output_path=output_path,
        sigma_value=sigma_value,
        thresh_adj_mult=thresh_adj_mult,
        min_obj_size=min_obj_size,
        overlap_thresh=overlap_thresh,
        transition_thresh=transition_thresh,
        window_size=window_size,
        max_workers=max_workers
    )
    
    print(f"\n Completed processing for sample {sample_num}, thresh_adj_mult={thresh_adj_mult:.1f}, tele_overlap_thresh={overlap_thresh:.2f}, peri_overlap_thresh={transition_thresh:.2f}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a single sample with a single parameter value')
    parser.add_argument('--data_dir', type=str, required=True, help='Root directory containing sample folders')
    parser.add_argument('--output_dir', type=str, required=True, help='Root directory for output results')
    parser.add_argument('--sample', type=int, required=True, help='Sample number to process')
    parser.add_argument('--thresh_adj_mult', type=float, required=True, help='Threshold adjustment multiplier value')
    parser.add_argument('--sigma', type=float, default=10, help='Sigma value for Gaussian filter')
    parser.add_argument('--min_obj_size', type=int, default=5000, help='Minimum object size to keep')
    parser.add_argument('--transition_thresh', type=float, required=True, help='Overlap threshold for region classification of transitional perinuclear mitochondria')
    parser.add_argument('--overlap_thresh', type=float, required=True, help='Overlap threshold for region classification of telenuclear mitochondria')
    parser.add_argument('--window_size', type=int, default=10, help='Window size for temporal averaging')
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
        transition_thresh=args.transition_thresh,
        overlap_thresh=args.overlap_thresh,
        window_size=args.window_size,
        max_workers=args.max_workers
    )
