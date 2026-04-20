
<p align="center">
  <img src="docs/assets/MitoMimics_Logo.png" alt="MitoMimics Logo">
</p>

<p align="center">
  <em>
    Synthetic Microscopy Timelapse Data Generation for Zero-Annotation AI Mitochondrial Segmentation and Tracking
  </em>
</p>

---
<div align="center">
  <video src="https://github.com/user-attachments/assets/f6d1f509-1c07-4afc-9459-7284797b4c50" 
    width="100%" 
    controls="controls" 
    autoplay="autoplay" 
    muted="muted" 
    loop="loop" 
    style="max-width:100%;">
  </video>
</div>

## Introduction
https://github.com/user-attachments/assets/f0d4397d-b33e-489a-8c3f-bf089cde3c9a

MitoMimics is designed to simulate temporal mitochondrial dynamics-based microscopy data. MitoMimics employs a Python-based rendering pipeline to generate synthetic widefield microscopy datasets of mitochondrial dynamics events. Our algorithmic toolset provides researchers a means for testing and validating AI-based models, providing both ease of use and flexibility.

<div align="center">
  <video src="https://github.com/user-attachments/assets/f0d4397d-b33e-489a-8c3f-bf089cde3c9a" 
    width="100%" 
    controls="controls" 
    autoplay="autoplay" 
    muted="muted" 
    loop="loop" 
    style="max-width:100%;">
  </video>
</div>

## Features

- Simulates mitochondrial motility, fission, and fusion in 2D space.
- Generates synthetic time-lapse microscopy data.
- Generates synthetic instance masks along with a JSON file that registers instances between frames, which can be used to validate segmentation and/or tracking models.
- Allows customization of simulation parameters, such as:
  - Mitochondrial Density
  - Mitochondrial Speed
  - Fission and Fusion Rates
  - Simulation Time Scale
  - Dimensions of the Simulation Space
- Allows customization of microscopy parameters, such as:
  - Point Spread Function (2D Gaussian PSF)
  - Noise (Background Offset, Camera Readout Noise, Poisson Noise)

## Installation

To install MitoMimics, follow these steps:

1. Clone the repository: `git clone https://github.com/aidanpcquinn/mito_sim_pack.git`
2. Navigate to the project directory: `cd MitoMimics/envs`
3. Install the suitable environment:
    - generic: `conda env create -f environment.yml`
    - m-series mac: `conda env create -f environment_mSeriesMac.yml`
4. Activate the conda environment with `conda activate mitodynamicsim` 

## Using the Simulation

To use MitoMimics for dataset generation, follow these steps:

1. Navigate to the MitoMimicsGeneration folder and run the simulation script: `python sim.py --seed 42`
   - This generates the underlying simulation of the mitochondria morphology and dynamics
   - Parameters can be modified in `parameters.yaml` or with `python parameter_edit_gui.py`
   - The files are saved in `MitoMimicsGeneration/sim_output/42`
   - batch scripts for running multiple jobs `gen_multi_batch_sim.sh`
2. Run the rendering script with `python renderer.py --seed 42 --cupy True --gpu 0`
   - This renders the simulation into the synthetic microscopy stacks
   - Parameters can be modified in `parameters_render.yaml` or with `python parameter_edit_gui.py`
   - The files are saved in `MitoMimicsGeneration/render_output/42`
   - batch scripts for running multiple jobs `gen_multi_batch_render.sh`
3. To view the simulated data, run `python sim_napari_viewer.py --loc render_output/42`
   - The user can vary the timescale indefinitely by changing the `save_data` or `sim_length_seconds` variables.

## Training With Simulated Data

1. Download a processed synthetic training dataset (`Training_Data.tar.gz`) from https://zenodo.org/uploads/19603477
2. Install Umamba (https://github.com/bowang-lab/U-Mamba) and follow instructions for dataset preperation and training
   - We trained with `nnUNetv2_train DATSET_ID 3d_fullres 0 -tr nnUNetTrainerUMambaEnc -num_gpus 4` and otherwise default parameters
3. Note: you will need to process and train on a dataset for both fullmasks and centerlines
4. To use your own tiff stacks, see `2_1_raw_to_intermediary_demo.ipynb` and `2_2_intermediary_prepro.ipynb` in `/RealDataProcessing`
  
## Inference and Post Processing

1. Download demo unlabaled photogentle sequences (`Unlabelled_Photogentle_Sequences.tar.gz`) from https://zenodo.org/uploads/19603477
2. Run inference with both trained models on real dataset
   - `nnUNetv2_predict -i raw_unlabaled_photogentle_sequences/ -o fullmask_out_location/ -d DATASET_ID -c 3d_fullres -f 0 -tr nnUNetTrainerUMambaEnc --disable_tta`
3. Run Post processing scripts `3_1 to 5_3` in `/RealDataProcessing`. Use `python 3_1...py --help` to view arguments
   - post processing needs the raw nifti stacks from step 1, and the outputs of both segmentation models in 2 

## Citation

## License

## Contact
For any questions, feedback, or potential collaborations, please contact the original authors of MitoMimics: 

- **Aidan Quinn** 

  📧 [aidanpcquinn@gmail.com](mailto:aidanpcquinn@gmail.com)

  🐙 Github: [@aidanpcquinn](https://github.com/aidanpcquinn)

- **Volkan Ozcoban** 

  📧 [volkanozcoban1@gmail.com](mailto:volkanozcoban1@gmail.com)

  🐙 Github: [@VolkanOzcoban](https://github.com/VolkanOzcoban)

Please include “MitoMimics” in the subject line when contacting us.
