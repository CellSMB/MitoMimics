# Mitochondrial Dynamics Microscopy Data Simulator (MitoDynaMicSim)


## Introduction

The Mitochondrial Dynamics Microscopy Data Simulator (MitoDynaMicSim) is designed to simulate temporal mitochondrial dynamics-based microscopy data. MitoDynaMicSim employs a Python-based rendering pipeline to generate synthetic widefield microscopy datasets of mitochondrial dynamics events. Our algorithmic toolset provides researchers a means for testing and validating AI-based models, providing both ease of use and flexibility.

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

To install MitoDynaMicSim, follow these steps:

1. Clone the repository: `git clone https://github.com/aidanpcquinn/mito_sim_pack.git`
2. Navigate to the project directory: `cd mito_sim_pack`
3. Install the required dependencies: `conda env create -f environment.yml`

## Using the Simulation

To use MitoDynaMicSim, follow these steps:

1. Open a terminal activate the conda environment with `conda activate mito_sim`
2. Run the simulation script: `python sim.py`
3. Specify simulation parameters by modifying the `sim.py` file directly.
   - JSON file configuration will be added soon.
5. Wait for the simulation to complete.
   - The user can vary the timescale indefinitely by changing the `save_data` or `sim_length_seconds` variables.
6. On simulation completion, generated data will automatically save in the output directory: `./out_stack`.

## Viewing Saved Data

1. See `view_output.ipynb`
2. Demo output at `./out_stack/test_state_output.pkl`
